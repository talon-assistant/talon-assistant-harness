"""Resume builder — selection-only tailoring from a pre-written bullet library.

Fabrication-proof design:
    - Bullets are NEVER generated. Only picked by index from the library.
    - Claude CLI returns JSON with selected indices per section.
    - Library at ~/OneDrive/Desktop/talon-assistant/docs/resumelibrary.md is
      the single source of truth for content. Read fresh every call.
    - Phase 1 output: markdown preview only. DOCX/PDF come in Phase 2.

Public API:
    ResumeLibrary(path).parse()  -> structured sections
    ResumeSelector(library).pick(job_description, jd_title, caps) -> dict
    render_preview(library, selection, jd_meta) -> str  (markdown)
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# Default library path (PII, outside repo)
DEFAULT_LIBRARY_PATH = (
    Path.home() / "OneDrive" / "Desktop" / "talon-assistant" / "docs" / "resumelibrary.md"
)

# Per-section bullet caps for a 2-page resume targeting leadership roles.
DEFAULT_CAPS: dict[str, int] = {
    "leadership_scope": 2,     # competencies line items
    "career_highlights": 2,    # summary callouts above role history
    "amherst": 6,
    "welldyne": 5,
    "cognizant": 3,
    "abercrombie": 4,
    "oarnet": 3,
    "talon": 4,
    "teaching": 0,             # optional, off by default
    "ic_library": 0,           # only for senior IC targets
}

# H2 header substring -> slug. First hit wins.
_HEADER_SLUGS: list[tuple[str, str]] = [
    ("leadership scope", "leadership_scope"),
    ("amherst", "amherst"),
    ("welldyne", "welldyne"),
    ("cognizant", "cognizant"),
    ("abercrombie", "abercrombie"),
    ("oarnet", "oarnet"),
    ("career highlights", "career_highlights"),
    ("ai development project", "talon"),
    ("teaching", "teaching"),
    ("ic bullet library", "ic_library"),
]

# Sections we parse but never feed to the selector.
_SKIP_SLUGS = {"earlier_career", "corrections_ledger"}


@dataclass
class Section:
    slug: str
    header: str
    context: str = ""                        # italics/prose above the bullets
    bullets: list[str] = field(default_factory=list)
    # Only used for the talon section, which has ### subsections
    subsections: dict[str, list[str]] = field(default_factory=dict)


class ResumeLibrary:
    """Parser for the bullet library markdown file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_LIBRARY_PATH
        self.sections: dict[str, Section] = {}
        self._raw = ""

    def parse(self) -> "ResumeLibrary":
        if not self.path.exists():
            raise FileNotFoundError(f"Resume library not found: {self.path}")
        self._raw = self.path.read_text(encoding="utf-8")
        self._parse_sections()
        return self

    def _slug_for(self, header: str) -> str | None:
        h = header.lower()
        if "corrections ledger" in h:
            return "corrections_ledger"
        if "earlier career" in h:
            return "earlier_career"
        for needle, slug in _HEADER_SLUGS:
            if needle in h:
                return slug
        return None

    def _parse_sections(self) -> None:
        lines = self._raw.splitlines()
        current: Section | None = None
        current_sub: str | None = None   # for ### subsections (talon)
        context_buf: list[str] = []
        in_bullets = False

        def flush_context() -> None:
            nonlocal context_buf
            if current and not in_bullets and context_buf:
                txt = " ".join(s.strip() for s in context_buf if s.strip())
                if txt and not current.context:
                    current.context = txt
            context_buf = []

        def commit_current() -> None:
            nonlocal current, current_sub
            if current:
                flush_context()
                if current.slug not in _SKIP_SLUGS:
                    self.sections[current.slug] = current
            current = None
            current_sub = None

        for raw in lines:
            line = raw.rstrip()

            # H2 = new section
            if line.startswith("## "):
                commit_current()
                header = line[3:].strip()
                slug = self._slug_for(header)
                if slug is None:
                    current = None
                    continue
                current = Section(slug=slug, header=header)
                current_sub = None
                in_bullets = False
                continue

            if current is None:
                continue

            # H3 (only meaningful inside talon: Taglines / Bullets / Closing lines)
            if line.startswith("### "):
                flush_context()
                sub = line[4:].strip().lower()
                if "tagline" in sub:
                    current_sub = "taglines"
                elif "closing" in sub:
                    current_sub = "closing"
                elif "bullet" in sub:
                    current_sub = "bullets"
                else:
                    current_sub = sub
                current.subsections.setdefault(current_sub, [])
                in_bullets = False
                continue

            # Hyphen bullet
            m = re.match(r"^-\s+(.*)$", line)
            if m:
                in_bullets = True
                text = m.group(1).strip()
                if not text:
                    continue
                if current.slug == "talon" and current_sub:
                    current.subsections.setdefault(current_sub, []).append(text)
                else:
                    current.bullets.append(text)
                continue

            # Horizontal rule ends nothing on its own; section end comes from next H2
            if line.strip() == "---":
                continue

            # Otherwise it's prose; only capture before the first bullet as context
            if not in_bullets and line.strip():
                context_buf.append(line.strip())

        commit_current()

        # Promote talon subsection bullets into the main bullets list too, so the
        # selector can pick from them uniformly.
        talon = self.sections.get("talon")
        if talon and not talon.bullets:
            talon.bullets = list(talon.subsections.get("bullets", []))

    # ── Convenience accessors ───────────────────────────────────────────────

    def get(self, slug: str) -> Section | None:
        return self.sections.get(slug)

    def bullets(self, slug: str) -> list[str]:
        s = self.sections.get(slug)
        return s.bullets if s else []

    def to_selector_payload(self) -> dict[str, Any]:
        """Structured form fed to the Claude CLI selector.

        Each bullet carries a stable (slug, index) id. Nothing is rewritten.
        """
        payload: dict[str, Any] = {"sections": []}
        for slug, section in self.sections.items():
            payload["sections"].append({
                "slug": slug,
                "header": section.header,
                "context": section.context,
                "bullets": [
                    {"id": f"{slug}:{i}", "index": i, "text": b}
                    for i, b in enumerate(section.bullets)
                ],
            })
        # Talon subsections (taglines, closing) are picked separately.
        talon = self.sections.get("talon")
        if talon:
            payload["talon_extras"] = {
                "taglines": [
                    {"id": f"talon.tagline:{i}", "index": i, "text": t}
                    for i, t in enumerate(talon.subsections.get("taglines", []))
                ],
                "closing": [
                    {"id": f"talon.closing:{i}", "index": i, "text": c}
                    for i, c in enumerate(talon.subsections.get("closing", []))
                ],
            }
        return payload


# ─────────────────────────────────────────────────────────────────────────────
# Selector
# ─────────────────────────────────────────────────────────────────────────────

_SELECTOR_SYSTEM = """You are tailoring a resume by SELECTING pre-written bullets from a library. You do not write, rewrite, or paraphrase. You only pick which bullet indices best fit the target job description.

Selection rules:
1. For each section, return a list of integer indices from that section's bullets array. Never exceed the per-section cap.
2. Prefer bullets that directly echo skills, tools, frameworks, or outcomes named in the job description.
3. Preserve chronological storytelling: the oldest roles get fewer bullets, the most recent get more, unless the JD clearly favors older work.
4. If a section has no relevant bullets, return an empty list.
5. Also pick exactly ONE talon tagline index and ONE talon closing index (from talon_extras).
6. Output ONLY valid JSON matching the schema. No preamble, no commentary, no markdown fences.

Output schema:
{
  "picks": { "<slug>": [<int>, ...], ... },
  "talon_tagline": <int>,
  "talon_closing": <int>,
  "rationale": "<one or two sentences on the overall angle>"
}
"""


@dataclass
class Selection:
    picks: dict[str, list[int]]
    talon_tagline: int | None
    talon_closing: int | None
    rationale: str
    raw: str = ""


class ResumeSelector:
    """Calls Claude CLI to pick bullet indices for a specific JD."""

    def __init__(self, library: ResumeLibrary, *, timeout: int = 180) -> None:
        self.library = library
        self.timeout = timeout

    def pick(
        self,
        *,
        job_title: str,
        company: str,
        job_description: str,
        caps: dict[str, int] | None = None,
    ) -> Selection:
        caps = {**DEFAULT_CAPS, **(caps or {})}
        payload = self.library.to_selector_payload()

        # Trim bullets that aren't in allowed sections (cap == 0)
        allowed_sections = [
            s for s in payload["sections"]
            if caps.get(s["slug"], 0) > 0
        ]

        prompt_obj = {
            "target": {
                "company": company,
                "title": job_title,
                "job_description": job_description[:6000],
            },
            "caps": {s["slug"]: caps.get(s["slug"], 0) for s in allowed_sections},
            "sections": allowed_sections,
            "talon_extras": payload.get("talon_extras", {}),
        }

        prompt = (
            _SELECTOR_SYSTEM
            + "\n\nLIBRARY AND TARGET (JSON):\n"
            + json.dumps(prompt_obj, indent=2)
            + "\n\nReturn the JSON object now."
        )

        claude_bin = shutil.which("claude")
        if not claude_bin:
            raise RuntimeError("Claude CLI not found in PATH.")

        result = subprocess.run(
            [claude_bin, "-p", "--output-format", "text"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=self.timeout,
            cwd=str(Path.home()),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Claude CLI failed: {result.stderr[:300] or result.stdout[:300]}"
            )

        raw = result.stdout.strip()
        data = _extract_json(raw)
        if not data:
            raise RuntimeError(f"Selector returned no JSON. Raw: {raw[:300]}")

        picks_raw = data.get("picks", {}) or {}
        picks: dict[str, list[int]] = {}
        for slug, idxs in picks_raw.items():
            if not isinstance(idxs, list):
                continue
            cap = caps.get(slug, 0)
            clean: list[int] = []
            section = self.library.get(slug)
            n = len(section.bullets) if section else 0
            for x in idxs:
                try:
                    i = int(x)
                except (TypeError, ValueError):
                    continue
                if 0 <= i < n and i not in clean:
                    clean.append(i)
                if len(clean) >= cap:
                    break
            if clean:
                picks[slug] = clean

        return Selection(
            picks=picks,
            talon_tagline=_safe_int(data.get("talon_tagline")),
            talon_closing=_safe_int(data.get("talon_closing")),
            rationale=str(data.get("rationale", "")).strip(),
            raw=raw,
        )


def _safe_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a text blob."""
    if not text:
        return None
    # Strip markdown fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    # Fall back: find the first balanced object
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Preview renderer
# ─────────────────────────────────────────────────────────────────────────────

# Display order matches the target resume template.
_PREVIEW_ORDER = [
    "leadership_scope",
    "career_highlights",
    "amherst",
    "welldyne",
    "cognizant",
    "abercrombie",
    "oarnet",
    "talon",
    "teaching",
    "ic_library",
]


def render_preview(
    library: ResumeLibrary,
    selection: Selection,
    *,
    company: str,
    job_title: str,
    job_url: str = "",
) -> str:
    """Render the selection as a markdown preview. Phase 1 output."""
    out: list[str] = []
    out.append(f"# Tailored Resume Preview — {company} / {job_title}")
    out.append("")
    if job_url:
        out.append(f"**Source:** {job_url}")
    if selection.rationale:
        out.append(f"**Angle:** {selection.rationale}")
    out.append("")
    out.append("---")
    out.append("")

    for slug in _PREVIEW_ORDER:
        picks = selection.picks.get(slug, [])
        section = library.get(slug)
        if not section:
            continue

        # Talon gets special treatment: tagline, bullets, closing
        if slug == "talon":
            if not picks and selection.talon_tagline is None:
                continue
            out.append(f"## {section.header}")
            taglines = section.subsections.get("taglines", [])
            closings = section.subsections.get("closing", [])
            if selection.talon_tagline is not None and 0 <= selection.talon_tagline < len(taglines):
                out.append(f"*{taglines[selection.talon_tagline]}*")
                out.append("")
            for i in picks:
                if 0 <= i < len(section.bullets):
                    out.append(f"- {section.bullets[i]}")
            if selection.talon_closing is not None and 0 <= selection.talon_closing < len(closings):
                out.append("")
                out.append(closings[selection.talon_closing])
            out.append("")
            continue

        if not picks:
            continue
        out.append(f"## {section.header}")
        if section.context:
            out.append(f"*{section.context}*")
            out.append("")
        for i in picks:
            if 0 <= i < len(section.bullets):
                out.append(f"- {section.bullets[i]}")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def render_selection_notes(
    library: ResumeLibrary,
    selection: Selection,
    *,
    company: str,
    job_title: str,
) -> str:
    """Sidecar explaining what was picked and why, for the output folder."""
    out: list[str] = []
    out.append(f"# Selection Notes — {company} / {job_title}")
    out.append("")
    if selection.rationale:
        out.append(f"**Angle:** {selection.rationale}")
        out.append("")

    total = sum(len(v) for v in selection.picks.values())
    out.append(f"**Total bullets picked:** {total}")
    out.append("")

    for slug in _PREVIEW_ORDER:
        picks = selection.picks.get(slug, [])
        section = library.get(slug)
        if not section or not picks:
            continue
        cap = DEFAULT_CAPS.get(slug, 0)
        out.append(f"## {section.header}  ({len(picks)}/{cap})")
        for i in picks:
            if 0 <= i < len(section.bullets):
                out.append(f"- [{slug}:{i}] {section.bullets[i]}")
        out.append("")

    talon = library.get("talon")
    if talon:
        if selection.talon_tagline is not None:
            taglines = talon.subsections.get("taglines", [])
            if 0 <= selection.talon_tagline < len(taglines):
                out.append(f"**Talon tagline:** {taglines[selection.talon_tagline]}")
        if selection.talon_closing is not None:
            closings = talon.subsections.get("closing", [])
            if 0 <= selection.talon_closing < len(closings):
                out.append(f"**Talon closing:** {closings[selection.talon_closing]}")
        out.append("")

    return "\n".join(out).rstrip() + "\n"
