"""Threat digest talent — surface newly-added CISA KEV entries as a feed.

Pulls the CISA Known Exploited Vulnerabilities (KEV) catalog and reports only
what is NEW since the last run, so it reads like a feed ("here's what dropped")
rather than a re-dump of the whole catalog. Seen CVE IDs are persisted to
``data/threat_digest_state.json``.

On the first run (no state yet) it shows the most recent N entries and seeds
the state, so every run after that is delta-only.

The KEV catalog is structured JSON from CISA, so entries are formatted from
their own fields directly — no LLM summarisation, which keeps it deterministic
and removes any prompt-injection surface from the fetched text.

Optional config in ``config/threat_digest.json``::

    {
      "kev_url": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
      "max_items": 20,
      "first_run_recent": 15,
      "watchlist": ["microsoft", "fortinet", "cisco", "ivanti"]
    }

``watchlist`` is optional: vendor/product keywords that flag entries as
relevant to your stack. v1 shows every new KEV entry regardless; the watchlist
just marks the ones that match.
"""
import os
import json

from talents.base import BaseTalent, TalentResult

import logging
log = logging.getLogger(__name__)


class ThreatDigestTalent(BaseTalent):
    name = "threat_digest"
    description = (
        "Report newly-added CISA Known Exploited Vulnerabilities (KEV) as a "
        "security threat feed. Shows what is new since the last check, with the "
        "CVE, affected vendor/product, a known-ransomware flag, and the CISA "
        "remediation due date. Use for 'what new vulnerabilities dropped' / "
        "'threat digest' / 'KEV' style requests."
    )
    keywords = [
        "kev", "threat digest", "threat feed", "exploited vulnerabilities",
        "exploited vulns", "cisa kev", "new cves", "vulnerability feed",
    ]
    examples = [
        "threat digest",
        "what's new in KEV",
        "kev digest",
        "any new exploited vulnerabilities",
        "show me the security threat feed",
        "what new CVEs dropped",
    ]
    priority = 53

    _STATE_FILE = os.path.join("data", "threat_digest_state.json")
    _CONFIG_PATH = os.path.join("config", "threat_digest.json")
    _DEFAULT_KEV_URL = (
        "https://www.cisa.gov/sites/default/files/feeds/"
        "known_exploited_vulnerabilities.json"
    )

    def __init__(self):
        super().__init__()
        self._kev_url = self._DEFAULT_KEV_URL
        self._max_items = 20
        self._first_run_recent = 15
        self._watchlist: list[str] = []
        self._load_config()

    # ── Config ─────────────────────────────────────────────────────

    def _load_config(self) -> None:
        try:
            with open(self._CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            cfg = {}
        self._kev_url = cfg.get("kev_url") or self._DEFAULT_KEV_URL
        try:
            self._max_items = max(1, int(cfg.get("max_items", 20)))
            self._first_run_recent = max(1, int(cfg.get("first_run_recent", 15)))
        except (TypeError, ValueError):
            self._max_items, self._first_run_recent = 20, 15
        self._watchlist = [str(w).lower() for w in cfg.get("watchlist", []) if w]

    def get_config_schema(self) -> dict:
        return {
            "fields": [
                {"key": "max_items", "label": "Max new items per digest",
                 "type": "int", "default": 20, "min": 1, "max": 100},
                {"key": "first_run_recent", "label": "Items to show on first run",
                 "type": "int", "default": 15, "min": 1, "max": 100},
                {"key": "watchlist", "label": "Stack watchlist (vendor/product keywords)",
                 "type": "list", "default": []},
            ]
        }

    # ── Routing ────────────────────────────────────────────────────

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    # ── Execution ──────────────────────────────────────────────────

    def execute(self, command: str, context: dict) -> TalentResult:
        try:
            vulns = self._fetch_kev()
        except Exception as e:
            return TalentResult(
                success=False,
                response=f"Couldn't reach the CISA KEV feed right now: {e}",
                actions_taken=[], spoken=False,
            )

        if not vulns:
            return TalentResult(
                success=False,
                response="The KEV feed returned no entries.",
                actions_taken=[], spoken=False,
            )

        seen = self._load_seen()
        first_run = not seen

        # Newest first by date added.
        vulns_sorted = sorted(
            vulns, key=lambda v: v.get("dateAdded", ""), reverse=True)

        if first_run:
            new_entries = vulns_sorted[: self._first_run_recent]
            header = (
                f"First run, so here are the {len(new_entries)} most recent KEV "
                f"entries. From now on you'll only see what's new since the last "
                f"check."
            )
        else:
            new_entries = [
                v for v in vulns_sorted if v.get("cveID") not in seen
            ][: self._max_items]
            if not new_entries:
                # Re-seed in case the catalog changed shape, then report quiet.
                self._save_seen({v.get("cveID") for v in vulns if v.get("cveID")})
                return TalentResult(
                    success=True,
                    response="No new KEV entries since the last check. You're current.",
                    actions_taken=[{"action": "threat_digest", "new": 0}],
                    spoken=False,
                )
            plural = "y" if len(new_entries) == 1 else "ies"
            header = f"{len(new_entries)} new KEV entr{plural} since the last check:"

        body = self._format(new_entries)
        self._save_seen({v.get("cveID") for v in vulns if v.get("cveID")})

        watch_hits = sum(1 for v in new_entries if self._matches_watchlist(v))
        actions = [{"action": "threat_digest", "new": len(new_entries),
                    "watchlist_hits": watch_hits}]
        return TalentResult(
            success=True,
            response=f"{header}\n\n{body}",
            actions_taken=actions,
            spoken=False,
        )

    # ── KEV fetch + format ─────────────────────────────────────────

    def _fetch_kev(self) -> list[dict]:
        import requests
        resp = requests.get(
            self._kev_url, headers={"User-Agent": "Talon/1.0"}, timeout=25)
        resp.raise_for_status()
        return resp.json().get("vulnerabilities", [])

    def _matches_watchlist(self, v: dict) -> bool:
        if not self._watchlist:
            return False
        haystack = f"{v.get('vendorProject', '')} {v.get('product', '')}".lower()
        return any(w in haystack for w in self._watchlist)

    def _format(self, entries: list[dict]) -> str:
        lines = []
        for v in entries:
            cve = v.get("cveID", "?")
            vendor = (v.get("vendorProject") or "").strip()
            product = (v.get("product") or "").strip()
            name = (v.get("vulnerabilityName") or "").strip()
            due = (v.get("dueDate") or "").strip()
            ransom = (v.get("knownRansomwareCampaignUse") or "").strip().lower()

            flags = []
            if ransom == "known":
                flags.append("RANSOMWARE")
            if self._matches_watchlist(v):
                flags.append("WATCHLIST")
            flag_str = f"  [{' / '.join(flags)}]" if flags else ""

            label = " ".join(p for p in (vendor, product) if p)
            detail = f"{label}: {name}" if (label and name) else (name or label)

            entry = f"- {cve}{flag_str}"
            if detail:
                entry += f"\n    {detail}"
            if due:
                entry += f"\n    Patch by {due}"
            lines.append(entry)
        return "\n".join(lines)

    # ── State (seen CVE IDs) ───────────────────────────────────────

    def _load_seen(self) -> set:
        try:
            with open(self._STATE_FILE, encoding="utf-8") as f:
                return set(json.load(f).get("seen", []))
        except (FileNotFoundError, json.JSONDecodeError):
            return set()

    def _save_seen(self, ids: set) -> None:
        try:
            os.makedirs(os.path.dirname(self._STATE_FILE), exist_ok=True)
            with open(self._STATE_FILE, "w", encoding="utf-8") as f:
                json.dump({"seen": sorted(i for i in ids if i)}, f)
        except OSError as e:
            log.warning(f"[ThreatDigest] couldn't save state: {e}")
