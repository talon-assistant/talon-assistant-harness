"""core/document_index.py — TOC and (eventually) Index parsing.

Pure functions over page-text dicts. No I/O, no DB.

Caller decides where the page text comes from (fitz at ingestion time, or
ChromaDB during backfill) and where to persist results. This keeps the
parser cheap to test in isolation.

Convention for page numbering:
    pdf_page_idx = printed_page + page_offset

So offset = 0 means "printed page 1 lives at PDF index 1" (one cover page
before content). Offset = -1 means no front matter at all. Offset > 0
means more front matter (credits, TOC, foreword, etc.) before printed
page 1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Regex for one TOC line.
#
# Matches:
#   "Clean the Blood ................................... 4"
#   "Mundane.............................................9"
#   "credits ....................................................................... 5"
#   "Branching Out ..................................................... 6"
#
# Notes:
#   - Title is non-greedy so it stops at the first dot/space-leader run.
#   - Leader must be 2+ dots/whitespace; we filter further for sanity.
#   - Page is 1-4 digits, must be at end of line.
# ---------------------------------------------------------------------------
TOC_LINE_RE = re.compile(
    r'^(?P<indent>\s*)'
    r'(?P<title>.+?)'
    r'\s*'
    r'(?P<leader>[\.\s]{2,})'
    r'\s*'
    r'(?P<page>\d{1,4})'
    r'\s*$'
)


@dataclass
class TocEntry:
    title: str
    level: int = 0
    # PDF target: page_printed is the printed page number; page_pdf is the
    # resolved PDF page index after offset detection.
    page_printed: int | None = None
    page_pdf: int | None = None
    # EPUB target: chapter_idx is the spine position of the chapter that
    # contains this section. NULL for PDFs.
    chapter_idx: int | None = None


# ---------------------------------------------------------------------------
# Line-level parsing
# ---------------------------------------------------------------------------

def parse_toc_lines(text: str) -> list[TocEntry]:
    """Parse TOC entries from a block of text (one or more pages joined).

    Filters out lines that match the regex but aren't real TOC entries:
      - Title with no letters at all (page numbers in headers, etc.)
      - Leader that's mostly whitespace with too few dots (prose with
        a trailing number)
      - Page number out of range
    """
    entries: list[TocEntry] = []
    seen: set[tuple[str, int]] = set()  # dedup across joined pages

    for line in text.splitlines():
        m = TOC_LINE_RE.match(line)
        if not m:
            continue

        title = m.group("title").strip()
        if not title or not re.search(r"[A-Za-z]", title):
            continue

        # Reject pure-noise titles (just punctuation + a few chars)
        if len(re.sub(r"[^A-Za-z]", "", title)) < 2:
            continue

        leader = m.group("leader")
        # Real TOC leaders either have 2+ dots OR are 5+ chars wide whitespace
        # (some books use spaces only when right-aligning).
        if leader.count(".") < 2 and len(leader.replace(".", "")) < 5:
            continue

        try:
            page_printed = int(m.group("page"))
        except ValueError:
            continue
        if page_printed < 1 or page_printed > 9999:
            continue

        # Indentation (rarely useful in fitz-extracted text since layout
        # collapses, but we capture it for the rare case it survives).
        indent_chars = len(m.group("indent").expandtabs(4))
        level = min(indent_chars // 2, 5)

        key = (title.lower(), page_printed)
        if key in seen:
            continue
        seen.add(key)

        entries.append(TocEntry(
            title=title, page_printed=page_printed, level=level
        ))

    return entries


# ---------------------------------------------------------------------------
# Page-level detection
# ---------------------------------------------------------------------------

def detect_toc_pages(pages: dict[int, str], scan_first_n: int = 25,
                     min_entries_per_page: int = 5) -> list[int]:
    """Return sorted PDF page indices that look like TOC pages.

    Scans only the first `scan_first_n` pages of the doc — TOCs always
    live in the front matter. A page qualifies if at least
    `min_entries_per_page` lines on it match the TOC pattern.
    """
    toc_pages: list[int] = []
    for idx in sorted(pages.keys()):
        if idx >= scan_first_n:
            break
        text = pages[idx]
        if not text:
            continue
        entries = parse_toc_lines(text)
        if len(entries) >= min_entries_per_page:
            toc_pages.append(idx)
    return toc_pages


def parse_toc(pages: dict[int, str]) -> tuple[list[int], list[TocEntry]]:
    """Detect TOC pages and parse all entries from them.

    Args:
        pages: dict of pdf_page_idx → raw text

    Returns:
        (toc_page_indices, entries) — entries have page_pdf=None until
        resolve_pdf_pages() is called with the detected offset.
    """
    toc_pages = detect_toc_pages(pages)
    if not toc_pages:
        return [], []

    # Concatenate all detected TOC pages so multi-page TOCs work as a unit.
    combined = "\n".join(pages[i] for i in toc_pages if pages.get(i))
    entries = parse_toc_lines(combined)
    return toc_pages, entries


# ---------------------------------------------------------------------------
# Offset detection
# ---------------------------------------------------------------------------

def detect_page_offset(entries: list[TocEntry], pages: dict[int, str],
                       sample_size: int = 10,
                       offset_range: tuple[int, int] = (-5, 25)
                       ) -> tuple[int, float, int]:
    """Find the offset such that pdf_page_idx = printed_page + offset.

    Strategy: pick distinctive TOC entries (longer titles), and for each
    candidate offset, count how many of those titles actually appear on
    the page they'd map to. Highest hit count wins.

    Returns:
        (offset, confidence, sample_used) where confidence is hits/samples
        (0.0 - 1.0). Caller should treat low confidence (<0.3) as
        "offset unknown" and skip storing it.
    """
    if not entries:
        return 0, 0.0, 0

    # Distinctive titles: long, contain uppercase letters, not generic
    # words like "Introduction" that might appear on many pages.
    GENERIC = {"introduction", "credits", "contents", "index", "appendix",
               "foreword", "preface", "acknowledgments"}
    candidates = [
        e for e in entries
        if len(e.title) >= 8
        and e.title.lower() not in GENERIC
    ]
    candidates.sort(key=lambda e: -len(e.title))
    sample = candidates[:sample_size] if candidates else entries[:sample_size]
    if not sample:
        return 0, 0.0, 0

    best_offset = 0
    best_hits = -1

    for offset in range(offset_range[0], offset_range[1] + 1):
        hits = 0
        for entry in sample:
            target_idx = entry.page_printed + offset
            page_text = pages.get(target_idx, "")
            if not page_text:
                continue
            # Match on first 4 distinctive words of the title (case-insensitive).
            # Section headings get rendered larger than body text so they
            # usually appear verbatim in the extracted page text.
            words = entry.title.split()[:4]
            distinctive = " ".join(words).lower()
            if len(distinctive) >= 5 and distinctive in page_text.lower():
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_offset = offset

    confidence = best_hits / len(sample) if sample else 0.0
    return best_offset, confidence, len(sample)


def resolve_pdf_pages(entries: list[TocEntry], offset: int) -> list[TocEntry]:
    """Apply offset to fill in page_pdf on each entry.

    Returns a new list; does not mutate input.
    """
    return [
        TocEntry(
            title=e.title,
            page_printed=e.page_printed,
            level=e.level,
            page_pdf=e.page_printed + offset
                     if e.page_printed is not None else None,
        )
        for e in entries
    ]


# ---------------------------------------------------------------------------
# EPUB TOC extraction (uses ebooklib's structured toc + spine)
# ---------------------------------------------------------------------------

def extract_epub_toc(book) -> list[TocEntry]:
    """Walk an ebooklib Book's TOC and return flat entries with chapter_idx.

    EPUBs ship with a structured table of contents (NCX or nav doc) that
    ebooklib exposes as `book.toc`. Each entry has a title and an href.
    The href points at a chapter file (possibly with a #fragment); we map
    the chapter file to its position in the spine to get a chapter_idx
    that matches our chunk metadata.

    TOC entries whose href doesn't resolve to any chapter (rare — usually
    cover.xhtml or notes pages excluded from the spine) are skipped.
    """
    try:
        import ebooklib
    except ImportError:
        return []

    # Build href → chapter_idx map. Chapters are walked in spine order.
    href_to_idx: dict[str, int] = {}
    for idx, item in enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT)):
        # ebooklib uses file_name, possibly with directory prefix
        href_to_idx[item.file_name] = idx
        # Also strip directory and try just the basename — toc hrefs are
        # often relative-from-OEBPS while spine items include the prefix
        bare = item.file_name.split("/")[-1]
        href_to_idx.setdefault(bare, idx)

    out: list[TocEntry] = []
    seen: set[tuple[str, int]] = set()

    def _walk(items, level: int) -> None:
        for entry in items:
            if isinstance(entry, tuple) and len(entry) == 2:
                # (Section/Link, [children])
                node, children = entry
                _emit(node, level)
                _walk(children, level + 1)
            else:
                _emit(entry, level)

    def _emit(node, level: int) -> None:
        title = (getattr(node, "title", "") or "").strip()
        href = getattr(node, "href", "") or ""
        if not title or not href:
            return
        # Strip #fragment — we resolve to the chapter file only
        href_file = href.split("#", 1)[0]
        idx = href_to_idx.get(href_file)
        if idx is None:
            # Try basename match as fallback
            bare = href_file.split("/")[-1]
            idx = href_to_idx.get(bare)
        if idx is None:
            return
        key = (title.lower(), idx)
        if key in seen:
            return
        seen.add(key)
        out.append(TocEntry(
            title=title,
            level=level,
            chapter_idx=idx,
        ))

    _walk(book.toc or [], 0)
    return out
