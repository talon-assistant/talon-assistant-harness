"""build_toc_index.py — Populate the book TOC SQLite from ChromaDB chunks.

PDFs: walks each book's chunks, parses TOC pages out of stored raw text,
detects page offset, writes the parsed entries to SQLite. No re-ingestion.

EPUBs: needs the original .epub source files (chunks lost the structured
TOC during ingestion). Match local files to ingested filenames by EPUB
metadata (title + creator) since renames are common.

Usage:
    # PDFs
    python build_toc_index.py "Wild_Life.pdf"
    python build_toc_index.py --all
    python build_toc_index.py --force "Book.pdf"

    # EPUBs (matching dry-run)
    python build_toc_index.py --epub-dir <path>

    # EPUBs (actually write after reviewing the dry-run)
    python build_toc_index.py --epub-dir <path> --apply
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import logging
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# Reuse the loader from parse_book_toc — keeps page-text extraction logic
# in one place.
from parse_book_toc import _load_pages_for, _list_pdfs, _default_chroma_path, _c

from core.document_index import (
    parse_toc, detect_page_offset, resolve_pdf_pages, extract_epub_toc
)
from core.toc_store import TocStore


def process_book(filename: str, store: TocStore, chroma_path: str,
                 force: bool = False, verbose: bool = False) -> dict:
    """Parse one book and write to the store. Returns a result summary."""
    if not force and store.has_book(filename):
        return {"filename": filename, "status": "skipped (already indexed)"}

    pages = _load_pages_for(filename, chroma_path)
    if not pages:
        return {"filename": filename, "status": "no chunks found"}

    toc_pages, entries = parse_toc(pages)
    if not toc_pages:
        store.mark_no_toc(filename)
        return {"filename": filename, "status": "no TOC", "entries": 0}

    offset, confidence, n_sample = detect_page_offset(entries, pages)
    resolved = resolve_pdf_pages(entries, offset)
    n_written = store.store_book(
        filename, resolved, toc_pages, offset, confidence
    )

    return {
        "filename": filename,
        "status": "ok",
        "entries": n_written,
        "toc_pages": toc_pages,
        "offset": offset,
        "confidence": confidence,
        "sample_size": n_sample,
    }


# ── EPUB backfill helpers ────────────────────────────────────────────────

_TOKEN_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "and", "or", "for", "to", "with",
    "by", "from", "at", "as", "is", "are", "be", "ed", "edition", "second",
    "third", "fourth", "fifth", "vol", "volume", "part", "book",
}


def _normalize_tokens(s: str) -> set[str]:
    """Lowercase, alphanumeric tokens of length >= 3, stopwords dropped."""
    return {
        t for t in re.findall(r"[a-z0-9]+", s.lower())
        if len(t) >= 3 and t not in _TOKEN_STOPWORDS
    }


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _epub_metadata_string(book) -> str:
    """Concatenate title(s) and creator(s) for matching."""
    parts: list[str] = []
    for entry in book.get_metadata("DC", "title") or []:
        if entry and entry[0]:
            parts.append(entry[0])
    for entry in book.get_metadata("DC", "creator") or []:
        if entry and entry[0]:
            parts.append(entry[0])
    return " ".join(parts)


def _list_ingested_epubs(chroma_path: str) -> list[str]:
    """Return all .epub filenames present in ChromaDB."""
    import chromadb
    client = chromadb.PersistentClient(path=chroma_path)
    col = client.get_collection("talon_documents")
    seen: set[str] = set()
    offset = 0
    batch = 5000
    while True:
        res = col.get(include=["metadatas"], limit=batch, offset=offset)
        metas = res.get("metadatas", [])
        if not metas:
            break
        for m in metas:
            fn = m.get("filename", "")
            if fn.lower().endswith(".epub"):
                seen.add(fn)
        offset += batch
        if len(metas) < batch:
            break
    return sorted(seen)


def _read_local_metadata(
    local_path: Path,
) -> tuple[set[str], str]:
    """Open an EPUB and return (token_set, metadata_string)."""
    from ebooklib import epub as _epub
    try:
        book = _epub.read_epub(str(local_path), options={"ignore_ncx": True})
    except Exception as exc:
        log.warning(f"  Could not open {local_path.name}: {exc}")
        return set(), ""
    meta_string = _epub_metadata_string(book)
    if not meta_string:
        return set(), ""
    return _normalize_tokens(meta_string), meta_string


def _assign_matches(
    local_paths: list[Path],
    ingested: list[str],
    threshold: float,
) -> tuple[list[tuple[Path, str | None, float, str]], list[str]]:
    """Greedy unique assignment: each local file gets at most one ingested
    target, each ingested entry gets at most one local file.

    Pairs are scored by Jaccard between local EPUB metadata tokens and the
    ingested filename tokens. We take the highest-scoring pair first, mark
    both as used, and repeat until no pairs above threshold remain.

    Returns:
        (matches, unused_ingested)
        matches: [(local_path, matched_ingested_or_None, score, metadata)]
                 — entries below threshold have None.
        unused_ingested: ingested filenames that no local file matched.
    """
    # Read each local file's metadata once
    local_data: list[tuple[Path, set[str], str]] = []
    for p in local_paths:
        tokens, meta = _read_local_metadata(p)
        local_data.append((p, tokens, meta))

    # Pre-tokenize ingested filenames (strip .epub extension)
    ingested_tokens = {
        fn: _normalize_tokens(fn[:-5] if fn.endswith(".epub") else fn)
        for fn in ingested
    }

    # Score every (local, ingested) pair
    pairs: list[tuple[float, int, str]] = []  # (score, local_idx, ingested_fn)
    for li, (_, tokens, _) in enumerate(local_data):
        if not tokens:
            continue
        for fn, fn_tokens in ingested_tokens.items():
            score = _jaccard(tokens, fn_tokens)
            if score > 0:
                pairs.append((score, li, fn))
    # Greedy: highest-scoring pair first
    pairs.sort(key=lambda t: -t[0])

    assigned_local: dict[int, tuple[str, float]] = {}
    assigned_ingested: set[str] = set()
    for score, li, fn in pairs:
        if li in assigned_local or fn in assigned_ingested:
            continue
        if score < threshold:
            break  # rest are worse
        assigned_local[li] = (fn, score)
        assigned_ingested.add(fn)

    matches: list[tuple[Path, str | None, float, str]] = []
    for li, (path, _tokens, meta) in enumerate(local_data):
        if li in assigned_local:
            fn, score = assigned_local[li]
            matches.append((path, fn, score, meta))
        else:
            matches.append((path, None, 0.0, meta))

    unused_ingested = [fn for fn in ingested if fn not in assigned_ingested]
    return matches, unused_ingested


def process_epub(local_path: Path, ingested_name: str,
                 store: TocStore, force: bool = False) -> dict:
    """Open one EPUB, extract TOC, store under ingested_name."""
    if not force and store.has_book(ingested_name):
        return {"filename": ingested_name, "status": "skipped (already indexed)"}

    from ebooklib import epub as _epub
    try:
        book = _epub.read_epub(str(local_path), options={"ignore_ncx": True})
    except Exception as exc:
        return {"filename": ingested_name, "status": f"error: {exc}"}

    entries = extract_epub_toc(book)
    if not entries:
        store.mark_no_toc(ingested_name, source_type="epub")
        return {"filename": ingested_name, "status": "no TOC", "entries": 0}

    n = store.store_book(ingested_name, entries, source_type="epub")
    return {
        "filename": ingested_name,
        "status": "ok",
        "entries": n,
        "local_path": str(local_path),
    }


def _load_epub_map(path: str) -> dict[str, str]:
    """Parse a manual map file. Format:

        # comments allowed
        local_filename.epub | ingested_filename.epub

    Returns: {local_filename: ingested_filename}
    """
    out: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--epub-map-file not found: {path}")
    for line_num, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "|" not in s:
            log.warning(f"  map line {line_num}: no `|` separator, skipped: {s!r}")
            continue
        local, ingested = (part.strip() for part in s.split("|", 1))
        if not local or not ingested:
            log.warning(f"  map line {line_num}: empty side, skipped")
            continue
        out[local] = ingested
    return out


def run_epub_backfill(epub_dir: Path, store: TocStore, chroma_path: str,
                      apply: bool, force: bool, threshold: float,
                      manual_map: dict[str, str] | None = None) -> None:
    epubs = sorted(epub_dir.glob("*.epub"))
    if not epubs:
        print(_c(f"No .epub files found in {epub_dir}", "red"))
        return

    print(_c(f"\nLocal EPUB files: {len(epubs)}", "bold"))
    ingested = _list_ingested_epubs(chroma_path)
    print(_c(f"Ingested EPUBs in ChromaDB: {len(ingested)}", "bold"))
    print()

    matches, unused_ingested = _assign_matches(epubs, ingested, threshold)

    # Apply manual overrides. For each map entry, find the local Path
    # and force its match to the specified ingested name. Adjusts
    # unused_ingested accordingly.
    manual_applied: list[tuple[str, str]] = []
    if manual_map:
        ingested_set = set(ingested)
        for i, (local, prev_match, score, meta) in enumerate(matches):
            override = manual_map.get(local.name)
            if not override:
                continue
            if override not in ingested_set:
                log.warning(f"  manual map: ingested '{override}' not in "
                            f"ChromaDB, skipping")
                continue
            # Update the match. Score 1.0 = manual.
            matches[i] = (local, override, 1.0, meta)
            manual_applied.append((local.name, override))
            # Free the previously assigned ingested (if any) so it can
            # surface as unused, and remove the new override from unused.
            if prev_match and prev_match in ingested_set:
                # No good way to know which OTHER local was using prev_match;
                # we just track unused below by recomputing.
                pass
        # Recompute unused after overrides
        used_ingested = {m[1] for m in matches if m[1] is not None}
        unused_ingested = [fn for fn in ingested if fn not in used_ingested]

    matched = [m for m in matches if m[1] is not None]
    unmatched = [m for m in matches if m[1] is None]

    manual_local_names = {local for local, _ in manual_applied}

    if manual_applied:
        print(_c(f"━━━ MANUAL OVERRIDES ({len(manual_applied)}) ━━━", "bold"))
        for local_name, ingested_name in manual_applied:
            print(f"  {_c('M', 'cyan')}    {local_name}")
            print(f"         → {ingested_name}")
        print()

    print(_c(f"━━━ ASSIGNED MATCHES (≥{threshold:.0%}, unique) ━━━", "bold"))
    for local, best, score, meta in matched:
        if local.name in manual_local_names:
            color, label = "cyan", "manual"
        elif score >= 0.7:
            color, label = "green", f"{score:.0%}"
        else:
            color, label = "yellow", f"{score:.0%}"
        print(f"  {_c(label, color)}  {local.name}")
        print(f"         → {best}")
        print(f"         metadata: {meta[:90]}")
    print()

    if unmatched:
        print(_c(f"━━━ UNMATCHED LOCAL FILES ━━━", "yellow"))
        for local, _best, _score, meta in unmatched:
            print(f"  {_c('?', 'red')}  {local.name}")
            print(f"         metadata: {meta[:90]}")
        print()

    if unused_ingested:
        print(_c(f"━━━ INGESTED ENTRIES WITH NO LOCAL MATCH ━━━", "yellow"))
        print(_c("(these are in ChromaDB but no local .epub mapped to them)", "dim"))
        for fn in unused_ingested:
            print(f"  {_c('?', 'yellow')}  {fn}")
        print()

    print(_c(f"Summary:", "bold"))
    print(f"  Matched:           {len(matched)}/{len(epubs)} local files")
    print(f"  Unmatched local:   {len(unmatched)} (no ingested file fit)")
    print(f"  Unmatched ingest:  {len(unused_ingested)} (no local file claimed)")

    if not apply:
        print()
        print(_c("Dry run only. Review the assignment above, then re-run "
                "with --apply to write TOC.", "cyan"))
        return

    print()
    print(_c("━━━ WRITING TOC FOR MATCHED FILES ━━━", "bold"))
    print()

    n_ok = 0
    n_no_toc = 0
    n_skip = 0
    n_err = 0
    total_entries = 0

    for local, best, score, meta in matched:
        result = process_epub(local, best, store, force=force)
        status = result["status"]
        if status == "ok":
            n_ok += 1
            total_entries += result["entries"]
            print(f"  {_c('✓', 'green')} {best}  ({result['entries']} entries)")
        elif status == "no TOC":
            n_no_toc += 1
            print(f"  {_c('-', 'yellow')} {best}  (no TOC)")
        elif status.startswith("skipped"):
            n_skip += 1
            print(f"  {_c('·', 'dim')} {best}  ({status})")
        else:
            n_err += 1
            print(f"  {_c('✗', 'red')} {best}  ({status})")

    print()
    print(_c("━" * 50, "dim"))
    print(f"  Indexed: {n_ok}  |  No TOC: {n_no_toc}  |  "
          f"Skipped: {n_skip}  |  Errors: {n_err}")
    print(f"  Total TOC entries written: {total_entries}")
    print(_c("━" * 50, "dim"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate book TOC index from ChromaDB chunks.",
    )
    parser.add_argument("filenames", nargs="*",
                        help="Book filenames to process (e.g. Wild_Life.pdf)")
    parser.add_argument("--all", action="store_true",
                        help="Process every PDF with page metadata in ChromaDB")
    parser.add_argument("--books-file", type=str,
                        help="File with one filename per line")
    parser.add_argument("--force", action="store_true",
                        help="Re-process books already in the store")
    parser.add_argument("--chroma", type=str, default=None,
                        help="ChromaDB path override")
    parser.add_argument("--db", type=str, default="data/talon_book_index.db",
                        help="SQLite output path (default: data/talon_book_index.db)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--epub-dir", type=str, default=None,
                        help="Directory of source .epub files for EPUB "
                             "backfill (matched to ingested filenames by "
                             "title+creator metadata).")
    parser.add_argument("--apply", action="store_true",
                        help="With --epub-dir: actually write the matches. "
                             "Without it the run is dry only.")
    parser.add_argument("--match-threshold", type=float, default=0.4,
                        help="Min Jaccard score to count as a confident "
                             "EPUB match (default: 0.4).")
    parser.add_argument("--epub-map-file", type=str, default=None,
                        help="File with manual EPUB mappings, one per line "
                             "as `<local_filename> | <ingested_filename>`. "
                             "Comments start with #. Mappings override "
                             "auto-matching for the listed local files.")
    args = parser.parse_args()

    chroma = args.chroma or _default_chroma_path()
    print(_c(f"Chroma DB: {chroma}", "dim"))
    print(_c(f"Output DB: {args.db}", "dim"))

    store = TocStore(args.db)

    # EPUB backfill flow takes precedence — separate path entirely
    if args.epub_dir:
        epub_dir = Path(args.epub_dir)
        if not epub_dir.is_dir():
            parser.error(f"--epub-dir not a directory: {epub_dir}")
        manual_map = _load_epub_map(args.epub_map_file) if args.epub_map_file else {}
        run_epub_backfill(
            epub_dir, store, chroma,
            apply=args.apply, force=args.force,
            threshold=args.match_threshold,
            manual_map=manual_map,
        )
        return

    # Build target list
    targets: list[str] = list(args.filenames)
    if args.books_file:
        targets += [
            line.strip()
            for line in Path(args.books_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    if args.all:
        pdfs = _list_pdfs(chroma)
        targets += [fn for fn, _, _ in pdfs]

    # Dedup, preserve order
    seen: set[str] = set()
    targets = [fn for fn in targets if not (fn in seen or seen.add(fn))]

    if not targets:
        parser.error("Specify book filenames, --all, or --books-file")

    print(_c(f"\nProcessing {len(targets)} book(s)...", "bold"))
    print()

    results = []
    for fn in targets:
        result = process_book(fn, store, chroma,
                              force=args.force, verbose=args.verbose)
        results.append(result)
        status = result["status"]
        if status == "ok":
            conf = result["confidence"]
            color = "green" if conf >= 0.7 else ("yellow" if conf >= 0.3 else "red")
            print(f"  {_c('✓', 'green')} {fn}")
            print(f"      entries: {result['entries']}, "
                  f"offset: {result['offset']}, "
                  f"confidence: {_c(f'{conf:.0%}', color)} "
                  f"({int(conf * result['sample_size'])}/{result['sample_size']})")
        elif status == "no TOC":
            print(f"  {_c('-', 'yellow')} {fn}  (no TOC detected)")
        elif status.startswith("skipped"):
            print(f"  {_c('·', 'dim')} {fn}  ({status})")
        else:
            print(f"  {_c('✗', 'red')} {fn}  ({status})")

    # Summary
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_no_toc = sum(1 for r in results if r["status"] == "no TOC")
    n_skip = sum(1 for r in results if "skipped" in r["status"])
    n_err = len(results) - n_ok - n_no_toc - n_skip
    total_entries = sum(r.get("entries", 0) for r in results)

    print()
    print(_c("━" * 50, "dim"))
    print(f"  Indexed: {n_ok}  |  No TOC: {n_no_toc}  |  "
          f"Skipped: {n_skip}  |  Errors: {n_err}")
    print(f"  Total TOC entries written: {total_entries}")
    print(_c("━" * 50, "dim"))


if __name__ == "__main__":
    main()
