"""build_toc_index.py — Populate the book TOC SQLite from ChromaDB chunks.

Walks one or more books in ChromaDB, parses TOC pages out of the stored
raw text, detects the page offset, and writes the parsed entries into
SQLite. No re-ingestion needed — this is a backfill.

Usage:
    python build_toc_index.py "Wild_Life.pdf"            # one book
    python build_toc_index.py --books-file books.txt     # batch from file
    python build_toc_index.py --all                       # every PDF in DB
    python build_toc_index.py --force "Book.pdf"         # re-process even if stored
    python build_toc_index.py --db data/test.db ...      # alternate output DB
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import logging
logging.basicConfig(level=logging.WARNING)

# Reuse the loader from parse_book_toc — keeps page-text extraction logic
# in one place.
from parse_book_toc import _load_pages_for, _list_pdfs, _default_chroma_path, _c

from core.document_index import (
    parse_toc, detect_page_offset, resolve_pdf_pages
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
    args = parser.parse_args()

    chroma = args.chroma or _default_chroma_path()
    print(_c(f"Chroma DB: {chroma}", "dim"))
    print(_c(f"Output DB: {args.db}", "dim"))

    store = TocStore(args.db)

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
