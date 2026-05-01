"""parse_book_toc.py — Standalone TOC extraction tester.

Pulls a book's pages from ChromaDB, runs the TOC parser, detects the
page offset, and prints the result. Mirrors the rquery.py / deep_query.py
debug-tool pattern.

Usage:
    python parse_book_toc.py "CAT28008_Wild_Life.pdf"
    python parse_book_toc.py --list                    # show available PDFs
    python parse_book_toc.py --all                     # run against every PDF
    python parse_book_toc.py --chroma <path> "<filename>"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Silence noisy loggers
import logging
logging.basicConfig(level=logging.WARNING)


def _c(text: str, color: str) -> str:
    """ANSI color wrapper. Disabled if output is redirected."""
    if not sys.stdout.isatty():
        return text
    codes = {
        "bold": "\033[1m", "dim": "\033[2m",
        "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
        "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
    }
    return f"{codes.get(color, '')}{text}\033[0m"


def _default_chroma_path() -> str:
    """Pick the chroma DB to read.

    Default: the Desktop runtime copy where the real data lives.
    Override with --chroma.
    """
    desktop = Path("C:/Users/you/OneDrive/Desktop/talon-assistant/data/chroma_db")
    if desktop.exists():
        return str(desktop)
    # Fallback: settings.json
    for name in ("config/settings.json", "config/settings.example.json"):
        p = Path(name)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg.get("memory", {}).get("chroma_path", "data/chroma_db")
    return "data/chroma_db"


def _load_pages_for(filename: str, chroma_path: str) -> dict[int, str]:
    """Pull all chunks for one filename from ChromaDB and group by page idx.

    Strips VISION blocks so we work with raw extracted text only — VISION
    descriptions can hallucinate TOC-shaped text and pollute the parse.
    """
    import chromadb
    client = chromadb.PersistentClient(path=chroma_path)
    col = client.get_collection("talon_documents")

    res = col.get(
        where={"filename": filename},
        include=["documents", "metadatas"],
        limit=10000,
    )
    by_page: dict[int, list[tuple[int, str]]] = {}
    for doc, meta in zip(res.get("documents", []), res.get("metadatas", [])):
        page_num = meta.get("page_number")
        if page_num is None:
            continue
        sub = meta.get("sub_chunk", 0)
        # Strip VISION block — keep only RAW TEXT
        text = doc
        if "RAW TEXT:" in text:
            text = text.split("RAW TEXT:", 1)[1]
        # Drop the [VISION: ...] header if it remained at the top
        if text.lstrip().startswith("[VISION:"):
            end = text.find("]")
            if end != -1:
                text = text[end + 1:]
        by_page.setdefault(page_num, []).append((sub, text.strip()))

    # Concatenate sub-chunks in order
    pages: dict[int, str] = {}
    for page_idx, parts in by_page.items():
        parts.sort(key=lambda t: t[0])
        pages[page_idx] = "\n\n".join(p for _, p in parts if p)
    return pages


def _list_pdfs(chroma_path: str) -> list[tuple[str, int, int]]:
    """Return [(filename, chunk_count, max_page_idx), ...] for PDFs."""
    import chromadb
    client = chromadb.PersistentClient(path=chroma_path)
    col = client.get_collection("talon_documents")

    counts: dict[str, int] = {}
    max_pages: dict[str, int] = {}
    offset = 0
    batch = 5000
    while True:
        res = col.get(include=["metadatas"], limit=batch, offset=offset)
        metas = res.get("metadatas", [])
        if not metas:
            break
        for m in metas:
            fn = m.get("filename", "")
            counts[fn] = counts.get(fn, 0) + 1
            pn = m.get("page_number")
            if pn is not None:
                max_pages[fn] = max(max_pages.get(fn, -1), pn)
        offset += batch
        if len(metas) < batch:
            break

    pdfs = [
        (fn, counts[fn], max_pages[fn])
        for fn in counts
        if fn.endswith(".pdf") and fn in max_pages
    ]
    return sorted(pdfs, key=lambda t: -t[1])


def _print_one_book(filename: str, chroma_path: str, verbose: bool) -> None:
    from core.document_index import (
        parse_toc, detect_page_offset, resolve_pdf_pages
    )

    pages = _load_pages_for(filename, chroma_path)
    if not pages:
        print(_c(f"  No pages found for {filename}", "red"))
        return

    toc_pages, entries = parse_toc(pages)
    if not toc_pages:
        print(_c(f"  {filename}: no TOC detected", "yellow"))
        return

    offset, confidence, n_sample = detect_page_offset(entries, pages)
    resolved = resolve_pdf_pages(entries, offset)

    if confidence >= 0.7:
        conf_color = "green"
    elif confidence >= 0.3:
        conf_color = "yellow"
    else:
        conf_color = "red"
    conf_str = _c(f"{confidence:.0%}", conf_color)

    print(_c(f"\n=== {filename} ===", "bold"))
    print(f"  TOC pages (PDF idx): {toc_pages}")
    print(f"  Entries parsed:       {len(entries)}")
    print(f"  Page offset:          {offset}  "
          f"(pdf_idx = printed_page + {offset})")
    print(f"  Offset confidence:    {conf_str} "
          f"({int(confidence * n_sample)}/{n_sample} sample hits)")
    if confidence < 0.3:
        print(_c(f"  ⚠ low confidence — offset may be wrong", "yellow"))

    if verbose:
        print()
        print(_c("  Entries:", "dim"))
        for e in resolved:
            indent = "    " * (e.level + 1)
            print(f"{indent}{_c(f'p{e.page_printed:>3}', 'cyan')} "
                  f"→ {_c(f'idx{e.page_pdf:>3}', 'magenta')} "
                  f"{e.title}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the TOC parser against books in ChromaDB.",
    )
    parser.add_argument("filename", nargs="?",
                        help="Filename (e.g. CAT28008_Wild_Life.pdf). "
                        "Omit with --list or --all.")
    parser.add_argument("--list", action="store_true",
                        help="List available PDFs in the DB and exit.")
    parser.add_argument("--all", action="store_true",
                        help="Run against every PDF in the DB (summary only).")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print every TOC entry, not just summary.")
    parser.add_argument("--chroma", type=str, default=None,
                        help="Override path to chroma_db.")
    args = parser.parse_args()

    chroma = args.chroma or _default_chroma_path()
    print(_c(f"Chroma DB: {chroma}", "dim"))

    if args.list:
        pdfs = _list_pdfs(chroma)
        print(_c(f"\nPDFs with page metadata: {len(pdfs)}\n", "bold"))
        for fn, n, max_p in pdfs:
            print(f"  {n:>5} chunks, {max_p+1:>4} pages   {fn}")
        return

    if args.all:
        pdfs = _list_pdfs(chroma)
        print(_c(f"\nRunning parser against {len(pdfs)} PDFs...", "bold"))
        for fn, _n, _p in pdfs:
            _print_one_book(fn, chroma, verbose=args.verbose)
        return

    if not args.filename:
        parser.error("filename required (or use --list / --all)")
    _print_one_book(args.filename, chroma, verbose=args.verbose)


if __name__ == "__main__":
    main()
