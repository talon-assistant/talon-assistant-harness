"""core/toc_store.py — SQLite persistence for parsed TOCs.

Two tables, both keyed by filename:

    book_metadata   — one row per book. Tracks offset, confidence,
                      TOC presence, and ingestion timestamp. Lets the
                      backfill skip books already processed.
    book_toc        — one row per TOC entry. Each entry knows its
                      printed page number, resolved PDF page index,
                      and indentation level.

Decoupled from the parser (core/document_index.py) so the parser stays
pure. Decoupled from the rest of the talon_memory.db schema so the TOC
table can be wiped and rebuilt without touching anything else.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from core.document_index import TocEntry

import logging
log = logging.getLogger(__name__)


class TocStore:
    """Read/write parsed TOC data to a SQLite file."""

    def __init__(self, db_path: str | Path = "data/talon_book_index.db"):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS book_metadata (
                    filename          TEXT PRIMARY KEY,
                    page_offset       INTEGER NOT NULL DEFAULT 0,
                    offset_confidence REAL    NOT NULL DEFAULT 0.0,
                    has_toc           INTEGER NOT NULL DEFAULT 0,
                    has_index         INTEGER NOT NULL DEFAULT 0,
                    toc_pages         TEXT,
                    entry_count       INTEGER NOT NULL DEFAULT 0,
                    indexed_at        TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS book_toc (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename      TEXT    NOT NULL,
                    title         TEXT    NOT NULL,
                    title_lower   TEXT    NOT NULL,
                    page_printed  INTEGER NOT NULL,
                    page_pdf      INTEGER NOT NULL,
                    level         INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_book_toc_filename
                    ON book_toc(filename);

                CREATE INDEX IF NOT EXISTS idx_book_toc_title_lower
                    ON book_toc(filename, title_lower);
            """)
            # Idempotent migrations for the EPUB path. ALTER ADD COLUMN
            # raises if the column exists; we swallow that to keep startup
            # safe across versions.
            for stmt in (
                "ALTER TABLE book_toc ADD COLUMN chapter_idx INTEGER",
                "ALTER TABLE book_metadata ADD COLUMN source_type TEXT "
                "NOT NULL DEFAULT 'pdf'",
            ):
                try:
                    c.execute(stmt)
                except sqlite3.OperationalError:
                    pass

    # ── Writers ───────────────────────────────────────────────────────────

    def store_book(
        self,
        filename: str,
        entries: list[TocEntry],
        toc_pages: list[int] | None = None,
        page_offset: int = 0,
        offset_confidence: float = 0.0,
        source_type: str = "pdf",
    ) -> int:
        """Store a parsed book. Replaces any existing rows for this filename.

        For PDFs, entries should have page_printed and page_pdf set.
        For EPUBs, entries should have chapter_idx set.

        Returns the number of entries written.
        """
        with self._conn() as c:
            # Replace any existing data for this book
            c.execute("DELETE FROM book_toc WHERE filename = ?", (filename,))
            c.execute("DELETE FROM book_metadata WHERE filename = ?", (filename,))

            c.execute(
                """INSERT INTO book_metadata
                   (filename, page_offset, offset_confidence, has_toc,
                    has_index, toc_pages, entry_count, indexed_at, source_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    filename,
                    page_offset,
                    offset_confidence,
                    1 if entries else 0,
                    0,  # has_index reserved
                    json.dumps(toc_pages or []),
                    len(entries),
                    datetime.now().isoformat(),
                    source_type,
                ),
            )

            written = 0
            for e in entries:
                # Skip entries without any usable target. PDFs need page_pdf;
                # EPUBs need chapter_idx.
                if e.page_pdf is None and e.chapter_idx is None:
                    continue
                # Use sentinels for the NOT NULL columns when the field is
                # not applicable for this source type.
                p_printed = e.page_printed if e.page_printed is not None else 0
                p_pdf = e.page_pdf if e.page_pdf is not None else -1
                c.execute(
                    """INSERT INTO book_toc
                       (filename, title, title_lower, page_printed,
                        page_pdf, level, chapter_idx)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        filename, e.title, e.title.lower(),
                        p_printed, p_pdf, e.level, e.chapter_idx,
                    ),
                )
                written += 1
            return written

    def mark_no_toc(self, filename: str, source_type: str = "pdf") -> None:
        """Record that a book was scanned and has no TOC.

        Lets backfill skip it on subsequent runs.
        """
        with self._conn() as c:
            c.execute("DELETE FROM book_toc WHERE filename = ?", (filename,))
            c.execute("DELETE FROM book_metadata WHERE filename = ?", (filename,))
            c.execute(
                """INSERT INTO book_metadata
                   (filename, has_toc, indexed_at, source_type)
                   VALUES (?, 0, ?, ?)""",
                (filename, datetime.now().isoformat(), source_type),
            )

    # ── Readers ───────────────────────────────────────────────────────────

    def has_book(self, filename: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM book_metadata WHERE filename = ?",
                (filename,),
            ).fetchone()
            return row is not None

    def get_metadata(self, filename: str) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM book_metadata WHERE filename = ?",
                (filename,),
            ).fetchone()
            return dict(row) if row else None

    def list_books(self) -> list[dict]:
        """Return all books with TOC entries, summary form."""
        with self._conn() as c:
            rows = c.execute(
                """SELECT filename, page_offset, offset_confidence,
                          has_toc, entry_count, indexed_at
                   FROM book_metadata
                   ORDER BY filename"""
            ).fetchall()
            return [dict(r) for r in rows]

    def lookup(self, filename: str, query: str,
               max_results: int = 12) -> list[dict]:
        """Search TOC entries within one book by title substring.

        Splits the query into keywords and finds entries that match all
        of them (AND-style). Returns entries sorted by:
            1. Exact-phrase match first
            2. Number of keywords matched
            3. Top of book (lower page) first

        Returns:
            [{title, page_printed, page_pdf, level}, ...]
        """
        if not query.strip():
            return []
        query_lower = query.lower().strip()
        # Tokenize, drop short/stopword tokens
        STOPWORDS = {"the", "a", "an", "of", "in", "on", "at", "to",
                     "and", "or", "for", "with", "from", "by"}
        tokens = [
            t for t in query_lower.replace(",", " ").replace("-", " ").split()
            if len(t) > 2 and t not in STOPWORDS
        ]
        if not tokens:
            tokens = [query_lower]

        with self._conn() as c:
            # Pull all entries for the book; rank in Python (small set per book)
            rows = c.execute(
                "SELECT title, title_lower, page_printed, page_pdf, level, "
                "chapter_idx "
                "FROM book_toc WHERE filename = ?",
                (filename,),
            ).fetchall()

        # Pre-compile word-boundary regexes for each token. Word-boundary
        # matching prevents short tokens (e.g. "run") from spuriously
        # matching inside longer words (e.g. "Grunt").
        token_res = [
            re.compile(rf"\b{re.escape(t)}\b") for t in tokens
        ]
        # For 2+ token queries, require at least 2 tokens to hit (or a
        # full phrase match). Single-token queries pass with 1 hit. This
        # filters out cases like query="smuggler's run" matching an
        # entry called "Run Compensation" via just the "run" token.
        min_required = min(len(tokens), 2)

        scored: list[tuple[int, int, int, str, dict]] = []
        for r in rows:
            title_lower = r["title_lower"]
            n_hits = sum(1 for tre in token_res if tre.search(title_lower))
            phrase_match = 1 if query_lower in title_lower else 0
            if not phrase_match and n_hits < min_required:
                continue
            # Sort key for "earlier in the book": prefer chapter_idx for
            # EPUBs, page_pdf for PDFs. Sentinel -1 from PDFs without
            # printed pages still sorts low which is acceptable.
            order_key = (r["chapter_idx"] if r["chapter_idx"] is not None
                         else r["page_pdf"])
            scored.append((
                -phrase_match,        # phrase match first (lower sort key)
                -n_hits,              # then more hits first
                order_key,            # then earlier in the book
                r["title"],           # stable tiebreaker so dicts never compared
                {
                    "title": r["title"],
                    "page_printed": (r["page_printed"]
                                     if r["page_printed"] else None),
                    "page_pdf": (r["page_pdf"] if r["page_pdf"] >= 0 else None),
                    "chapter_idx": r["chapter_idx"],
                    "level": r["level"],
                },
            ))
        scored.sort()
        return [d for *_, d in scored[:max_results]]

    def all_books_with_toc(self) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT filename FROM book_metadata WHERE has_toc = 1"
            ).fetchall()
            return [r["filename"] for r in rows]
