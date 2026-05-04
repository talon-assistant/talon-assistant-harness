"""core/document_retriever.py — RAG retrieval pipeline for document chunks.

Extracted from core/memory.py to separate storage (CRUD) from retrieval
intelligence (multi-query expansion, RRF fusion, cross-encoder reranking,
multi-hop entity lookup).
"""

import re
import json
from core import embeddings as _emb
from core import reranker as _reranker

import logging
log = logging.getLogger(__name__)


class DocumentRetriever:
    """Retrieve and rank document chunks from a ChromaDB collection."""

    def __init__(self, docs_collection, embed_model: str, reranker_model: str,
                 toc_store_path: str = "data/talon_book_index.db"):
        self._docs = docs_collection
        self._embed_model = embed_model
        self._reranker_model = reranker_model
        self._toc_store_path = toc_store_path
        self._toc_store = None  # lazy: loaded on first TOC lookup

    # ── Public API ────────────────────────────────────────────────────────

    def get_document_context(self, query: str, explicit: bool = False,
                             alt_queries: list | None = None,
                             multi_hop: bool = False,
                             synthesis: bool = False) -> str:
        """Retrieve document chunks for RAG injection into the conversation path.

        Args:
            query:       Primary embedding query (expanded from user command).
            explicit:    If True, user explicitly asked for document search —
                         use loose distance cap (1.8) and return up to 12 chunks.
                         If False (ambient), only inject if distance <= 0.55 and
                         return at most 2 chunks.
            alt_queries: Optional list of alternate queries (synonyms / related
                         terms) whose results are unioned with the primary query
                         and deduplicated. Explicit mode only.
            multi_hop:   If True, fire a second retrieval pass using entity names
                         extracted from the top chunks (explicit mode only).
            synthesis:   If True, use explicit-grade retrieval (wide cap, RRF) but
                         skip multi-hop. For compare/list-all queries.

        Returns:
            Formatted string ready for injection, or "" if nothing qualifies.
        """
        if len(query.strip()) < 4:
            return ""

        use_explicit = explicit or synthesis
        n_results = 12 if use_explicit else 2
        max_distance = 1.8 if use_explicit else 0.55
        meta_cache: dict[str, dict] = {}  # text[:100] → full ChromaDB metadata dict

        def _run_query(q: str) -> list[tuple]:
            """Run one ChromaDB query; cache metadata; return (fn, text, dist, pg) tuples."""
            try:
                results = self._docs.query(
                    query_embeddings=_emb.embed_queries([q], self._embed_model),
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"],
                )
                if not results["documents"] or not results["documents"][0]:
                    return []
                hits = []
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    if dist <= max_distance:
                        meta_cache[doc[:100]] = meta
                        hits.append((meta.get("filename", "unknown file"), doc, dist,
                                     meta.get("page_number")))
                return hits
            except Exception:
                return []

        try:
            # ── Phase 1: gather candidates ────────────────────────────────
            all_chunks = _run_query(query)

            # Alt queries (explicit mode only) — union and deduplicate
            if use_explicit and alt_queries:
                seen: set[str] = {text[:100] for _, text, _, _pg in all_chunks}
                for aq in alt_queries:
                    if len(aq.strip()) < 4:
                        continue
                    for filename, text, dist, page_num in _run_query(aq):
                        key = text[:100]
                        if key not in seen:
                            seen.add(key)
                            all_chunks.append((filename, text, dist, page_num))

            # Text-match fallback (explicit mode only): use ChromaDB $contains
            # to pull chunks with exact keyword matches, bypassing the semantic
            # distance cutoff.  Sparse stat-block chunks embed poorly but are
            # textually exact — this guarantees they surface even when the
            # embedding distance would otherwise exclude them.
            if use_explicit:
                all_terms = [query] + (alt_queries or [])
                text_kws = sorted(
                    {w for t in all_terms for w in t.split() if len(w) > 3},
                    key=len, reverse=True,
                )[:8]
                seen_txt: set[str] = {t[:100] for _, t, _, _pg in all_chunks}
                for kw in text_kws:
                    for variant in {kw, kw.title()}:
                        try:
                            hits = self._docs.get(
                                where_document={"$contains": variant},
                                limit=6,
                                include=["documents", "metadatas"],
                            )
                            for doc, meta in zip(
                                hits.get("documents", []),
                                hits.get("metadatas", []),
                            ):
                                key = doc[:100]
                                if key not in seen_txt:
                                    seen_txt.add(key)
                                    meta_cache[key] = meta
                                    all_chunks.append(
                                        (meta.get("filename", "unknown file"), doc, 1.0,
                                         meta.get("page_number"))
                                    )
                        except Exception:
                            pass

            if not all_chunks:
                mode = "explicit" if use_explicit else "ambient"
                log.debug(f"[RAG] No chunks passed threshold "
                      f"(mode={mode}, threshold={max_distance:.2f})")
                return ""

            # ── Phase 2: ranking ──────────────────────────────────────────
            all_terms = [query] + (alt_queries or [])

            # Strip common English filler words that inflate keyword scores
            # on generic content. Only topic-specific terms should drive
            # scoring — "provide detail" shouldn't boost a credits page.
            _STOPWORDS = {
                "what", "when", "where", "which", "who", "whom", "whose",
                "that", "this", "these", "those", "there", "here",
                "have", "has", "had", "been", "being", "were", "was",
                "will", "would", "could", "should", "shall", "might",
                "does", "did", "done", "doing", "make", "made",
                "give", "gave", "given", "take", "took", "taken",
                "tell", "told", "about", "with", "from", "into",
                "much", "many", "more", "most", "some", "such",
                "also", "just", "than", "then", "only", "very",
                "like", "well", "even", "back", "over", "after",
                "each", "every", "other", "both", "same", "know",
                "provide", "detail", "details", "detailed", "explain",
                "explain", "describe", "please", "possible", "list",
                "show", "find", "help", "need", "want", "look",
                "information", "info",
                # Domain-generic: appears in every book in the collection
                "shadowrun",
            }

            keywords = set(
                w.lower() for t in all_terms for w in t.split()
                if len(w) > 3 and w.lower() not in _STOPWORDS
            )

            def _keyword_score(chunk_text: str) -> int:
                lower = chunk_text.lower()
                return sum(1 for kw in keywords if kw in lower)

            if use_explicit:
                # RRF fusion: combine semantic distance ranking with
                # keyword overlap ranking. Chunks from $contains get
                # distance = 1.0 and go into the keyword pool only.
                # Semantic chunks (distance < 1.0) go into the semantic
                # pool, BUT chunks with strong keyword overlap also
                # appear in the keyword pool so they get a double RRF
                # boost — prevents high-kw chunks at mediocre semantic
                # distance from being buried.
                semantic_pool = sorted(
                    [c for c in all_chunks if c[2] < 1.0], key=lambda x: x[2])
                keyword_pool  = sorted(
                    [c for c in all_chunks if c[2] >= 1.0],
                    key=lambda x: -_keyword_score(x[1]))

                # Promote semantic chunks with strong keyword overlap
                # into the keyword pool for a double RRF boost.
                if keywords:
                    kw_threshold = max(2, len(keywords) // 2)
                    for chunk in semantic_pool:
                        if _keyword_score(chunk[1]) >= kw_threshold:
                            keyword_pool.append(chunk)
                    keyword_pool.sort(key=lambda x: -_keyword_score(x[1]))

                all_chunks = self._rrf_fuse(semantic_pool, keyword_pool)
                all_chunks = self._jaccard_dedup(all_chunks)
            else:
                all_chunks.sort(key=lambda x: x[2])

            # Candidate pool for reranker: substantially larger than final
            # cap so the cross-encoder (our best ranker) sees enough
            # diversity. With 35K+ chunks in the collection, RRF fusion
            # of semantic + $contains can push relevant chunks past
            # position 12, so a wider pool prevents the cross-encoder
            # from missing them.
            RERANK_POOL = 20 if use_explicit else 2
            FINAL_CAP   = 8  if use_explicit else 2
            all_chunks = all_chunks[:RERANK_POOL]

            # ── Phase 2.5: cross-encoder reranking (explicit mode) ────────
            # Score each (query, chunk) pair jointly — much more accurate than
            # cosine distance, especially when query and document vocabularies
            # differ.  Only applied in explicit mode where latency tolerance is
            # higher (user explicitly asked for document search).
            #
            # min_score=-1.0: bge-reranker-base raw logits.  Clearly irrelevant
            # pairs score ≈ -10 to -3; borderline ≈ -3 to 0; relevant > 0.
            # -1.0 cuts out most noise while keeping strong marginal matches.
            RERANK_MIN_SCORE = -1.0
            # Keyword boost: cross-encoder can produce near-zero scores on
            # short queries where several chunks look loosely similar. The
            # boost adds (kw_hits / n_keywords) * 0.2 to each chunk's score,
            # promoting chunks with actual query-term presence over chunks
            # the reranker marked as "kinda related". min_score is applied
            # to the RAW reranker score so boost can't rescue true junk.
            KW_BOOST = 0.2
            if use_explicit and len(all_chunks) > 1:
                n_before = len(all_chunks)
                all_chunks = _reranker.rerank(
                    query, all_chunks, self._reranker_model,
                    top_k=FINAL_CAP, min_score=RERANK_MIN_SCORE,
                    keywords=keywords, kw_boost=KW_BOOST,
                )
                log.info(f"[RAG] Cross-encoder reranked {n_before}→{len(all_chunks)} chunks "
                      f"(min_score={RERANK_MIN_SCORE}, kw_boost={KW_BOOST})")
                if not all_chunks:
                    log.warning("[RAG] All chunks below reranker threshold — skipping injection")
                    return ""

            # ── Phase 3: source concentration + page vicinity ────────────
            # When results cluster around one document, filter out noise
            # from other books and fetch neighboring pages to ensure
            # content spanning page boundaries is fully captured.
            #
            # Pick the primary source by TOTAL keyword score, not chunk count.
            # A document with 3 highly-relevant chunks (Bestial Nature with
            # "shifter", "feathery") beats a document with 5 generic chunks
            # (Berlin Edition with "credits", "table of contents").
            if use_explicit and all_chunks:
                from collections import defaultdict
                source_kw_totals: dict[str, int] = defaultdict(int)
                source_counts: dict[str, int] = defaultdict(int)
                for fn, txt, _, _ in all_chunks:
                    source_kw_totals[fn] += _keyword_score(txt)
                    source_counts[fn] += 1

                # Pick source with highest total keyword score (ties broken by count)
                top_source = max(
                    source_kw_totals,
                    key=lambda fn: (source_kw_totals[fn], source_counts[fn]),
                )
                top_count = source_counts[top_source]

                # Only concentrate if the source has ≥2 chunks AND meaningful keyword overlap
                if top_count >= 2 and source_kw_totals[top_source] >= 4:
                    primary_chunks = [c for c in all_chunks if c[0] == top_source]
                    other_chunks = [c for c in all_chunks if c[0] != top_source]

                    # Page-vicinity: find pages adjacent to retrieved pages
                    # from the same document. Stat blocks often span 2-3 pages.
                    retrieved_pages = {pg for _, _, _, pg in primary_chunks
                                       if pg is not None}
                    vicinity_pages = set()
                    for pg in retrieved_pages:
                        vicinity_pages.update(range(max(0, pg - 1), pg + 3))
                    missing_pages = vicinity_pages - retrieved_pages
                    seen_keys: set[str] = {t[:100] for _, t, _, _ in all_chunks}

                    if missing_pages:
                        for pg in sorted(missing_pages):
                            try:
                                # Fetch up to 5 chunks per page to handle
                                # duplicates — pick the one with the most
                                # keyword hits (not longest, which may lack
                                # the relevant section).
                                hits = self._docs.get(
                                    where={"$and": [
                                        {"filename": top_source},
                                        {"page_number": pg},
                                    ]},
                                    limit=5,
                                    include=["documents", "metadatas"],
                                )
                                best_doc = None
                                best_score = -1
                                for doc, meta in zip(
                                    hits.get("documents", []),
                                    hits.get("metadatas", []),
                                ):
                                    key = doc[:100]
                                    if key not in seen_keys:
                                        score = _keyword_score(doc)
                                        if score > best_score or (
                                            score == best_score and
                                            (best_doc is None or len(doc) > len(best_doc))
                                        ):
                                            best_doc = doc
                                            best_score = score
                                if best_doc:
                                    seen_keys.add(best_doc[:100])
                                    primary_chunks.append(
                                        (top_source, best_doc, 0.9, pg)
                                    )
                            except Exception:
                                pass

                        if len(primary_chunks) > len([c for c in all_chunks if c[0] == top_source]):
                            added = len(primary_chunks) - top_count
                            log.info(f"[RAG] Page-vicinity added {added} "
                                  f"adjacent page(s) from {top_source}")

                    # Reranker-selected pages have priority over vicinity
                    # filler. If we need to cap, drop vicinity pages first.
                    # Then sort by page number for reading order.
                    reranker_pages = {pg for _, _, _, pg in primary_chunks
                                      if pg is not None and any(
                                          c[3] == pg and c[2] != 0.9
                                          for c in primary_chunks)}
                    vicinity_only = [c for c in primary_chunks
                                     if c[3] not in reranker_pages]
                    reranker_kept = [c for c in primary_chunks
                                     if c[3] in reranker_pages]

                    # Fill to FINAL_CAP: all reranker pages + vicinity to fill
                    remaining_slots = FINAL_CAP - len(reranker_kept)
                    if remaining_slots > 0:
                        # Sort vicinity by proximity to reranker pages
                        if reranker_pages:
                            mid = sum(reranker_pages) / len(reranker_pages)
                            vicinity_only.sort(
                                key=lambda c: abs((c[3] or 999) - mid))
                        final_primary = reranker_kept + vicinity_only[:remaining_slots]
                    else:
                        final_primary = reranker_kept[:FINAL_CAP]

                    # Sort by page number for reading order
                    final_primary.sort(
                        key=lambda c: c[3] if c[3] is not None else 999)

                    all_chunks = final_primary
                    remaining = FINAL_CAP - len(all_chunks)
                    if remaining > 0 and other_chunks:
                        all_chunks.extend(other_chunks[:min(2, remaining)])

                    log.info(f"[RAG] Source concentration: {top_count} from "
                          f"'{top_source}', {len(all_chunks)} total after vicinity")

            # ── Phase 4: format ───────────────────────────────────────────
            log.debug(f"[RAG] Injecting {len(all_chunks)} unique chunk(s) "
                  f"(explicit={explicit}, synthesis={synthesis}, "
                  f"best_dist={all_chunks[0][2]:.3f})")
            for _fn, _txt, _d, _pg in all_chunks:
                pg_label = f" p{_pg + 1}" if _pg is not None else ""
                log.info(f"   {_d:.3f} kw={_keyword_score(_txt)}  "
                      f"{_fn}{pg_label}  |  {_txt[:60].replace(chr(10), ' ')!r}")

            if use_explicit:
                lines = [
                    "DOCUMENT EXCERPTS (source of truth — report ALL details found):"
                ]
            else:
                lines = [
                    "The following document excerpts may be relevant — "
                    "use them if helpful, ignore if not:"
                ]
            for filename, text, dist, page_num in all_chunks:
                # Vision-ingested chunks have format:
                #   [VISION: <hallucinated description>]\n\nRAW TEXT:\n<actual content>
                # Strip VISION, use only RAW TEXT for injection.
                raw_marker = "\nRAW TEXT:\n"
                if raw_marker in text:
                    raw_part = text.split(raw_marker, 1)[1]
                    if len(raw_part.strip()) > 50:
                        text = raw_part

                # Explicit mode: pass full page content — no truncation.
                # Chunks are page-sized (~400-800 words). Truncating at
                # any fixed limit risks cutting off stat blocks that appear
                # at the tail end of a page (feathery shifter starts at
                # char ~3200 on a 4075-char page). With 16K context and
                # 8 chunks, full pages fit comfortably.
                # Ambient mode: keep brief to avoid bloating casual replies.
                if not use_explicit:
                    text = text[:800] + "..." if len(text) > 800 else text
                truncated = text
                source = (f"{filename} (page {page_num + 1})"
                          if page_num is not None else filename)
                lines.append(f"- From {source}: {truncated}")

            return "\n".join(lines) + "\n"

        except Exception as e:
            log.error(f"[RAG] Document context error: {e}")
            return ""

    # ── Public API for agentic retrieval ──────────────────────────────────

    def _get_toc_store(self):
        """Lazily open the TocStore. Returns None on any failure."""
        if self._toc_store is not None:
            return self._toc_store
        try:
            from core.toc_store import TocStore
            self._toc_store = TocStore(self._toc_store_path)
            return self._toc_store
        except Exception as e:
            log.warning(f"[RAG] TocStore unavailable: {e}")
            return None

    def lookup_in_toc(self, filename: str, query: str,
                      max_results: int = 8) -> list[dict]:
        """Look up TOC entries in one book matching `query`.

        Returns entries with chunk_ids that the agent can pass to read_page.
        Builds a page-style chunk_id (`p{N+1}`) for PDF entries and a
        chapter-style chunk_id (`ch{N}`) for EPUB entries.

        Returns:
            [{chunk_id, title, page_printed, page_pdf, chapter_idx,
              level, filename}, ...]
            sorted by relevance (phrase match > keyword count > earlier in book).
            Empty list if no TOC indexed for this book or no matches.
        """
        store = self._get_toc_store()
        if store is None:
            return []
        matches = store.lookup(filename, query, max_results=max_results)
        out: list[dict] = []
        for m in matches:
            if m.get("chapter_idx") is not None:
                chunk_id = self._build_chunk_id(
                    filename, m["chapter_idx"], 0,
                    position_type="chapter")
            elif m.get("page_pdf") is not None:
                chunk_id = self._build_chunk_id(
                    filename, m["page_pdf"], 0,
                    position_type="page")
            else:
                continue  # malformed entry — skip
            out.append({
                "chunk_id": chunk_id,
                "title": m["title"],
                "page_printed": m.get("page_printed"),
                "page_pdf": m.get("page_pdf"),
                "chapter_idx": m.get("chapter_idx"),
                "level": m["level"],
                "filename": filename,
            })
        return out

    def has_toc(self, filename: str) -> bool:
        """Is there a parsed TOC available for this book?"""
        store = self._get_toc_store()
        if store is None:
            return False
        meta = store.get_metadata(filename)
        return bool(meta and meta.get("has_toc"))

    # Cap on how many pages a single read_section call can join, so a
    # 100-page chapter doesn't blow up the agent's context budget.
    _MAX_SECTION_PAGES = 8

    def _find_section_for_position(
        self, filename: str, position: int, position_type: str
    ) -> tuple[dict, dict | None] | None:
        """Find the TOC entry containing `position` and the next entry.

        Returns (current_entry, next_entry_or_None) or None if no TOC
        section covers the position.
        """
        store = self._get_toc_store()
        if store is None:
            return None
        try:
            entries = store.entries_for_book(filename)
        except Exception:
            return None
        if not entries:
            return None

        # Filter to entries of the matching position_type. PDF entries
        # have page_pdf set; EPUB entries have chapter_idx set.
        if position_type == "chapter":
            candidates = [e for e in entries if e["chapter_idx"] is not None]
        else:
            candidates = [e for e in entries if e["page_pdf"] is not None]
        if not candidates:
            return None

        # Find the latest entry whose position is <= target position.
        # That entry is the "section" containing the target.
        current = None
        current_idx = -1
        for i, e in enumerate(candidates):
            if e["position"] <= position:
                current = e
                current_idx = i
            else:
                break
        if current is None:
            return None

        next_entry = (candidates[current_idx + 1]
                      if current_idx + 1 < len(candidates) else None)
        return (current, next_entry)

    def read_section(self, chunk_id: str) -> dict | None:
        """Read the entire TOC section that `chunk_id` belongs to.

        Mirrors how a human navigates by structure: "give me the whole
        chapter on Combat Spells" rather than "give me page 134."
        Section boundaries come from the TOC: section starts at its
        TOC entry's page, ends one page before the next entry. For
        EPUBs the section is the single chapter (already atomic).

        Capped at _MAX_SECTION_PAGES pages joined to protect context.
        """
        parsed = self._parse_chunk_id(chunk_id)
        if not parsed:
            return None
        filename, position_type, position, _sub = parsed
        if position is None:
            return None

        # EPUB: chapter is the section. Just return the chapter chunk.
        if position_type == "chapter":
            return self.get_chunk_by_id(chunk_id)

        section_info = self._find_section_for_position(
            filename, position, position_type)
        if not section_info:
            return None
        current, next_entry = section_info

        start = current["position"]
        # End: page before next section, or +5 pages if no next entry
        if next_entry is not None:
            end = next_entry["position"] - 1
        else:
            end = start + 5
        # Cap section length
        end = min(end, start + self._MAX_SECTION_PAGES - 1)
        if end < start:
            end = start

        # Join all pages in [start, end]
        parts: list[tuple[int, dict]] = []
        for pg in range(start, end + 1):
            cid = self._build_chunk_id(filename, pg, 0,
                                       position_type=position_type)
            chunk = self.get_chunk_by_id(cid)
            if chunk:
                parts.append((pg, chunk))
        if not parts:
            return None

        lines: list[str] = [f"[Section: {current['title']!r}]\n"]
        pages_included: list[int] = []
        for pg, chunk in parts:
            lines.append(f"\n[• page {pg + 1}]\n{chunk.get('text', '')}")
            pages_included.append(pg + 1)
        if next_entry and end == start + self._MAX_SECTION_PAGES - 1 \
                and end < next_entry["position"] - 1:
            lines.append(
                f"\n[Section continues to page {next_entry['position']} — "
                f"truncated at {self._MAX_SECTION_PAGES} pages. Call "
                f"read_nearby for more.]"
            )

        return {
            "chunk_id": chunk_id,
            "source": filename,
            "page": position + 1,
            "section_title": current["title"],
            "section_page_start": start + 1,
            "section_page_end": end + 1,
            "next_section_title": next_entry["title"] if next_entry else None,
            "pages_included": pages_included,
            "text": "".join(lines).strip(),
            "spans": len(parts),
        }

    def next_section(self, chunk_id: str) -> dict | None:
        """Jump to the start of the section after `chunk_id`'s section.

        Reads the first page (PDF) or chapter chunk (EPUB) of the next
        TOC entry. The agent uses this for "what comes after this part?"
        questions and for forward navigation through structured docs.
        """
        parsed = self._parse_chunk_id(chunk_id)
        if not parsed:
            return None
        filename, position_type, position, _sub = parsed
        if position is None:
            return None

        section_info = self._find_section_for_position(
            filename, position, position_type)
        if not section_info:
            return None
        _current, next_entry = section_info
        if next_entry is None:
            return None

        next_pos = next_entry["position"]
        next_cid = self._build_chunk_id(filename, next_pos, 0,
                                        position_type=position_type)
        chunk = self.get_chunk_by_id(next_cid)
        if not chunk:
            return None

        return {
            "chunk_id": next_cid,
            "source": filename,
            "page": next_pos + 1 if position_type == "page" else None,
            "chapter": next_pos if position_type == "chapter" else None,
            "section_title": next_entry["title"],
            "text": chunk.get("text", ""),
        }

    def get_neighboring_chunks(self, chunk_id: str,
                               before: int = 1, after: int = 1) -> dict | None:
        """Read pages adjacent to `chunk_id` and return them joined.

        Mirrors how a human flips forward/back from a known location.
        Walks position_type-aware: ±N pages for PDFs, ±N chapters for
        EPUBs. Skips out-of-range positions. Joins each page's text in
        natural order with section markers.

        Returns:
            {chunk_id, source, text, pages_included: [...], ...} or None
            if the chunk_id is malformed or yields no chunks.
        """
        parsed = self._parse_chunk_id(chunk_id)
        if not parsed:
            return None
        filename, position_type, position, _sub = parsed
        if position is None:
            return None

        # Clamp the range so we don't ask for negative pages
        before = max(0, int(before))
        after = max(0, int(after))
        if before == 0 and after == 0:
            return self.get_chunk_by_id(chunk_id)

        parts: list[tuple[int, dict]] = []
        for offset in range(-before, after + 1):
            target_pos = position + offset
            if target_pos < 0:
                continue
            target_id = self._build_chunk_id(
                filename, target_pos, 0, position_type=position_type)
            chunk = self.get_chunk_by_id(target_id)
            if chunk:
                parts.append((offset, chunk))

        if not parts:
            return None

        # Build joined text with explicit page markers so the agent can
        # tell where one page ends and the next begins.
        lines: list[str] = []
        pages_included: list[int] = []
        chapters_included: list[int] = []
        for offset, chunk in parts:
            tag = "←" if offset < 0 else ("→" if offset > 0 else "•")
            if position_type == "chapter":
                label = f"{tag} chapter {position + offset}"
                chapters_included.append(position + offset)
            else:
                label = f"{tag} page {position + offset + 1}"
                pages_included.append(position + offset + 1)
            lines.append(f"\n[{label}]\n{chunk.get('text', '')}")
        joined = "".join(lines).strip()

        # Use the centerpoint chunk's metadata as the canonical source
        center = next((c for off, c in parts if off == 0),
                      parts[0][1])
        return {
            "chunk_id": chunk_id,
            "source": center.get("source", filename),
            "page": center.get("page"),
            "chapter": center.get("chapter"),
            "text": joined,
            "pages_included": pages_included,
            "chapters_included": chapters_included,
            "spans": len(parts),
        }

    def follow_index_reference(self, filename: str, query: str,
                               index_text: str,
                               max_followups: int = 2) -> list[dict]:
        """When `index_text` is a back-of-book index, follow page refs.

        Extracts the meaningful subject words from the user's query
        (stripping question prefixes like "in book X what is..."), looks
        them up in the index, applies the book's stored page_offset, and
        fetches the resolved chunks. Returns a list (possibly empty) of
        followed chunks with their printed/pdf page numbers.
        """
        from core.document_index import extract_index_pages

        # ── Strip question-style filler so the index lookup uses just
        # the subject. The user typically says something like
        # "in shadowrun berlin edition what is a mana bolt" — only
        # "mana bolt" should drive the index lookup.
        _FILLER_WORDS = {
            "what", "who", "where", "when", "why", "how", "which",
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "of", "in", "on", "at", "to", "for", "with", "from", "by",
            "and", "or", "but", "as", "if", "do", "does", "did",
            "tell", "me", "about", "explain", "describe", "show",
            "i", "you", "he", "she", "it", "we", "they",
            "edition", "book", "chapter", "section", "page",
            # Common book/series tokens that shouldn't drive index lookup
            "shadowrun", "applied", "cryptography", "berlin",
        }
        cleaned_words = [
            w for w in re.findall(r"[a-zA-Z][\w'-]*", query.lower())
            if len(w) >= 3 and w not in _FILLER_WORDS
        ]
        if not cleaned_words:
            return []

        # Build the term list. Prefer multi-word phrases first (more
        # specific), then individual words, then a concat variant for
        # compound nouns the author might have indexed as one word.
        terms: list[str] = []
        if 2 <= len(cleaned_words) <= 4:
            phrase = " ".join(cleaned_words)
            terms.append(phrase)
            concat = "".join(cleaned_words)
            if concat and concat != phrase:
                terms.insert(0, concat)  # try concat first
        terms.extend(cleaned_words)

        printed_pages = extract_index_pages(index_text, terms)
        if not printed_pages:
            return []

        # Look up the book's printed→pdf offset from the TOC store
        store = self._get_toc_store()
        offset = 0
        if store is not None:
            meta = store.get_metadata(filename)
            if meta:
                offset = meta.get("page_offset", 0) or 0

        results: list[dict] = []
        seen_chunks: set[str] = set()
        for printed_page in printed_pages[:max_followups]:
            pdf_idx = printed_page + offset
            chunk_id = self._build_chunk_id(filename, pdf_idx, 0)
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            chunk = self.get_chunk_by_id(chunk_id)
            if chunk and chunk.get("text"):
                results.append({
                    "chunk_id": chunk_id,
                    "printed_page": printed_page,
                    "pdf_idx": pdf_idx,
                    "text": chunk["text"],
                })
        return results

    def list_sources(self) -> list[dict]:
        """Return a list of all source files with chunk counts.

        Used by DeepSearchAgent so the LLM can see what documents exist
        before deciding where to search. The `has_toc` flag tells the
        agent which books support `lookup_in_toc` for direct navigation.

        Returns:
            [{filename, chunks, has_toc}, ...] sorted by chunk count descending.
        """
        try:
            # Page through metadata — ChromaDB has a bind-variable cap that
            # breaks at ~30k+ chunks if we ask for everything at once.
            counts: dict[str, int] = {}
            offset = 0
            batch = 5000
            while True:
                results = self._docs.get(
                    include=["metadatas"], limit=batch, offset=offset,
                )
                metas = results.get("metadatas", [])
                if not metas:
                    break
                for meta in metas:
                    fn = meta.get("filename", "unknown")
                    counts[fn] = counts.get(fn, 0) + 1
                offset += batch
                if len(metas) < batch:
                    break
            store = self._get_toc_store()
            toc_books: set[str] = set()
            if store is not None:
                try:
                    toc_books = set(store.all_books_with_toc())
                except Exception:
                    pass
            # Sort: TOC-indexed books first (the agent can navigate them
            # via lookup_in_toc), then by chunk count descending. This
            # keeps high-leverage books at the top of the agent's view
            # even when an EPUB happens to have more chunks.
            return sorted(
                [{"filename": fn, "chunks": n, "has_toc": fn in toc_books}
                 for fn, n in counts.items()],
                key=lambda r: (not r["has_toc"], -r["chunks"]),
            )
        except Exception as e:
            log.error(f"[RAG] list_sources failed: {e}")
            return []

    def get_raw_chunks(self, query: str, top_k: int = 8,
                       source_filter: str | None = None,
                       apply_reranker: bool = True) -> list[dict]:
        """Agentic entry point: return raw chunks with metadata.

        Args:
            query:          Search query string.
            top_k:          Max chunks to return.
            source_filter:  If set, restrict search to this filename only.
            apply_reranker: If True, apply cross-encoder reranking.

        Returns:
            [{chunk_id, source, page, text, preview, dist, kw_score, rerank_score?}, ...]
            where chunk_id is "{filename}::p{page}::s{sub_chunk}" for stable reference.
        """
        if len(query.strip()) < 2:
            return []

        # Build keyword set (lowercase, stopword-filtered) for scoring
        _STOPWORDS = {
            "what", "when", "where", "which", "who", "whom", "whose",
            "that", "this", "these", "those", "there", "here",
            "have", "has", "had", "been", "being", "were", "was",
            "will", "would", "could", "should", "shall", "might",
            "does", "did", "done", "make", "made", "give", "gave",
            "tell", "told", "about", "with", "from", "into",
            "much", "many", "more", "most", "some", "such",
            "also", "just", "than", "then", "only", "very",
            "provide", "detail", "details", "explain", "describe",
            "please", "possible", "list", "show", "find", "help",
            "information", "info", "shadowrun",
        }
        keywords = {
            w.lower() for w in query.split()
            if len(w) > 3 and w.lower() not in _STOPWORDS
        }

        # Compound-word variant: PDFs sometimes extract space-separated
        # terms as one word ("Mana Bolt" → "Manabolt"). When the query has
        # 2-3 alphabetic words, also search for the concatenated form so
        # those chunks make the candidate pool. The original phrase still
        # drives reranker scoring.
        query_variants: list[str] = [query]
        words_for_variant = query.split()
        if 2 <= len(words_for_variant) <= 3 and all(
                w.replace("-", "").replace("'", "").isalpha()
                for w in words_for_variant):
            concat = "".join(words_for_variant).lower()
            # Add the concatenation as long as it's distinct from the
            # original query string (which it always is when the query
            # has spaces — the earlier check compared concat against the
            # space-stripped query, which made this branch a no-op).
            if concat and concat != query.lower():
                query_variants.append(concat)
                if len(concat) > 3 and concat not in _STOPWORDS:
                    keywords.add(concat)

        def _kw_score(txt: str) -> int:
            lower = txt.lower()
            return sum(1 for kw in keywords if kw in lower)

        # Build where clause for source filter
        where = {"filename": source_filter} if source_filter else None

        # When filtering by source, widen the pool aggressively. A single
        # document has far fewer chunks, so pulling 32 candidates still
        # lets the reranker choose from the whole book's most relevant
        # pages rather than just the top 16.
        n_semantic = 32 if source_filter else max(top_k * 2, 16)
        contains_limit = 12 if source_filter else 6

        try:
            # Semantic search across all variants — ChromaDB returns parallel
            # result lists, one per query embedding, which we union+dedupe.
            results = self._docs.query(
                query_embeddings=_emb.embed_queries(
                    query_variants, self._embed_model),
                n_results=n_semantic,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            hits: list[tuple] = []
            meta_by_key: dict[str, dict] = {}
            seen_keys: set[str] = set()
            if results.get("documents"):
                for docs_q, metas_q, dists_q in zip(
                    results["documents"],
                    results["metadatas"],
                    results["distances"],
                ):
                    for doc, meta, dist in zip(docs_q, metas_q, dists_q):
                        key = doc[:100]
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        meta_by_key[key] = meta
                        hits.append((
                            meta.get("filename", "?"), doc, dist,
                            meta.get("page_number"),
                        ))

            # $contains fallback for exact keyword matches
            seen = {t[:100] for _, t, _, _ in hits}
            for kw in list(keywords)[:4]:
                for variant in {kw, kw.title()}:
                    try:
                        where_doc = {"$contains": variant}
                        if source_filter:
                            cont_results = self._docs.get(
                                where={"filename": source_filter},
                                where_document=where_doc,
                                limit=contains_limit,
                                include=["documents", "metadatas"],
                            )
                        else:
                            cont_results = self._docs.get(
                                where_document=where_doc,
                                limit=contains_limit,
                                include=["documents", "metadatas"],
                            )
                        for doc, meta in zip(
                            cont_results.get("documents", []),
                            cont_results.get("metadatas", []),
                        ):
                            key = doc[:100]
                            if key not in seen:
                                seen.add(key)
                                meta_by_key[key] = meta
                                hits.append((
                                    meta.get("filename", "?"), doc, 1.0,
                                    meta.get("page_number"),
                                ))
                    except Exception:
                        pass

            if not hits:
                return []

            # Reranking — but only when we don't have strong keyword
            # evidence already. Cross-encoders are great when query
            # vocabulary differs from document vocabulary, and bad on
            # pages where the answer term is one entry among many (the
            # multi-spell page case): the page's overall embedding
            # doesn't smell like the query, so the reranker drops it
            # below chunks that are more topically focused but don't
            # actually contain the answer.
            n_kw = len(keywords) if keywords else 0
            rerank_pool_size = 40 if source_filter else 20
            # Strong-signal check runs over the FULL hits list, not just
            # the rerank pool — $contains-only matches (like "Manabolt"
            # found by "Bolt" substring) are appended after the semantic
            # union and can be past position 40.
            has_strong_kw_signal = (
                n_kw > 0
                and any(
                    sum(1 for k in keywords if k in (txt or "").lower())
                    >= n_kw
                    for _fn, txt, _d, _p in hits
                )
            )

            if apply_reranker and len(hits) > 1 and not has_strong_kw_signal:
                try:
                    rerank_top = max(top_k * 2, top_k + 8)
                    reranked = _reranker.rerank(
                        query, hits[:rerank_pool_size], self._reranker_model,
                        top_k=rerank_top, min_score=-1.0,
                        keywords=keywords, kw_boost=0.5,
                    )
                    hits = reranked
                except Exception as e:
                    log.warning(f"[RAG] Reranker failed in agentic mode: {e}")
                    hits.sort(key=lambda x: x[2])
                    hits = hits[:rerank_pool_size]
            else:
                # Strong keyword signal — keep ALL hits (incl. tail $contains
                # matches that may sit beyond the cross-encoder's pool) and
                # let the quality-scoring pass below choose the winner.
                pass

            # ── Post-rerank: search-engine-style scoring pass ──────────
            # IDF weighting, heading boost, phrase bonus. The reranker is
            # great at semantic relevance but can be fooled by chunks
            # whose embedding is diluted (e.g., a spell-catalog page where
            # one entry is the answer but five other entries crowd out the
            # signal). These signals reward chunks that look like real
            # answers regardless of embedding density.
            if hits and len(hits) > 1 and keywords:
                try:
                    idf = self._compute_idf_weights(keywords)
                except Exception as e:
                    log.warning(f"[RAG] IDF computation failed: {e}")
                    idf = {}
                phrase = query.lower().strip()
                has_phrase = (
                    len(phrase.split()) >= 2
                    and not phrase.startswith(("what ", "who ", "where ",
                                               "when ", "why ", "how ",
                                               "which ", "tell ", "explain "))
                )

                def quality(hit: tuple) -> float:
                    _fn, txt, _dist, _pg = hit
                    text_lower = txt.lower()
                    score = 0.0
                    # IDF-weighted keyword presence — rare term hits dominate
                    for kw in keywords:
                        if kw in text_lower:
                            score += idf.get(kw, 1.0)
                    # Heading hit: keyword in an ALL-CAPS or short title line
                    if self._has_keyword_in_heading(txt, keywords):
                        score += 4.0
                    # Phrase match: exact multi-word query in chunk
                    if has_phrase and phrase in text_lower:
                        score += 5.0
                    return score

                # Re-sort by quality DESC, breaking ties with original order
                indexed = list(enumerate(hits))
                indexed.sort(key=lambda i_h: (-quality(i_h[1]), i_h[0]))
                hits = [h for _, h in indexed[:top_k]]
            else:
                hits = hits[:top_k]

            # Convert to dict format with chunk_id
            out: list[dict] = []
            for fn, txt, dist, pg in hits:
                meta = meta_by_key.get(txt[:100], {})
                sub = meta.get("sub_chunk", 0)
                # EPUB chunks use chapter, not page_number
                position_type = ("chapter" if meta.get("chapter") is not None
                                 else "page")
                position = (meta.get("chapter") if position_type == "chapter"
                            else pg)
                chunk_id = self._build_chunk_id(
                    fn, position, sub, position_type=position_type)

                # Strip VISION section for preview / text
                raw_marker = "\nRAW TEXT:\n"
                display_text = txt
                if raw_marker in txt:
                    raw_part = txt.split(raw_marker, 1)[1]
                    if len(raw_part.strip()) > 50:
                        display_text = raw_part

                preview = self._keyword_preview(display_text, keywords)

                out.append({
                    "chunk_id": chunk_id,
                    "source": fn,
                    "page": (pg + 1) if pg is not None else None,
                    "chapter": meta.get("chapter"),
                    "text": display_text,
                    "preview": preview,
                    "dist": dist,
                    "kw_score": _kw_score(txt),
                })
            return out
        except Exception as e:
            log.error(f"[RAG] get_raw_chunks failed: {e}")
            return []

    # ── Search-engine-style scoring helpers ──────────────────────────────
    #
    # Borrowed from how Google ranks results: rare terms carry more signal
    # than common ones (IDF weighting), terms in headings outrank terms in
    # body text (field-aware boost), and exact phrase matches outrank
    # scattered-word matches (phrase bonus). Applied as a post-reranker
    # re-ordering pass.

    def _compute_idf_weights(
        self, keywords: set[str], total_chunks: int | None = None
    ) -> dict[str, float]:
        """For each keyword, compute log(N / df) — rare terms score high.

        df = number of chunks containing the term (capped at 5000 for cost).
        Falls back to weight 1.0 on any error.
        """
        if not keywords:
            return {}
        if total_chunks is None:
            try:
                total_chunks = max(1, self._docs.count())
            except Exception:
                total_chunks = 35000  # rough library size fallback
        import math
        weights: dict[str, float] = {}
        for kw in keywords:
            try:
                res = self._docs.get(
                    where_document={"$contains": kw},
                    limit=5000,
                    include=[],
                )
                df = max(1, len(res.get("ids", [])))
            except Exception:
                df = 1
            weights[kw] = max(0.1, math.log(total_chunks / df))
        return weights

    @staticmethod
    def _has_keyword_in_heading(text: str, keywords: set[str]) -> bool:
        """Detect a query keyword sitting in a section-heading line.

        Heading detection is harder than it looks: visually-all-caps
        headings in PDFs ("MANABOLT") often get extracted in mixed case
        ("Manabolt") because the font renders small-caps over lowercase
        glyphs. We use three signals to compensate:

          1. ALL CAPS short line containing the keyword (best signal).
          2. Short line where the keyword is among the first few tokens
             AND the line isn't an index entry ("Term, 133" pattern).
          3. Keyword followed by a parenthetical descriptor on a short
             line ("Manabolt (Direct Combat)") — common spell/entity
             format in reference books.
        """
        if not text or not keywords:
            return False

        kw_lower = {k.lower() for k in keywords if k}

        # Pattern that matches an index-style "Term, 133" line — we
        # reject those so the index page doesn't get a heading bonus.
        index_line_re = re.compile(r"\b\w+\s*,\s*\d{1,4}\b")

        for raw in text.split("\n"):
            line = raw.strip()
            if not line or len(line) > 80:
                continue
            line_lower = line.lower()
            if not any(kw in line_lower for kw in kw_lower):
                continue
            if index_line_re.search(line):
                # Looks like an index entry, not a section heading
                continue

            alpha_chars = [c for c in line if c.isalpha()]
            if len(alpha_chars) < 3:
                continue
            upper_ratio = sum(1 for c in alpha_chars
                              if c.isupper()) / len(alpha_chars)

            # Signal 1: real ALL CAPS line
            if upper_ratio >= 0.7:
                return True
            # Signal 2: keyword in first ~3 tokens of a short line
            first_tokens = " ".join(line_lower.split()[:3])
            if any(kw in first_tokens for kw in kw_lower):
                return True
            # Signal 3: keyword followed by " (" pattern (spell-entry style)
            for kw in kw_lower:
                paren_idx = line_lower.find(kw + " (")
                if paren_idx == -1:
                    paren_idx = line_lower.find(kw + "(")
                if paren_idx >= 0:
                    return True
        return False

    @staticmethod
    def _keyword_preview(text: str, keywords: set[str],
                         window: int = 200) -> str:
        """Build a preview centered on the first keyword match.

        If `text` contains any keyword (case-insensitive substring), return
        a `window`-char snippet starting a bit before the first match so
        the matched term is clearly visible. Falls back to the leading
        `window` chars when no keyword is found, which is the previous
        behavior.

        This makes "the agent picks the right chunk" much more reliable —
        instead of seeing first-paragraph text that may not mention the
        target term, the agent sees an excerpt around where the term
        actually appears.
        """
        if not text:
            return ""
        clean = text.replace("\n", " ").strip()
        if not keywords:
            return (clean[:window] + ("..." if len(clean) > window else ""))

        lower = clean.lower()
        # Prefer LONGER/more-specific keywords first. "manabolt" (concat
        # variant) is more distinctive than "mana" or "bolt", so its match
        # position should drive the preview window over a generic substring
        # match elsewhere in the chunk.
        sorted_keywords = sorted(
            (k for k in keywords if k),
            key=lambda k: -len(k),
        )
        best_pos = -1
        for kw in sorted_keywords:
            i = lower.find(kw.lower())
            if i >= 0:
                best_pos = i
                break
        if best_pos < 0:
            return (clean[:window] + ("..." if len(clean) > window else ""))

        # Window starts ~50 chars before the match so the matched term sits
        # near the front of the preview but with a tiny bit of lead-in.
        start = max(0, best_pos - 50)
        end = min(len(clean), start + window)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(clean) else ""
        return f"{prefix}{clean[start:end]}{suffix}"

    # Cap on how many sub-chunks we'll concatenate when read_page resolves
    # to a long page or EPUB chapter. Each sub-chunk is ~400 words; 10 caps
    # the joined text at ~4000 words which fits comfortably in the agent's
    # answer-generation context window.
    _MAX_SUBCHUNKS_PER_READ = 10

    def get_chunk_by_id(self, chunk_id: str) -> dict | None:
        """Fetch a chunk by its {filename}::{p|ch}{N}::s{sub} id.

        Routes to either page_number (PDFs) or chapter (EPUBs) metadata
        depending on the chunk_id's position_type. Returns ALL sub-chunks
        for the page or chapter joined in order — the sub_chunk number
        embedded in the chunk_id is treated as a hint, not a filter, so
        the agent gets the full page even if a specific sub-chunk number
        was returned by search. Embeddings are sub-chunk-scoped for
        retrieval precision; reading should always return the whole page.
        """
        parsed = self._parse_chunk_id(chunk_id)
        if not parsed:
            return None
        filename, position_type, position, _sub_hint = parsed
        try:
            where_clauses = [{"filename": filename}]
            if position is not None:
                if position_type == "chapter":
                    where_clauses.append({"chapter": position})
                else:
                    where_clauses.append({"page_number": position})

            where = (where_clauses[0] if len(where_clauses) == 1
                     else {"$and": where_clauses})
            results = self._docs.get(
                where=where,
                limit=self._MAX_SUBCHUNKS_PER_READ,
                include=["documents", "metadatas"],
            )
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
            if not docs:
                return None

            # Sort by sub_chunk so concatenation reflects original order
            ordered = sorted(
                zip(docs, metas),
                key=lambda dm: dm[1].get("sub_chunk", 0),
            )

            # Strip VISION block per chunk and join RAW TEXT in order. The
            # 50-word overlap between sub-chunks shows up as repeated text
            # at boundaries; we accept that to keep the join simple.
            raw_marker = "\nRAW TEXT:\n"
            parts: list[str] = []
            for d, _m in ordered:
                text = d
                if raw_marker in d:
                    raw_part = d.split(raw_marker, 1)[1]
                    if len(raw_part.strip()) > 50:
                        text = raw_part
                parts.append(text.strip())
            joined = "\n\n".join(p for p in parts if p)

            # Use first chunk's metadata for the return shape; all sub-chunks
            # share filename/page_number/chapter.
            meta = ordered[0][1]
            page_disp = None
            if meta.get("page_number") is not None:
                page_disp = meta.get("page_number") + 1

            return {
                "chunk_id": chunk_id,
                "source": meta.get("filename", "?"),
                "page": page_disp,
                "chapter": meta.get("chapter"),
                "text": joined,
                "sub_chunks_joined": len(parts),
            }
        except Exception as e:
            log.error(f"[RAG] get_chunk_by_id failed for {chunk_id}: {e}")
            return None

    @staticmethod
    def _build_chunk_id(filename: str, position: int | None,
                        sub: int | None = 0,
                        position_type: str = "page") -> str:
        """Stable chunk id: {filename}::{p{N+1}|ch{N}}::s{sub}.

        position_type:
          "page"    — for PDFs. Displayed 1-based (p1, p2, ...).
          "chapter" — for EPUBs. Displayed 0-based (ch0, ch1, ...).
        """
        if position_type == "chapter":
            pos_s = f"ch{position}" if position is not None else "ch?"
        else:
            pos_s = f"p{position + 1}" if position is not None else "p?"
        sub_s = f"s{sub if sub is not None else 0}"
        return f"{filename}::{pos_s}::{sub_s}"

    @staticmethod
    def _parse_chunk_id(chunk_id: str
                        ) -> tuple[str, str, int | None, int | None] | None:
        """Reverse of _build_chunk_id.

        Returns:
            (filename, position_type, position_idx, sub) where position_type
            is 'page' or 'chapter', position_idx is always 0-based regardless
            of the displayed format. None if the id is malformed.
        """
        try:
            parts = chunk_id.split("::")
            if len(parts) != 3:
                return None
            filename = parts[0]
            position = None
            position_type = "page"
            tag = parts[1]
            if tag.startswith("ch"):
                position_type = "chapter"
                if tag[2:] != "?":
                    position = int(tag[2:])
            elif tag.startswith("p"):
                position_type = "page"
                if tag[1:] != "?":
                    # Display is 1-based, stored 0-based
                    position = int(tag[1:]) - 1
            sub = None
            if parts[2].startswith("s"):
                sub = int(parts[2][1:])
            return filename, position_type, position, sub
        except Exception:
            return None

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _extract_relevant_paragraphs(
        text: str,
        keywords: set[str],
        context_paragraphs: int = 1,
        max_chars: int = 2000,
    ) -> str:
        """Extract paragraphs containing query keywords with surrounding context.

        Instead of injecting an entire page, find the paragraphs that contain
        query-relevant terms and include ±context_paragraphs of surrounding
        content.  This focuses the model's attention on the specific section
        that matches (e.g. the Feathery section, not Furry/Scaly).

        Falls back to the original text if no keywords match or the result
        would be too short.
        """
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        if len(paragraphs) <= 3:
            return text  # Already short enough

        # Score each paragraph by keyword overlap
        hits: set[int] = set()
        for i, para in enumerate(paragraphs):
            lower = para.lower()
            if any(kw in lower for kw in keywords):
                hits.add(i)

        if not hits:
            return text  # No keyword matches — return full text

        # Expand each hit by ±context_paragraphs
        included: set[int] = set()
        for idx in hits:
            for offset in range(-context_paragraphs, context_paragraphs + 1):
                pos = idx + offset
                if 0 <= pos < len(paragraphs):
                    included.add(pos)

        # Build result preserving order, with "..." between gaps
        parts: list[str] = []
        prev = -2
        for idx in sorted(included):
            if idx > prev + 1:
                if parts:
                    parts.append("...")
            parts.append(paragraphs[idx])
            prev = idx

        result = "\n".join(parts)
        # Fall back if the extraction is too short (lost important context)
        if len(result) < 100:
            return text
        return result[:max_chars]

    def _rrf_fuse(
        self,
        list_a: list[tuple],
        list_b: list[tuple],
        k: int = 60,
    ) -> list[tuple]:
        """Reciprocal Rank Fusion of two pre-sorted chunk lists.

        Both lists contain (filename, text, dist, page_num) tuples.
        Position 0 = rank 1 in each list. Chunks appearing in both lists
        receive contributions from both rankings.

        Args:
            list_a: First ranked list (e.g. semantic hits sorted by distance).
            list_b: Second ranked list (e.g. keyword hits sorted by hit count).
            k:      RRF constant (default 60, standard value).

        Returns:
            Fused list of unique chunks sorted by descending RRF score.
        """
        rank_a = {c[1][:100]: i + 1 for i, c in enumerate(list_a)}
        rank_b = {c[1][:100]: i + 1 for i, c in enumerate(list_b)}

        # Preserve first-seen tuple for each unique key
        chunk_index: dict[str, tuple] = {}
        for c in list_a + list_b:
            key = c[1][:100]
            if key not in chunk_index:
                chunk_index[key] = c

        def _score(key: str) -> float:
            s = 0.0
            if key in rank_a:
                s += 1.0 / (k + rank_a[key])
            if key in rank_b:
                s += 1.0 / (k + rank_b[key])
            return s

        return sorted(chunk_index.values(), key=lambda c: _score(c[1][:100]), reverse=True)

    def _jaccard_dedup(
        self,
        chunks: list[tuple],
        threshold: float = 0.85,
    ) -> list[tuple]:
        """Remove near-duplicate chunks from the same source document.

        Compares word-set Jaccard similarity between chunks sharing the same
        filename. Higher-ranked chunks (earlier in list) are always kept.

        Args:
            chunks:    Ranked list of (filename, text, dist, page_num) tuples.
            threshold: Jaccard >= this value -> considered a duplicate.

        Returns:
            Deduplicated list preserving original rank order.
        """
        accepted: list[tuple] = []
        accepted_by_file: dict[str, list[set[str]]] = {}

        for chunk in chunks:
            filename, text, dist, page_num = chunk
            word_set = set(text.lower().split())
            is_dup = False
            for existing_words in accepted_by_file.get(filename, []):
                if not existing_words or not word_set:
                    continue
                union = len(word_set | existing_words)
                if union > 0 and len(word_set & existing_words) / union >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                accepted.append(chunk)
                accepted_by_file.setdefault(filename, []).append(word_set)

        return accepted

    def _parse_entity_names_from_chunk(self, text: str, meta: dict) -> list[str]:
        """Extract entity names from a chunk for multi-hop queries.

        Prefers the 'entity_names' metadata field (written by --mdextraction).
        Falls back to parsing [METADATA: {...}] embedded in chunk text.

        Returns list of name strings, empty list on failure.
        """
        if meta.get("entity_names"):
            return [n.strip() for n in meta["entity_names"].split(",") if n.strip()]

        match = re.search(r'\[METADATA:\s*(\{.*?\})\]', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                return [e["name"] for e in parsed.get("entities", []) if e.get("name")]
            except Exception:
                pass
        return []

    def _second_hop_query(
        self,
        entity_names: list[str],
        original_query: str,
        seen_keys: set[str],
        n_results: int = 2,
    ) -> list[tuple]:
        """Retrieve additional chunks referenced by named entities in top results.

        For each entity, tries metadata-field lookup first (requires --mdextraction
        ingest), then falls back to semantic search. Skips entities already present
        in the original query (trivial re-fetch). Returns at most 4 new chunks.

        Args:
            entity_names:    Candidates from top chunk metadata/text.
            original_query:  Skip entities that already appear in it.
            seen_keys:       text[:100] keys already in result set; updated in-place.
            n_results:       Max semantic fallback results per entity.
        """
        query_lower = original_query.lower()
        hop_chunks: list[tuple] = []
        entities_tried = 0

        for entity in entity_names:
            if entities_tried >= 3 or len(hop_chunks) >= 4:
                break
            if entity.lower() in query_lower:
                continue

            entities_tried += 1
            new_chunks: list[tuple] = []

            # Attempt 1: structured metadata lookup (only works for --mdextraction docs)
            try:
                hits = self._docs.get(
                    where={"entity_names": {"$contains": entity}},
                    limit=4,
                    include=["documents", "metadatas"],
                )
                for doc, meta in zip(hits.get("documents", []), hits.get("metadatas", [])):
                    key = doc[:100]
                    if key not in seen_keys:
                        seen_keys.add(key)
                        new_chunks.append((
                            meta.get("filename", "unknown file"),
                            doc, 0.5, meta.get("page_number"),
                        ))
            except Exception:
                pass

            # Attempt 2: semantic fallback when no metadata hits
            if not new_chunks:
                try:
                    results = self._docs.query(
                        query_embeddings=_emb.embed_queries([entity], self._embed_model),
                        n_results=n_results,
                        include=["documents", "metadatas", "distances"],
                    )
                    if results["documents"] and results["documents"][0]:
                        for doc, meta, dist in zip(
                            results["documents"][0],
                            results["metadatas"][0],
                            results["distances"][0],
                        ):
                            if dist <= 1.4:
                                key = doc[:100]
                                if key not in seen_keys:
                                    seen_keys.add(key)
                                    new_chunks.append((
                                        meta.get("filename", "unknown file"),
                                        doc, dist, meta.get("page_number"),
                                    ))
                except Exception:
                    pass

            hop_chunks.extend(new_chunks)

        return hop_chunks[:4]
