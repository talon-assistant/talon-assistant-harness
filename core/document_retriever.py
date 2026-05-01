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
        Empty list if no TOC indexed for this book or no matches.

        Returns:
            [{chunk_id, title, page_printed, page_pdf, level, filename}, ...]
            sorted by relevance (phrase match > keyword count > earlier page).
        """
        store = self._get_toc_store()
        if store is None:
            return []
        matches = store.lookup(filename, query, max_results=max_results)
        out: list[dict] = []
        for m in matches:
            chunk_id = self._build_chunk_id(filename, m["page_pdf"], 0)
            out.append({
                "chunk_id": chunk_id,
                "title": m["title"],
                "page_printed": m["page_printed"],
                "page_pdf": m["page_pdf"],
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
            # Semantic search
            results = self._docs.query(
                query_embeddings=_emb.embed_queries(
                    [query], self._embed_model),
                n_results=n_semantic,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            hits: list[tuple] = []
            meta_by_key: dict[str, dict] = {}
            if results.get("documents") and results["documents"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    key = doc[:100]
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

            # Optional reranking
            if apply_reranker and len(hits) > 1:
                # Wider reranker pool when filtering by source (small doc).
                rerank_pool_size = 40 if source_filter else 20
                try:
                    reranked = _reranker.rerank(
                        query, hits[:rerank_pool_size], self._reranker_model,
                        top_k=top_k, min_score=-1.0,
                        keywords=keywords, kw_boost=0.2,
                    )
                    hits = reranked
                except Exception as e:
                    log.warning(f"[RAG] Reranker failed in agentic mode: {e}")
                    hits.sort(key=lambda x: x[2])
                    hits = hits[:top_k]
            else:
                hits.sort(key=lambda x: x[2])
                hits = hits[:top_k]

            # Convert to dict format with chunk_id
            out: list[dict] = []
            for fn, txt, dist, pg in hits:
                meta = meta_by_key.get(txt[:100], {})
                sub = meta.get("sub_chunk", 0)
                chunk_id = self._build_chunk_id(fn, pg, sub)

                # Strip VISION section for preview / text
                raw_marker = "\nRAW TEXT:\n"
                display_text = txt
                if raw_marker in txt:
                    raw_part = txt.split(raw_marker, 1)[1]
                    if len(raw_part.strip()) > 50:
                        display_text = raw_part

                preview = display_text[:200].replace("\n", " ").strip()
                if len(display_text) > 200:
                    preview += "..."

                out.append({
                    "chunk_id": chunk_id,
                    "source": fn,
                    "page": (pg + 1) if pg is not None else None,
                    "text": display_text,
                    "preview": preview,
                    "dist": dist,
                    "kw_score": _kw_score(txt),
                })
            return out
        except Exception as e:
            log.error(f"[RAG] get_raw_chunks failed: {e}")
            return []

    def get_chunk_by_id(self, chunk_id: str) -> dict | None:
        """Fetch a specific chunk by its {filename}::p{page}::s{sub} id.

        Returns the full (untruncated) chunk text so the agent can read
        a page in detail after seeing its preview in a search result.
        """
        parsed = self._parse_chunk_id(chunk_id)
        if not parsed:
            return None
        filename, page, sub = parsed
        try:
            where_clauses = [{"filename": filename}]
            if page is not None:
                where_clauses.append({"page_number": page})
            if sub is not None:
                where_clauses.append({"sub_chunk": sub})

            where = (where_clauses[0] if len(where_clauses) == 1
                     else {"$and": where_clauses})
            results = self._docs.get(
                where=where,
                limit=5,  # multiple sub-chunks possible
                include=["documents", "metadatas"],
            )
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
            if not docs:
                return None

            # Pick the longest match (best content for the page)
            best_idx = 0
            for i, d in enumerate(docs):
                if len(d) > len(docs[best_idx]):
                    best_idx = i
            doc = docs[best_idx]
            meta = metas[best_idx]

            # Strip VISION section for display
            raw_marker = "\nRAW TEXT:\n"
            display_text = doc
            if raw_marker in doc:
                raw_part = doc.split(raw_marker, 1)[1]
                if len(raw_part.strip()) > 50:
                    display_text = raw_part

            return {
                "chunk_id": chunk_id,
                "source": meta.get("filename", "?"),
                "page": (meta.get("page_number") + 1)
                        if meta.get("page_number") is not None else None,
                "text": display_text,
            }
        except Exception as e:
            log.error(f"[RAG] get_chunk_by_id failed for {chunk_id}: {e}")
            return None

    @staticmethod
    def _build_chunk_id(filename: str, page: int | None,
                        sub: int | None = 0) -> str:
        """Stable chunk id: {filename}::p{page}::s{sub}."""
        page_s = f"p{page + 1}" if page is not None else "p?"
        sub_s = f"s{sub if sub is not None else 0}"
        return f"{filename}::{page_s}::{sub_s}"

    @staticmethod
    def _parse_chunk_id(chunk_id: str) -> tuple[str, int | None, int | None] | None:
        """Reverse of _build_chunk_id. Returns (filename, page_idx, sub) or None."""
        try:
            parts = chunk_id.split("::")
            if len(parts) != 3:
                return None
            filename = parts[0]
            page = None
            if parts[1].startswith("p") and parts[1][1:] != "?":
                page = int(parts[1][1:]) - 1  # display is 1-based, stored 0-based
            sub = None
            if parts[2].startswith("s"):
                sub = int(parts[2][1:])
            return filename, page, sub
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
