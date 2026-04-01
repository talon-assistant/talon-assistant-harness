"""core/document_retriever.py — RAG retrieval pipeline for document chunks.

Extracted from core/memory.py to separate storage (CRUD) from retrieval
intelligence (multi-query expansion, RRF fusion, cross-encoder reranking,
multi-hop entity lookup).
"""

import re
import json
from core import embeddings as _emb
from core import reranker as _reranker


class DocumentRetriever:
    """Retrieve and rank document chunks from a ChromaDB collection."""

    def __init__(self, docs_collection, embed_model: str, reranker_model: str):
        self._docs = docs_collection
        self._embed_model = embed_model
        self._reranker_model = reranker_model

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
        n_results = 8 if use_explicit else 2
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
                print(f"   [RAG] No chunks passed threshold "
                      f"(mode={mode}, threshold={max_distance:.2f})")
                return ""

            # ── Phase 2: ranking ──────────────────────────────────────────
            all_terms = [query] + (alt_queries or [])
            keywords = set(w.lower() for t in all_terms for w in t.split() if len(w) > 3)

            def _keyword_score(chunk_text: str) -> int:
                lower = chunk_text.lower()
                return sum(1 for kw in keywords if kw in lower)

            if use_explicit:
                # RRF fusion: split by real distance vs. artificial $contains distance
                semantic_pool = sorted(
                    [c for c in all_chunks if c[2] < 1.0], key=lambda x: x[2])
                keyword_pool  = sorted(
                    [c for c in all_chunks if c[2] >= 1.0],
                    key=lambda x: -_keyword_score(x[1]))
                all_chunks = self._rrf_fuse(semantic_pool, keyword_pool)
                all_chunks = self._jaccard_dedup(all_chunks)
            else:
                all_chunks.sort(key=lambda x: x[2])

            # Candidate pool for reranker: slightly larger than final cap so
            # the cross-encoder has enough to choose from.
            RERANK_POOL = 12 if use_explicit else 2
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
            if use_explicit and len(all_chunks) > 1:
                n_before = len(all_chunks)
                all_chunks = _reranker.rerank(
                    query, all_chunks, self._reranker_model,
                    top_k=FINAL_CAP, min_score=RERANK_MIN_SCORE,
                )
                print(f"   [RAG] Cross-encoder reranked {n_before}→{len(all_chunks)} chunks "
                      f"(min_score={RERANK_MIN_SCORE})")
                if not all_chunks:
                    print("   [RAG] All chunks below reranker threshold — skipping injection")
                    return ""

            # ── Phase 3: multi-hop ────────────────────────────────────────
            if multi_hop and explicit and all_chunks:
                seen_keys: set[str] = {c[1][:100] for c in all_chunks}
                hop_entities: list[str] = []
                for _fn, text, _d, _pg in all_chunks[:3]:
                    hop_entities.extend(
                        self._parse_entity_names_from_chunk(
                            text, meta_cache.get(text[:100], {}))
                    )
                unique_entities = list(dict.fromkeys(hop_entities))
                if unique_entities:
                    hop_chunks = self._second_hop_query(unique_entities[:6], query, seen_keys)
                    if hop_chunks:
                        # Discount hop chunks: inflate distance → weaker RRF contribution
                        discounted = [(fn, txt, dist / 0.7, pg)
                                      for fn, txt, dist, pg in hop_chunks]
                        all_chunks = self._rrf_fuse(all_chunks, discounted)
                        all_chunks = self._jaccard_dedup(all_chunks)
                        all_chunks = all_chunks[:FINAL_CAP + 4]
                        print(f"   [RAG] Multi-hop added {len(hop_chunks)} chunk(s) "
                              f"from entities: {unique_entities[:3]}")

            # ── Phase 4: format ───────────────────────────────────────────
            print(f"   [RAG] Injecting {len(all_chunks)} unique chunk(s) "
                  f"(explicit={explicit}, synthesis={synthesis}, "
                  f"best_dist={all_chunks[0][2]:.3f})")
            for _fn, _txt, _d, _pg in all_chunks:
                pg_label = f" p{_pg + 1}" if _pg is not None else ""
                print(f"      {_d:.3f} kw={_keyword_score(_txt)}  "
                      f"{_fn}{pg_label}  |  {_txt[:60].replace(chr(10), ' ')!r}")

            if use_explicit:
                lines = [
                    "The following excerpts are from the user's own documents. "
                    "Prioritize this content — use it directly and cite the source filename. "
                    "Use ONLY what is explicitly stated in these excerpts. "
                    "For any specific stat, number, rule, or structured value — if it is not "
                    "present in the excerpts, say it was not found rather than substituting "
                    "from general knowledge. General knowledge may contradict the document."
                ]
            else:
                lines = [
                    "The following document excerpts may be relevant — "
                    "use them if helpful, ignore if not:"
                ]
            for filename, text, dist, page_num in all_chunks:
                truncated = text[:800] + "..." if len(text) > 800 else text
                source = (f"{filename} (page {page_num + 1})"
                          if page_num is not None else filename)
                lines.append(f"- From {source}: {truncated}")

            return "\n".join(lines) + "\n"

        except Exception as e:
            print(f"   [RAG] Document context error: {e}")
            return ""

    # ── Internal helpers ──────────────────────────────────────────────────

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
