"""Cross-encoder reranker for Talon's RAG stack.

After initial vector retrieval, the reranker scores each (query, chunk) pair
jointly — producing a substantially more accurate relevance ranking than
cosine distance alone, particularly when the query vocabulary differs from
the document vocabulary.

Used in explicit-mode get_document_context() after Phase-2 RRF fusion.

Model default: BAAI/bge-reranker-base (~278 MB, CPU-friendly).
Upgrade to bge-reranker-large for better quality at higher latency cost.
"""
from __future__ import annotations

import logging
log = logging.getLogger(__name__)

_model = None
_model_name: str | None = None


def _get_model(model_name: str):
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from sentence_transformers import CrossEncoder
        # CPU to avoid competing with KoboldCpp VRAM.
        _model = CrossEncoder(model_name, device="cpu")
        _model_name = model_name
        log.info(f"[Reranker] Loaded {model_name} on cpu")
    return _model


def rerank(
    query: str,
    chunks: list[tuple],   # (filename, text, dist, page_num)
    model_name: str,
    top_k: int = 8,
    min_score: float = -2.0,
    keywords: set[str] | None = None,
    kw_boost: float = 0.0,
) -> list[tuple]:
    """Rerank retrieved chunks using a cross-encoder, filtering low-scoring ones.

    Args:
        query:       The user's natural-language query string.
        chunks:      List of (filename, text, dist, page_num) tuples as
                     returned by get_document_context's retrieval phase.
        model_name:  HuggingFace cross-encoder model id.
        top_k:       Maximum number of chunks to return after reranking.
        min_score:   Minimum raw logit score to keep a chunk.  bge-reranker-base
                     outputs raw logits: clearly irrelevant pairs score around
                     -10 to -3; marginally relevant around -3 to 0; relevant
                     is positive.  Default -2.0 drops obvious noise while
                     preserving borderline matches.  Set to None to disable.
        keywords:    Optional set of lowercased query keywords.  Required
                     when kw_boost > 0.  Used to count keyword presence per
                     chunk for the boost calculation.
        kw_boost:    If > 0, add (kw_hits / len(keywords)) * kw_boost to
                     each chunk's reranker score before final sort.  Promotes
                     chunks that contain actual query terms over chunks the
                     reranker scored as loosely semantically related.
                     Typical values: 0.0 (off), 0.1-0.3 (gentle), 0.5+ (strong).
                     min_score is applied to the RAW reranker score so boost
                     cannot rescue genuinely irrelevant chunks.

    Returns:
        Top-k chunks that pass min_score, sorted by score (highest first).
        Returns empty list if nothing clears the threshold — the caller then
        skips RAG injection entirely.
        The original dist field is preserved for debug output.
    """
    if not chunks:
        return chunks

    model = _get_model(model_name)
    pairs = [(query, chunk[1]) for chunk in chunks]
    raw_scores = model.predict(pairs).tolist()

    # Compute final scores with optional keyword boost.
    if kw_boost > 0 and keywords:
        n_kw = max(1, len(keywords))
        final_scores = []
        for raw_s, chunk in zip(raw_scores, chunks):
            lower = chunk[1].lower()
            kw_hits = sum(1 for kw in keywords if kw in lower)
            bonus = (kw_hits / n_kw) * kw_boost
            final_scores.append(raw_s + bonus)
    else:
        final_scores = raw_scores

    # Sort descending by final score; apply min_score to RAW reranker score
    # so that kw_boost cannot rescue genuinely irrelevant chunks.
    triples = sorted(
        zip(final_scores, raw_scores, chunks),
        key=lambda x: x[0], reverse=True,
    )
    if min_score is not None:
        triples = [(fs, rs, c) for fs, rs, c in triples if rs >= min_score]

    return [chunk for _fs, _rs, chunk in triples[:top_k]]
