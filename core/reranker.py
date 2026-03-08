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

_model = None
_model_name: str | None = None


def _get_model(model_name: str):
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from sentence_transformers import CrossEncoder
        # CPU to avoid competing with KoboldCpp VRAM.
        _model = CrossEncoder(model_name, device="cpu")
        _model_name = model_name
        print(f"   [Reranker] Loaded {model_name} on cpu")
    return _model


def rerank(
    query: str,
    chunks: list[tuple],   # (filename, text, dist, page_num)
    model_name: str,
    top_k: int = 8,
    min_score: float = -2.0,
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
    scores = model.predict(pairs).tolist()

    # Sort descending by reranker score; apply score threshold.
    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    if min_score is not None:
        scored = [(s, c) for s, c in scored if s >= min_score]

    return [chunk for _score, chunk in scored[:top_k]]
