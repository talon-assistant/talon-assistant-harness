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
) -> list[tuple]:
    """Rerank retrieved chunks using a cross-encoder.

    Args:
        query:       The user's natural-language query string.
        chunks:      List of (filename, text, dist, page_num) tuples as
                     returned by get_document_context's retrieval phase.
        model_name:  HuggingFace cross-encoder model id.
        top_k:       Maximum number of chunks to return after reranking.

    Returns:
        Top-k chunks sorted by cross-encoder score (highest first).
        The original dist field is preserved for debug output.
    """
    if not chunks:
        return chunks

    model = _get_model(model_name)
    pairs = [(query, chunk[1]) for chunk in chunks]
    scores = model.predict(pairs).tolist()

    # Sort descending by reranker score; preserve original (fn, text, dist, pg) tuple.
    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _score, chunk in scored[:top_k]]
