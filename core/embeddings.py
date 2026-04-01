"""BGE embedding model wrapper for Talon's RAG stack.

Replaces ChromaDB's default all-MiniLM-L6-v2 with a BGE model that applies
asymmetric retrieval: queries receive an instruction prefix that significantly
improves recall for diverse natural-language questions, while documents are
stored without a prefix.

Design: embeddings are always computed externally and passed to ChromaDB via
the ``embeddings=`` / ``query_embeddings=`` parameters.  ChromaDB is used
purely as a vector store — its internal EF is never invoked.

Usage:
    from core import embeddings as _emb

    # At ingest / add time:
    vecs = _emb.embed_documents(["chunk text ..."], model_name)
    collection.add(embeddings=vecs, documents=[...], ...)

    # At query time:
    qvec = _emb.embed_queries(["user question"], model_name)
    collection.query(query_embeddings=qvec, ...)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import logging
log = logging.getLogger(__name__)

_model = None
_model_name: str | None = None

# BGE instruction prefix applied to queries only (not documents).
# This asymmetric setup consistently improves top-k recall for retrieval tasks.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _get_model(model_name: str):
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from sentence_transformers import SentenceTransformer
        # Run on CPU by default — KoboldCpp holds the GPU for inference.
        _model = SentenceTransformer(model_name, device="cpu")
        _model_name = model_name
        log.info(f"[Embeddings] Loaded {model_name} on cpu")
    return _model


def embed_documents(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed document texts without any prefix (used at add/ingest time)."""
    vecs = _get_model(model_name).encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    return vecs.tolist()


def embed_queries(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed query strings with the BGE instruction prefix (used at retrieval time)."""
    prefixed = [_BGE_QUERY_PREFIX + t for t in texts]
    vecs = _get_model(model_name).encode(
        prefixed, normalize_embeddings=True, show_progress_bar=False
    )
    return vecs.tolist()


class TalonEmbeddings:
    """Thin class wrapper around the module-level embedding functions.

    Exists so callers that prefer an object interface (e.g. SecurityClassifier)
    can import and instantiate this instead of referencing the functions directly.
    The underlying model cache is shared with the module-level functions.
    """

    def embed_documents(self, texts: list[str], model_name: str) -> list[list[float]]:
        return embed_documents(texts, model_name)

    def embed_queries(self, texts: list[str], model_name: str) -> list[list[float]]:
        return embed_queries(texts, model_name)
