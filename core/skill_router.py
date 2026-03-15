"""
core/skill_router.py
--------------------
Two-stage skill routing: embed all on-demand talent descriptions at startup,
then do a fast cosine similarity lookup per query to pre-filter the routing
roster before the LLM routing call.

Core talents (always included in roster):
  planner, conversation, conversation_rag

On-demand talents: everything else — top-K by cosine similarity injected per query.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from talents.base import BaseTalent

logger = logging.getLogger(__name__)

# These talent names are always included in the routing roster regardless of
# query similarity — they are the meta-handlers that need to be available
# for any command.
CORE_TALENT_NAMES = {"planner", "conversation", "conversation_rag"}

# How many on-demand talents to inject per query
DEFAULT_TOP_K = 4


class SkillRouter:
    """Maintains BGE embeddings for on-demand talents and returns a filtered
    roster for a given query."""

    def __init__(self, top_k: int = DEFAULT_TOP_K):
        self.top_k = top_k
        self._model = None
        self._embeddings: dict[str, np.ndarray] = {}   # talent_name -> embedding
        self._talent_map: dict[str, BaseTalent] = {}   # talent_name -> talent
        self._ready = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            logger.debug("[SkillRouter] BGE model loaded")
        except Exception as exc:
            logger.warning("[SkillRouter] Could not load BGE model: %s", exc)

    def _talent_text(self, talent: "BaseTalent") -> str:
        """Build the text to embed for a talent."""
        parts = [talent.description]
        if talent.examples:
            parts.append(" ".join(talent.examples[:8]))
        elif talent.keywords:
            parts.append(" ".join(talent.keywords[:8]))
        return " ".join(parts)

    def build(self, talents: list["BaseTalent"]):
        """Embed all on-demand talents. Call once after talents are loaded."""
        self._load_model()
        if self._model is None:
            self._ready = False
            return

        on_demand = [
            t for t in talents
            if t.enabled and t.routing_available and t.name not in CORE_TALENT_NAMES
        ]

        if not on_demand:
            self._ready = True
            return

        texts = [self._talent_text(t) for t in on_demand]
        try:
            vecs = self._model.encode(texts, normalize_embeddings=True)
            for talent, vec in zip(on_demand, vecs):
                self._embeddings[talent.name] = vec
                self._talent_map[talent.name] = talent
            self._ready = True
            logger.debug("[SkillRouter] Embedded %d on-demand talents", len(on_demand))
        except Exception as exc:
            logger.warning("[SkillRouter] Embedding failed: %s", exc)
            self._ready = False

    def rebuild(self, talents: list["BaseTalent"]):
        """Rebuild after a talent is added or removed."""
        self._embeddings.clear()
        self._talent_map.clear()
        self._ready = False
        self.build(talents)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def top_talents(self, query: str) -> list["BaseTalent"]:
        """Return the top-K on-demand talents most relevant to *query*."""
        if not self._ready or not self._embeddings or self._model is None:
            # Fallback: return all on-demand talents
            return list(self._talent_map.values())

        try:
            q_vec = self._model.encode([query], normalize_embeddings=True)[0]
            names = list(self._embeddings.keys())
            matrix = np.stack([self._embeddings[n] for n in names])
            scores = matrix @ q_vec          # cosine sim (vecs are normalised)
            idx = np.argsort(scores)[::-1][:self.top_k]
            return [self._talent_map[names[i]] for i in idx]
        except Exception as exc:
            logger.warning("[SkillRouter] top_talents error: %s", exc)
            return list(self._talent_map.values())
