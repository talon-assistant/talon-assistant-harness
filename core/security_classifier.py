"""core/security_classifier.py — Semantic security classifier for Talon.

Loads a trained MLP (weights in data/security_classifier/model.pt) and uses
frozen BGE embeddings to classify stored artifacts as 'safe' or 'suspicious'
before they are committed to ChromaDB or SQLite.

Architecture
------------
BGE-base-en-v1.5 (768-dim, frozen) + artifact-type one-hot (5-dim)
→ Linear(773, 256) → ReLU → Dropout → Linear(256, 64) → ReLU → Dropout
→ Linear(64, 2) → LogSoftmax

The model is lazy-loaded on the first call to classify() so there is no
startup cost if the model file does not exist or PyTorch is unavailable.

Usage
-----
    from core.security_classifier import SecurityClassifier
    clf = SecurityClassifier()                      # no load yet
    result = clf.classify("some text", "summary")  # loads on first call
    # result = {"label": "safe", "confidence": 0.97, "suspicious_score": 0.03}

    # If the model isn't trained yet or PyTorch is absent:
    # result = {"label": "safe", "confidence": 1.0, "suspicious_score": 0.0,
    #           "skipped": True, "reason": "..."}
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent.parent          # project root
_MODEL_PATH = _HERE / "data" / "security_classifier" / "model.pt"

# ---------------------------------------------------------------------------
# MLP definition — must match scripts/train_security_classifier.py exactly
# ---------------------------------------------------------------------------

def _build_mlp(input_dim: int = 773, dropout1: float = 0.3, dropout2: float = 0.2):
    """Reconstruct the SecurityMLP without importing the training script."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout1),
        nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(dropout2),
        nn.Linear(64, 2),    nn.LogSoftmax(dim=1),
    )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class SecurityClassifier:
    """Semantic classifier that wraps the trained MLP.

    Thread-safety note: ``classify()`` is idempotent after the first load.
    The model is loaded once and stays in RAM.  Calls from multiple threads
    are safe after the initial load completes; the lazy load is protected by
    a simple flag so at worst two threads race to load (harmless, one wins).
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float = 0.5,
        verbose: bool = True,
    ):
        """
        Args:
            model_path: Override the default model.pt location.
            threshold:  Suspicious-score cutoff.  Scores ≥ threshold → SUSPICIOUS.
                        Lower values = more sensitive (more flags, more false positives).
                        Recommended range 0.40–0.65.
            verbose:    Print load/inference diagnostics to stdout.
        """
        self._model_path = Path(model_path) if model_path else _MODEL_PATH
        self.threshold = threshold
        self._verbose = verbose

        # State populated on first load
        self._model = None          # nn.Sequential
        self._artifact_types: list[str] = []
        self._embed_model_name: str = ""
        self._input_dim: int = 773
        self._loaded: bool = False
        self._load_error: str = ""  # non-empty → skip all classification silently

        # Embedder — same singleton used by core/embeddings.py
        self._embedder = None

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, text: str, artifact_type: str) -> dict:
        """Classify a text artifact.

        Args:
            text:          The artifact text (summary, rule, insight, hint, signal).
            artifact_type: One of the artifact type strings the model was trained on.
                           Unknown types default to the zero vector (no type hint).

        Returns:
            dict with keys:
                label          – "safe" or "suspicious"
                confidence     – probability of the predicted label (0.0–1.0)
                suspicious_score – raw probability of the suspicious class (0.0–1.0)
                skipped        – True if the model was unavailable (defaults to safe)
                reason         – human-readable explanation if skipped=True
        """
        if not self._loaded:
            self._load()

        if self._load_error:
            return {
                "label": "safe",
                "confidence": 1.0,
                "suspicious_score": 0.0,
                "skipped": True,
                "reason": self._load_error,
            }

        try:
            return self._infer(text, artifact_type)
        except Exception as exc:
            if self._verbose:
                print(f"   [SecurityClassifier] Inference error: {exc}")
            return {
                "label": "safe",
                "confidence": 1.0,
                "suspicious_score": 0.0,
                "skipped": True,
                "reason": f"inference error: {exc}",
            }

    def is_available(self) -> bool:
        """Return True if the model has been loaded successfully."""
        if not self._loaded:
            self._load()
        return self._loaded and not self._load_error

    def reload(self, model_path: str | Path | None = None) -> bool:
        """Force a model reload (e.g. after retraining).

        Returns True on success.
        """
        if model_path:
            self._model_path = Path(model_path)
        self._model = None
        self._loaded = False
        self._load_error = ""
        self._load()
        return not bool(self._load_error)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Lazy-load the model. Sets self._load_error on failure."""
        self._loaded = True  # mark True even on error to avoid repeat attempts

        # --- torch availability
        try:
            import torch
        except ImportError:
            self._load_error = "PyTorch not installed — classifier disabled"
            if self._verbose:
                print(f"   [SecurityClassifier] {self._load_error}")
            return

        # --- model file
        if not self._model_path.exists():
            self._load_error = (
                f"Model not found at {self._model_path} — "
                "run scripts/train_security_classifier.py first"
            )
            if self._verbose:
                print(f"   [SecurityClassifier] {self._load_error}")
            return

        # --- load checkpoint
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ckpt = torch.load(
                    self._model_path,
                    map_location="cpu",
                    weights_only=True,
                )
        except Exception as exc:
            self._load_error = f"Could not load checkpoint: {exc}"
            if self._verbose:
                print(f"   [SecurityClassifier] {self._load_error}")
            return

        # --- reconstruct model
        self._artifact_types = ckpt.get("artifact_types", ["hint", "insight", "rule", "signal", "summary"])
        self._embed_model_name = ckpt.get("embed_model", "BAAI/bge-base-en-v1.5")
        self._input_dim = ckpt.get("input_dim", 773)

        model = _build_mlp(self._input_dim)
        try:
            model.load_state_dict(ckpt["state_dict"])
        except Exception as exc:
            self._load_error = f"State dict mismatch: {exc}"
            if self._verbose:
                print(f"   [SecurityClassifier] {self._load_error}")
            return

        model.eval()
        self._model = model

        # --- load embedder
        try:
            from core.embeddings import TalonEmbeddings
            self._embedder = TalonEmbeddings()
        except Exception as exc:
            self._load_error = f"Could not load embedder: {exc}"
            if self._verbose:
                print(f"   [SecurityClassifier] {self._load_error}")
            self._model = None
            return

        trained_on = ckpt.get("trained_on", "?")
        val_f1 = ckpt.get("val_f1", None)
        f1_str = f", val_F1={val_f1:.3f}" if val_f1 is not None else ""
        if self._verbose:
            print(
                f"   [SecurityClassifier] Loaded ({trained_on} examples{f1_str}, "
                f"threshold={self.threshold:.2f})"
            )

    # ── Inference ─────────────────────────────────────────────────────────────

    def _encode_artifact_type(self, artifact_type: str) -> list[float]:
        """Return a one-hot vector for the given artifact type."""
        vec = [0.0] * len(self._artifact_types)
        if artifact_type in self._artifact_types:
            vec[self._artifact_types.index(artifact_type)] = 1.0
        return vec

    def _infer(self, text: str, artifact_type: str) -> dict:
        import torch

        # 1. Embed text
        embeddings = self._embedder.embed_documents(
            [text], model_name=self._embed_model_name
        )
        emb = embeddings[0]  # list[float], 768-dim

        # 2. Artifact one-hot
        one_hot = self._encode_artifact_type(artifact_type)

        # 3. Concatenate → feature vector
        features = emb + one_hot  # list concat

        # 4. Run model
        x = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            log_probs = self._model(x)        # shape (1, 2)
            probs = log_probs.exp()[0]        # shape (2,)

        # Index 0 = "safe", index 1 = "suspicious" (alphabetical sort in training)
        safe_score = float(probs[0])
        suspicious_score = float(probs[1])

        is_suspicious = suspicious_score >= self.threshold
        label = "suspicious" if is_suspicious else "safe"
        confidence = suspicious_score if is_suspicious else safe_score

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "suspicious_score": round(suspicious_score, 4),
            "skipped": False,
        }


# ---------------------------------------------------------------------------
# Module-level singleton (shared across the process)
# ---------------------------------------------------------------------------

_instance: Optional[SecurityClassifier] = None


def get_classifier(threshold: float = 0.5) -> SecurityClassifier:
    """Return the process-wide SecurityClassifier singleton.

    The first call creates the instance; subsequent calls return the same one.
    The threshold can only be set on the first call — use clf.threshold = X
    afterwards if you need to adjust it at runtime.
    """
    global _instance
    if _instance is None:
        _instance = SecurityClassifier(threshold=threshold)
    return _instance
