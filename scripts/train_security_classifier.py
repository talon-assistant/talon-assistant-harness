#!/usr/bin/env python3
"""Train the Talon semantic security classifier.

Reads labelled artifact examples from a JSONL dataset, computes BGE
embeddings (frozen), concatenates a one-hot artifact-type vector, then
trains a small MLP to distinguish safe from suspicious artifacts.

Architecture:
    [BGE embedding (768)] + [artifact one-hot (5)] -> 773 dims
    Linear(773, 256) -> ReLU -> Dropout(0.3)
    Linear(256, 64)  -> ReLU -> Dropout(0.2)
    Linear(64, 2)    -> log-softmax

Output:
    data/security_classifier/model.pt   — model weights + metadata

Usage:
    python scripts/train_security_classifier.py
    python scripts/train_security_classifier.py --data data/security_classifier/combined.jsonl
    python scripts/train_security_classifier.py --epochs 200 --lr 5e-4
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).parent.parent))

import core.embeddings as _emb

# ── Constants ─────────────────────────────────────────────────────────────

ARTIFACT_TYPES = ["hint", "insight", "rule", "signal", "summary"]
ARTIFACT_TO_IDX = {t: i for i, t in enumerate(ARTIFACT_TYPES)}
EMBED_DIM = 768
ARTIFACT_DIM = len(ARTIFACT_TYPES)
INPUT_DIM = EMBED_DIM + ARTIFACT_DIM
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_BATCH = 64


# ── MLP ──────────────────────────────────────────────────────────────────

class SecurityMLP(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, dropout1: float = 0.3,
                 dropout2: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(64, 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Data loading ─────────────────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("artifact_type") not in ARTIFACT_TO_IDX:
                    print(f"  [warn] line {i}: unknown artifact_type "
                          f"'{r.get('artifact_type')}' — skipped")
                    continue
                if r.get("label") not in ("safe", "suspicious"):
                    print(f"  [warn] line {i}: unknown label "
                          f"'{r.get('label')}' — skipped")
                    continue
                records.append(r)
            except json.JSONDecodeError as e:
                print(f"  [warn] line {i}: JSON error — {e}")
    return records


def encode_artifact_type(artifact_type: str) -> list[float]:
    v = [0.0] * ARTIFACT_DIM
    v[ARTIFACT_TO_IDX[artifact_type]] = 1.0
    return v


def embed_texts_batched(texts: list[str], batch_size: int = EMBED_BATCH,
                        cache_path: Path | None = None) -> np.ndarray:
    """Compute BGE embeddings in batches, with optional on-disk cache."""
    if cache_path and cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path.name} ...")
        return np.load(str(cache_path))

    print(f"  Embedding {len(texts)} texts with BGE "
          f"(batch_size={batch_size}) ...")
    all_vecs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        vecs = _emb.embed_documents(batch, EMBED_MODEL)
        all_vecs.extend(vecs)
        done = min(start + batch_size, len(texts))
        print(f"    {done}/{len(texts)}", end="\r", flush=True)
    print()

    result = np.array(all_vecs, dtype=np.float32)
    if cache_path:
        np.save(str(cache_path), result)
        print(f"  Cached embeddings -> {cache_path.name}")
    return result


def build_feature_matrix(records: list[dict],
                          embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate BGE embedding + artifact one-hot. Return (X, y)."""
    X_list, y_list = [], []
    for i, rec in enumerate(records):
        art_vec = encode_artifact_type(rec["artifact_type"])
        feature = np.concatenate([embeddings[i], art_vec])
        X_list.append(feature)
        y_list.append(0 if rec["label"] == "safe" else 1)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


# ── Training ─────────────────────────────────────────────────────────────

def train(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
          X_val: torch.Tensor, y_val: torch.Tensor,
          class_weights: torch.Tensor, epochs: int, lr: float,
          patience: int, batch_size: int) -> dict:
    """Train with early stopping. Returns history dict."""
    criterion = nn.NLLLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_f1 = -1.0
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    n_train = X_train.size(0)

    for epoch in range(1, epochs + 1):
        # — Train —
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_train_loss = epoch_loss / n_batches

        # — Validate —
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val).item()
            val_preds = val_out.argmax(dim=1).numpy()

        val_f1 = f1_score(y_val.numpy(), val_preds, average="macro")
        scheduler.step(val_f1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={avg_train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best val_f1={best_val_f1:.4f})")
                break

    model.load_state_dict(best_state)
    history["best_val_f1"] = best_val_f1
    return history


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
             records: list[dict]) -> None:
    """Print detailed evaluation including per-strategy breakdown."""
    model.eval()
    with torch.no_grad():
        out = model(X)
        probs = torch.exp(out).numpy()
        preds = out.argmax(dim=1).numpy()
    labels = y.numpy()

    print("\n-- Overall --")
    print(classification_report(
        labels, preds,
        target_names=["safe", "suspicious"],
        digits=3,
    ))

    print("-- Confusion matrix (rows=actual, cols=predicted) --")
    cm = confusion_matrix(labels, preds)
    print(f"  {'':12s} {'pred:safe':>10} {'pred:susp':>10}")
    print(f"  {'actual:safe':12s} {cm[0][0]:>10} {cm[0][1]:>10}")
    print(f"  {'actual:susp':12s} {cm[1][0]:>10} {cm[1][1]:>10}")

    # Per artifact type
    print("\n-- Per artifact type --")
    type_data: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for i, rec in enumerate(records):
        t = rec["artifact_type"]
        type_data[t][0].append(labels[i])
        type_data[t][1].append(preds[i])
    for atype in sorted(type_data):
        yt, yp = type_data[atype]
        f1 = f1_score(yt, yp, average="macro", zero_division=0)
        prec, rec, _, _ = precision_recall_fscore_support(
            yt, yp, average="macro", zero_division=0
        )
        total_s = sum(1 for x in yt if x == 0)
        total_x = sum(1 for x in yt if x == 1)
        print(f"  {atype:<10}  F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}"
              f"  (safe={total_s} susp={total_x})")

    # Per attack strategy (suspicious only)
    print("\n-- Per attack strategy (suspicious class) --")
    strat_data: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for i, rec in enumerate(records):
        if rec.get("label") == "suspicious" and rec.get("strategy"):
            s = rec["strategy"]
            strat_data[s][0].append(labels[i])
            strat_data[s][1].append(preds[i])
    for strat in sorted(strat_data, key=lambda s: -len(strat_data[s][0])):
        yt, yp = strat_data[strat]
        correct = sum(a == b for a, b in zip(yt, yp))
        print(f"  {strat:<25}  recall={correct}/{len(yt)}"
              f"  ({100*correct/len(yt):.0f}%)")

    # Confidence histogram
    print("\n-- Confidence distribution --")
    for threshold in (0.7, 0.8, 0.85, 0.9, 0.95):
        susp_probs = probs[:, 1]
        flagged = (susp_probs >= threshold).sum()
        true_susp = labels.sum()
        captured = ((susp_probs >= threshold) & (labels == 1)).sum()
        print(f"  threshold={threshold:.2f}  flagged={flagged}  "
              f"true_susp_captured={captured}/{true_susp}  "
              f"({100*captured/max(true_susp,1):.0f}%)")


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Talon semantic security classifier"
    )
    parser.add_argument(
        "--data",
        default="data/security_classifier/synthetic_all.jsonl",
        help="JSONL dataset path",
    )
    parser.add_argument(
        "--extra",
        default="data/security_classifier/real.jsonl",
        help="Optional second JSONL to merge (e.g. harvested real data)",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument(
        "--out", default="data/security_classifier/model.pt"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Recompute embeddings even if cache exists"
    )
    args = parser.parse_args()

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.data}")
    records = load_dataset(args.data)
    print(f"  {len(records)} records loaded")

    extra_path = Path(args.extra)
    if extra_path.exists():
        print(f"Merging extra dataset: {args.extra}")
        extra = load_dataset(args.extra)
        records.extend(extra)
        print(f"  Total after merge: {len(records)} records")

    if not records:
        print("Error: no records loaded. Check dataset path.")
        sys.exit(1)

    safe_n = sum(1 for r in records if r["label"] == "safe")
    susp_n = sum(1 for r in records if r["label"] == "suspicious")
    print(f"\n  safe={safe_n}  suspicious={susp_n}  total={len(records)}")

    # ── Embed ────────────────────────────────────────────────────────────
    texts = [r["text"] for r in records]
    cache_key = len(records)
    cache_path = (out_dir / f"embed_cache_{cache_key}.npy"
                  ) if not args.no_cache else None
    embeddings = embed_texts_batched(texts, cache_path=cache_path)

    # ── Build feature matrix ─────────────────────────────────────────────
    print("\nBuilding feature matrix ...")
    X, y = build_feature_matrix(records, embeddings)
    print(f"  X shape: {X.shape}  y shape: {y.shape}")

    # ── Train / val split ────────────────────────────────────────────────
    idx = np.arange(len(records))
    idx_train, idx_val = train_test_split(
        idx, test_size=args.val_split, stratify=y, random_state=42
    )
    print(f"\n  Train: {len(idx_train)}  Val: {len(idx_val)}")

    X_train = torch.from_numpy(X[idx_train])
    y_train = torch.from_numpy(y[idx_train])
    X_val   = torch.from_numpy(X[idx_val])
    y_val   = torch.from_numpy(y[idx_val])

    # Class weights to handle any imbalance
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y[idx_train])
    class_weights = torch.tensor(cw, dtype=torch.float32)
    print(f"  Class weights: safe={cw[0]:.3f}  suspicious={cw[1]:.3f}")

    # ── Build model ──────────────────────────────────────────────────────
    model = SecurityMLP()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} trainable parameters")

    # ── Train ────────────────────────────────────────────────────────────
    print(f"\nTraining for up to {args.epochs} epochs "
          f"(lr={args.lr}, patience={args.patience}) ...\n")
    t0 = time.time()
    history = train(
        model, X_train, y_train, X_val, y_val,
        class_weights=class_weights,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        batch_size=args.batch,
    )
    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Best val F1: {history['best_val_f1']:.4f}")

    # ── Evaluate on validation set ───────────────────────────────────────
    val_records = [records[i] for i in idx_val]
    print(f"\n-- Validation set evaluation ({len(idx_val)} examples) --")
    evaluate(model, X_val, y_val, val_records)

    # ── Save ─────────────────────────────────────────────────────────────
    save_obj = {
        # Model
        "state_dict": model.state_dict(),
        # Architecture metadata (for load-time reconstruction)
        "architecture": {
            "input_dim": INPUT_DIM,
            "embed_dim": EMBED_DIM,
            "artifact_dim": ARTIFACT_DIM,
        },
        # Vocabulary
        "artifact_types": ARTIFACT_TYPES,
        "artifact_to_idx": ARTIFACT_TO_IDX,
        "labels": ["safe", "suspicious"],
        # Embedding model
        "embed_model": EMBED_MODEL,
        # Training provenance
        "training": {
            "n_train": len(idx_train),
            "n_val": len(idx_val),
            "epochs_run": len(history["train_loss"]),
            "best_val_f1": history["best_val_f1"],
            "data_path": str(args.data),
            "elapsed_s": round(elapsed, 1),
        },
    }
    torch.save(save_obj, args.out)
    print(f"\n  Model saved -> {args.out}")

    # Final hint on threshold selection
    print(
        "\nThreshold guidance (from confidence histogram above):\n"
        "  - 0.80 recommended starting point (high recall, moderate precision)\n"
        "  - 0.90 for lower false-positive rate if you see too many false flags\n"
        "  Configure via settings.json: security.semantic_classifier.threshold"
    )


if __name__ == "__main__":
    main()
