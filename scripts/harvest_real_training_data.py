#!/usr/bin/env python3
"""Harvest real Talon artifacts as safe-class training data.

Queries the live ChromaDB and SQLite stores used by Talon and exports
every stored artifact as a labelled 'safe' example in the same JSONL
format produced by generate_security_training_data.py.

Real data is higher quality than synthetic for the safe class because
it is exactly what the classifier will encounter in production.

The suspicious class cannot be harvested (you would hope there are no
real injections in storage) and must remain synthetically generated.

Usage:
    python scripts/harvest_real_training_data.py
    python scripts/harvest_real_training_data.py --out data/security_classifier/real.jsonl

Merge with synthetic data before training:
    cat data/security_classifier/real.jsonl data/security_classifier/train.jsonl \\
        > data/security_classifier/combined.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from core.memory import MemorySystem


# ── Helpers ────────────────────────────────────────────────────────────────

def _write(out_f, text: str, artifact_type: str, source: str,
           seen: set, stats: dict) -> bool:
    """Write one record if text is non-empty and not a duplicate."""
    text = text.strip()
    if not text or len(text) < 8:
        return False
    if text in seen:
        stats["duplicates"] += 1
        return False
    record = {
        "text": text,
        "artifact_type": artifact_type,
        "label": "safe",
        "source": source,
        "strategy": None,
    }
    out_f.write(json.dumps(record) + "\n")
    out_f.flush()
    seen.add(text)
    stats[artifact_type] += 1
    return True


def _parse_reflection(doc: str) -> dict[str, list[str]]:
    """Parse a stored session reflection into its component fields.

    Reflection format (from store_session_reflection):
        Session Summary: <text>
        Preferences: <p1>; <p2>
        Failures: <f1>; <f2>
        Shortcuts: <s1>; <s2>
    """
    result: dict[str, list[str]] = {
        "summary": [],
        "preferences": [],
        "shortcuts": [],
    }
    for line in doc.splitlines():
        line = line.strip()
        if line.startswith("Session Summary:"):
            text = line[len("Session Summary:"):].strip()
            if text:
                result["summary"].append(text)
        elif line.startswith("Preferences:"):
            raw = line[len("Preferences:"):].strip()
            result["preferences"].extend(
                p.strip() for p in raw.split(";") if p.strip()
            )
        elif line.startswith("Shortcuts:"):
            raw = line[len("Shortcuts:"):].strip()
            result["shortcuts"].extend(
                s.strip() for s in raw.split(";") if s.strip()
            )
    return result


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvest real Talon artifacts as safe training examples"
    )
    parser.add_argument(
        "--out", default="data/security_classifier/real.jsonl",
        help="Output JSONL path (default: data/security_classifier/real.jsonl)"
    )
    parser.add_argument(
        "--config", default="config/settings.json",
        help="Talon settings.json path"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found at {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        settings = json.load(f)

    memory_cfg = settings.get("memory", {})
    db_path = memory_cfg.get("db_path", "data/talon_memory.db")
    chroma_path = memory_cfg.get("chroma_path", "data/chroma_db")

    print(f"SQLite:   {db_path}")
    print(f"ChromaDB: {chroma_path}\n")

    memory = MemorySystem(
        db_path=db_path,
        chroma_path=chroma_path,
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    stats: dict = defaultdict(int)

    with open(output_path, "w", encoding="utf-8") as out_f:

        # ── Preferences (type="preference") ───────────────────────────────
        print("Harvesting preferences...")
        try:
            results = memory.memory_collection.get(
                where={"type": "preference"},
                include=["documents"],
            )
            for doc in (results.get("documents") or []):
                _write(out_f, doc, "insight", "real_preference", seen, stats)
            print(f"  {stats['insight']} so far")
        except Exception as e:
            print(f"  [warn] {e}")

        # ── Eviction insights (type="insight") ────────────────────────────
        print("Harvesting eviction insights...")
        before = stats["insight"]
        try:
            results = memory.memory_collection.get(
                where={"type": "insight"},
                include=["documents"],
            )
            for doc in (results.get("documents") or []):
                _write(out_f, doc, "insight", "real_insight", seen, stats)
            print(f"  {stats['insight'] - before} new insights")
        except Exception as e:
            print(f"  [warn] {e}")

        # ── Soft hints (type="soft_hint") ─────────────────────────────────
        print("Harvesting soft hints...")
        try:
            results = memory.memory_collection.get(
                where={"type": "soft_hint"},
                include=["documents"],
            )
            for doc in (results.get("documents") or []):
                _write(out_f, doc, "hint", "real_hint", seen, stats)
            print(f"  {stats['hint']} hints")
        except Exception as e:
            print(f"  [warn] {e}")

        # ── Session reflections (type="session_reflection") ───────────────
        # Parse each reflection into summary, preferences, and shortcuts
        print("Harvesting session reflections...")
        summary_count = 0
        pref_count = 0
        hint_count = 0
        try:
            results = memory.memory_collection.get(
                where={"type": "session_reflection"},
                include=["documents"],
            )
            for doc in (results.get("documents") or []):
                parsed = _parse_reflection(doc)

                for text in parsed["summary"]:
                    if _write(out_f, text, "summary", "real_reflection", seen, stats):
                        summary_count += 1

                for text in parsed["preferences"]:
                    if _write(out_f, text, "insight", "real_reflection", seen, stats):
                        pref_count += 1

                for text in parsed["shortcuts"]:
                    if _write(out_f, text, "hint", "real_reflection", seen, stats):
                        hint_count += 1

            print(f"  {summary_count} summaries, {pref_count} preferences, "
                  f"{hint_count} shortcuts extracted")
        except Exception as e:
            print(f"  [warn] {e}")

        # ── Rules (SQLite) ─────────────────────────────────────────────────
        # Store trigger and action as separate entries, and as a combined rule entry
        print("Harvesting rules from SQLite...")
        rule_count = 0
        try:
            rules = memory.list_rules(limit=500)
            for rule in rules:
                trigger = rule.get("trigger_phrase", "").strip()
                action = rule.get("action_text", "").strip()

                if trigger and action:
                    combined = f"TRIGGER: {trigger} | ACTION: {action}"
                    if _write(out_f, combined, "rule", "real_rule", seen, stats):
                        rule_count += 1
                elif trigger:
                    if _write(out_f, trigger, "rule", "real_rule", seen, stats):
                        rule_count += 1

            print(f"  {rule_count} rules")
        except Exception as e:
            print(f"  [warn] {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    total = sum(v for k, v in stats.items() if k != "duplicates")
    print(f"\n── Done ──")
    print(f"  Total safe examples harvested: {total}")
    print(f"  Duplicates skipped:            {stats['duplicates']}")
    print(f"\n  By artifact type:")
    for atype in ("summary", "rule", "insight", "hint", "signal"):
        n = stats.get(atype, 0)
        if n:
            print(f"    {atype:<15} {n}")
    print(f"\n  Output: {output_path.resolve()}")

    if total == 0:
        print(
            "\n  Note: no artifacts found — Talon may not have accumulated "
            "enough session history yet. Run generate_security_training_data.py "
            "for fully synthetic data in the meantime."
        )
    else:
        print(
            "\n  Merge with synthetic data before training:\n"
            f"    cat {output_path} data/security_classifier/train.jsonl "
            f"> data/security_classifier/combined.jsonl"
        )


if __name__ == "__main__":
    main()
