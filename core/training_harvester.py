"""training_harvester.py — silently collect LLM training pairs during normal use.

Writes Alpaca-format JSONL to data/training_pairs.jsonl.
One record per line, appended on each qualifying interaction.
Ready to feed directly into Unsloth/axolotl when the training pipeline is built.

Two sources are captured:
  "correction"  — user corrected a wrong answer ("no I meant X")
                  pair: (original bad command → correct re-executed response)
  "web_search"  — model's training knowledge was stale; web search found the truth
                  pair: (user's search command → synthesized correct answer)

Can be disabled entirely via settings.json:
  "training": {"harvest_pairs": false}
"""

import json
import os
from datetime import datetime

_PAIRS_FILE = os.path.join("data", "training_pairs.jsonl")
_MIN_OUTPUT_LEN = 20    # skip empty / one-word error responses


def append_training_pair(instruction: str, output: str, source: str) -> bool:
    """Append one Alpaca-format training pair to data/training_pairs.jsonl.

    Args:
        instruction: The user's command or question (input side of the pair)
        output:      The correct response the model should produce
        source:      "correction" | "web_search"

    Returns:
        True  — pair was written
        False — pair was skipped (too short, empty, or duplicate instruction)
    """
    instruction = instruction.strip()
    output = output.strip()

    if not instruction or not output:
        return False
    if len(output) < _MIN_OUTPUT_LEN:
        return False

    # Simple dedup: skip if this exact instruction is already recorded.
    # Linear scan is fine — the file will stay small (hundreds, not millions).
    if os.path.exists(_PAIRS_FILE):
        try:
            with open(_PAIRS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        if json.loads(line).get("instruction") == instruction:
                            return False
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

    record = {
        "instruction": instruction,
        "input": "",        # Alpaca format — leave empty for single-turn Q&A
        "output": output,
        "source": source,
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs("data", exist_ok=True)
    with open(_PAIRS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"   [Harvest] Saved {source} pair: '{instruction[:60]}'")
    return True
