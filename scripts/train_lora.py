"""train_lora.py — Fine-tune the local LLM on collected training pairs using LoRA.

Usage:
    python scripts/train_lora.py \
        --pairs data/training_pairs.jsonl \
        --base-model /path/to/hf/model \
        --output-dir data/lora_adapters \
        --config '{"lora_r":16,"lora_alpha":16,"epochs":3,"batch_size":2,"max_seq_length":2048}'

Outputs:
    {output_dir}/adapter/       — HuggingFace LoRA adapter (safetensors + config)
    {output_dir}/adapter.gguf   — GGUF adapter for KoboldCpp (if convert tool found)

Sentinel lines printed to stdout (parsed by LoraTrainTalent):
    ADAPTER_PATH:{output_dir}/adapter
    GGUF_PATH:{output_dir}/adapter.gguf

Requires: pip install unsloth trl
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _load_pairs(pairs_file: str) -> list[dict]:
    """Load Alpaca-format JSONL training pairs."""
    records = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _format_as_chatml(record: dict) -> str:
    """Format an Alpaca record as ChatML for Qwen2.5 / llama-chat models."""
    instruction = record.get("instruction", "").strip()
    output = record.get("output", "").strip()
    return (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


def train(
    pairs_file: str,
    base_model: str,
    output_dir: str,
    lora_r: int = 16,
    lora_alpha: int = 16,
    epochs: int = 3,
    batch_size: int = 2,
    max_seq_length: int = 2048,
    bin_path: str = "bin",
) -> None:
    """Run LoRA fine-tuning and save adapter."""
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # ── Load and format dataset ──────────────────────────────────
    print(f"[Train] Loading pairs from {pairs_file}...")
    records = _load_pairs(pairs_file)
    if not records:
        raise ValueError(f"No valid training pairs found in {pairs_file}")
    print(f"[Train] {len(records)} pairs loaded.")

    texts = [_format_as_chatml(r) for r in records]
    dataset = Dataset.from_dict({"text": texts})

    # ── Load base model in 4-bit ─────────────────────────────────
    print(f"[Train] Loading base model: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,  # auto-detect
    )

    # ── Apply LoRA adapters ──────────────────────────────────────
    print(f"[Train] Applying LoRA (r={lora_r}, alpha={lora_alpha})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )

    # ── Train ────────────────────────────────────────────────────
    adapter_path = str(Path(output_dir) / "adapter")
    print(f"[Train] Training for {epochs} epoch(s) on {len(records)} pairs...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=max(5, len(records) // 20),
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="no",           # save manually below
            report_to="none",
        ),
    )
    trainer.train()

    # ── Save HuggingFace adapter ─────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"ADAPTER_PATH:{adapter_path}")

    # ── Attempt GGUF conversion ──────────────────────────────────
    convert_script = Path(bin_path) / "convert_lora_to_gguf.py"
    if not convert_script.exists():
        # Also check one level up (bin/ relative to project root or cwd)
        for candidate in [
            Path("bin") / "convert_lora_to_gguf.py",
            Path("bin") / "llama.cpp" / "convert_lora_to_gguf.py",
        ]:
            if candidate.exists():
                convert_script = candidate
                break

    if convert_script.exists():
        gguf_path = str(Path(output_dir) / "adapter.gguf")
        print(f"[Train] Converting adapter to GGUF: {gguf_path}")
        result = subprocess.run(
            [
                sys.executable, str(convert_script),
                "--base", base_model,
                "--outfile", gguf_path,
                adapter_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"GGUF_PATH:{gguf_path}")
        else:
            print(f"[Train] GGUF conversion failed (non-fatal):\n{result.stderr[:500]}")
    else:
        print(
            "[Train] convert_lora_to_gguf.py not found in bin/. "
            "To use with KoboldCpp, convert manually:\n"
            "  python llama.cpp/convert_lora_to_gguf.py "
            f"--base {base_model} --outfile {output_dir}/adapter.gguf {adapter_path}"
        )

    print("[Train] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Talon")
    parser.add_argument("--pairs",      required=True,  help="Path to training_pairs.jsonl")
    parser.add_argument("--base-model", required=True,  help="HuggingFace model path or name")
    parser.add_argument("--output-dir", required=True,  help="Directory for adapter output")
    parser.add_argument("--bin-path",   default="bin",  help="llama.cpp bin directory")
    parser.add_argument("--config",     default="{}",   help="JSON string of training config")
    args = parser.parse_args()

    cfg = {}
    try:
        cfg = json.loads(args.config)
    except json.JSONDecodeError:
        print("[Train] Warning: could not parse --config JSON, using defaults.", file=sys.stderr)

    train(
        pairs_file=args.pairs,
        base_model=args.base_model,
        output_dir=args.output_dir,
        bin_path=args.bin_path,
        lora_r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        epochs=int(cfg.get("epochs", 3)),
        batch_size=int(cfg.get("batch_size", 2)),
        max_seq_length=int(cfg.get("max_seq_length", 2048)),
    )


if __name__ == "__main__":
    main()
