"""train_reflection_lora.py — Train a LoRA adapter to fix reflection quality.

Standalone script that trains on curated seed examples to shift the model's
weight-level bias away from contemplative/abstract monologue ("soft breath
era") toward diverse, concrete, grounded reflections.

Usage:
    python scripts/train_reflection_lora.py

    # With custom settings:
    python scripts/train_reflection_lora.py --epochs 5 --rank 16 --lr 2e-4

    # Dry run (just show data stats, don't train):
    python scripts/train_reflection_lora.py --dry-run

Requires:
    pip install unsloth trl datasets

Output:
    data/lora/peft_adapter/     — HuggingFace PEFT adapter
    data/lora/peft_adapter/adapter.gguf  — GGUF for KoboldCpp (if converter found)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS_FILE = PROJECT_ROOT / "data" / "lora" / "reflection_seeds.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "lora" / "peft_adapter"
GGUF_OUTPUT = OUTPUT_DIR / "adapter.gguf"

# Defaults
# Use the pre-quantized 4-bit version — much smaller download (~5GB vs ~16GB)
DEFAULT_BASE_MODEL = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
DEFAULT_RANK = 16
DEFAULT_ALPHA = 32
DEFAULT_EPOCHS = 10
DEFAULT_LR = 2e-4
DEFAULT_BATCH = 2
DEFAULT_MAX_SEQ = 2048


def load_seeds(path: Path) -> list[dict]:
    """Load curated seed examples from JSONL."""
    if not path.exists():
        print(f"[Error] Seed file not found: {path}")
        sys.exit(1)

    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "conversations" in record:
                    examples.append(record)
                else:
                    print(f"  [Warning] Line {i}: missing 'conversations' key, skipped")
            except json.JSONDecodeError as e:
                print(f"  [Warning] Line {i}: JSON error: {e}")
    return examples


def format_chatml(examples: list[dict]) -> list[str]:
    """Convert conversation examples to ChatML format strings."""
    texts = []
    for ex in examples:
        chatml = ""
        for msg in ex["conversations"]:
            role = msg["role"]
            content = msg["content"]
            chatml += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        texts.append(chatml)
    return texts


def train(args):
    """Run the training pipeline."""
    # Step 1: Load data
    print(f"\n{'='*60}")
    print(f"  Talon Reflection LoRA Training")
    print(f"{'='*60}\n")

    print(f"[1/6] Loading seed examples from {SEEDS_FILE}...")
    examples = load_seeds(SEEDS_FILE)
    print(f"       {len(examples)} examples loaded.")

    # Show stats
    total_chars = sum(
        len(msg["content"])
        for ex in examples
        for msg in ex["conversations"]
        if msg["role"] == "assistant"
    )
    avg_chars = total_chars // len(examples) if examples else 0
    print(f"       Total assistant text: {total_chars:,} chars")
    print(f"       Average response: {avg_chars:,} chars")

    if args.dry_run:
        print(f"\n[Dry run] Would train on {len(examples)} examples.")
        print(f"  Base model:  {args.base_model}")
        print(f"  Rank:        {args.rank}")
        print(f"  Alpha:       {args.alpha}")
        print(f"  Epochs:      {args.epochs}")
        print(f"  LR:          {args.lr}")
        print(f"  Batch size:  {args.batch}")
        print(f"  Output:      {OUTPUT_DIR}")
        return

    # Step 2: Import and load model
    print(f"\n[2/6] Loading base model: {args.base_model}...")
    print(f"       (First run downloads the model — ~5GB for 4-bit quantized)")

    is_vision = "VL" in args.base_model.upper()

    if is_vision:
        from unsloth import FastVisionModel as ModelClass
        print(f"       Using FastVisionModel (vision-language)")
    else:
        from unsloth import FastLanguageModel as ModelClass
        print(f"       Using FastLanguageModel (text-only)")

    model, tokenizer = ModelClass.from_pretrained(
        model_name=args.base_model,
        max_seq_length=DEFAULT_MAX_SEQ,
        load_in_4bit=True,
        dtype=None,
    )

    # Step 3: Apply LoRA
    print(f"\n[3/6] Applying LoRA (rank={args.rank}, alpha={args.alpha})...")

    if is_vision:
        model = ModelClass.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=0,
            bias="none",
            use_rslora=False,
            use_gradient_checkpointing="unsloth",
        )
    else:
        model = ModelClass.get_peft_model(
            model,
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

    # Step 4: Prepare dataset
    print(f"\n[4/6] Preparing dataset ({len(examples)} conversations)...")
    from datasets import Dataset

    texts = format_chatml(examples)
    dataset = Dataset.from_dict({"text": texts})

    # Step 5: Train
    print(f"\n[5/6] Training for {args.epochs} epochs...")
    print(f"       batch_size={args.batch}, lr={args.lr}")

    from trl import SFTTrainer, SFTConfig

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=max(1, 4 // args.batch),
        learning_rate=args.lr,
        warmup_steps=min(10, len(examples) // args.batch),
        logging_steps=5,
        save_strategy="no",  # Save manually at the end
        fp16=True,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=DEFAULT_MAX_SEQ,
        packing=not is_vision,
    )

    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": dataset,
        "args": sft_config,
    }

    if is_vision:
        from unsloth.trainer import UnslothVisionDataCollator
        trainer_kwargs["data_collator"] = UnslothVisionDataCollator(
            model, tokenizer
        )

    trainer = SFTTrainer(**trainer_kwargs)

    train_result = trainer.train()
    print(f"\n       Training complete! Final loss: {train_result.training_loss:.4f}")

    # Save adapter
    print(f"\n[6/6] Saving adapter to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"       PEFT adapter saved!")

    # Attempt GGUF conversion
    converter_candidates = [
        PROJECT_ROOT / "bin" / "convert_lora_to_gguf.py",
        Path.home() / "llama.cpp" / "convert_lora_to_gguf.py",
    ]
    converter = None
    for c in converter_candidates:
        if c.is_file():
            converter = c
            break

    if not converter:
        # Try downloading
        try:
            import urllib.request
            converter = PROJECT_ROOT / "bin" / "convert_lora_to_gguf.py"
            converter.parent.mkdir(parents=True, exist_ok=True)
            print(f"       Downloading GGUF converter...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/ggerganov/llama.cpp/"
                "master/convert_lora_to_gguf.py",
                str(converter),
            )
        except Exception as e:
            print(f"       Could not download converter: {e}")
            converter = None

    if converter:
        print(f"       Converting to GGUF...")
        result = subprocess.run(
            [sys.executable, str(converter),
             "--outfile", str(GGUF_OUTPUT),
             str(OUTPUT_DIR)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            print(f"       GGUF adapter: {GGUF_OUTPUT}")
        else:
            print(f"       GGUF conversion failed: {result.stderr[:300]}")
            print(f"       (PEFT adapter still usable — convert manually)")
    else:
        print(f"       No GGUF converter found — convert manually later:")
        print(f"       python convert_lora_to_gguf.py --outfile {GGUF_OUTPUT} {OUTPUT_DIR}")

    # Free GPU
    del model, tokenizer, trainer
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"\n{'='*60}")
    print(f"  Done! Restart KoboldCpp to load the adapter.")
    print(f"  The bat file will auto-detect it at:")
    print(f"  {GGUF_OUTPUT}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter to fix Talon's reflection quality"
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help=f"HuggingFace model name (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK,
                        help=f"LoRA rank (default: {DEFAULT_RANK})")
    parser.add_argument("--alpha", type=int, default=DEFAULT_ALPHA,
                        help=f"LoRA alpha (default: {DEFAULT_ALPHA})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help=f"Batch size (default: {DEFAULT_BATCH})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show data stats without training")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
