"""lora_train.py — Trigger LoRA fine-tuning on accumulated training pairs.

Disabled by default. Enable via Settings → Talent Config → lora_train,
then set base_model_path to your HuggingFace model directory.

Flow:
  1. Prerequisites check (pairs, base model, unsloth)
  2. Stop inference server (if running)
  3. Launch scripts/train_lora.py in a background thread
  4. Return immediately — notify + restart server when done

Requires: pip install unsloth trl
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from talents.base import BaseTalent


class LoraTrainTalent(BaseTalent):
    name = "lora_train"
    description = "Fine-tune the local LLM on collected training pairs using LoRA"
    keywords = [
        "train from history", "start lora training", "fine tune", "retrain",
        "train the model", "lora training", "improve from corrections",
        "run training", "start fine tuning", "finetune",
    ]
    examples = [
        "start LoRA training",
        "train from history",
        "fine tune the model",
        "retrain from corrections",
        "run LoRA fine-tuning",
    ]
    priority = 60

    def get_config_schema(self) -> dict:
        return {
            "fields": [
                {"key": "enabled",         "label": "Enable LoRA Training",      "type": "bool",   "default": False},
                {"key": "base_model_path", "label": "HuggingFace Model Path",    "type": "string", "default": ""},
                {"key": "output_dir",      "label": "Adapter Output Directory",  "type": "string", "default": "data/lora_adapters"},
                {"key": "min_pairs",       "label": "Minimum Pairs to Train",    "type": "int",    "default": 100},
                {"key": "lora_r",          "label": "LoRA Rank",                 "type": "int",    "default": 16},
                {"key": "lora_alpha",      "label": "LoRA Alpha",                "type": "int",    "default": 16},
                {"key": "epochs",          "label": "Training Epochs",           "type": "int",    "default": 3},
                {"key": "batch_size",      "label": "Batch Size",                "type": "int",    "default": 2},
                {"key": "max_seq_length",  "label": "Max Sequence Length",       "type": "int",    "default": 2048},
            ]
        }

    def can_handle(self, command: str) -> bool:
        if not self.talent_config.get("enabled", False):
            return False
        return self.keyword_match(command)

    # ── Helpers ────────────────────────────────────────────────────

    def _count_pairs(self, pairs_file: Path) -> int:
        try:
            return sum(1 for line in pairs_file.open(encoding="utf-8") if line.strip())
        except OSError:
            return 0

    def _rough_eta(self, pair_count: int, epochs: int) -> str:
        """Very rough estimate: ~50 pairs/min on mobile 4090 at batch_size=2."""
        minutes = max(1, (pair_count * epochs) // 50)
        if minutes < 60:
            return f"~{minutes} min"
        return f"~{minutes // 60}h {minutes % 60}min"

    # ── Execute ────────────────────────────────────────────────────

    def execute(self, command: str, context: dict) -> dict:
        cfg            = self.talent_config
        server_manager = context.get("server_manager")
        notify_cb      = context.get("notify")

        # ── 1. Prerequisites ───────────────────────────────────────
        pairs_file = Path("data/training_pairs.jsonl")
        if not pairs_file.exists():
            return {
                "success": False,
                "response": (
                    "No training pairs collected yet. Use Talon normally "
                    "(corrections and web searches accumulate pairs automatically)."
                ),
                "actions_taken": [],
            }

        pair_count = self._count_pairs(pairs_file)
        min_pairs  = int(cfg.get("min_pairs", 100))
        if pair_count < min_pairs:
            return {
                "success": False,
                "response": (
                    f"Only {pair_count} training pair(s) collected "
                    f"({min_pairs} minimum required). "
                    "Keep using Talon to accumulate more data."
                ),
                "actions_taken": [],
            }

        base_model = cfg.get("base_model_path", "").strip()
        if not base_model:
            return {
                "success": False,
                "response": (
                    "No base model path set. Go to Settings → Talent Config → lora_train "
                    "and set 'HuggingFace Model Path' to your local model directory "
                    "(e.g. C:/models/Qwen2.5-7B-Instruct)."
                ),
                "actions_taken": [],
            }

        if not Path(base_model).exists():
            return {
                "success": False,
                "response": (
                    f"Base model path not found: {base_model}\n"
                    "Download the HuggingFace weights first "
                    "(not the GGUF — the full safetensors model)."
                ),
                "actions_taken": [],
            }

        try:
            import unsloth  # noqa: F401
        except ImportError:
            return {
                "success": False,
                "response": (
                    "Unsloth is not installed. Run:\n"
                    "  pip install unsloth trl\n"
                    "then try again."
                ),
                "actions_taken": [],
            }

        # ── 2. Stop inference server ───────────────────────────────
        was_running = bool(server_manager and server_manager.is_running())
        if was_running:
            print("   [LoRA] Stopping inference server for training...")
            server_manager.stop()
            time.sleep(2)

        # ── 3. Build training config ───────────────────────────────
        output_dir   = cfg.get("output_dir", "data/lora_adapters")
        train_config = {
            "lora_r":          int(cfg.get("lora_r",          16)),
            "lora_alpha":      int(cfg.get("lora_alpha",       16)),
            "epochs":          int(cfg.get("epochs",           3)),
            "batch_size":      int(cfg.get("batch_size",       2)),
            "max_seq_length":  int(cfg.get("max_seq_length",   2048)),
        }
        bin_path = context.get("config", {}).get("llm_server", {}).get("bin_path", "bin")

        # ── 4. Launch training in background thread ────────────────
        def _train_and_notify():
            gguf_path = None
            try:
                script = Path(__file__).parent.parent / "scripts" / "train_lora.py"
                result = subprocess.run(
                    [
                        sys.executable, str(script),
                        "--pairs",      str(pairs_file),
                        "--base-model", base_model,
                        "--output-dir", output_dir,
                        "--bin-path",   bin_path,
                        "--config",     json.dumps(train_config),
                    ],
                    capture_output=False,   # Let output stream to console
                    timeout=7200,           # 2-hour hard cap
                )

                if result.returncode == 0:
                    # Look for GGUF adapter output
                    gguf_candidate = Path(output_dir) / "adapter.gguf"
                    if gguf_candidate.exists():
                        gguf_path = str(gguf_candidate)

                    msg = f"LoRA training complete! {pair_count} pairs, {train_config['epochs']} epoch(s)."
                    if gguf_path:
                        msg += f"\nAdapter GGUF: {gguf_path}\nAdd to Settings → extra_args: --lora {gguf_path}"
                    else:
                        adapter_hf = str(Path(output_dir) / "adapter")
                        msg += f"\nHuggingFace adapter: {adapter_hf}"
                    print(f"   [LoRA] {msg}")
                    if notify_cb:
                        try:
                            notify_cb("LoRA Training Complete", msg[:120])
                        except Exception:
                            pass
                else:
                    print(f"   [LoRA] Training failed (exit {result.returncode}).")
                    if notify_cb:
                        try:
                            notify_cb("LoRA Training Failed", "Check the console for details.")
                        except Exception:
                            pass

            except subprocess.TimeoutExpired:
                print("   [LoRA] Training timed out after 2 hours.")
            except Exception as e:
                print(f"   [LoRA] Training error: {e}")
            finally:
                if was_running and server_manager:
                    print("   [LoRA] Restarting inference server...")
                    try:
                        server_manager.start()
                    except Exception as e:
                        print(f"   [LoRA] Server restart failed: {e}")

        threading.Thread(target=_train_and_notify, daemon=True).start()

        eta = self._rough_eta(pair_count, train_config["epochs"])
        return {
            "success": True,
            "response": (
                f"Fine-tuning started on {pair_count} training pair(s) "
                f"({train_config['epochs']} epoch(s), {eta}). "
                "Inference server paused. I'll notify you when training is complete "
                "and restart the server automatically."
            ),
            "actions_taken": [
                {
                    "action": {"type": "lora_training", "pairs": pair_count},
                    "result": "training started",
                    "success": True,
                }
            ],
        }
