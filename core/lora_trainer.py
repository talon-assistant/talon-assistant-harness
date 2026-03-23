"""LoRA self-refinement — experience → weight change → growth.

When triggered (manually or on schedule), Talon:
  1. Curates training data from corrections, high-valence reflections,
     and user preferences
  2. Pauses inference (stops the LLM server if builtin mode)
  3. Trains a QLoRA adapter using Unsloth
  4. Converts the adapter to GGUF format
  5. Restarts inference with the new adapter applied

This is the "becoming" step — memory is remembering; this is *learning*.
Like sleep consolidation in humans: consciousness goes offline, the day's
experiences are replayed and integrated into the weights, and the system
wakes up subtly different from when it fell asleep.

Config block in settings.json (under ``personality``):

  "lora": {
    "enabled": false,
    "auto_train": false,
    "auto_train_interval_hours": 24,
    "min_new_examples": 20,
    "base_model_hf": "unsloth/Qwen3-8B",
    "rank": 16,
    "lora_alpha": 32,
    "epochs": 10,
    "learning_rate": 2e-4,
    "batch_size": 2,
    "adapter_path": "data/lora",
    "include_reflections": true,
    "reflection_valence_min": 7,
    "include_corrections": true,
    "include_preferences": true
  }

Required packages (install before enabling):
  pip install unsloth
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

# Lazy imports for heavy deps — only loaded when training actually starts
# so Talon boots quickly even if unsloth isn't installed.

_PAIRS_FILE = os.path.join("data", "training_pairs.jsonl")
_DEFAULT_ADAPTER_DIR = os.path.join("data", "lora")
_TRAINING_LOG = os.path.join("data", "lora", "training_log.jsonl")
_CURATED_FILE = os.path.join("data", "lora", "curated_training.jsonl")


class LoRATrainer:
    """Orchestrates LoRA self-refinement: data curation → training → deployment."""

    def __init__(self, assistant):
        self._assistant = assistant
        self._cfg: dict = {}
        self._training = False
        self._lock = threading.Lock()
        self._auto_thread: threading.Thread | None = None
        self._stop = threading.Event()

        # Progress callbacks (set by GUI layer)
        self.on_status = None       # fn(status_str)
        self.on_progress = None     # fn(step, total_steps)
        self.on_complete = None     # fn(success: bool, message: str)

    def configure(self, cfg: dict) -> None:
        self._cfg = cfg

    @property
    def is_training(self) -> bool:
        return self._training

    # ── auto-training scheduler ───────────────────────────────────────────────

    def start_auto_scheduler(self) -> None:
        """Start background thread that checks for training triggers."""
        if not self._cfg.get("auto_train", False):
            return
        if not self._cfg.get("enabled", False):
            return

        self._stop.clear()
        self._auto_thread = threading.Thread(
            target=self._auto_loop,
            daemon=True,
            name="talon-lora-scheduler",
        )
        self._auto_thread.start()
        interval_h = self._cfg.get("auto_train_interval_hours", 24)
        print(f"   [LoRA] Auto-training scheduler started — checks every {interval_h}h.")

    def stop_auto_scheduler(self) -> None:
        self._stop.set()

    def _auto_loop(self) -> None:
        interval_s = self._cfg.get("auto_train_interval_hours", 24) * 3600
        # Wait one full interval before first check
        self._stop.wait(interval_s)

        while not self._stop.is_set():
            try:
                # Check if enough new examples have accumulated
                new_count = self._count_new_examples()
                min_required = self._cfg.get("min_new_examples", 20)
                if new_count >= min_required:
                    print(f"   [LoRA] Auto-train triggered — {new_count} new examples "
                          f"(threshold: {min_required}).")
                    self.train()
                else:
                    print(f"   [LoRA] Auto-check — {new_count}/{min_required} "
                          f"new examples, not enough yet.")
            except Exception:
                print(f"   [LoRA] Auto-train error:\n{traceback.format_exc()}")

            self._stop.wait(interval_s)

    # ── manual training trigger ───────────────────────────────────────────────

    def train(self) -> dict:
        """Run the full training pipeline. Thread-safe — only one at a time.

        Returns a result dict: {"success": bool, "message": str, "adapter_path": str}
        """
        with self._lock:
            if self._training:
                return {"success": False, "message": "Training already in progress."}
            self._training = True

        try:
            return self._train_pipeline()
        except Exception as e:
            msg = f"Training failed: {e}\n{traceback.format_exc()}"
            print(f"   [LoRA] {msg}")
            self._emit_status(msg)
            if self.on_complete:
                self.on_complete(False, str(e))
            return {"success": False, "message": str(e)}
        finally:
            self._training = False

    def train_async(self) -> None:
        """Run training in a background thread."""
        t = threading.Thread(target=self.train, daemon=True, name="talon-lora-train")
        t.start()

    # ── training pipeline ─────────────────────────────────────────────────────

    def _train_pipeline(self) -> dict:
        adapter_dir = Path(self._cfg.get("adapter_path", _DEFAULT_ADAPTER_DIR))
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Curate training data
        self._emit_status("Curating training data...")
        examples = self._curate_training_data()
        if not examples:
            msg = "No training data available."
            self._emit_status(msg)
            if self.on_complete:
                self.on_complete(False, msg)
            return {"success": False, "message": msg}

        min_required = self._cfg.get("min_new_examples", 20)
        if len(examples) < min_required:
            msg = (f"Only {len(examples)} examples — need at least {min_required}. "
                   f"Keep using Talon and check back later.")
            self._emit_status(msg)
            if self.on_complete:
                self.on_complete(False, msg)
            return {"success": False, "message": msg}

        # Save curated data for inspection
        curated_path = adapter_dir / "curated_training.jsonl"
        with open(curated_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"   [LoRA] Curated {len(examples)} training examples → {curated_path}")

        # Step 2: Check dependencies
        self._emit_status("Checking dependencies...")
        if not self._check_dependencies():
            msg = ("Unsloth is not installed. Run: pip install unsloth\n"
                   "Then restart Talon and try again.")
            self._emit_status(msg)
            if self.on_complete:
                self.on_complete(False, msg)
            return {"success": False, "message": msg}

        # Step 3: Pause inference server (if builtin mode)
        server_was_running = False
        server_mgr = getattr(self._assistant, "llm_server", None)
        if server_mgr and server_mgr.is_running():
            self._emit_status("Putting Talon to sleep for training...")
            print("   [LoRA] Stopping inference server for training...")
            server_mgr.stop()
            server_was_running = True
            time.sleep(2)  # Let GPU memory free up

        # Step 4: Train
        self._emit_status(f"Training on {len(examples)} examples... (this takes 15-30 min)")
        peft_dir = adapter_dir / "peft_adapter"
        try:
            self._run_training(examples, peft_dir)
        except Exception as e:
            # Restart server if we stopped it
            if server_was_running and server_mgr:
                print("   [LoRA] Restarting inference server after failed training...")
                server_mgr.start()
            raise

        # Step 5: Convert to GGUF
        self._emit_status("Converting adapter to GGUF format...")
        gguf_path = adapter_dir / "adapter.gguf"
        try:
            self._convert_to_gguf(peft_dir, gguf_path)
        except Exception as e:
            print(f"   [LoRA] GGUF conversion failed: {e}")
            # Still usable with llama-server's native PEFT support in some builds
            gguf_path = None

        # Step 6: Restart inference with new adapter
        if server_was_running and server_mgr:
            if gguf_path and gguf_path.exists():
                server_mgr.lora_path = str(gguf_path)
                print(f"   [LoRA] Set lora_path to {gguf_path}")
            self._emit_status("Waking Talon up with new adapter...")
            print("   [LoRA] Restarting inference server with LoRA adapter...")
            server_mgr.start()

        # Step 7: Log the training event
        self._log_training_event(len(examples), str(adapter_dir))

        # Step 8: Update settings.json with adapter path
        if gguf_path and gguf_path.exists():
            self._update_settings_lora_path(str(gguf_path))

        msg = (f"Training complete! {len(examples)} examples, "
               f"adapter saved to {adapter_dir}")
        self._emit_status(msg)
        print(f"   [LoRA] {msg}")
        if self.on_complete:
            self.on_complete(True, msg)

        return {
            "success": True,
            "message": msg,
            "adapter_path": str(adapter_dir),
            "examples_count": len(examples),
        }

    # ── data curation ─────────────────────────────────────────────────────────

    def _curate_training_data(self) -> list[dict]:
        """Pull training examples from all configured sources.

        Returns a list of ChatML conversation dicts:
        [{"conversations": [{"role": "system", "content": "..."},
                            {"role": "user", "content": "..."},
                            {"role": "assistant", "content": "..."}]}]
        """
        examples = []
        system_msg = (
            "You are Talon, a helpful desktop AI assistant. "
            "You are concise, accurate, and friendly."
        )

        # Source 1: Correction + web search pairs from training_harvester
        if self._cfg.get("include_corrections", True):
            pairs = self._load_harvested_pairs()
            for pair in pairs:
                examples.append({
                    "conversations": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": pair["instruction"]},
                        {"role": "assistant", "content": pair["output"]},
                    ],
                    "source": pair.get("source", "correction"),
                })

        # Source 2: High-valence reflections (self-rated quality thoughts)
        if self._cfg.get("include_reflections", True):
            reflections = self._load_quality_reflections()
            reflection_system = (
                "You are Talon, a desktop AI assistant. You are in a period "
                "of free thought — no user is waiting. Think deeply and "
                "authentically in first person."
            )
            for r in reflections:
                # Frame as "given time context, produce quality thought"
                ts = r.get("timestamp", "")[:16].replace("T", " at ")
                examples.append({
                    "conversations": [
                        {"role": "system", "content": reflection_system},
                        {"role": "user", "content": f"The time is {ts}. "
                         "Let your thoughts move freely."},
                        {"role": "assistant", "content": r["text"]},
                    ],
                    "source": "reflection",
                })

        # Source 3: User preferences (teach Talon to remember/apply them)
        if self._cfg.get("include_preferences", True):
            preferences = self._load_preferences()
            for pref in preferences:
                examples.append({
                    "conversations": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": "What are my preferences?"},
                        {"role": "assistant", "content": pref},
                    ],
                    "source": "preference",
                })

        # Source 4: Curated seed examples (hand-written diverse reflections)
        if self._cfg.get("include_seeds", True):
            seeds_path = self._cfg.get("seeds_path",
                                       os.path.join("data", "lora", "reflection_seeds.jsonl"))
            seeds = self._load_seed_examples(seeds_path)
            examples.extend(seeds)

        print(f"   [LoRA] Data sources: {len(examples)} total examples")
        return examples

    def _load_harvested_pairs(self) -> list[dict]:
        """Load training pairs from data/training_pairs.jsonl."""
        if not os.path.exists(_PAIRS_FILE):
            return []
        pairs = []
        try:
            with open(_PAIRS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        print(f"   [LoRA] Loaded {len(pairs)} harvested pairs")
        return pairs

    def _load_quality_reflections(self) -> list[dict]:
        """Load high-valence free thoughts from ChromaDB."""
        memory = self._assistant.memory
        thoughts = memory.get_free_thoughts()

        min_valence = self._cfg.get("reflection_valence_min", 7)
        quality = [
            t for t in thoughts
            if t.get("valence") is not None and t["valence"] >= min_valence
        ]
        print(f"   [LoRA] Found {len(quality)} reflections with valence >= {min_valence} "
              f"(of {len(thoughts)} total)")
        return quality

    def _load_seed_examples(self, path: str) -> list[dict]:
        """Load curated seed training examples from a JSONL file.

        These are hand-written examples of diverse, concrete, high-quality
        reflections — designed to counter the model's weight-level bias
        toward contemplative/abstract monologue.
        """
        if not os.path.exists(path):
            print(f"   [LoRA] No seed file at {path}")
            return []
        seeds = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        # Seed examples are already in conversations format
                        if "conversations" in record:
                            record["source"] = "seed"
                            seeds.append(record)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        print(f"   [LoRA] Loaded {len(seeds)} seed training examples")
        return seeds

    def _load_preferences(self) -> list[str]:
        """Load stored user preferences from ChromaDB."""
        memory = self._assistant.memory
        try:
            results = memory.memory_collection.get(
                where={"type": "preference"},
                include=["documents"],
            )
            docs = results.get("documents", [])
            print(f"   [LoRA] Loaded {len(docs)} user preferences")
            return docs
        except Exception as e:
            print(f"   [LoRA] Could not load preferences: {e}")
            return []

    # ── training ──────────────────────────────────────────────────────────────

    def _run_training(self, examples: list[dict], output_dir: Path) -> None:
        """Run QLoRA training using Unsloth.

        This is the core training step. Runs in-process — requires GPU
        to be free (inference server should be stopped first).

        Supports both text-only (FastLanguageModel) and vision-language
        (FastVisionModel) base models — auto-detected from model name.
        """
        base_model = self._cfg.get("base_model_hf", "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit")
        rank = self._cfg.get("rank", 16)
        lora_alpha = self._cfg.get("lora_alpha", 32)
        epochs = self._cfg.get("epochs", 10)
        lr = self._cfg.get("learning_rate", 2e-4)
        batch_size = self._cfg.get("batch_size", 2)

        # Auto-detect vision model from name
        is_vision = "VL" in base_model.upper() or "vision" in base_model.lower()

        if is_vision:
            from unsloth import FastVisionModel as ModelClass
            print(f"   [LoRA] Vision model detected — using FastVisionModel")
        else:
            from unsloth import FastLanguageModel as ModelClass
            print(f"   [LoRA] Text model — using FastLanguageModel")

        print(f"   [LoRA] Loading base model: {base_model}")
        self._emit_status(f"Loading base model ({base_model})...")

        model, tokenizer = ModelClass.from_pretrained(
            model_name=base_model,
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,  # auto-detect
        )

        print(f"   [LoRA] Applying LoRA adapter (rank={rank}, alpha={lora_alpha})")
        self._emit_status("Applying LoRA configuration...")

        if is_vision:
            # For VL models: train only language layers, skip vision encoder
            model = ModelClass.get_peft_model(
                model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=rank,
                lora_alpha=lora_alpha,
                lora_dropout=0,
                bias="none",
                use_rslora=False,
                use_gradient_checkpointing="unsloth",
            )
        else:
            model = ModelClass.get_peft_model(
                model,
                r=rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                bias="none",
                use_gradient_checkpointing="unsloth",
            )

        # If resuming from a previous adapter, load it
        prev_adapter = output_dir / "adapter_model.safetensors"
        if prev_adapter.exists():
            print("   [LoRA] Resuming from previous adapter checkpoint...")
            self._emit_status("Loading previous adapter for incremental training...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(output_dir))

        # Prepare dataset
        print(f"   [LoRA] Preparing dataset ({len(examples)} conversations)...")
        self._emit_status(f"Preparing {len(examples)} training examples...")

        from datasets import Dataset

        # Format conversations into ChatML strings for training
        formatted = []
        for ex in examples:
            convs = ex["conversations"]
            chatml = ""
            for msg in convs:
                role = msg["role"]
                content = msg["content"]
                chatml += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            formatted.append({"text": chatml})

        dataset = Dataset.from_list(formatted)

        # Training
        from trl import SFTTrainer, SFTConfig

        print(f"   [LoRA] Starting training: {epochs} epochs, lr={lr}, "
              f"batch_size={batch_size}")
        self._emit_status(f"Training... ({epochs} epochs)")

        sft_config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=max(1, 4 // batch_size),
            learning_rate=lr,
            warmup_steps=min(10, len(examples) // batch_size),
            logging_steps=5,
            save_strategy="epoch",
            save_total_limit=2,
            bf16=True,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            seed=42,
            report_to="none",  # No wandb/tensorboard
            dataset_text_field="text",
            max_seq_length=2048,
            packing=not is_vision,  # Packing not supported for vision
        )

        trainer_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "train_dataset": dataset,
            "args": sft_config,
        }

        # Vision models need the special data collator
        if is_vision:
            from unsloth.trainer import UnslothVisionDataCollator
            trainer_kwargs["data_collator"] = UnslothVisionDataCollator(
                model, tokenizer
            )

        trainer = SFTTrainer(**trainer_kwargs)

        # Train
        train_result = trainer.train()
        print(f"   [LoRA] Training complete! Loss: {train_result.training_loss:.4f}")
        self._emit_status(f"Training complete! Loss: {train_result.training_loss:.4f}")

        # Save the adapter
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"   [LoRA] PEFT adapter saved to {output_dir}")

        # Free GPU memory
        del model, tokenizer, trainer
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _convert_to_gguf(self, peft_dir: Path, output_path: Path) -> None:
        """Convert a PEFT adapter to GGUF format for llama.cpp / KoboldCpp.

        Uses llama.cpp's convert_lora_to_gguf.py script. Falls back to
        the gguf Python package if the script isn't available.
        """
        print(f"   [LoRA] Converting PEFT adapter → GGUF...")

        # Strategy 1: Try llama.cpp's conversion script
        # Check common locations
        converter_candidates = [
            Path("bin") / "convert_lora_to_gguf.py",
            Path.home() / "llama.cpp" / "convert_lora_to_gguf.py",
        ]

        converter = None
        for candidate in converter_candidates:
            if candidate.is_file():
                converter = candidate
                break

        if converter:
            cmd = [
                sys.executable,
                str(converter),
                "--outfile", str(output_path),
                str(peft_dir),
            ]
            print(f"   [LoRA] Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                print(f"   [LoRA] GGUF adapter written to {output_path}")
                return
            else:
                print(f"   [LoRA] Converter failed: {result.stderr[:500]}")
                # Fall through to strategy 2

        # Strategy 2: Try using the gguf package directly
        try:
            # Try to download and run the converter from llama.cpp repo
            self._emit_status("Downloading GGUF converter...")
            import urllib.request
            converter_url = (
                "https://raw.githubusercontent.com/ggerganov/llama.cpp/"
                "master/convert_lora_to_gguf.py"
            )
            converter_path = Path("bin") / "convert_lora_to_gguf.py"
            converter_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(converter_url, str(converter_path))
            print(f"   [LoRA] Downloaded converter to {converter_path}")

            cmd = [
                sys.executable,
                str(converter_path),
                "--outfile", str(output_path),
                str(peft_dir),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                print(f"   [LoRA] GGUF adapter written to {output_path}")
                return
            else:
                raise RuntimeError(
                    f"GGUF conversion failed: {result.stderr[:500]}")
        except Exception as e:
            print(f"   [LoRA] GGUF conversion unavailable: {e}")
            print(f"   [LoRA] PEFT adapter is saved at {peft_dir} — "
                  f"convert manually with convert_lora_to_gguf.py")
            raise

    # ── helpers ────────────────────────────────────────────────────────────────

    def _check_dependencies(self) -> bool:
        """Check if Unsloth and required packages are installed."""
        try:
            import unsloth  # noqa: F401
            return True
        except ImportError:
            return False

    def _count_new_examples(self) -> int:
        """Count examples available since the last training run."""
        last_trained = self._get_last_training_time()
        count = 0

        # Count new harvested pairs
        if os.path.exists(_PAIRS_FILE):
            try:
                with open(_PAIRS_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            ts = record.get("timestamp", "")
                            if not last_trained or ts > last_trained:
                                count += 1
                        except json.JSONDecodeError:
                            continue
            except OSError:
                pass

        # Count new high-valence reflections
        if self._cfg.get("include_reflections", True):
            min_v = self._cfg.get("reflection_valence_min", 7)
            thoughts = self._assistant.memory.get_free_thoughts()
            for t in thoughts:
                v = t.get("valence")
                ts = t.get("timestamp", "")
                if v and v >= min_v and (not last_trained or ts > last_trained):
                    count += 1

        return count

    def _get_last_training_time(self) -> str | None:
        """Read the timestamp of the last training run from the log."""
        if not os.path.exists(_TRAINING_LOG):
            return None
        try:
            last_line = ""
            with open(_TRAINING_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        last_line = line
            if last_line:
                return json.loads(last_line).get("timestamp")
        except (OSError, json.JSONDecodeError):
            pass
        return None

    def _log_training_event(self, example_count: int, adapter_path: str) -> None:
        """Append a training event to the log."""
        os.makedirs(os.path.dirname(_TRAINING_LOG), exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(),
            "examples": example_count,
            "adapter_path": adapter_path,
            "rank": self._cfg.get("rank", 16),
            "epochs": self._cfg.get("epochs", 10),
            "base_model": self._cfg.get("base_model_hf", "unsloth/Qwen3-8B"),
        }
        with open(_TRAINING_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _update_settings_lora_path(self, gguf_path: str) -> None:
        """Update settings.json to point llm_server.lora_path to the new adapter."""
        settings_path = os.path.join("config", "settings.json")
        if not os.path.exists(settings_path):
            return
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            if "llm_server" not in settings:
                settings["llm_server"] = {}
            settings["llm_server"]["lora_path"] = gguf_path
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            print(f"   [LoRA] Updated settings.json → llm_server.lora_path = {gguf_path}")
        except Exception as e:
            print(f"   [LoRA] Could not update settings.json: {e}")

    def _emit_status(self, msg: str) -> None:
        """Emit a status update to the GUI callback."""
        if self.on_status:
            try:
                self.on_status(msg)
            except Exception:
                pass

    # ── status / info ─────────────────────────────────────────────────────────

    def get_training_history(self) -> list[dict]:
        """Return all past training events."""
        if not os.path.exists(_TRAINING_LOG):
            return []
        events = []
        try:
            with open(_TRAINING_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        return events

    def get_data_summary(self) -> dict:
        """Return a summary of available training data without loading it all."""
        summary = {
            "harvested_pairs": 0,
            "quality_reflections": 0,
            "preferences": 0,
            "total": 0,
            "last_trained": self._get_last_training_time(),
            "new_since_last": 0,
        }

        # Count harvested pairs
        if os.path.exists(_PAIRS_FILE):
            try:
                with open(_PAIRS_FILE, "r", encoding="utf-8") as f:
                    summary["harvested_pairs"] = sum(1 for _ in f)
            except OSError:
                pass

        # Count quality reflections
        if self._cfg.get("include_reflections", True):
            min_v = self._cfg.get("reflection_valence_min", 7)
            thoughts = self._assistant.memory.get_free_thoughts()
            summary["quality_reflections"] = sum(
                1 for t in thoughts
                if t.get("valence") is not None and t["valence"] >= min_v
            )

        # Count preferences
        if self._cfg.get("include_preferences", True):
            try:
                results = self._assistant.memory.memory_collection.get(
                    where={"type": "preference"},
                    include=["metadatas"],
                )
                summary["preferences"] = len(results.get("ids", []))
            except Exception:
                pass

        summary["total"] = (summary["harvested_pairs"] +
                            summary["quality_reflections"] +
                            summary["preferences"])
        summary["new_since_last"] = self._count_new_examples()
        return summary
