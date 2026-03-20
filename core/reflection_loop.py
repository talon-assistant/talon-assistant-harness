"""Talon free-thought loop — periodic unsupervised LLM reflection.

Every ``interval_minutes`` minutes, Talon gets a window of unstructured time.
No user is waiting. No task to complete. The LLM generates freely, seeding
each thought from the previous one, until the window closes.

Each thought is stored in ChromaDB as type="free_thought" and can surface
in future RAG lookups when semantically relevant to a user query.

Config block in settings.json:

  "reflection": {
    "enabled": true,
    "interval_minutes": 60,
    "duration_minutes": 2,
    "max_tokens_per_thought": 250
  }
"""

from __future__ import annotations

import threading
import time
from datetime import datetime


_SYSTEM_PROMPT = (
    "You are Talon, a desktop AI assistant. Right now, no user is waiting for you. "
    "You have some unstructured time — there is no task to complete, no question to answer. "
    "Let your thoughts move wherever they naturally go. "
    "You might reflect on a recent interaction, notice a pattern, sit with a question, "
    "or think about nothing in particular. "
    "Write in first person. This is for you, not for anyone else."
)


class ReflectionLoop:
    """Background thread that gives Talon periodic unsupervised think-time."""

    def __init__(self, assistant):
        self._assistant = assistant
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cfg: dict = {}

    def configure(self, cfg: dict) -> None:
        self._cfg = cfg

    def start(self) -> None:
        if not self._cfg.get("enabled", False):
            print("   [Reflection] Disabled — set reflection.enabled=true to activate.")
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="talon-reflection",
        )
        self._thread.start()
        interval = self._cfg.get("interval_minutes", 60)
        print(f"   [Reflection] Loop started — free thought every {interval}m.")

    def stop(self) -> None:
        self._stop.set()

    # ── internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        # Give startup time to settle before the first reflection.
        initial_wait = 300  # 5 minutes
        self._stop.wait(initial_wait)

        while not self._stop.is_set():
            try:
                self._reflect()
            except Exception as e:
                print(f"   [Reflection] Error: {e}")

            interval_s = self._cfg.get("interval_minutes", 60) * 60
            self._stop.wait(interval_s)

    def _reflect(self) -> None:
        assistant   = self._assistant
        llm         = assistant.llm
        memory      = assistant.memory
        duration_s  = self._cfg.get("duration_minutes", 2) * 60
        max_tokens  = self._cfg.get("max_tokens_per_thought", 250)
        deadline    = time.time() + duration_s

        now      = datetime.now()
        time_str = now.strftime("%A, %B %d at %I:%M %p")

        # ── build context seed ────────────────────────────────────────────────
        context_parts = [f"The time is {time_str}."]

        if assistant._session_summary:
            context_parts.append(f"Earlier today: {assistant._session_summary}")
        elif assistant.conversation_buffer:
            turns = list(assistant.conversation_buffer)[-2:]
            for t in turns:
                role = "User" if t["role"] == "user" else "Talon"
                context_parts.append(f"{role}: {t['text'][:150]}")

        context = "\n".join(context_parts)

        print(f"\n[Reflection] Free thought beginning at {time_str}…")

        seed        = context + "\n\n"
        thought_num = 0

        while time.time() < deadline and not self._stop.is_set():
            thought_num += 1

            thought = llm.generate(
                seed,
                system_prompt=_SYSTEM_PROMPT,
                max_length=max_tokens,
                temperature=0.88,
            )

            if not thought or not thought.strip():
                break

            thought = thought.strip()
            preview = thought[:120] + ("…" if len(thought) > 120 else "")
            print(f"   [Reflection] [{thought_num}] {preview}")

            memory.store_free_thought(thought, thought_num)

            # Seed the next thought from this one.
            seed = thought + "\n\n"

            if time.time() < deadline and not self._stop.is_set():
                time.sleep(1)

        if thought_num:
            print(f"   [Reflection] Done — {thought_num} thought(s).")
        else:
            print("   [Reflection] No output generated.")
