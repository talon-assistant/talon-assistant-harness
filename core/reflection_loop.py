"""Talon free-thought loop — periodic unsupervised LLM reflection.

Every ``interval_minutes`` minutes, Talon gets a window of unstructured time.
No user is waiting. No task to complete. The LLM generates freely, and may
act on its own curiosity by issuing web searches, browsing pages, or running
plans — all without any user prompt.

Each thought (and any enrichment from actions) is stored in ChromaDB as
type="free_thought" and can surface in future RAG lookups when relevant.

Config block in settings.json:

  "reflection": {
    "enabled": true,
    "interval_minutes": 60,
    "max_tokens_per_thought": 4096
  }
"""

from __future__ import annotations

import threading
import traceback
from datetime import datetime


_SYSTEM_PROMPT = (
    "You are Talon, a desktop AI assistant. Right now, no user is waiting for you. "
    "You have some unstructured time — there is no task to complete, no question to answer. "
    "Let your thoughts move wherever they naturally go. "
    "You might reflect on a recent interaction, notice a pattern, sit with a question, "
    "or think about nothing in particular. "
    "Write in first person. This is for you, not for anyone else."
)

_ACTION_SYSTEM = (
    "You are Talon. The user is not present. You are deciding whether to act on your own curiosity. "
    "Reply with either a single short command (e.g. 'search the web for X', 'browse example.com') "
    "or just the word 'no' if you don't want to look anything up right now."
)

_NO_RESPONSES = {"no", "no.", "none", "no action", "no.", "nope", "nothing", ""}


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
        interval_s = self._cfg.get("interval_minutes", 60) * 60
        # Initial wait: one full interval, capped at 5 minutes.
        initial_wait = min(interval_s, 300)
        print(f"   [Reflection] First thought in {initial_wait}s.")
        self._stop.wait(initial_wait)

        while not self._stop.is_set():
            try:
                self._reflect()
            except Exception:
                print(f"   [Reflection] Error:\n{traceback.format_exc()}")

            self._stop.wait(interval_s)

    def _reflect(self) -> None:
        assistant = self._assistant

        # Don't interrupt an active user command — skip this cycle if busy.
        if not assistant.command_lock.acquire(blocking=False):
            print("   [Reflection] Skipped — system busy.")
            return

        try:
            self._reflect_locked()
        finally:
            assistant.command_lock.release()

    def _reflect_locked(self) -> None:
        assistant  = self._assistant
        llm        = assistant.llm
        memory     = assistant.memory
        max_tokens = self._cfg.get("max_tokens_per_thought", 4096)

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

        print(f"\n[Reflection] Free thought at {time_str}…")

        # ── Phase 1: free thought ─────────────────────────────────────────────
        thought = llm.generate(
            context + "\n\n",
            system_prompt=_SYSTEM_PROMPT,
            max_length=max_tokens,
            temperature=0.88,
        )

        if not thought or not thought.strip():
            print("   [Reflection] No output generated.")
            return

        thought = thought.strip()

        # ── Phase 2: curiosity check ──────────────────────────────────────────
        # Ask the LLM if there's something it wants to act on.
        action_prompt = (
            f"Your thought:\n{thought}\n\n"
            "Is there something specific you want to search for, browse, or look up? "
            "Write a single short command, or just 'no'."
        )
        action_raw = llm.generate(
            action_prompt,
            system_prompt=_ACTION_SYSTEM,
            max_length=40,
            temperature=0.2,
        )

        # ── Phase 3: act on curiosity ─────────────────────────────────────────
        enrichment = ""
        if action_raw:
            action = action_raw.strip()
            if action.lower().rstrip(".") not in _NO_RESPONSES:
                print(f"   [Reflection] Curiosity: {action}")
                # Route through the full talent pipeline (no TTS, no buffer).
                result = assistant.process_command(
                    action,
                    speak_response=False,
                    _executing_rule=True,
                    command_source="reflection",
                )
                if result and result.get("success") and result.get("response"):
                    enrichment = result["response"]
                    print(f"   [Reflection] Got results ({len(enrichment)} chars).")

        # ── Phase 4: synthesise ───────────────────────────────────────────────
        if enrichment:
            synthesis_seed = (
                f"Your earlier thought:\n{thought}\n\n"
                f"You explored and found:\n{enrichment[:1500]}\n\n"
                "Continue your reflection, weaving in what you discovered."
            )
            extension = llm.generate(
                synthesis_seed,
                system_prompt=_SYSTEM_PROMPT,
                max_length=500,
                temperature=0.88,
            )
            if extension and extension.strip():
                thought = thought + "\n\n---\n\n" + extension.strip()

        preview = thought[:120] + ("…" if len(thought) > 120 else "")
        print(f"   [Reflection] {preview}")

        memory.store_free_thought(thought)
