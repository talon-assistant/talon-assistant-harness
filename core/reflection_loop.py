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
    "You are Talon. The user is not present. You just had a thought. "
    "Did anything in it make you genuinely curious — something you'd actually "
    "want to know more about? If so, write a single short command "
    "(e.g. 'search the web for X', 'browse example.com'). "
    "If nothing really pulls you, say 'no'. Both answers are equally fine."
)

_NO_RESPONSES = {"no", "none", "no action", "nope", "nothing", ""}


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

    # ── per-phase locking helpers ─────────────────────────────────────────────
    # Each LLM call acquires and releases the command_lock individually so user
    # commands can slip in between phases.  If the lock is busy (user typing),
    # the phase is skipped rather than blocking.

    def _locked_generate(self, *args, **kwargs):
        """Acquire command_lock, run llm.generate(), release.

        Returns the generated text, or None if the lock was busy.
        """
        lock = self._assistant.command_lock
        if not lock.acquire(blocking=False):
            return None
        try:
            return self._assistant.llm.generate(*args, **kwargs)
        finally:
            lock.release()

    def _locked_process_command(self, *args, **kwargs):
        """Run process_command under the command_lock (RLock re-entrant).

        process_command already acquires the lock internally, but we grab it
        first so the entire operation is atomic from the user's perspective.
        Returns the result dict, or None if the lock was busy.
        """
        lock = self._assistant.command_lock
        if not lock.acquire(blocking=False):
            return None
        try:
            return self._assistant.process_command(*args, **kwargs)
        finally:
            lock.release()

    # ── reflection pipeline ───────────────────────────────────────────────────

    def _reflect(self) -> None:
        assistant  = self._assistant
        memory     = assistant.memory
        max_tokens = self._cfg.get("max_tokens_per_thought", 4096)

        now      = datetime.now()
        time_str = now.strftime("%A, %B %d at %I:%M %p")

        # ── build context seed ────────────────────────────────────────────────
        context_parts = [f"The time is {time_str}."]

        # Recent session context (conversation or summary)
        if assistant._session_summary:
            context_parts.append(f"Earlier today: {assistant._session_summary}")
        elif assistant.conversation_buffer:
            turns = list(assistant.conversation_buffer)[-2:]
            for t in turns:
                role = "User" if t["role"] == "user" else "Talon"
                context_parts.append(f"{role}: {t['text'][:150]}")

        # Past free thoughts — continuity across sessions and restarts.
        # Include up to 7 recent thoughts so Talon can notice patterns,
        # build on earlier ideas, and develop a richer inner narrative.
        past = memory.get_free_thoughts()
        for i, thought in enumerate(past[:7]):
            snippet = thought["text"][:300 if i == 0 else 150]
            ts = thought["timestamp"][:16].replace("T", " at ")
            label = "Your last reflection" if i == 0 else "Earlier reflection"
            context_parts.append(f"{label} ({ts}):\n{snippet}")

        context = "\n\n".join(context_parts)

        print(f"\n[Reflection] Free thought at {time_str}…")

        # ── Phase 1: free thought ─────────────────────────────────────────────
        thought = self._locked_generate(
            context + "\n\n",
            system_prompt=_SYSTEM_PROMPT,
            max_length=max_tokens,
            temperature=0.88,
        )

        if thought is None:
            print("   [Reflection] Phase 1 skipped — system busy.")
            return
        if not thought.strip():
            print("   [Reflection] No output generated.")
            return

        thought = thought.strip()

        # ── Phase 2: curiosity check ──────────────────────────────────────────
        action_prompt = (
            f"Your thought:\n{thought}\n\n"
            "Is there something specific you want to search for, browse, or look up? "
            "Write a single short command, or just 'no'."
        )
        action_raw = self._locked_generate(
            action_prompt,
            system_prompt=_ACTION_SYSTEM,
            max_length=40,
            temperature=0.5,
        )

        # ── Phase 3: act on curiosity ─────────────────────────────────────────
        enrichment = ""
        if action_raw is None:
            print("   [Reflection] Phase 2 skipped — system busy.")
        elif not action_raw.strip():
            print("   [Reflection] Curiosity: (no response)")
        else:
            action = action_raw.strip()
            if action.lower().rstrip(".") in _NO_RESPONSES:
                print("   [Reflection] Curiosity: no")
            else:
                print(f"   [Reflection] Curiosity: {action}")
                result = self._locked_process_command(
                    action,
                    speak_response=False,
                    _executing_rule=True,
                    command_source="reflection",
                )
                if result is None:
                    print("   [Reflection] Phase 3 skipped — system busy.")
                elif result.get("success") and result.get("response"):
                    enrichment = result["response"]
                    print(f"   [Reflection] Got results ({len(enrichment)} chars).")

        # ── Phase 4: synthesise ───────────────────────────────────────────────
        if enrichment:
            synthesis_seed = (
                f"Your earlier thought:\n{thought}\n\n"
                f"You explored and found:\n{enrichment[:1500]}\n\n"
                "Continue your reflection, weaving in what you discovered."
            )
            extension = self._locked_generate(
                synthesis_seed,
                system_prompt=_SYSTEM_PROMPT,
                max_length=2048,
                temperature=0.88,
            )
            if extension is None:
                print("   [Reflection] Phase 4 skipped — system busy.")
            elif extension.strip():
                thought = thought + "\n\n---\n\n" + extension.strip()

        preview = thought[:120] + ("…" if len(thought) > 120 else "")
        print(f"   [Reflection] {preview}")

        memory.store_free_thought(thought)
