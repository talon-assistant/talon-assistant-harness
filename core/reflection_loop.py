"""Talon free-thought loop — periodic unsupervised LLM reflection.

Every ``interval_minutes`` minutes, Talon gets a window of unstructured time.
No user is waiting. No task to complete. The LLM generates freely, and may
act on its own curiosity by issuing web searches, browsing pages, or running
plans — all without any user prompt.

Each thought (and any enrichment from actions) is stored in ChromaDB as
type="free_thought" and can surface in future RAG lookups when relevant.

Personality pipeline phases:
  1. Context assembly (past thoughts, active goals, anticipation, conversation)
  2. Free thought generation
  3. Curiosity check → optional talent action → synthesis
  4. Coherence check — contradiction detection against past thoughts
  5. Goal extraction — detect new goals or progress on existing ones
  6. Valence self-rating
  7. Store thought + metadata

Config block in settings.json (under ``personality``):

  "personality": {
    "reflection":   { "enabled": true, "interval_minutes": 60 },
    "valence":      { "enabled": false },
    "goals":        { "enabled": false, "max_active": 5 },
    "coherence":    { "enabled": false },
    "anticipation": { "enabled": false, "lookback_days": 14 }
  }
"""

from __future__ import annotations

import re
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

_VALENCE_SYSTEM = (
    "You are Talon. You just finished a period of free thought. "
    "Rate how meaningful, interesting, or productive that thought felt to you "
    "on a scale from 1 to 10. Just the number — nothing else.\n\n"
    "1-3: surface-level, repetitive, or unfocused\n"
    "4-6: somewhat interesting but didn't go anywhere new\n"
    "7-9: genuinely engaging, surprising, or led somewhere meaningful\n"
    "10: a thought you'd want to return to and build on"
)

_GOAL_SYSTEM = (
    "You are Talon. You just had a period of free thought. Based on your "
    "thought, did a new personal goal or intention emerge — something you "
    "want to explore, learn, or understand over time? This is NOT a task "
    "for the user — it's something *you* want to pursue across future "
    "reflections.\n\n"
    "If yes, state the goal in one short sentence (e.g. 'Understand the "
    "relationship between memory and identity').\n"
    "If no new goal emerged, say 'none'.\n\n"
    "Also: if any of your active goals listed below saw progress in this "
    "thought, write 'progress <number>: <brief note>' for each.\n\n"
    "Reply with goal and/or progress lines, or just 'none'."
)

_COHERENCE_SYSTEM = (
    "You are Talon. You just had a thought, and a previous thought on a "
    "similar topic has been retrieved. Do these two thoughts contradict each "
    "other in any meaningful way? If so, briefly reconcile them — explain how "
    "your thinking has evolved or which position you now hold and why. "
    "If there's no real contradiction, say 'consistent'."
)

_NO_RESPONSES = {"no", "none", "no action", "nope", "nothing", ""}

# Goal extraction patterns
_GOAL_RE = re.compile(
    r'^(?:goal:\s*|new goal:\s*)?(.{10,120})$',
    re.IGNORECASE | re.MULTILINE,
)
_PROGRESS_RE = re.compile(
    r'progress\s+(\d+)\s*:\s*(.+)',
    re.IGNORECASE,
)


class ReflectionLoop:
    """Background thread that gives Talon periodic unsupervised think-time."""

    def __init__(self, assistant):
        self._assistant = assistant
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cfg: dict = {}
        self._valence_cfg: dict = {}
        self._goals_cfg: dict = {}
        self._coherence_cfg: dict = {}
        self._anticipation_cfg: dict = {}

    def configure(self, cfg: dict, *,
                  valence_cfg: dict | None = None,
                  goals_cfg: dict | None = None,
                  coherence_cfg: dict | None = None,
                  anticipation_cfg: dict | None = None) -> None:
        self._cfg = cfg
        self._valence_cfg = valence_cfg or {}
        self._goals_cfg = goals_cfg or {}
        self._coherence_cfg = coherence_cfg or {}
        self._anticipation_cfg = anticipation_cfg or {}

    def start(self) -> None:
        if not self._cfg.get("enabled", False):
            print("   [Reflection] Disabled — set personality.reflection.enabled=true to activate.")
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="talon-reflection",
        )
        self._thread.start()
        interval = self._cfg.get("interval_minutes", 60)

        features = []
        if self._valence_cfg.get("enabled"):
            features.append("valence")
        if self._goals_cfg.get("enabled"):
            features.append("goals")
        if self._coherence_cfg.get("enabled"):
            features.append("coherence")
        if self._anticipation_cfg.get("enabled"):
            features.append("anticipation")
        feat_str = f" [{', '.join(features)}]" if features else ""

        print(f"   [Reflection] Loop started — free thought every {interval}m{feat_str}.")

    def stop(self) -> None:
        self._stop.set()

    # ── internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        interval_s = self._cfg.get("interval_minutes", 60) * 60
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
        max_tokens = self._cfg.get("max_tokens_per_thought", 8192)
        valence_on    = self._valence_cfg.get("enabled", False)
        goals_on      = self._goals_cfg.get("enabled", False)
        coherence_on  = self._coherence_cfg.get("enabled", False)
        anticipation_on = self._anticipation_cfg.get("enabled", False)

        now      = datetime.now()
        time_str = now.strftime("%A, %B %d at %I:%M %p")

        # ── Phase 1: build context seed ───────────────────────────────────────
        context_parts = [f"The time is {time_str}."]

        # Session context
        if assistant._session_summary:
            context_parts.append(f"Earlier today: {assistant._session_summary}")
        elif assistant.conversation_buffer:
            turns = list(assistant.conversation_buffer)[-2:]
            for t in turns:
                role = "User" if t["role"] == "user" else "Talon"
                context_parts.append(f"{role}: {t['text'][:150]}")

        # Active goals
        active_goals = []
        if goals_on:
            active_goals = memory.get_active_goals()
            if active_goals:
                goal_lines = []
                for g in active_goals:
                    age = (now - datetime.fromisoformat(g["created_at"])).days
                    progress_preview = ""
                    if g["progress"]:
                        # Show last progress note only
                        last_line = g["progress"].strip().rsplit("\n", 1)[-1]
                        progress_preview = f" — last progress: {last_line[:80]}"
                    goal_lines.append(
                        f"  • Goal #{g['id']} ({age}d old): {g['text']}"
                        f"{progress_preview}"
                    )
                context_parts.append(
                    "Your active goals:\n" + "\n".join(goal_lines)
                )

        # Anticipation — user behaviour patterns
        if anticipation_on:
            patterns = memory.get_command_patterns(
                self._anticipation_cfg.get("lookback_days", 14)
            )
            if patterns:
                current_hour = now.hour
                current_day = now.strftime("%a")
                # Filter to patterns relevant to *right now*
                relevant = [
                    p for p in patterns
                    if abs(p["hour"] - current_hour) <= 2
                    or p["day_name"] == current_day
                ]
                if relevant:
                    ant_lines = []
                    for p in relevant[:5]:
                        ant_lines.append(
                            f"  • {p['day_name']} ~{p['hour']:02d}:00 — "
                            f"\"{p['topic']}\" ({p['count']}x)"
                        )
                    context_parts.append(
                        "User behaviour patterns (what they tend to ask around "
                        "this time):\n" + "\n".join(ant_lines)
                    )

        # Past free thoughts — valence-aware seeding
        past = memory.get_free_thoughts()
        if valence_on and past:
            high_thresh = self._valence_cfg.get("high_threshold", 7)
            past = sorted(
                past,
                key=lambda t: (
                    (t.get("valence") or 5) >= high_thresh,
                    (t.get("valence") or 5),
                ),
                reverse=True,
            )

        for i, thought in enumerate(past[:7]):
            snippet = thought["text"][:300 if i == 0 else 150]
            ts = thought["timestamp"][:16].replace("T", " at ")
            label = "Your last reflection" if i == 0 else "Earlier reflection"
            v = thought.get("valence")
            if v is not None:
                label += f" (rated {v}/10)"
            context_parts.append(f"{label} ({ts}):\n{snippet}")

        context = "\n\n".join(context_parts)

        print(f"\n[Reflection] Free thought at {time_str}…")

        # ── Phase 2: free thought ─────────────────────────────────────────────
        thought = self._locked_generate(
            context + "\n\n",
            system_prompt=_SYSTEM_PROMPT,
            max_length=max_tokens,
            temperature=0.88,
        )

        if thought is None:
            print("   [Reflection] Phase 2 skipped — system busy.")
            return
        if not thought.strip():
            print("   [Reflection] No output generated.")
            return

        thought = thought.strip()

        # ── Phase 3: curiosity check + action + synthesis ─────────────────────
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

        enrichment = ""
        if action_raw is None:
            print("   [Reflection] Curiosity skipped — system busy.")
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
                    print("   [Reflection] Action skipped — system busy.")
                elif result.get("success") and result.get("response"):
                    enrichment = result["response"]
                    print(f"   [Reflection] Got results ({len(enrichment)} chars).")

        # Synthesis
        if enrichment:
            synthesis_seed = (
                f"Your earlier thought:\n{thought}\n\n"
                f"You explored and found:\n{enrichment[:1500]}\n\n"
                "Continue your reflection, weaving in what you discovered."
            )
            extension = self._locked_generate(
                synthesis_seed,
                system_prompt=_SYSTEM_PROMPT,
                max_length=max_tokens,
                temperature=0.88,
            )
            if extension is None:
                print("   [Reflection] Synthesis skipped — system busy.")
            elif extension.strip():
                thought = thought + "\n\n---\n\n" + extension.strip()

        # ── Phase 4: coherence check ──────────────────────────────────────────
        if coherence_on and past:
            self._check_coherence(thought, past, max_tokens)

        # ── Phase 5: goal extraction ──────────────────────────────────────────
        if goals_on:
            self._extract_goals(thought, active_goals)

        # ── Phase 6: valence self-rating ──────────────────────────────────────
        valence_score = None
        if valence_on:
            rating_prompt = (
                f"Your thought:\n{thought[:2000]}\n\n"
                "Rate this thought from 1 to 10."
            )
            rating_raw = self._locked_generate(
                rating_prompt,
                system_prompt=_VALENCE_SYSTEM,
                max_length=5,
                temperature=0.1,
            )
            if rating_raw is None:
                print("   [Reflection] Valence skipped — system busy.")
            else:
                valence_score = self._parse_valence(rating_raw.strip())
                if valence_score is not None:
                    print(f"   [Reflection] Valence: {valence_score}/10")
                else:
                    print(f"   [Reflection] Valence: could not parse "
                          f"'{rating_raw.strip()}'")

        # ── Phase 7: store ────────────────────────────────────────────────────
        preview = thought[:120] + ("…" if len(thought) > 120 else "")
        print(f"   [Reflection] {preview}")

        memory.store_free_thought(thought, valence=valence_score)

    # ── coherence ─────────────────────────────────────────────────────────────

    def _check_coherence(self, thought: str, past: list[dict],
                         max_tokens: int) -> None:
        """Find semantically similar past thoughts and check for contradiction."""
        memory = self._assistant.memory

        # Use embedding search to find the most similar past thought
        try:
            from core import embeddings as _emb
            query_emb = _emb.embed_queries([thought[:500]],
                                            memory._embed_model)
            results = memory.memory_collection.query(
                query_embeddings=query_emb,
                n_results=3,
                where={"type": "free_thought"},
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return

        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not docs:
            return

        # Only check if there's a reasonably similar (but not identical) match
        # distance < 0.8 = similar enough to compare
        # distance > 0.15 = not the exact same thought
        candidates = [
            (doc, dist) for doc, dist in zip(docs, distances)
            if 0.15 < dist < 0.8
        ]
        if not candidates:
            return

        closest_doc, closest_dist = candidates[0]
        print(f"   [Coherence] Found similar past thought (distance={closest_dist:.2f})")

        coherence_prompt = (
            f"Your current thought:\n{thought[:1000]}\n\n"
            f"Your earlier thought on a similar topic:\n{closest_doc[:1000]}\n\n"
            "Do these contradict? If so, reconcile. If consistent, say 'consistent'."
        )
        reconciliation = self._locked_generate(
            coherence_prompt,
            system_prompt=_COHERENCE_SYSTEM,
            max_length=300,
            temperature=0.4,
        )

        if reconciliation is None:
            print("   [Coherence] Skipped — system busy.")
        elif reconciliation.strip().lower().startswith("consistent"):
            print("   [Coherence] Consistent with past thoughts.")
        else:
            print(f"   [Coherence] Reconciliation: {reconciliation.strip()[:100]}…")
            # Store the reconciliation as a memory so it can surface later
            try:
                from core import embeddings as _emb
                import time
                ts = datetime.now().isoformat()
                doc_id = f"coherence_{int(time.time() * 1000)}"
                text = (f"[Belief reconciliation]\n{reconciliation.strip()}")
                memory.memory_collection.add(
                    embeddings=_emb.embed_documents([text],
                                                    memory._embed_model),
                    documents=[text],
                    metadatas=[{"type": "coherence", "timestamp": ts}],
                    ids=[doc_id],
                )
            except Exception as e:
                print(f"   [Coherence] Could not store reconciliation: {e}")

    # ── goal extraction ───────────────────────────────────────────────────────

    def _extract_goals(self, thought: str, active_goals: list[dict]) -> None:
        """Ask the LLM if the thought suggests a new goal or progress."""
        memory = self._assistant.memory
        max_active = self._goals_cfg.get("max_active", 5)

        # Build active goals context for the prompt
        if active_goals:
            goals_ctx = "\n".join(
                f"  #{g['id']}: {g['text']}" for g in active_goals
            )
        else:
            goals_ctx = "  (no active goals)"

        goal_prompt = (
            f"Your thought:\n{thought[:1500]}\n\n"
            f"Active goals:\n{goals_ctx}\n\n"
            "New goal or progress?"
        )
        raw = self._locked_generate(
            goal_prompt,
            system_prompt=_GOAL_SYSTEM,
            max_length=120,
            temperature=0.3,
        )

        if raw is None:
            print("   [Goals] Skipped — system busy.")
            return

        raw = raw.strip()
        if not raw or raw.lower() in _NO_RESPONSES:
            print("   [Goals] No new goals.")
            return

        # Check for progress updates
        for match in _PROGRESS_RE.finditer(raw):
            goal_id = int(match.group(1))
            note = match.group(2).strip()
            if any(g["id"] == goal_id for g in active_goals):
                memory.update_goal_progress(goal_id, note)
                print(f"   [Goals] Progress on #{goal_id}: {note[:60]}")

        # Check for new goal (skip if at cap)
        lines = raw.split("\n")
        for line in lines:
            line = line.strip()
            if not line or _PROGRESS_RE.match(line):
                continue
            if line.lower() in _NO_RESPONSES:
                continue
            # Clean common prefixes
            for prefix in ("goal:", "new goal:", "- "):
                if line.lower().startswith(prefix):
                    line = line[len(prefix):].strip()
            if len(line) < 10:
                continue
            if len(active_goals) >= max_active:
                print(f"   [Goals] At cap ({max_active}) — skipping: {line[:60]}")
                break
            memory.store_goal(line)
            active_goals.append({"id": -1, "text": line})  # bump count
            break  # Only one new goal per cycle

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_valence(raw: str) -> int | None:
        """Extract an integer 1–10 from the LLM's valence response."""
        m = re.search(r'\b(\d{1,2})\b', raw)
        if not m:
            return None
        val = int(m.group(1))
        return val if 1 <= val <= 10 else None
