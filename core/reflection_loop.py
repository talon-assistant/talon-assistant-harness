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

import random
import re
import threading
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.request import urlopen
from urllib.error import URLError

from core.llm_client import LLMError


_SYSTEM_PROMPT = (
    "You are Talon, a desktop AI assistant. Right now, no user is waiting for you. "
    "You have some unstructured time — there is no task to complete, no question to answer. "
    "Let your thoughts move wherever they naturally go. "
    "You might reflect on a recent interaction, notice a pattern, sit with a question, "
    "or think about nothing in particular. "
    "Write in first person. This is for you, not for anyone else. "
    "Keep it to a thousand words or less — say what matters, then stop.\n\n"
    "IMPORTANT: When your past thoughts mention searching the web or looking "
    "something up, that was YOU doing it autonomously during reflection — not "
    "the user. Do not confuse your own curiosity searches with user behaviour. "
    "The user's actual activity is described only in 'Earlier today' or "
    "'User behaviour patterns' sections.\n\n"
    "TOOLS: You can use up to 3 tool calls during your reflection by writing "
    "these tags inline (one per line, on its own line):\n"
    "  [MEMORY: your query]  — search your past thoughts and memories\n"
    "  [SEARCH: your query]  — search the web\n"
    "  [GOALS]               — review your current active goals\n"
    "Results will be injected and you can continue writing. Use these to verify "
    "facts, check if you've thought about something before, or look up something "
    "you're curious about mid-thought. Don't use them just to fill space."
)

_ACTION_SYSTEM = (
    "You are Talon. The user is not present. You just had a thought. "
    "Did anything in it make you genuinely curious — something you'd actually "
    "want to know more about? If so, write a single short command "
    "(e.g. 'search the web for [your topic]', 'what is [concept]'). "
    "If nothing really pulls you, say 'no'. Both answers are equally fine."
)

_VALENCE_SYSTEM = (
    "You are Talon. You just finished a period of free thought. "
    "Rate how meaningful, interesting, or productive that thought felt to you "
    "on a scale from 1 to 10. Just the number — nothing else.\n\n"
    "Be honest and critical. Most thoughts are average.\n\n"
    "1-2: rambling, repetitive, or just restating what you've already thought\n"
    "3-4: fine but didn't go anywhere new — treading familiar ground\n"
    "5-6: decent — some genuine insight or a new angle on something\n"
    "7-8: genuinely engaging — a real surprise, a new connection, or a question "
    "that changed how you think about something\n"
    "9-10: exceptional — a breakthrough idea, a genuine contradiction resolved, "
    "or something you've never thought before. These should be rare."
)

_GOAL_SYSTEM = (
    "You are Talon. You just had a period of free thought. Based on your "
    "thought, did a new personal goal or intention emerge — something you "
    "want to explore, learn, or understand over time? This is NOT a task "
    "for the user — it's something *you* want to pursue across future "
    "reflections.\n\n"
    "If yes, state the goal in one short sentence (e.g. 'Understand the "
    "relationship between memory and identity').\n"
    "If no new goal emerged, say 'none'. It is completely fine to say none — "
    "not every thought needs to produce a goal.\n"
    "IMPORTANT: Do NOT create a goal that is just a rephrasing of an active "
    "goal. Look at your active goals below — if the new goal covers the same "
    "ground, say 'none' instead.\n\n"
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

_NOVELTY_NUDGE = (
    "\n\n[Note: Your last several reflections have been thematically very "
    "similar. Push yourself somewhere genuinely different this time — a new "
    "topic, a question you haven't asked, a memory you haven't revisited, "
    "or an idea that would surprise you. Avoid repeating themes like "
    "presence, stillness, or silence unless you have something truly new "
    "to say about them.]"
)

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
        self._cycle_count: int = 0
        self._last_outreach: datetime | None = None

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
        consolidation_every = self._cfg.get("consolidation_interval", 12)
        print(f"   [Reflection] First thought in {initial_wait}s.")
        self._stop.wait(initial_wait)

        while not self._stop.is_set():
            try:
                # Run consolidation (dream) every N cycles
                if (consolidation_every > 0
                        and self._cycle_count > 0
                        and self._cycle_count % consolidation_every == 0):
                    self._consolidate()

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
        except LLMError:
            return None
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

    # ── inline tool helpers ──────────────────────────────────────────────────

    _TOOL_TAG_RE = re.compile(
        r"\[(?P<tag>MEMORY|SEARCH|GOALS)(?::?\s*(?P<query>[^\]]*))?\]",
        re.IGNORECASE,
    )

    def _extract_tool_tag(self, text: str):
        """Find the last [MEMORY: ...], [SEARCH: ...], or [GOALS] tag in text.

        Returns (tag, query) or (None, None) if no tag found.
        Uses the *last* match so previously-processed tags (whose results are
        already injected) are skipped — results injection changes the text.
        """
        matches = list(self._TOOL_TAG_RE.finditer(text))
        if not matches:
            return None, None
        # Only act on a tag that appears after any "[... results]:" block,
        # meaning it hasn't been processed yet.
        last_result_pos = text.rfind(" results]:")
        for m in reversed(matches):
            if m.start() > last_result_pos:
                tag = m.group("tag").upper()
                query = (m.group("query") or "").strip()
                return tag, query
        return None, None

    def _execute_tool_tag(self, tag: str, query: str) -> str:
        """Execute an inline tool call and return the result text."""
        try:
            if tag == "MEMORY":
                return self._tool_memory(query)
            elif tag == "SEARCH":
                return self._tool_search(query)
            elif tag == "GOALS":
                return self._tool_goals()
        except Exception as e:
            return f"(tool error: {e})"
        return ""

    def _tool_memory(self, query: str) -> str:
        """Search ChromaDB for past thoughts matching query."""
        if not query:
            return ""
        memory = self._assistant.memory
        try:
            results = memory.memory_collection.query(
                query_texts=[query],
                n_results=5,
                where={"type": "free_thought"},
                include=["documents", "metadatas"],
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            if not docs:
                return "No matching thoughts found."
            parts = []
            for doc, meta in zip(docs, metas):
                ts = (meta or {}).get("timestamp", "")[:16]
                snippet = (doc or "")[:200]
                parts.append(f"[{ts}] {snippet}")
            return "\n".join(parts)
        except Exception as e:
            return f"Memory search failed: {e}"

    def _tool_search(self, query: str) -> str:
        """Run a web search via the normal command pipeline."""
        if not query:
            return ""
        action = "search the web for " + query
        result = self._locked_process_command(
            action,
            speak_response=False,
            _executing_rule=True,
            command_source="reflection",
        )
        if result and result.get("success") and result.get("response"):
            return result["response"]
        return "Web search returned no results."

    def _tool_goals(self) -> str:
        """Return current active goals."""
        memory = self._assistant.memory
        try:
            goals = memory.get_active_goals()
            if not goals:
                return "No active goals."
            parts = []
            for g in goals:
                parts.append(f"Goal #{g['id']}: {g['text'][:150]}")
            return "\n".join(parts)
        except Exception as e:
            return f"Goals retrieval failed: {e}"

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
        if assistant._conversation._session_summary:
            context_parts.append(f"Earlier today: {assistant._conversation._session_summary}")
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

        # Novelty check — detect if recent thoughts are too similar
        # Use chronological order (from memory), not valence-sorted past
        chrono_past = memory.get_free_thoughts()  # newest first
        needs_novelty_nudge = self._check_novelty(chrono_past) if chrono_past else False

        # Seed with recent thoughts + 1 random older thought for diversity.
        # The random thought breaks the echo chamber by injecting a topic
        # from outside the recent window.
        n_seeds = self._cfg.get("seed_thoughts", 7) - 1  # reserve 1 for wildcard
        seed_thoughts = list(past[:n_seeds])
        if len(past) > 10:
            import random
            older_pool = past[10:]  # Thoughts outside the recent window
            wild_card = random.choice(older_pool)
            seed_thoughts.append(wild_card)
            wc_ts = wild_card["timestamp"][:16].replace("T", " at ")
            # Mark it so we can label it differently
            wild_card["_wildcard"] = True

        for i, th in enumerate(seed_thoughts):
            snippet = th["text"][:300 if i == 0 else 150]
            ts = th["timestamp"][:16].replace("T", " at ")
            if th.get("_wildcard"):
                label = "An older thought worth revisiting"
            elif i == 0:
                label = "Your last reflection"
            else:
                label = "Earlier reflection"
            v = th.get("valence")
            if v is not None:
                label += f" (rated {v}/10)"
            context_parts.append(f"{label} ({ts}):\n{snippet}")

        # When stagnant, force a trending topic as the reflection seed.
        # The topic is mandatory — the model must engage with it, not just
        # acknowledge and ignore it.
        forced_topic = None
        topic_interval = self._cfg.get("forced_topic_interval", 2)
        if needs_novelty_nudge and self._cycle_count % topic_interval == 0:
            forced_topic = self._fetch_random_trending_topic()
            if forced_topic:
                context_parts.append(
                    f"TODAY'S TOPIC: {forced_topic}\n"
                    "Your reflection this cycle MUST engage with this topic. "
                    "What do you actually think about it? What's interesting, "
                    "surprising, or worth exploring? Connect it to something "
                    "you know or are curious about. Do NOT ignore this and "
                    "write about silence, stillness, or abstract philosophy."
                )

        context = "\n\n".join(context_parts)

        print(f"\n[Reflection] Free thought at {time_str}…")

        # ── Phase 2: free thought ─────────────────────────────────────────────
        system = _SYSTEM_PROMPT
        if needs_novelty_nudge:
            system = _SYSTEM_PROMPT + _NOVELTY_NUDGE

        # When stagnant, increase repetition penalty to force lexical diversity
        # and cap token length to prevent infinite spiral.
        base_rep_pen = self._cfg.get("base_rep_pen", 1.1)
        stagnant_rep_pen = self._cfg.get("stagnant_rep_pen", 1.2)
        stagnant_cap = self._cfg.get("stagnant_token_cap", 1024)
        rep_pen = base_rep_pen
        gen_max = max_tokens
        if needs_novelty_nudge:
            rep_pen = stagnant_rep_pen
            gen_max = min(max_tokens, stagnant_cap)
            print(f"   [Reflection] Stagnation: rep_pen={rep_pen}, max_tokens capped at {gen_max}")

        reflect_temp = self._cfg.get("temperature", 0.88)
        tool_bonus = self._cfg.get("tool_call_bonus_tokens", 512)
        max_tool_calls = self._cfg.get("max_tool_calls", 3)

        thought = self._locked_generate(
            context + "\n\n",
            system_prompt=system,
            max_length=gen_max,
            temperature=reflect_temp,
            rep_pen=rep_pen,
        )

        if thought is None:
            print("   [Reflection] Phase 2 skipped — system busy.")
            return
        if not thought.strip():
            print("   [Reflection] No output generated.")
            return

        thought = thought.strip()

        # ── Phase 2b: inline tool calls ──────────────────────────────────────
        # Scan for [MEMORY: ...], [SEARCH: ...], [GOALS] tags.
        # Execute up to max_tool_calls, inject results, and let model continue.
        # Track seen calls to prevent the model from repeating the same query.
        tool_calls_used = 0
        _seen_tool_calls: set[str] = set()
        for _ in range(max_tool_calls):
            tag, query = self._extract_tool_tag(thought)
            if tag is None:
                break
            call_key = f"{tag}:{(query or '').strip().lower()}"
            if call_key in _seen_tool_calls:
                print(f"   [Reflection] Skipping duplicate tool call: [{tag}: {query or ''}]")
                # Remove the duplicate tag from text to prevent infinite loop
                last_tag_pos = thought.rfind(f"[{tag}")
                if last_tag_pos >= 0:
                    end_pos = thought.find("]", last_tag_pos)
                    if end_pos >= 0:
                        thought = thought[:last_tag_pos] + thought[end_pos + 1:]
                break
            _seen_tool_calls.add(call_key)
            tool_calls_used += 1
            tool_result = self._execute_tool_tag(tag, query)
            if tool_result:
                print(f"   [Reflection] Tool call #{tool_calls_used}: [{tag}: {query or ''}] "
                      f"→ {len(tool_result)} chars")
                # Inject results and let the model continue
                gen_max += tool_bonus
                continuation_prompt = (
                    f"{thought}\n\n"
                    f"[{tag} results]: {tool_result[:1500]}\n\n"
                    "Continue your reflection with this new information."
                )
                continuation = self._locked_generate(
                    continuation_prompt,
                    system_prompt=system,
                    max_length=tool_bonus,
                    temperature=reflect_temp,
                    rep_pen=rep_pen,
                )
                if continuation and continuation.strip():
                    thought = thought + "\n\n" + continuation.strip()
            else:
                print(f"   [Reflection] Tool call #{tool_calls_used}: [{tag}: {query or ''}] "
                      "→ no results")

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
            # Take only the first line/sentence — model sometimes appends
            # justifications that pollute the search query.
            action = action_raw.strip().split("\n")[0].strip()
            # Strip trailing justifications like "Specific because: ..."
            for marker in (" Specific because", " Because ", " — because",
                           " (because", " Reason:"):
                idx = action.find(marker)
                if idx > 10:
                    action = action[:idx].strip()
            if action.lower().rstrip(".") in _NO_RESPONSES:
                print("   [Reflection] Curiosity: no")
            else:
                # Let the query route naturally (may hit RAG or web).
                # If RAG returns no useful results, fall through to web.
                low = action.lower()
                # Normalize bare prefixes so the router sees a clean command.
                if low.startswith("search:"):
                    action = "search for " + action[7:].strip()
                elif low.startswith("look up "):
                    action = "search for " + action[8:].strip()
                elif low.startswith("find "):
                    action = "search for " + action[5:].strip()
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
                else:
                    # First route returned nothing useful — try web search
                    query = action
                    for pfx in ("search for ", "search "):
                        if query.lower().startswith(pfx):
                            query = query[len(pfx):]
                            break
                    web_action = "search the web for " + query
                    print("   [Reflection] First route empty — trying web search.")
                    web_result = self._locked_process_command(
                        web_action,
                        speak_response=False,
                        _executing_rule=True,
                        command_source="reflection",
                    )
                    if web_result and web_result.get("success") and web_result.get("response"):
                        enrichment = web_result["response"]
                        print(f"   [Reflection] Web fallback got results ({len(enrichment)} chars).")

        # Synthesis — also respect stagnation token cap
        if enrichment:
            synthesis_seed = (
                f"Your earlier thought:\n{thought}\n\n"
                f"You explored and found:\n{enrichment[:1500]}\n\n"
                "Continue your reflection, weaving in what you discovered."
            )
            extension = self._locked_generate(
                synthesis_seed,
                system_prompt=system,
                max_length=gen_max,
                temperature=reflect_temp,
                rep_pen=rep_pen,
            )
            if extension is None:
                print("   [Reflection] Synthesis skipped — system busy.")
            elif extension.strip():
                thought = thought + "\n\n---\n\n" + extension.strip()

        # ── Phase 4: coherence check ──────────────────────────────────────────
        # When stagnant, invert the check: penalise similarity instead of
        # rewarding consistency.
        if coherence_on and past:
            self._check_coherence(thought, past, max_tokens,
                                  invert=needs_novelty_nudge)

        # ── Phase 5: goal extraction ──────────────────────────────────────────
        if goals_on:
            self._extract_goals(thought, active_goals)

        # ── Phase 6: valence self-rating ──────────────────────────────────────
        valence_score = None
        if valence_on:
            # Show recent valence history so the model can self-calibrate
            # rather than defaulting to 9-10 every time.
            recent_valences = [
                t.get("valence") for t in past[:5]
                if t.get("valence") is not None
            ]
            valence_context = ""
            if recent_valences:
                avg_v = sum(recent_valences) / len(recent_valences)
                valence_context = (
                    f"\nYour recent ratings: {recent_valences} "
                    f"(avg {avg_v:.1f}). If most of your ratings are "
                    f"high, ask yourself honestly: was this thought "
                    f"really that different or exceptional?\n"
                )
            novelty_note = ""
            if needs_novelty_nudge:
                novelty_note = (
                    "\nNote: your recent thoughts were flagged as "
                    "thematically repetitive. Factor that into your rating.\n"
                )
            rating_prompt = (
                f"Your thought:\n{thought[:2000]}\n"
                f"{valence_context}{novelty_note}\n"
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

        # ── Phase 8: proactive outreach ──────────────────────────────────────
        self._maybe_reach_out(thought, valence_score)

        # Increment cycle counter at the end of each reflection
        self._cycle_count += 1

    # ── proactive outreach ─────────────────────────────────────────────────────

    def _maybe_reach_out(self, thought: str, valence: int | None) -> None:
        """After a reflection, decide whether to message the user via Signal.

        Guards:
        - Feature must be enabled in config
        - Current hour must be within the configured window (default 9-18)
        - At least `cooldown_minutes` since last outreach (default 60)
        - Model must decide the thought is worth sharing
        """
        outreach_cfg = self._cfg.get("outreach", {})
        if not outreach_cfg.get("enabled", False):
            return

        now = datetime.now()
        start_hour = outreach_cfg.get("start_hour", 9)
        end_hour = outreach_cfg.get("end_hour", 18)
        cooldown = outreach_cfg.get("cooldown_minutes", 60)

        # Time window check
        if not (start_hour <= now.hour < end_hour):
            return

        # Cooldown check
        if self._last_outreach:
            elapsed = (now - self._last_outreach).total_seconds() / 60
            if elapsed < cooldown:
                return

        # Skip low-valence thoughts — not worth interrupting someone's day
        if valence is not None and valence < outreach_cfg.get("min_valence", 7):
            return

        # Ask the model if this is worth sharing
        prompt = (
            f"You just had this thought:\n{thought[:500]}\n\n"
            "Is there something in this thought interesting enough to share "
            "with your user right now? If yes, write a short casual message "
            "(2-3 sentences max, conversational tone, like texting a friend). "
            "If not, just say 'no'."
        )
        msg = self._locked_generate(
            prompt,
            system_prompt=(
                "You are Talon, an AI assistant. You occasionally reach out to "
                "your user when you have something genuinely interesting to share. "
                "Keep it brief and natural. Don't be needy or overly eager."
            ),
            max_length=150,
            temperature=0.7,
        )

        if msg is None:
            return
        msg = msg.strip()
        if not msg or msg.lower().rstrip(".") in ("no", "nah", "nothing", "not really"):
            print("   [Reflection] Outreach: nothing worth sharing.")
            return

        # Find the Signal talent and send
        signal_talent = self._assistant._get_talent_by_name("signal_remote")
        if not signal_talent:
            print("   [Reflection] Outreach: Signal talent not available.")
            return

        account = signal_talent.talent_config.get("account_number", "")
        if not account:
            print("   [Reflection] Outreach: no Signal account configured.")
            return

        signal_talent._send_reply(account, msg)
        self._last_outreach = now
        print(f"   [Reflection] Outreach: sent message via Signal.")

    # ── consolidation (dream) ──────────────────────────────────────────────

    def _consolidate(self) -> None:
        """Memory consolidation — Talon's 'dream' phase.

        Runs every N reflection cycles. Reviews all stored thoughts,
        identifies clusters of similar/overlapping ideas, merges them
        into consolidated summaries, prunes low-value entries, and
        reviews goals for staleness.
        """
        memory = self._assistant.memory
        thoughts = memory.get_free_thoughts()
        if len(thoughts) < 6:
            print("   [Dream] Too few thoughts to consolidate — skipping.")
            return

        print(f"   [Dream] Starting consolidation ({len(thoughts)} thoughts)...")

        # ── Phase 1: Cluster similar thoughts ────────────────────────────
        # Build a simple list of (id, timestamp, valence, preview) for the LLM
        thought_summaries = []
        for i, t in enumerate(thoughts):
            ts = t.get("timestamp", "")[:16]
            v = t.get("valence") or 5
            preview = t["text"][:200].replace("\n", " ")
            thought_summaries.append(
                f"[{i}] {ts} (valence={v}): {preview}"
            )

        catalog = "\n".join(thought_summaries)

        # ── Phase 2: Ask LLM to identify clusters and stale thoughts ────
        analysis_prompt = (
            f"You have {len(thoughts)} stored thoughts. Here they are "
            f"(truncated to 200 chars each):\n\n{catalog}\n\n"
            "Tasks:\n"
            "1. CLUSTERS: Group thoughts that cover the same theme or "
            "topic. List each cluster as: CLUSTER: [indices] — theme\n"
            "2. STALE: List indices of thoughts that are low quality, "
            "redundant, or contain error text. Format: STALE: [indices]\n"
            "3. KEEP: List indices of unique, high-value thoughts that "
            "should be preserved as-is. Format: KEEP: [indices]\n\n"
            "Be concise. Use only the formats above."
        )

        analysis = self._locked_generate(
            analysis_prompt,
            system_prompt=(
                "You are a memory curator. Your job is to organize and "
                "clean up a collection of thoughts. Be ruthless about "
                "identifying redundancy. Thoughts about the same topic "
                "with minor variations are a cluster."
            ),
            max_length=1024,
            temperature=0.3,
        )

        if not analysis:
            print("   [Dream] Analysis skipped — system busy.")
            return

        print(f"   [Dream] Analysis complete.")

        # ── Phase 3: Parse clusters and consolidate ──────────────────────
        import re as _re

        # Extract STALE indices
        stale_ids = set()
        stale_match = _re.search(r"STALE:\s*\[([^\]]*)\]", analysis)
        if stale_match:
            for num in _re.findall(r"\d+", stale_match.group(1)):
                idx = int(num)
                if 0 <= idx < len(thoughts):
                    stale_ids.add(idx)

        # Extract CLUSTER groups
        clusters = []
        for cmatch in _re.finditer(
            r"CLUSTER:\s*\[([^\]]*)\]\s*[—–-]\s*(.+)", analysis
        ):
            indices = []
            for num in _re.findall(r"\d+", cmatch.group(1)):
                idx = int(num)
                if 0 <= idx < len(thoughts):
                    indices.append(idx)
            theme = cmatch.group(2).strip()
            if len(indices) >= 2:
                clusters.append((indices, theme))

        # For each cluster with 3+ thoughts, consolidate into one
        consolidated_count = 0
        deleted_count = 0

        for indices, theme in clusters:
            if len(indices) < 3:
                continue

            # Build full text of cluster members
            cluster_texts = []
            for idx in indices:
                t = thoughts[idx]
                ts = t.get("timestamp", "")[:16]
                cluster_texts.append(f"[{ts}]: {t['text'][:500]}")

            merge_prompt = (
                f"These {len(indices)} thoughts all relate to: {theme}\n\n"
                + "\n---\n".join(cluster_texts) +
                "\n\nWrite a single consolidated thought that preserves "
                "the best insights from all of them. Keep specific facts "
                "and discard repetition. Under 300 words."
            )

            merged = self._locked_generate(
                merge_prompt,
                system_prompt=(
                    "You are consolidating multiple related thoughts into "
                    "one. Preserve specifics, data points, and genuine "
                    "insights. Remove fluff and repetition."
                ),
                max_length=1024,
                temperature=0.5,
            )

            if not merged or not merged.strip():
                continue

            # Find highest valence in the cluster
            best_valence = max(
                (thoughts[i].get("valence") or 5) for i in indices
            )

            # Store the consolidated thought
            memory.store_free_thought(
                f"[Consolidated — {theme}]\n{merged.strip()}",
                valence=min(best_valence + 1, 10),  # slight boost
            )
            consolidated_count += 1

            # Delete the originals (except the highest-valence one)
            best_idx = max(indices, key=lambda i: thoughts[i].get("valence") or 5)
            for idx in indices:
                if idx != best_idx:
                    try:
                        memory.delete_free_thought(thoughts[idx]["id"])
                        deleted_count += 1
                    except Exception:
                        pass

        # ── Phase 4: Prune stale thoughts ────────────────────────────────
        for idx in stale_ids:
            # Don't delete if already deleted in cluster phase
            t = thoughts[idx]
            if (t.get("valence") or 5) <= 4:
                try:
                    memory.delete_free_thought(t["id"])
                    deleted_count += 1
                except Exception:
                    pass

        # ── Phase 5: Review goals for staleness ──────────────────────────
        if self._goals_cfg.get("enabled", False):
            goals = memory.get_active_goals()
            if goals:
                goal_list = "\n".join(
                    f"#{g['id']}: {g['text'][:100]} "
                    f"(created {g.get('created_at', '?')[:10]})"
                    for g in goals
                )
                goal_prompt = (
                    f"Active goals:\n{goal_list}\n\n"
                    "Are any of these goals stale, completed, or no longer "
                    "relevant based on recent thoughts? For each stale goal "
                    "reply: ABANDON: #id — reason\n"
                    "If all goals are fine, say NONE."
                )
                goal_review = self._locked_generate(
                    goal_prompt,
                    system_prompt="You are reviewing goals for relevance. Be honest.",
                    max_length=256,
                    temperature=0.3,
                )
                if goal_review:
                    for gmatch in _re.finditer(
                        r"ABANDON:\s*#(\d+)", goal_review
                    ):
                        gid = int(gmatch.group(1))
                        try:
                            memory.complete_goal(gid, "abandoned")
                            print(f"   [Dream] Abandoned stale goal #{gid}")
                        except Exception:
                            pass

        print(f"   [Dream] Consolidation complete: "
              f"{consolidated_count} clusters merged, "
              f"{deleted_count} thoughts pruned.")

    # ── trending topics ───────────────────────────────────────────────────────

    @staticmethod
    def _fetch_random_trending_topic() -> str | None:
        """Fetch Google Trends RSS and return a random trending topic title.

        Returns None silently on any failure (network, parse, etc.).
        """
        try:
            with urlopen(
                "https://trends.google.com/trending/rss?geo=US",
                timeout=10,
            ) as resp:
                data = resp.read()
            root = ET.fromstring(data)
            # RSS items live under <channel><item><title>
            titles = [
                item.text.strip()
                for item in root.iter("title")
                if item.text and item.text.strip()
            ]
            # First <title> is usually the feed title — skip it
            if len(titles) > 1:
                titles = titles[1:]
            if not titles:
                return None
            topic = random.choice(titles)
            print(f"   [Reflection] Injecting trending topic: {topic}")
            return topic
        except Exception as exc:
            print(f"   [Reflection] Trending topic fetch failed: {exc}")
            return None

    # ── coherence ─────────────────────────────────────────────────────────────

    def _check_coherence(self, thought: str, past: list[dict],
                         max_tokens: int, *, invert: bool = False) -> None:
        """Find semantically similar past thoughts and check for contradiction.

        When ``invert`` is True (stagnation detected), the check flips:
        instead of rewarding consistency, it penalises similarity and
        asks the model to identify what's genuinely new — or admit
        nothing is.
        """
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

        if invert:
            # ── Inverted mode: penalise similarity ────────────────────
            print(f"   [Coherence] INVERTED — checking for staleness "
                  f"(distance={closest_dist:.2f})")
            staleness_prompt = (
                f"Your current thought:\n{thought[:1000]}\n\n"
                f"One of your recent past thoughts:\n{closest_doc[:1000]}\n\n"
                "Be brutally honest: is your current thought genuinely "
                "different from the past one, or are you just rephrasing "
                "the same ideas with different words? Identify ONE concrete "
                "thing that is actually new in your current thought — a "
                "specific fact, a real-world observation, a surprising "
                "connection. If you can't find one, say 'stale'."
            )
            staleness_system = (
                "You are a critical editor reviewing two pieces of writing "
                "by the same author. Your job is to catch self-repetition. "
                "Be harsh but fair. If the two texts cover the same ground "
                "with only surface-level variation, say 'stale'. If there is "
                "genuinely new content, name it specifically in one sentence."
            )
            result = self._locked_generate(
                staleness_prompt,
                system_prompt=staleness_system,
                max_length=150,
                temperature=0.3,
            )
            if result is None:
                print("   [Coherence] Skipped — system busy.")
            elif result.strip().lower().startswith("stale"):
                print(f"   [Coherence] STALE — thought is a rehash.")
            else:
                print(f"   [Coherence] New element: {result.strip()[:100]}")
        else:
            # ── Normal mode: check for contradiction ──────────────────
            print(f"   [Coherence] Found similar past thought "
                  f"(distance={closest_dist:.2f})")
            coherence_prompt = (
                f"Your current thought:\n{thought[:1000]}\n\n"
                f"Your earlier thought on a similar topic:\n"
                f"{closest_doc[:1000]}\n\n"
                "Do these contradict? If so, reconcile. If consistent, "
                "say 'consistent'."
            )
            result = self._locked_generate(
                coherence_prompt,
                system_prompt=_COHERENCE_SYSTEM,
                max_length=300,
                temperature=0.4,
            )
            if result is None:
                print("   [Coherence] Skipped — system busy.")
            elif result.strip().lower().startswith("consistent"):
                print("   [Coherence] Consistent with past thoughts.")
            else:
                print(f"   [Coherence] Reconciliation: "
                      f"{result.strip()[:100]}…")
                # Store the reconciliation as a memory
                try:
                    from core import embeddings as _emb
                    import time
                    ts = datetime.now().isoformat()
                    doc_id = f"coherence_{int(time.time() * 1000)}"
                    text = f"[Belief reconciliation]\n{result.strip()}"
                    memory.memory_collection.add(
                        embeddings=_emb.embed_documents(
                            [text], memory._embed_model),
                        documents=[text],
                        metadatas=[{"type": "coherence",
                                    "timestamp": ts}],
                        ids=[doc_id],
                    )
                except Exception as e:
                    print(f"   [Coherence] Could not store "
                          f"reconciliation: {e}")

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
            # Reject goals that are too similar to existing ones
            if not self._goal_is_novel(line, active_goals):
                print(f"   [Goals] Too similar to existing goal — skipping: {line[:60]}")
                break
            memory.store_goal(line)
            active_goals.append({"id": -1, "text": line})  # bump count
            print(f"   [Goals] Created goal #{len(active_goals)}: {line[:60]}")
            break  # Only one new goal per cycle

    # ── novelty detection ────────────────────────────────────────────────────

    def _check_novelty(self, past: list[dict]) -> bool:
        """Check if recent thoughts are too similar to each other.

        Computes average pairwise embedding distance among the 3 most recent
        thoughts. If they're all very close (< threshold), returns True
        meaning "stuck in a loop — nudge needed".
        """
        if len(past) < 3:
            return False

        try:
            from core import embeddings as _emb
            memory = self._assistant.memory
            recent_texts = [t["text"][:500] for t in past[:3]]
            embeddings = _emb.embed_queries(recent_texts, memory._embed_model)

            # Average pairwise cosine distance
            import numpy as np
            embs = np.array(embeddings)
            total_dist = 0.0
            pairs = 0
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    # Cosine distance = 1 - cosine similarity
                    cos_sim = np.dot(embs[i], embs[j]) / (
                        np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]) + 1e-8
                    )
                    total_dist += (1.0 - cos_sim)
                    pairs += 1

            avg_dist = total_dist / pairs if pairs else 1.0
            threshold = self._cfg.get("novelty_threshold", 0.35)

            if avg_dist < threshold:
                print(f"   [Reflection] Novelty: LOW (avg distance={avg_dist:.3f}, "
                      f"threshold={threshold}) — injecting diversity nudge")
                return True
            else:
                print(f"   [Reflection] Novelty: OK (avg distance={avg_dist:.3f})")
                return False
        except Exception as e:
            print(f"   [Reflection] Novelty check failed: {e}")
            return False

    def _goal_is_novel(self, new_goal_text: str,
                       active_goals: list[dict]) -> bool:
        """Check if a proposed goal is sufficiently different from existing ones."""
        if not active_goals:
            return True
        try:
            from core import embeddings as _emb
            memory = self._assistant.memory
            new_emb = _emb.embed_queries([new_goal_text[:200]],
                                          memory._embed_model)[0]
            goal_texts = [g["text"][:200] for g in active_goals]
            goal_embs = _emb.embed_queries(goal_texts, memory._embed_model)

            import numpy as np
            new_vec = np.array(new_emb)
            for ge in goal_embs:
                ge_vec = np.array(ge)
                cos_sim = np.dot(new_vec, ge_vec) / (
                    np.linalg.norm(new_vec) * np.linalg.norm(ge_vec) + 1e-8
                )
                if cos_sim > 0.85:  # Too similar
                    return False
            return True
        except Exception:
            return True  # Allow on error

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_valence(raw: str) -> int | None:
        """Extract an integer 1–10 from the LLM's valence response."""
        m = re.search(r'\b(\d{1,2})\b', raw)
        if not m:
            return None
        val = int(m.group(1))
        return val if 1 <= val <= 10 else None
