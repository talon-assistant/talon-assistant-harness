"""
PlannerTalent — multi-step routine execution.

Detects commands that require more than one talent and breaks them into an
ordered sequence of sub-commands, each routed through the normal
process_command() pipeline. This means every existing talent, memory lookup,
and routing rule applies automatically to each step.

Examples:
  "good morning routine"
  "set up movie night"
  "turn on the lights and check the weather"
  "evening wind-down"
"""

import json
import re
from talents.base import BaseTalent


class PlannerTalent(BaseTalent):
    name = "planner"
    description = (
        "Execute multi-step routines by breaking a request into individual "
        "actions and running each one in sequence"
    )
    keywords = [
        "routine", "sequence", "setup", "set up", "mode",
        "morning", "evening", "night", "wind down", "wind-down",
        "and then", "followed by", "after that",
    ]
    examples = [
        "good morning routine",
        "set up movie night",
        "turn on the lights and check the weather",
        "evening routine",
        "do my morning checks",
        "set up my workspace",
        "wind down for the night",
    ]
    priority = 85  # Above all built-in talents (news=80 is the previous max)

    # ── Prompt ────────────────────────────────────────────────────────────────

    _PLANNER_SYSTEM_PROMPT = """\
You are a planning assistant for a desktop AI called Talon.
Your job is to decide if a user request requires multiple sequential actions,
and if so, break it down into individual, self-contained sub-commands.

Available handlers (what Talon can do):
{roster}

Rules:
- Only output a plan if the request clearly needs 2 or more DIFFERENT handlers (e.g. lights AND weather). If all parts of the request use the same handler (e.g. open calculator, type, press enter, read screen — all desktop_control), set is_multi_step to false.
- Steps are executed in order; keep them short and specific.
- Maximum 8 steps.
- If the request is really just a single action, set is_multi_step to false.
- If a required detail is missing (e.g. which folder, which device, which contact), add a step BEFORE the steps that need it using the prefix "ask_user: <clear question>". Later steps that depend on the answer should include the placeholder {{user_input}} where the answer will be substituted at runtime. Only ask when genuinely needed — do not ask for things that can be reasonably inferred.
- If a step produces a file path or output that the NEXT step needs, write the next step using the placeholder {{last_result}} exactly where that value belongs. The placeholder is replaced at runtime with the full response text of the previous step. EXAMPLE — "generate morning news digest" produces a path; the email step must be written as: "email the file at {{last_result}} to user@example.com". Do NOT write "attach the generated digest" without the placeholder — that gives the email talent no path to attach.

Respond ONLY with a JSON object — no markdown, no explanation:

If multi-step:
{{"is_multi_step": true, "summary": "<short title>", "steps": ["step1", "step2", ...]}}

If single-step:
{{"is_multi_step": false}}
"""

    _MAX_STEPS = 8

    # ── Routing ───────────────────────────────────────────────────────────────

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    # ── Main execution ────────────────────────────────────────────────────────

    def execute(self, command: str, context: dict) -> dict:
        llm = context["llm"]
        voice = context.get("voice")
        speak = context.get("speak_response", True)
        notify = context.get("notify")
        assistant = context.get("assistant")
        command_source = context.get("command_source", "local")

        if not assistant:
            # Should never happen — context always includes assistant
            return {
                "success": False,
                "response": "Planner unavailable: no assistant reference in context.",
                "actions_taken": [],
                "spoken": False,
            }

        # ── Step 1: Ask LLM to build a plan ──────────────────────────────────
        roster = self._build_roster(assistant)
        system_prompt = self._PLANNER_SYSTEM_PROMPT.format(roster=roster)

        raw = llm.generate(
            f"User request: {command}",
            system_prompt=system_prompt,
            temperature=0.2,
            max_length=512,
        )

        plan = self._parse_plan(raw)

        if plan is None:
            # LLM returned unparseable output — decline so normal routing handles it
            print("   [Planner] Could not parse plan JSON, declining.")
            return {
                "success": False,
                "response": "",
                "actions_taken": [],
                "spoken": False,
            }

        if not plan.get("is_multi_step"):
            # LLM says this is a single-step request — decline gracefully
            print("   [Planner] Single-step request detected, declining to normal routing.")
            return {
                "success": False,
                "response": "",
                "actions_taken": [],
                "spoken": False,
            }

        steps = plan.get("steps", [])[:self._MAX_STEPS]
        summary = plan.get("summary", "Running routine")

        if not steps:
            return {
                "success": False,
                "response": "Planner generated an empty plan.",
                "actions_taken": [],
                "spoken": False,
            }

        print(f"   [Planner] Plan: '{summary}' — {len(steps)} step(s)")
        for i, s in enumerate(steps, 1):
            print(f"   [Planner]   {i}. {s}")

        # ── Step 2: Announce plan ─────────────────────────────────────────────
        intro = f"{summary}. {len(steps)} step{'s' if len(steps) != 1 else ''}."
        if speak and voice:
            voice.speak(intro)

        # ── Step 3: Execute each step ─────────────────────────────────────────
        from talents.plan_executor import execute_plan_steps

        step_results = execute_plan_steps(
            steps=steps,
            assistant=assistant,
            speak=speak,
            notify=notify,
            command_source=command_source,
        )
        all_ok = all(sr["success"] for sr in step_results)

        # ── Step 4: Build summary response ────────────────────────────────────
        lines = [f"**{summary}** — {len(steps)} steps completed:\n"]
        for sr in step_results:
            icon = "✓" if sr["success"] else "✗"
            lines.append(f"{icon} {sr['command']}")
            if sr["response"]:
                lines.append(f"  → {sr['response']}")

        combined_response = "\n".join(lines)

        actions_taken = [
            {
                "action": {"type": "plan_step", "step": sr["step"], "command": sr["command"]},
                "result": sr["response"],
                "success": sr["success"],
            }
            for sr in step_results
        ]

        return {
            "success": all_ok,
            "response": combined_response,
            "actions_taken": actions_taken,
            "spoken": True,  # Each step's talent already spoke its own result
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_roster(self, assistant) -> str:
        """Build a short talent list for the planning prompt."""
        lines = []
        for talent in assistant.talents:
            if not talent.enabled or not talent.routing_available:
                continue
            if talent.name == self.name:
                continue  # Skip planner itself
            if talent.examples:
                ex = "; ".join(talent.examples[:3])
                lines.append(f"- {talent.name}: {talent.description} (e.g. {ex})")
            else:
                kws = ", ".join(talent.keywords[:4])
                lines.append(f"- {talent.name}: {talent.description} (keywords: {kws})")
        lines.append("- conversation: General chat, questions, anything else")
        return "\n".join(lines)

    def _parse_plan(self, raw: str) -> dict | None:
        """Parse the LLM plan JSON, stripping markdown fences if present."""
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```[a-z]*\n?", "", clean)
                clean = re.sub(r"\n?```$", "", clean.strip())

            match = re.search(r'\{.*\}', clean, re.DOTALL)
            if not match:
                return None

            return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            return None
