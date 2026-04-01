"""TaskAssist — collaborative screen-aware writing and coding assistant.

Supports two modes:
  - **Quick draft** (legacy): one-shot LLM call → draft → review dialog
  - **Agentic mode**: LLM analyzes context → proposes a plan → user approves
    → steps execute through talent pipeline → review dialog

The mode is selected in the pre-dialog by the user (toggle button).

Example triggers
----------------
  "task assist"
  "help me with this"
  "help me write this letter"
  "help me with this code"
  "work on this with me"
  "review this with me"
  Ctrl+Alt+J (global hotkey, configurable)
"""

import base64
import io
import json
import re

import pyperclip
from PIL import Image

from talents.base import BaseTalent
from core.llm_client import LLMError


class TaskAssistTalent(BaseTalent):
    """Collaborative screen-aware assistant — capture context, draft, review."""

    name = "task_assist"
    description = (
        "Collaborative task assistant: captures the current screen context, "
        "reads clipboard content, and produces a draft the user can accept, "
        "revise, or decline"
    )
    keywords = [
        "task assist",
        "help me with this",
        "help me write",
        "help me with this code",
        "help me with this document",
        "help me with this letter",
        "help me with this email",
        "work on this with me",
        "review this with me",
        "draft something for this",
        "help me draft",
        "cowork",
        "co-work",
        "collaborate on this",
    ]
    examples = [
        "task assist",
        "help me write this letter",
        "help me with this code",
        "help me draft a response to this",
        "work on this with me",
        "review this document with me",
    ]
    priority = 75  # Between clipboard_transform (68) and planner (85)

    _MAX_CLIP_CHARS = 3000
    _SCREENSHOT_MAX_PX = 1280  # cap longest side before encoding

    # ── Planning prompt for agentic mode ────────────────────────────────────

    _PLAN_SYSTEM_PROMPT = """\
You are a planning assistant for a desktop AI called Talon.
The user pressed a hotkey to ask for help with something on their screen.
You can see their screenshot and clipboard contents.

Your job is to:
1. Assess the SITUATION — what is the user looking at, what app are they in,
   what do they likely need help with?
2. Create a PLAN — a short sequence of concrete steps Talon can execute to
   help the user.  Each step should be a natural-language command that Talon's
   talent system can route (web search, memory lookup, draft text, etc.).

Available Talon capabilities:
{roster}

Active window: {app_title} ({process_name})

Rules:
- Maximum 6 steps.
- Each step must be a self-contained command Talon can execute.
- If a step produces output the next step needs, use {{last_result}} as a
  placeholder in the next step.
- The LAST step should always be the final drafting/output step.
- If you need information from the user's memory or past conversations,
  include a "recall from memory..." step.
- If you need information from the web, include a "search the web for..." step.

Respond ONLY with a JSON object — no markdown, no explanation:
{{"situation": "short description of what user is doing",
  "plan_summary": "what you'll do for them",
  "steps": ["step 1", "step 2", ...]}}
"""

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    def execute(self, command: str, context: dict) -> dict:
        assistant = context.get("assistant")
        agentic = getattr(assistant, "_pending_task_assist_agentic", False)
        if assistant:
            assistant._pending_task_assist_agentic = False

        if agentic:
            return self._execute_agentic(command, context)
        else:
            return self._execute_quick_draft(command, context)

    # ── Quick draft mode (legacy) ───────────────────────────────────────────

    def _execute_quick_draft(self, command: str, context: dict) -> dict:
        llm = context["llm"]
        vision = context.get("vision")
        assistant = context.get("assistant")

        task = self._get_task(assistant, llm, command)
        clip_text = self._get_clipboard()

        if clip_text:
            blocked = self._check_clipboard_security(clip_text)
            if blocked:
                return blocked

        screenshot_b64 = self._get_screenshot(assistant, vision)

        # Build prompt
        parts = [f"Task: {task}"]
        if clip_text:
            parts.append(
                f"\n[CLIPBOARD CONTENT]\n{clip_text}\n[END CLIPBOARD]"
            )
        screen_note = " and the clipboard content above" if clip_text else ""
        parts.append(
            f"\nBased on what you can see on screen{screen_note}, "
            "help the user with their task. "
            "Produce a draft, edit, or response as appropriate. "
            "Return ONLY the result — no preamble, no explanation."
        )
        prompt = "\n".join(parts)

        system_prompt = (
            "You are a collaborative writing and coding assistant. "
            "The user has shared their screen with you. "
            "Produce exactly what they asked for — no meta-commentary."
        )

        try:
            draft = llm.generate(
                prompt,
                use_vision=bool(screenshot_b64),
                screenshot_b64=screenshot_b64,
                max_length=900,
                temperature=0.7,
                system_prompt=system_prompt,
            )
            draft = (draft or "").strip()
        except LLMError as e:
            return {
                "success": False,
                "response": f"LLM unavailable: {e}",
                "actions_taken": [],
                "spoken": False,
            }
        except Exception as e:
            return {
                "success": False,
                "response": f"Task Assist failed: {e}",
                "actions_taken": [],
            }

        if not draft:
            return {
                "success": False,
                "response": "Got an empty draft — try rephrasing your request.",
                "actions_taken": [],
            }

        return {
            "success": True,
            "response": "Opening Task Assist review...",
            "actions_taken": [{"action": "task_assist", "task": task}],
            "pending_task_assist": {
                "task": task,
                "draft": draft,
                "screenshot_b64": screenshot_b64,
            },
        }

    # ── Agentic mode ────────────────────────────────────────────────────────

    def _execute_agentic(self, command: str, context: dict) -> dict:
        llm = context["llm"]
        vision = context.get("vision")
        assistant = context.get("assistant")

        task = self._get_task(assistant, llm, command)
        clip_text = self._get_clipboard()

        if clip_text:
            blocked = self._check_clipboard_security(clip_text)
            if blocked:
                return blocked

        screenshot_b64 = self._get_screenshot(assistant, vision)
        window_info = {}
        if assistant:
            window_info = getattr(
                assistant, "_pending_task_assist_window_info", {}
            ) or {}
            assistant._pending_task_assist_window_info = None

        # Build the planning prompt
        roster = self._build_roster(assistant)
        system_prompt = self._PLAN_SYSTEM_PROMPT.format(
            roster=roster,
            app_title=window_info.get("app_title", "Unknown"),
            process_name=window_info.get("process_name", "unknown"),
        )

        parts = [f"User's task: {task}"]
        if clip_text:
            parts.append(
                f"\n[CLIPBOARD CONTENT]\n{clip_text}\n[END CLIPBOARD]"
            )
        parts.append(
            "\nAnalyze the screenshot and context, then create a plan."
        )
        prompt = "\n".join(parts)

        try:
            raw = llm.generate(
                prompt,
                use_vision=bool(screenshot_b64),
                screenshot_b64=screenshot_b64,
                max_length=512,
                temperature=0.3,
                system_prompt=system_prompt,
            )
        except LLMError as e:
            return {
                "success": False,
                "response": f"LLM unavailable: {e}",
                "actions_taken": [],
                "spoken": False,
            }
        except Exception as e:
            return {
                "success": False,
                "response": f"Task Assist planning failed: {e}",
                "actions_taken": [],
            }

        plan = self._parse_plan(raw)
        if plan is None or not plan.get("steps"):
            # Fall back to quick draft if planning fails
            print("   [TaskAssist] Plan parsing failed, falling back to quick draft.")
            return self._execute_quick_draft(command, context)

        print(f"   [TaskAssist] Plan: '{plan.get('plan_summary', '')}'")
        for i, s in enumerate(plan["steps"], 1):
            print(f"   [TaskAssist]   {i}. {s}")

        return {
            "success": True,
            "response": "Opening Task Assist plan review...",
            "actions_taken": [{"action": "task_assist_plan", "task": task}],
            "pending_task_assist_plan": {
                "task": task,
                "situation": plan.get("situation", ""),
                "plan_summary": plan.get("plan_summary", ""),
                "steps": plan["steps"][:6],
                "screenshot_b64": screenshot_b64,
                "clip_text": clip_text,
            },
        }

    # ── Shared helpers ──────────────────────────────────────────────────────

    def _get_task(self, assistant, llm, command: str) -> str:
        task = None
        if assistant and getattr(assistant, "_pending_task_assist_task", None):
            task = assistant._pending_task_assist_task
            assistant._pending_task_assist_task = None
        if not task:
            task = self._extract_arg(
                llm, command,
                "the task or goal the user wants help with — one short sentence",
                max_length=80,
            ) or command.strip()
        return task

    def _get_clipboard(self) -> str:
        try:
            clip_text = (pyperclip.paste() or "").strip()
            if len(clip_text) > self._MAX_CLIP_CHARS:
                clip_text = clip_text[:self._MAX_CLIP_CHARS]
            return clip_text
        except Exception:
            return ""

    def _check_clipboard_security(self, clip_text: str) -> dict | None:
        from core.security import get_security_filter as _gsf
        _sf = _gsf()
        if _sf:
            _blocked, _alert = _sf.check_semantic_input(clip_text, "web")
            if _blocked:
                return {
                    "success": False,
                    "response": "[Task Assist blocked: clipboard content flagged by security filter]",
                    "actions_taken": [],
                }
        return None

    def _get_screenshot(self, assistant, vision) -> str | None:
        screenshot_b64 = None
        if assistant and getattr(
            assistant, "_pending_task_assist_screenshot", None
        ):
            screenshot_b64 = assistant._pending_task_assist_screenshot
            assistant._pending_task_assist_screenshot = None
        elif vision:
            try:
                raw_b64 = vision.capture_screenshot()
                screenshot_b64 = self._resize_screenshot(raw_b64)
            except Exception as e:
                print(f"   [TaskAssist] Screenshot failed: {e}")
        return screenshot_b64

    def _resize_screenshot(self, b64_png: str) -> str:
        """Resize screenshot so the longest side is at most _SCREENSHOT_MAX_PX."""
        try:
            raw = base64.b64decode(b64_png)
            img = Image.open(io.BytesIO(raw))
            img.thumbnail(
                (self._SCREENSHOT_MAX_PX, self._SCREENSHOT_MAX_PX),
                Image.LANCZOS,
            )
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return b64_png

    def _build_roster(self, assistant) -> str:
        """Build a short talent list for the planning prompt."""
        if not assistant:
            return "- conversation: General chat, questions, anything else"
        lines = []
        for talent in assistant.talents:
            if not talent.enabled or not talent.routing_available:
                continue
            if talent.name in (self.name, "planner"):
                continue
            if talent.examples:
                ex = "; ".join(talent.examples[:3])
                lines.append(f"- {talent.name}: {talent.description} (e.g. {ex})")
            else:
                kws = ", ".join(talent.keywords[:4])
                lines.append(
                    f"- {talent.name}: {talent.description} (keywords: {kws})"
                )
        lines.append("- conversation: General chat, questions, anything else")
        return "\n".join(lines)

    def _parse_plan(self, raw: str) -> dict | None:
        """Parse the LLM plan JSON, stripping markdown fences if present."""
        if not raw:
            return None
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
