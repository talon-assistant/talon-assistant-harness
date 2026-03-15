"""TaskAssist — collaborative screen-aware writing and coding assistant.

The user activates Task Assist (text command, toolbar button, or global
hotkey) and Talon captures the current screen context plus clipboard,
sends both to the vision model, and presents a review dialog where the
user can Accept, Revise, or Decline the draft.

Example triggers
----------------
  "task assist"
  "help me with this"
  "help me write this letter"
  "help me with this code"
  "work on this with me"
  "review this with me"
  Ctrl+Shift+T (global hotkey, configurable)
"""

import base64
import io

import pyperclip
from PIL import Image

from talents.base import BaseTalent


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

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    def execute(self, command: str, context: dict) -> dict:
        llm = context["llm"]
        vision = context.get("vision")

        # ── Extract task description ───────────────────────────────────────────
        task = self._extract_arg(
            llm, command,
            "the task or goal the user wants help with — one short sentence",
            max_length=80,
        ) or command.strip()

        # ── Clipboard ─────────────────────────────────────────────────────────
        clip_text = ""
        try:
            clip_text = (pyperclip.paste() or "").strip()
            if len(clip_text) > self._MAX_CLIP_CHARS:
                clip_text = clip_text[:self._MAX_CLIP_CHARS]
        except Exception:
            pass

        # ── Semantic check on clipboard content ───────────────────────────────
        if clip_text:
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

        # ── Screenshot ────────────────────────────────────────────────────────
        # Use pre-captured screenshot from hotkey listener if available
        # (captured before any window switch, so shows the user's active app)
        assistant = context.get("assistant")
        screenshot_b64 = None
        if assistant and getattr(assistant, "_pending_task_assist_screenshot", None):
            screenshot_b64 = assistant._pending_task_assist_screenshot
            assistant._pending_task_assist_screenshot = None
        elif vision:
            try:
                raw_b64 = vision.capture_screenshot()
                screenshot_b64 = self._resize_screenshot(raw_b64)
            except Exception as e:
                print(f"   [TaskAssist] Screenshot failed: {e}")

        # ── Build prompt ───────────────────────────────────────────────────────
        parts = [f"Task: {task}"]
        if clip_text:
            parts.append(
                f"\n[CLIPBOARD CONTENT — provided by user]\n{clip_text}\n[END CLIPBOARD]"
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

        # ── LLM call ──────────────────────────────────────────────────────────
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

    # ── helpers ───────────────────────────────────────────────────────────────

    def _resize_screenshot(self, b64_png: str) -> str:
        """Resize screenshot so the longest side is at most _SCREENSHOT_MAX_PX."""
        try:
            raw = base64.b64decode(b64_png)
            img = Image.open(io.BytesIO(raw))
            img.thumbnail((self._SCREENSHOT_MAX_PX, self._SCREENSHOT_MAX_PX), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return b64_png  # fall back to original if resize fails
