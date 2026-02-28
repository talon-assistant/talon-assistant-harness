"""ClipboardTransform — apply LLM transforms to whatever is in the clipboard.

The user copies some text in any app, then tells Talon what to do with it.
Talon reads the clipboard, transforms it, and writes the result back.
The user just pastes wherever they want it.

Example commands
----------------
  "rewrite this more formally"
  "summarize this"
  "fix the grammar"
  "translate this to Spanish"
  "make this shorter"
  "turn this into bullet points"
  "explain this code in plain English"
  "make this more casual"
  "improve the writing"
  "proofread this"
"""

import pyperclip

from talents.base import BaseTalent


class ClipboardTransformTalent(BaseTalent):
    """Read clipboard text, apply a natural-language transform, write result back."""

    name = "clipboard_transform"
    description = (
        "Transform clipboard text on demand: rewrite, summarize, translate, "
        "fix grammar, shorten, or any other LLM-powered edit"
    )
    keywords = [
        "rewrite this", "rewrite that",
        "summarize this", "summarize that", "summarize what",
        "fix the grammar", "fix grammar", "fix my grammar",
        "translate this", "translate that",
        "make this shorter", "make that shorter",
        "make this longer", "make that longer",
        "make this more formal", "make this more casual",
        "make it more formal", "make it more casual",
        "make this formal", "make this casual",
        "explain this", "explain that", "explain this code",
        "bullet points", "turn this into",
        "clean up this", "clean this up",
        "rephrase this", "paraphrase this",
        "improve this", "improve the writing",
        "proofread this", "proofread that",
        "shorten this", "expand this",
    ]
    examples = [
        "rewrite this more formally",
        "summarize what I just copied",
        "fix the grammar in this",
        "translate this to French",
        "make this shorter and punchier",
        "turn this into bullet points",
        "explain this code in plain English",
        "make this more casual",
        "proofread this and fix any mistakes",
        "improve the writing in this",
    ]
    priority = 68   # Between hue_lights (70) and reminder (65)

    _MAX_CHARS = 3000   # Truncate clipboard beyond this to stay within LLM context

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    def execute(self, command: str, context: dict) -> dict:
        llm = context["llm"]

        # ── Read clipboard ────────────────────────────────────────────────────
        try:
            text = pyperclip.paste()
        except Exception as e:
            return {
                "success": False,
                "response": f"Couldn't read the clipboard: {e}",
                "actions_taken": [],
            }

        if not text or not text.strip():
            return {
                "success": False,
                "response": (
                    "The clipboard is empty. Copy some text first, "
                    "then ask me to transform it."
                ),
                "actions_taken": [],
            }

        # ── Truncate if necessary ─────────────────────────────────────────────
        truncated = False
        if len(text) > self._MAX_CHARS:
            text = text[:self._MAX_CHARS]
            truncated = True

        # ── Transform ─────────────────────────────────────────────────────────
        # Pass the raw command as the instruction — the LLM handles any phrasing
        # naturally without needing to classify the transform type.
        prompt = (
            f"Apply the following instruction to the text below.\n"
            f"Return ONLY the transformed text. No explanation, no preamble, no quotes.\n\n"
            f"Instruction: {command}\n\n"
            f"Text:\n{text}"
        )

        try:
            result = llm.generate(prompt, max_length=700, temperature=0.7)
            result = result.strip()
        except Exception as e:
            return {
                "success": False,
                "response": f"Transform failed: {e}",
                "actions_taken": [],
            }

        if not result:
            return {
                "success": False,
                "response": "Got an empty result — try rephrasing your instruction.",
                "actions_taken": [],
            }

        # ── Write result back to clipboard ────────────────────────────────────
        try:
            pyperclip.copy(result)
        except Exception as e:
            return {
                "success": False,
                "response": f"Transform succeeded but couldn't write to clipboard: {e}",
                "actions_taken": [],
            }

        # ── Build response ────────────────────────────────────────────────────
        word_count = len(result.split())
        truncate_note = " (input was long so I only used the first 3000 characters)" if truncated else ""
        response = f"Done{truncate_note} — {word_count} words ready to paste."

        return {
            "success": True,
            "response": response,
            "actions_taken": [
                {"action": "clipboard_transform", "instruction": command, "success": True}
            ],
        }
