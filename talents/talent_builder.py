"""TalentBuilder — self-writing talent plugin generator.

The user describes what they want in plain English; this talent generates,
validates, writes, and hot-loads a new talent plugin without restarting.

Flow
----
1. _extract_arg() → short description from the user's command
2. LLM call 1 (requirements) → structured JSON spec
3. LLM call 2 (code gen) → complete Python class using few-shot examples
4. Syntax validation via py_compile; safety scan via string pattern check
5. Optional fix pass (one attempt) if validation fails
6. Write to talents/user/<name>.py
7. Hot-load via context["assistant"].load_user_talent(filepath)
8. Respond with confirmation + example commands
"""

import json
import os
import py_compile
import re
import tempfile
from pathlib import Path

from talents.base import BaseTalent


# ── Few-shot examples embedded in the generation system prompt ───────────────
# These are intentionally short complete talents that demonstrate every pattern
# an LLM-generated talent needs to follow.

_EXAMPLE_1 = '''\
import random
from talents.base import BaseTalent

class CoinFlipTalent(BaseTalent):
    name = "coin_flip"
    description = "Flip a coin or roll dice on request"
    keywords = ["flip a coin", "roll dice", "roll a die"]
    examples = ["flip a coin", "roll a dice", "roll a 20-sided die"]
    priority = 50

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    def execute(self, command: str, context: dict) -> dict:
        llm = context["llm"]
        cmd = command.lower()
        if "dice" in cmd or "die" in cmd or "roll" in cmd:
            sides_str = self._extract_arg(llm, command, "number of sides",
                                          max_length=5) or "6"
            try:
                n = int(sides_str)
            except ValueError:
                n = 6
            result = random.randint(1, n)
            response = f"Rolled a {n}-sided die: {result}"
        else:
            result = random.choice(["Heads", "Tails"])
            response = f"Coin flip: {result}!"
        return {"success": True, "response": response, "actions_taken": []}
'''

_EXAMPLE_2 = '''\
import requests
from talents.base import BaseTalent

class DadJokeTalent(BaseTalent):
    name = "dad_joke"
    description = "Fetch a random dad joke from the internet"
    keywords = ["dad joke", "tell me a joke", "give me a joke"]
    examples = ["tell me a dad joke", "give me a funny joke", "I need a laugh"]
    priority = 50

    def get_config_schema(self) -> dict:
        return {"fields": [
            {"key": "language", "label": "Language", "type": "string", "default": "en"},
        ]}

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    def execute(self, command: str, context: dict) -> dict:
        llm = context["llm"]
        try:
            r = requests.get(
                "https://icanhazdadjoke.com/",
                headers={"Accept": "application/json"},
                timeout=5,
            )
            joke = r.json().get("joke", "Why don\'t scientists trust atoms? They make up everything.")
        except Exception:
            joke = "I would tell a chemistry joke but I know it wouldn\'t get a reaction."
        response = llm.generate(
            f"Deliver this joke naturally in one short sentence: {joke}",
            max_length=100, temperature=0.7,
        )
        return {"success": True, "response": response, "actions_taken": []}
'''

_GEN_SYSTEM_PROMPT = f"""\
You are generating a Python plugin for Talon, a personal AI desktop assistant.
Generate a complete, working Python class that extends BaseTalent.

BaseTalent interface (always import from talents.base):
  - Class attributes: name (str snake_case), description (str), keywords (list[str]),
    examples (list[str] — 3-5 natural command phrases), priority (int, default 50)
  - def can_handle(self, command: str) -> bool  — usually: return self.keyword_match(command)
  - def execute(self, command: str, context: dict) -> dict
      context keys available: llm, memory, voice, config, speak_response, assistant
      Must return: {{"success": bool, "response": str, "actions_taken": list}}
  - def get_config_schema(self) -> dict  — include ONLY when the talent needs user-supplied
      credentials or settings:
      return {{"fields": [{{"key": "api_key", "label": "API Key", "type": "password", "default": ""}}]}}
      Access at runtime via: self.talent_config.get("api_key", "")
  - self._extract_arg(llm, command, "what to extract", max_length=20, temperature=0.0) -> str|None
      Use this to pull a single value (location, number, name, etc.) from the command.
  - Use requests for HTTP calls. Always wrap in try/except with a sensible fallback response.

--- EXAMPLE 1: Simple talent with no external API ---
{_EXAMPLE_1}
--- EXAMPLE 2: Talent with HTTP API and config schema ---
{_EXAMPLE_2}
--- END EXAMPLES ---

Rules:
- Generate ONLY the Python source code — no markdown fences, no explanation.
- Start with import statements.
- The class must subclass BaseTalent.
- Keep execute() under 60 lines. Use helper methods if needed.
- Always return a dict with success, response, and actions_taken keys.
- If the talent needs credentials, include get_config_schema() and read from self.talent_config.
"""

_REQUIREMENTS_SYSTEM_PROMPT = """\
You are designing a plugin specification for Talon, a personal AI desktop assistant.
Given a plain-English description of what a new plugin should do, infer a complete
specification and return it as a JSON object.

Return ONLY valid JSON, no explanation, no markdown.
"""

_FORBIDDEN_PATTERNS = [
    "import subprocess",
    "os.system(",
    "os.popen(",
    "os.execv(",
    "os.execl(",
    "os.execle(",
    "__import__(",
    "eval(",
    "exec(",
]


class TalentBuilderTalent(BaseTalent):
    """Generate, validate, and hot-load new talent plugins from plain-English descriptions."""

    name = "talent_builder"
    description = "Create new talent plugins from a plain-English description"
    keywords = [
        "create a talent", "build a talent", "make a talent",
        "create a plugin", "build a plugin", "make a plugin",
        "create a skill", "build a skill", "make a skill",
        "write a talent", "generate a talent", "new talent plugin",
    ]
    examples = [
        "create a talent that checks my Jira tickets",
        "build a talent that can send Slack messages",
        "make a talent that reads my RSS feeds",
        "create a plugin that monitors my server uptime",
        "build a talent that shows me stock prices",
        "make a talent that tells me trivia questions",
    ]
    priority = 82   # High — just below planner (85)

    # ── public interface ──────────────────────────────────────────────────────

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    def execute(self, command: str, context: dict) -> dict:
        llm = context["llm"]
        assistant = context.get("assistant")

        # ── Step 1: extract the user's description ────────────────────────────
        description = self._extract_arg(
            llm, command,
            "description of what the new talent or plugin should do",
            max_length=100, temperature=0.0,
        )
        if not description:
            return {
                "success": False,
                "response": "What should the new talent do? Describe it and I'll build it.",
                "actions_taken": [],
            }

        print(f"   [TalentBuilder] Description: {description}")

        # ── Step 2: requirements extraction ──────────────────────────────────
        requirements = self._extract_requirements(llm, description)
        if not requirements:
            return {
                "success": False,
                "response": "I couldn't work out the requirements. Try describing the talent more specifically.",
                "actions_taken": [],
            }

        talent_name = requirements.get("name", "").strip().lower().replace(" ", "_")
        if not talent_name or not re.match(r'^[a-z][a-z0-9_]*$', talent_name):
            talent_name = re.sub(r'[^a-z0-9_]', '_', description.lower()[:20]).strip('_') or "custom_talent"

        print(f"   [TalentBuilder] Requirements for '{talent_name}': "
              f"needs_config={requirements.get('needs_config')}")

        # Handle name collision
        dest_path = Path("talents/user") / f"{talent_name}.py"
        if dest_path.exists():
            talent_name = f"{talent_name}_2"
            dest_path = Path("talents/user") / f"{talent_name}.py"
            requirements["name"] = talent_name
            if "class_name" in requirements:
                requirements["class_name"] = requirements["class_name"].rstrip("2") + "2Talent"

        # ── Step 3: code generation ───────────────────────────────────────────
        code = self._generate_code(llm, requirements)
        if not code:
            return {
                "success": False,
                "response": "Code generation failed. Try rephrasing the talent description.",
                "actions_taken": [],
            }

        # ── Step 4: validate ──────────────────────────────────────────────────
        valid, error = self._validate_code(code)

        # ── Step 5: one fix pass if needed ────────────────────────────────────
        if not valid:
            print(f"   [TalentBuilder] Validation failed ({error}), attempting fix...")
            code = self._fix_code(llm, code, error)
            valid, error = self._validate_code(code)

        # ── Step 6: write to disk ─────────────────────────────────────────────
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(code, encoding="utf-8")
        print(f"   [TalentBuilder] Written to {dest_path}")

        # ── Step 7: hot-load ──────────────────────────────────────────────────
        if valid and assistant and hasattr(assistant, "load_user_talent"):
            load_result = assistant.load_user_talent(str(dest_path))
            if load_result.get("success"):
                return self._success_response(load_result, requirements, dest_path)
            else:
                # File written but load failed
                return {
                    "success": True,
                    "response": (
                        f"Wrote '{talent_name}' to {dest_path} but couldn't load it: "
                        f"{load_result.get('error', 'unknown error')}. "
                        f"Try restarting Talon to pick it up."
                    ),
                    "actions_taken": [{"action": "write_talent", "result": str(dest_path), "success": True}],
                }
        elif not valid:
            return {
                "success": True,   # File saved — user can fix manually
                "response": (
                    f"Generated code for '{talent_name}' but hit a validation issue: {error}\n"
                    f"Saved to: {dest_path}\n"
                    f"Review and fix it manually, then restart Talon."
                ),
                "actions_taken": [{"action": "write_talent", "result": str(dest_path), "success": False}],
            }
        else:
            return {
                "success": True,
                "response": (
                    f"Created '{talent_name}' and saved to {dest_path}. "
                    f"Restart Talon to activate it."
                ),
                "actions_taken": [{"action": "write_talent", "result": str(dest_path), "success": True}],
            }

    # ── private helpers ───────────────────────────────────────────────────────

    def _extract_requirements(self, llm, description: str) -> dict | None:
        """LLM call 1 — infer a structured JSON spec from the description."""
        user_prompt = (
            f'Given this description of a new Talon talent plugin:\n\n"{description}"\n\n'
            "Return ONLY valid JSON with this exact structure:\n"
            "{\n"
            '  "class_name": "PascalCaseTalent",\n'
            '  "name": "snake_case",\n'
            '  "priority": 50,\n'
            '  "description": "one-line description",\n'
            '  "keywords": ["keyword1", "keyword2"],\n'
            '  "examples": ["example command 1", "example command 2", "example command 3"],\n'
            '  "needs_config": false,\n'
            '  "config_fields": [],\n'
            '  "external_api": "brief description or none",\n'
            '  "execute_logic": "step-by-step plain English of what execute() should do"\n'
            "}"
        )
        try:
            raw = llm.generate(
                user_prompt,
                system_prompt=_REQUIREMENTS_SYSTEM_PROMPT,
                max_length=350,
                temperature=0.0,
            )
            clean = raw.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```[a-z]*\n?", "", clean)
                clean = re.sub(r"\n?```$", "", clean.strip())
            match = re.search(r'\{.*\}', clean, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            print(f"   [TalentBuilder] Requirements extraction error: {e}")
        return None

    def _generate_code(self, llm, requirements: dict) -> str:
        """LLM call 2 — generate the full Python talent class."""
        user_prompt = (
            "Now generate a new talent with these requirements:\n"
            f"{json.dumps(requirements, indent=2)}"
        )
        try:
            raw = llm.generate(
                user_prompt,
                system_prompt=_GEN_SYSTEM_PROMPT,
                max_length=1200,
                temperature=0.1,
            )
            # Strip markdown fences if the model wrapped the output
            code = raw.strip()
            if code.startswith("```"):
                code = re.sub(r"^```[a-z]*\n?", "", code)
                code = re.sub(r"\n?```$", "", code.strip())
            return code
        except Exception as e:
            print(f"   [TalentBuilder] Code generation error: {e}")
            return ""

    def _validate_code(self, code: str) -> tuple[bool, str]:
        """Syntax check (py_compile) + safety scan.  Returns (valid, error_message)."""
        if not code.strip():
            return False, "Empty code generated"

        # 1. Syntax check — write to temp file and compile
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                tmp = f.name
            py_compile.compile(tmp, doraise=True)
        except py_compile.PyCompileError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

        # 2. Safety scan
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern in code:
                return False, f"Forbidden pattern: {pattern!r}"

        # 3. Must subclass BaseTalent
        if "BaseTalent" not in code:
            return False, "No BaseTalent subclass found"

        return True, ""

    def _fix_code(self, llm, code: str, error: str) -> str:
        """LLM call 3 — one-shot fix attempt for a syntax / validation error."""
        fix_prompt = (
            f"Fix this Python code. The error is:\n{error}\n\n"
            f"Return ONLY the corrected Python code, no explanation, no markdown:\n\n"
            f"{code}"
        )
        try:
            raw = llm.generate(fix_prompt, max_length=1200, temperature=0.0)
            fixed = raw.strip()
            if fixed.startswith("```"):
                fixed = re.sub(r"^```[a-z]*\n?", "", fixed)
                fixed = re.sub(r"\n?```$", "", fixed.strip())
            return fixed
        except Exception as e:
            print(f"   [TalentBuilder] Fix pass error: {e}")
            return code  # Return original if fix call itself fails

    @staticmethod
    def _success_response(load_result: dict, requirements: dict, dest_path: Path) -> dict:
        """Build the human-readable success message."""
        name = load_result["name"]
        examples = load_result.get("examples", requirements.get("examples", []))
        examples_str = ", ".join(f'"{e}"' for e in examples[:3])
        needs_config = load_result.get("needs_config") or requirements.get("needs_config", False)

        lines = [f"Created and loaded '{name}' talent!"]
        if examples_str:
            lines.append(f"Try: {examples_str}")
        if needs_config:
            lines.append(
                f"⚙️  Needs configuration — go to Settings → Talent Config → {name} "
                f"to add your credentials."
            )

        return {
            "success": True,
            "response": "\n".join(lines),
            "actions_taken": [{
                "action": "create_talent",
                "result": str(dest_path),
                "success": True,
            }],
        }
