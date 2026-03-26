"""TalentBuilder — self-writing talent plugin generator.

The user describes what they want in plain English; this talent generates,
validates, writes, and hot-loads a new talent plugin without restarting.

Strategy: Template Assembly (not free-form class generation)
------------------------------------------------------------
Small models (4/8B) struggle to reliably produce a correct Python class in one
shot — wrong return dict, missing imports, truncated code, bad indentation, etc.

Instead the responsibility is split:

  Python builds:   class header, all class attributes, can_handle(), the full
                   execute() wrapper (try/except, return dict), get_config_schema(),
                   and all module-level imports — all deterministic, always correct.

  LLM generates:   ONLY the ~10-40 line body of execute() logic.
                   Task: "fetch/compute data, end with response = '...'"
                   No class structure, no return, no try/except needed.

This makes the LLM's job radically simpler and nearly impossible to structurally fail.

Flow
----
1. _extract_arg()             → short description from the user's command
2. LLM call 1 (requirements)  → structured JSON spec (name, keywords, config…)
3. LLM call 2 (execute body)  → ~20-40 lines of logic ending with response = "..."
4. Body sanity check           → response= present, no forbidden patterns
5. _assemble_talent_file()     → Python builds the complete .py file
6. Syntax check (py_compile)   + safety scan
7. Optional fix pass           → one attempt on the body if assembly fails
8. Write to talents/user/<name>.py
9. Hot-load via context["assistant"].load_user_talent()
"""

import json
import os
import py_compile
import re
import tempfile
import textwrap
from pathlib import Path

from talents.base import BaseTalent


# ── Module-level imports that every assembled talent gets ─────────────────────
# All commonly needed stdlib + requests are pre-imported so the model never
# needs to write import statements.

_TEMPLATE_IMPORTS = """\
import datetime
import json
import os
import random
import re
import time

import requests
from talents.base import BaseTalent
"""

# ── System prompts ─────────────────────────────────────────────────────────────

_REQUIREMENTS_SYSTEM_PROMPT = """\
You are designing a plugin specification for Talon, a personal AI desktop assistant.
Given a plain-English description of what a new plugin should do, infer a complete
specification and return it as a JSON object.

Return ONLY valid JSON, no explanation, no markdown.
"""

_EXEC_BODY_SYSTEM_PROMPT = """\
You are writing the body of a Python execute() function for a Talon voice assistant plugin.

Variables already available — do NOT re-declare or import these:
  llm         call: llm.generate(prompt, max_length=150, temperature=0.7) -> str
  command     the user's voice command string
  self.talent_config   dict of user credentials/settings — read with .get("key", "")
  self._extract_arg(llm, command, "what to extract", max_length=20) -> str or None
  requests    for HTTP calls — always use timeout=10

Pre-imported, use freely without import statements:
  datetime, json, os, random, re, time

Rules (follow all of them):
  1. The LAST line of your code must be:  response = "..."
  2. Do NOT write:  return  /  try  /  except  /  def  /  class  /  import
  3. Keep it under 40 lines
  4. If credentials needed: check self.talent_config.get("key",""), set response
     to a config reminder message if the value is empty

--- EXAMPLE 1: Extract topic from command and ask LLM ---
topic = self._extract_arg(llm, command, "topic to explain", max_length=30) or "that"
response = llm.generate(
    f"Explain '{topic}' clearly in 2 short sentences.",
    max_length=150,
    temperature=0.7,
)

--- EXAMPLE 2: HTTP API call with credential guard ---
api_key = self.talent_config.get("api_key", "")
if not api_key:
    response = "API key not configured. Go to Settings → Talent Config to add it."
else:
    r = requests.get(
        "https://api.example.com/v1/data",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    data = r.json()
    value = data.get("result", "No data found")
    response = f"Result: {value}"

Write ONLY the body code. No imports, no def, no class, no return, no try/except.
Final line must be:  response = "..."
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

        # ── Step 1: extract what the user wants ───────────────────────────────
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

        # ── Step 2: LLM call 1 — requirements JSON ────────────────────────────
        requirements = self._extract_requirements(llm, description)
        if not requirements:
            return {
                "success": False,
                "response": "Couldn't determine requirements. Try describing the talent more specifically.",
                "actions_taken": [],
            }

        # Sanitise and normalise the name
        talent_name = self._safe_name(
            requirements.get("name", ""), description
        )
        requirements["name"] = talent_name
        requirements["class_name"] = self._class_name(
            requirements.get("class_name", ""), talent_name
        )
        print(f"   [TalentBuilder] Building '{talent_name}' "
              f"(needs_config={requirements.get('needs_config')})")

        # Resolve name collision by appending _2
        dest_path = Path("talents/user") / f"{talent_name}.py"
        if dest_path.exists():
            talent_name += "_2"
            requirements["name"] = talent_name
            requirements["class_name"] = requirements["class_name"][:-6] + "2Talent"
            dest_path = Path("talents/user") / f"{talent_name}.py"

        # ── Step 3: generate, validate, refine loop (up to 3 attempts) ──────
        max_attempts = 3
        code = ""
        valid = False
        error = ""
        test_error = ""

        for attempt in range(1, max_attempts + 1):
            print(f"   [TalentBuilder] Attempt {attempt}/{max_attempts}")

            # Generate execute body (with error context on retries)
            if attempt == 1:
                execute_body = self._generate_execute_body(llm, requirements)
            else:
                # Refine: send the previous body + error back
                fix_context = error or test_error
                print(f"   [TalentBuilder] Refining — error: {fix_context}")
                execute_body = self._fix_execute_body(
                    llm, execute_body, fix_context
                )

            if not execute_body:
                error = "Empty execute body"
                continue

            # Body sanity check
            body_ok, body_err = self._validate_execute_body(execute_body)
            if not body_ok:
                print(f"   [TalentBuilder] Body issue: {body_err}")
                execute_body = self._fix_execute_body(
                    llm, execute_body, body_err
                )

            # Assemble + syntax check
            code = self._assemble_talent_file(requirements, execute_body)
            valid, error = self._validate_code(code)

            if not valid:
                print(f"   [TalentBuilder] Syntax error: {error}")
                continue

            # Self-test: write to temp, import, try can_handle + execute
            test_ok, test_error = self._self_test(
                code, requirements, llm, assistant
            )
            if test_ok:
                print(f"   [TalentBuilder] Self-test passed on attempt {attempt}")
                break
            else:
                print(f"   [TalentBuilder] Self-test failed: {test_error}")
                # On last attempt, keep the code anyway — user can fix in manager
                if attempt < max_attempts:
                    valid = True  # Code compiles, just test failed

        # ── Write to disk ───────────────────────────────────────────────────
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(code, encoding="utf-8")
        print(f"   [TalentBuilder] Written to {dest_path}")

        # ── Hot-load ────────────────────────────────────────────────────────
        if valid and assistant and hasattr(assistant, "load_user_talent"):
            load_result = assistant.load_user_talent(str(dest_path))
            if load_result.get("success"):
                resp = self._success_response(
                    load_result, requirements, dest_path
                )
                if test_error:
                    resp["response"] += (
                        f"\n\nNote: self-test had an issue: {test_error}\n"
                        "You can edit it in Tools > Talent Manager."
                    )
                return resp
            return {
                "success": True,
                "response": (
                    f"Wrote '{talent_name}' to {dest_path} but couldn't "
                    f"load it: {load_result.get('error', 'unknown')}. "
                    "Open Tools > Talent Manager to edit and reload."
                ),
                "actions_taken": [
                    {"action": "write_talent", "result": str(dest_path),
                     "success": True}
                ],
            }

        if not valid:
            return {
                "success": True,
                "response": (
                    f"Saved '{talent_name}' to {dest_path} but validation "
                    f"failed: {error}\n"
                    "Open Tools > Talent Manager to edit and fix it."
                ),
                "actions_taken": [
                    {"action": "write_talent", "result": str(dest_path),
                     "success": False}
                ],
            }

        return {
            "success": True,
            "response": (
                f"Created '{talent_name}' and saved to {dest_path}. "
                "Restart Talon to activate it, or use Tools > Talent Manager."
            ),
            "actions_taken": [
                {"action": "write_talent", "result": str(dest_path),
                 "success": True}
            ],
        }

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _safe_name(raw: str, description: str) -> str:
        """Return a valid snake_case talent name."""
        name = raw.strip().lower().replace(" ", "_").replace("-", "_")
        if name and re.match(r'^[a-z][a-z0-9_]*$', name):
            return name
        # Fallback: derive from first 20 chars of description
        fallback = re.sub(r'[^a-z0-9_]', '_', description.lower()[:20]).strip('_')
        return fallback or "custom_talent"

    @staticmethod
    def _class_name(raw: str, talent_name: str) -> str:
        """Return a valid PascalCase class name ending in Talent."""
        if raw and re.match(r'^[A-Z][A-Za-z0-9]+Talent$', raw):
            return raw
        return "".join(w.capitalize() for w in talent_name.split("_")) + "Talent"

    def _extract_requirements(self, llm, description: str) -> dict | None:
        """LLM call 1 — infer a structured JSON spec from the description."""
        user_prompt = (
            f'Given this description of a new Talon talent plugin:\n\n"{description}"\n\n'
            "Return ONLY valid JSON with this exact structure:\n"
            '{\n'
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
            '}'
        )
        try:
            raw = llm.generate(
                user_prompt,
                system_prompt=_REQUIREMENTS_SYSTEM_PROMPT,
                max_length=350,
                temperature=0.0,
            )
            clean = re.sub(r'^```[a-z]*\n?', '', raw.strip())
            clean = re.sub(r'\n?```$', '', clean.strip())
            m = re.search(r'\{.*\}', clean, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception as e:
            print(f"   [TalentBuilder] Requirements error: {e}")
        return None

    def _generate_execute_body(self, llm, requirements: dict) -> str:
        """LLM call 2 — generate ONLY the execute() body (~10-40 lines).

        The model writes pure logic: no class, no def, no return, no try/except.
        It must end with:  response = "..."
        """
        config_fields = requirements.get("config_fields", [])
        config_hint = ""
        if config_fields:
            keys = [f'self.talent_config.get({f["key"]!r}, "") → {f["label"]}'
                    for f in config_fields]
            config_hint = "\nConfig keys:\n  " + "\n  ".join(keys) + "\n"

        logic = requirements.get("execute_logic") or requirements.get("description", "")

        user_prompt = (
            f"Write the execute() body for a talent that does this:\n\n"
            f"{logic}\n"
            f"{config_hint}\n"
            "Remember:\n"
            "  - End with  response = \"...\"\n"
            "  - No imports, no def, no class, no return, no try/except"
        )
        try:
            raw = llm.generate(
                user_prompt,
                system_prompt=_EXEC_BODY_SYSTEM_PROMPT,
                max_length=800,
                temperature=0.1,
            )
            return self._extract_code_block(raw)
        except Exception as e:
            print(f"   [TalentBuilder] Body gen error: {e}")
            return ""

    @staticmethod
    def _extract_code_block(raw: str) -> str:
        """Pull Python code from model output, stripping markdown and prose."""
        text = raw.strip()
        # Prefer an explicit fenced code block
        m = re.search(r'```(?:python)?\n?(.*?)```', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Otherwise drop leading prose: find first code-like line
        lines = text.split("\n")
        for i, line in enumerate(lines):
            s = line.strip()
            if s and re.match(r'^[a-zA-Z_#]', s):
                return "\n".join(lines[i:]).strip()
        return text

    @staticmethod
    def _validate_execute_body(body: str) -> tuple[bool, str]:
        """Quick checks on the raw body before assembly."""
        if not body.strip():
            return False, "Empty execute body"
        if "response" not in body:
            return False, "Body does not set 'response' variable"
        for p in _FORBIDDEN_PATTERNS:
            if p in body:
                return False, f"Forbidden pattern: {p!r}"
        return True, ""

    @staticmethod
    def _assemble_talent_file(requirements: dict, execute_body: str) -> str:
        """Build the complete Python file from the fixed template + LLM body.

        Python is responsible for all boilerplate — only the execute logic is
        LLM-generated.  The result is always structurally correct.
        """
        name        = requirements["name"]
        class_name  = requirements["class_name"]
        description = requirements.get("description", "Custom generated talent")
        keywords    = requirements.get("keywords", [name.replace("_", " ")])
        examples    = requirements.get("examples", [])
        priority    = int(requirements.get("priority", 50))
        needs_cfg   = requirements.get("needs_config", False)
        cfg_fields  = requirements.get("config_fields", [])

        # Strip any import lines the model may have added to the body
        # (all needed imports are already in _TEMPLATE_IMPORTS)
        raw_lines   = execute_body.strip().split("\n")
        body_lines  = [l for l in raw_lines
                       if not re.match(r'^\s*(import |from \w+ import )', l)]
        clean_body  = "\n".join(body_lines)

        # Dedent model output, then re-indent to 12 spaces (class→method→try block)
        dedented    = textwrap.dedent(clean_body)
        body_block  = "\n".join(
            ("            " + line) if line.strip() else ""
            for line in dedented.split("\n")
        )

        # Build the parts list — avoids f-string escaping of literal { }
        parts = [_TEMPLATE_IMPORTS, "\n\n"]

        # Class header + attributes
        parts += [
            f"class {class_name}(BaseTalent):\n",
            f"    name = {name!r}\n",
            f"    description = {description!r}\n",
            f"    keywords = {keywords!r}\n",
            f"    examples = {examples!r}\n",
            f"    priority = {priority}\n",
        ]

        # Optional config schema (only when talent needs credentials / settings)
        if needs_cfg and cfg_fields:
            # repr() produces valid Python literals from the JSON-parsed dicts
            parts += [
                "\n",
                "    def get_config_schema(self) -> dict:\n",
                f"        return {{'fields': {cfg_fields!r}}}\n",
            ]

        # can_handle (always keyword_match)
        parts += [
            "\n",
            "    def can_handle(self, command: str) -> bool:\n",
            "        return self.keyword_match(command)\n",
        ]

        # execute() with LLM body wrapped in try/except + return
        # Note: the {e} below is plain string content — not an f-string expression
        parts += [
            "\n",
            "    def execute(self, command: str, context: dict) -> dict:\n",
            '        llm = context["llm"]\n',
            "        try:\n",
            body_block, "\n",
            "        except Exception as e:\n",
            '            return {"success": False, "response": f"Error: {e}", "actions_taken": []}\n',
            '        return {"success": True, "response": response, "actions_taken": []}\n',
        ]

        return "".join(parts)

    @staticmethod
    def _validate_code(code: str) -> tuple[bool, str]:
        """Syntax check (py_compile) + safety scan on the assembled file."""
        if not code.strip():
            return False, "Empty code"

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

        for p in _FORBIDDEN_PATTERNS:
            if p in code:
                return False, f"Forbidden pattern: {p!r}"

        if "BaseTalent" not in code:
            return False, "No BaseTalent subclass found"

        return True, ""

    def _fix_execute_body(self, llm, body: str, error: str) -> str:
        """One-shot fix attempt: send the body + error back to the model."""
        fix_prompt = (
            f"Fix this Python code. Error: {error}\n\n"
            "Rules: No imports, no def, no class, no return, no try/except.\n"
            "Final line must be:  response = \"...\"\n\n"
            f"Broken code:\n{body}\n\nFixed code:"
        )
        try:
            raw = llm.generate(fix_prompt, max_length=800, temperature=0.0)
            return self._extract_code_block(raw)
        except Exception as e:
            print(f"   [TalentBuilder] Fix error: {e}")
            return body

    def _self_test(self, code: str, requirements: dict,
                   llm, assistant) -> tuple[bool, str]:
        """Import the assembled code in isolation and run a basic smoke test.

        Returns (success, error_message).
        """
        import importlib
        import sys

        tmp = None
        module_name = f"_talent_test_{requirements['name']}"

        try:
            # Write to temp file
            with tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", delete=False, encoding="utf-8",
                dir=str(Path("talents/user")),
            ) as f:
                f.write(code)
                tmp = f.name

            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, tmp)
            if not spec or not spec.loader:
                return False, "Could not create module spec"
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Find the BaseTalent subclass
            talent_cls = None
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if (isinstance(obj, type) and issubclass(obj, BaseTalent)
                        and obj is not BaseTalent):
                    talent_cls = obj
                    break

            if not talent_cls:
                return False, "No BaseTalent subclass found in generated code"

            # Instantiate
            talent = talent_cls()
            talent.initialize({})

            # Test can_handle with first example
            examples = requirements.get("examples", [])
            if examples:
                test_cmd = examples[0]
                if not talent.can_handle(test_cmd):
                    return False, (
                        f"can_handle('{test_cmd}') returned False — "
                        "keywords may not match examples"
                    )

            # Test execute with a simple command (skip if no assistant/llm)
            if llm and examples:
                context = {
                    "llm": llm,
                    "config": assistant.config if assistant else {},
                    "assistant": assistant,
                    "speak_response": False,
                    "command_source": "test",
                }
                result = talent.execute(examples[0], context)
                if not result.get("success"):
                    resp = result.get("response", "unknown error")
                    return False, f"execute() failed: {resp}"

            return True, ""

        except Exception as e:
            return False, str(e)
        finally:
            # Cleanup
            if module_name in sys.modules:
                del sys.modules[module_name]
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except Exception:
                    pass

    @staticmethod
    def _success_response(load_result: dict, requirements: dict, dest_path: Path) -> dict:
        """Build the human-readable success confirmation."""
        name        = load_result["name"]
        examples    = load_result.get("examples") or requirements.get("examples", [])
        ex_str      = ", ".join(f'"{e}"' for e in examples[:3])
        needs_cfg   = load_result.get("needs_config") or requirements.get("needs_config", False)

        lines = [f"Created and loaded '{name}' talent!"]
        if ex_str:
            lines.append(f"Try: {ex_str}")
        if needs_cfg:
            lines.append(
                f"⚙️  Needs configuration — Settings → Talent Config → {name}"
            )

        return {
            "success": True,
            "response": "\n".join(lines),
            "actions_taken": [
                {"action": "create_talent", "result": str(dest_path), "success": True}
            ],
        }
