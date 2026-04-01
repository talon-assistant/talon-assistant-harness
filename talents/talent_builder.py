"""TalentBuilder — generate talent plugins from plain-English descriptions.

Conversational flow:
1. User: "create a talent that checks Bitcoin prices"
2. LLM generates complete .py file
3. Talon shows the code and asks: "install it, or describe changes"
4. User: "install it" → writes + hot-loads
   OR "change it to use USD" → LLM refines → show again
"""

import json
import os
import py_compile
import re
import tempfile
from pathlib import Path

from talents.base import BaseTalent
from core.llm_client import LLMError

import logging
log = logging.getLogger(__name__)


# Dangerous patterns that are never allowed in generated code
_FORBIDDEN_PATTERNS = [
    "os.system(",
    "os.popen(",
    "os.execv(", "os.execl(", "os.execle(",
    "__import__(",
    "eval(",
    "exec(",
    "subprocess.call(", "subprocess.run(",  # but subprocess.Popen is OK if no shell=True
    "shutil.rmtree(",
    "open('/etc", "open('C:\\\\Windows",
]

# System prompt with BaseTalent API + 3 working examples
_GENERATION_SYSTEM_PROMPT = """\
You are a Python code generator for Talon, a desktop voice assistant. Generate a complete, working talent plugin file.

## BaseTalent API

Every talent must:
- Subclass `BaseTalent` (from `talents.base import BaseTalent`)
- Set class attributes: `name` (snake_case), `description` (one line), `keywords` (list), `examples` (3+ example commands), `priority` (int, 50 is default)
- Implement `execute(self, command: str, context: dict) -> dict`
- Return dict: `{"success": True/False, "response": "text for user", "actions_taken": [...], "spoken": False}`

Available in execute():
- `context["llm"]` — LLM client. Call: `llm.generate(prompt, max_length=200, temperature=0.7)` returns str
- `self._extract_arg(llm, command, "what to extract", max_length=30)` — extract a value from the command via LLM
- `self.talent_config` — dict of per-talent config (API keys, settings). Read with `.get("key", "")`
- `self._config` — same as talent_config

For config, define `get_config_schema()` returning `{"fields": [{"key": "api_key", "label": "API Key", "type": "password", "default": ""}]}`

For heavy C-extension libraries (pandas, numpy, yfinance), set `subprocess_isolated = True`.

For pip packages beyond stdlib, set `required_packages = ["package_name"]`.

## Rules
1. Write a COMPLETE Python file with imports at top, class definition, all methods
2. Use try/except for network calls and error-prone operations
3. Return the standard result dict from execute()
4. If config (API keys) is needed, check for empty values and return a helpful config reminder
5. Keep the code clean, concise, and well-structured
6. Do NOT use: os.system(), eval(), exec(), __import__()

## Example 1: Simple API talent with config

```python
import requests
from talents.base import BaseTalent


class BitcoinPriceTalent(BaseTalent):
    name = "bitcoin_price"
    description = "Check current Bitcoin and cryptocurrency prices"
    keywords = ["bitcoin", "btc", "crypto", "cryptocurrency", "coin price"]
    examples = [
        "what's the Bitcoin price",
        "check BTC price",
        "how much is Ethereum worth",
    ]
    priority = 50

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    def execute(self, command: str, context: dict) -> dict:
        llm = context.get("llm")
        coin = self._extract_arg(llm, command, "cryptocurrency name", max_length=20) or "bitcoin"
        coin = coin.lower().strip()

        try:
            resp = requests.get(
                f"https://api.coingecko.com/api/v3/simple/price",
                params={"ids": coin, "vs_currencies": "usd", "include_24hr_change": "true"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            if coin not in data:
                return {"success": False, "response": f"Couldn't find '{coin}'. Try 'bitcoin', 'ethereum', etc.",
                        "actions_taken": [], "spoken": False}

            price = data[coin]["usd"]
            change = data[coin].get("usd_24h_change", 0)
            arrow = "\\u2191" if change >= 0 else "\\u2193"
            return {
                "success": True,
                "response": f"{coin.title()}: ${price:,.2f} ({arrow} {change:+.1f}% 24h)",
                "actions_taken": [{"action": "crypto_price", "coin": coin}],
                "spoken": False,
            }
        except Exception as e:
            return {"success": False, "response": f"Error fetching price: {e}",
                    "actions_taken": [], "spoken": False}
```

## Example 2: LLM-powered text tool

```python
import pyperclip
from talents.base import BaseTalent


class SummarizeClipboardTalent(BaseTalent):
    name = "summarize_clipboard"
    description = "Summarize or transform text from the clipboard using the LLM"
    keywords = ["summarize clipboard", "summarize text", "clipboard summary", "rewrite clipboard"]
    examples = [
        "summarize my clipboard",
        "rewrite the clipboard text to be more formal",
        "translate clipboard to Spanish",
    ]
    priority = 45

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    def execute(self, command: str, context: dict) -> dict:
        llm = context.get("llm")
        if not llm:
            return {"success": False, "response": "LLM not available.", "actions_taken": [], "spoken": False}

        try:
            text = pyperclip.paste()
        except Exception:
            text = ""

        if not text or len(text.strip()) < 10:
            return {"success": False, "response": "Clipboard is empty or too short to summarize.",
                    "actions_taken": [], "spoken": False}

        # Truncate if very long
        if len(text) > 3000:
            text = text[:3000] + "..."

        task = self._extract_arg(llm, command, "what to do with the text", max_length=50) or "summarize"

        prompt = f"Task: {task}\\n\\nText:\\n{text}\\n\\nProvide the result:"
        try:
            result = llm.generate(prompt, max_length=500, temperature=0.5)
            return {
                "success": True,
                "response": result,
                "actions_taken": [{"action": "clipboard_transform", "task": task}],
                "spoken": False,
            }
        except Exception as e:
            return {"success": False, "response": f"Error: {e}", "actions_taken": [], "spoken": False}
```

## Example 3: Local automation

```python
import os
import glob
from talents.base import BaseTalent


class RecentFilesTotalent(BaseTalent):
    name = "recent_files"
    description = "List recently modified files in a directory"
    keywords = ["recent files", "latest files", "new files", "modified files"]
    examples = [
        "show recent files on my desktop",
        "what files were modified today in Downloads",
        "list latest files in Documents",
    ]
    priority = 45

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    def execute(self, command: str, context: dict) -> dict:
        llm = context.get("llm")
        folder = self._extract_arg(llm, command, "folder name like Desktop, Downloads, Documents", max_length=30)

        # Map common names to paths
        home = os.path.expanduser("~")
        folder_map = {
            "desktop": os.path.join(home, "Desktop"),
            "downloads": os.path.join(home, "Downloads"),
            "documents": os.path.join(home, "Documents"),
        }
        path = folder_map.get((folder or "desktop").lower(), os.path.join(home, folder or "Desktop"))

        if not os.path.isdir(path):
            return {"success": False, "response": f"Folder not found: {path}",
                    "actions_taken": [], "spoken": False}

        try:
            files = []
            for f in os.listdir(path):
                full = os.path.join(path, f)
                if os.path.isfile(full):
                    mtime = os.path.getmtime(full)
                    files.append((f, mtime))

            files.sort(key=lambda x: x[1], reverse=True)
            top = files[:10]

            if not top:
                return {"success": True, "response": f"No files found in {path}",
                        "actions_taken": [], "spoken": False}

            import datetime
            lines = [f"Recent files in {os.path.basename(path)}:\\n"]
            for name, mtime in top:
                dt = datetime.datetime.fromtimestamp(mtime)
                lines.append(f"  {name}  ({dt.strftime('%b %d %H:%M')})")

            return {
                "success": True,
                "response": "\\n".join(lines),
                "actions_taken": [{"action": "list_files", "path": path}],
                "spoken": False,
            }
        except Exception as e:
            return {"success": False, "response": f"Error: {e}", "actions_taken": [], "spoken": False}
```

Return ONLY the Python code for the talent file. No explanation, no markdown fences around the whole thing.
"""


class TalentBuilderTalent(BaseTalent):
    """Generate, validate, and hot-load new talent plugins from plain-English descriptions.

    Supports a conversational flow:
    - "create a talent that..." → generates code, shows for review
    - "install it" / "looks good" → writes to disk and hot-loads
    - "change X" / "fix Y" → refines and shows again
    """

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
    ]
    priority = 82

    # Pending code state for review/install flow
    _pending_code: str | None = None
    _pending_name: str | None = None
    _pending_description: str | None = None

    # Phrases that mean "install the pending code"
    _INSTALL_PHRASES = [
        "install it", "install that", "save it", "save that",
        "looks good", "looks great", "perfect", "do it",
        "load it", "activate it", "yes", "yep", "yeah",
    ]

    # Phrases that mean "refine the pending code"
    _REFINE_INDICATORS = [
        "change", "fix", "update", "modify", "add", "remove",
        "instead", "actually", "but", "also", "don't", "wrong",
        "should", "shouldn't", "needs to", "make it",
    ]

    def can_handle(self, command: str) -> bool:
        # If we have pending code, intercept install/refine commands
        if self._pending_code:
            cmd = command.lower().strip()
            if any(p in cmd for p in self._INSTALL_PHRASES):
                return True
            if any(p in cmd for p in self._REFINE_INDICATORS):
                return True
        return self.keyword_match(command)

    def execute(self, command: str, context: dict) -> dict:
        llm = context.get("llm")
        assistant = context.get("assistant")
        cmd_lower = command.lower().strip()

        if not llm:
            return self._fail("LLM not available.")

        # ── Handle pending code: install or refine ──
        if self._pending_code:
            if any(p in cmd_lower for p in self._INSTALL_PHRASES):
                return self._install_pending(assistant)
            if any(p in cmd_lower for p in self._REFINE_INDICATORS):
                return self._refine_pending(llm, command)
            # If neither, clear pending and treat as new request
            self._clear_pending()

        # ── New talent request ──
        description = self._extract_arg(
            llm, command,
            "description of what the new talent or plugin should do",
            max_length=120, temperature=0.0,
        )
        if not description:
            return self._fail("What should the new talent do? Describe it and I'll build it.")

        log.info(f"[TalentBuilder] Generating: {description}")

        # Generate complete talent code
        code = self._generate_talent_code(llm, description)
        if not code:
            return self._fail("Couldn't generate the talent code. Try describing it differently.")

        # Validate
        valid, error = self._validate_code(code)
        if not valid:
            # Try one fix attempt
            log.warning(f"[TalentBuilder] Validation failed: {error}, attempting fix")
            code = self._fix_code(llm, code, error)
            valid, error = self._validate_code(code)

        if not valid:
            return self._fail(f"Generated code has issues: {error}\nTry describing the talent differently.")

        # Extract talent name from code
        name = self._extract_name_from_code(code) or self._safe_name("", description)

        # Store as pending and show for review
        self._pending_code = code
        self._pending_name = name
        self._pending_description = description

        return {
            "success": True,
            "response": (
                f"Here's the **{name}** talent I generated:\n\n"
                f"```python\n{code}\n```\n\n"
                "Say **install it** to save and activate, or describe what to change."
            ),
            "actions_taken": [{"action": "talent_draft", "name": name}],
            "spoken": False,
        }

    # ── Install pending code ──

    def _install_pending(self, assistant) -> dict:
        code = self._pending_code
        name = self._pending_name or "custom_talent"

        # Resolve destination
        dest_path = Path("talents/user") / f"{name}.py"
        if dest_path.exists():
            name += "_2"
            dest_path = Path("talents/user") / f"{name}.py"
            # Update the class name in code too
            code = self._update_name_in_code(code, name)

        # Write
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(code, encoding="utf-8")
        log.info(f"[TalentBuilder] Written to {dest_path}")

        # Hot-load
        loaded = False
        if assistant and hasattr(assistant, "load_user_talent"):
            result = assistant.load_user_talent(str(dest_path))
            loaded = result.get("success", False)
            if not loaded:
                err = result.get("error", "unknown")
                self._clear_pending()
                return {
                    "success": True,
                    "response": (
                        f"Saved '{name}' to {dest_path} but couldn't load it: {err}\n"
                        "Open Tools > Talent Manager to edit and reload."
                    ),
                    "actions_taken": [{"action": "create_talent", "result": str(dest_path)}],
                    "spoken": False,
                }

        self._clear_pending()

        examples = self._extract_examples_from_code(code)
        ex_str = ", ".join(f'"{e}"' for e in examples[:3])

        lines = [f"Installed and loaded '{name}' talent!"]
        if ex_str:
            lines.append(f"Try: {ex_str}")
        if "get_config_schema" in code:
            lines.append(f"This talent needs configuration — go to Settings > Talent Config > {name}")

        return {
            "success": True,
            "response": "\n".join(lines),
            "actions_taken": [{"action": "create_talent", "result": str(dest_path), "success": True}],
            "spoken": False,
        }

    # ── Refine pending code ──

    def _refine_pending(self, llm, feedback: str) -> dict:
        log.info(f"[TalentBuilder] Refining: {feedback}")

        prompt = (
            f"Here is a Talon talent plugin:\n\n```python\n{self._pending_code}\n```\n\n"
            f"The user wants this change: {feedback}\n\n"
            "Generate the complete updated file. Return ONLY Python code."
        )

        try:
            raw = llm.generate(
                prompt,
                system_prompt=_GENERATION_SYSTEM_PROMPT,
                max_length=2048,
                temperature=0.1,
            )
            code = self._extract_code_block(raw)
        except LLMError as e:
            return self._fail(f"LLM unavailable: {e}")

        if not code:
            return self._fail("Couldn't generate the updated code. Try describing the change differently.")

        valid, error = self._validate_code(code)
        if not valid:
            code = self._fix_code(llm, code, error)
            valid, error = self._validate_code(code)

        if not valid:
            return self._fail(f"Updated code has issues: {error}")

        name = self._extract_name_from_code(code) or self._pending_name
        self._pending_code = code
        self._pending_name = name

        return {
            "success": True,
            "response": (
                f"Updated **{name}**:\n\n"
                f"```python\n{code}\n```\n\n"
                "Say **install it** to save, or describe more changes."
            ),
            "actions_taken": [{"action": "talent_refined", "name": name}],
            "spoken": False,
        }

    # ── Generation ──

    def _generate_talent_code(self, llm, description: str) -> str | None:
        prompt = (
            f"Create a Talon talent plugin that does this:\n\n{description}\n\n"
            "Generate the complete Python file. Return ONLY the code, no markdown fences."
        )
        try:
            raw = llm.generate(
                prompt,
                system_prompt=_GENERATION_SYSTEM_PROMPT,
                max_length=2048,
                temperature=0.1,
            )
            return self._extract_code_block(raw)
        except LLMError as e:
            log.warning(f"[TalentBuilder] LLM unavailable: {e}")
            return None
        except Exception as e:
            log.error(f"[TalentBuilder] Generation error: {e}")
            return None

    def _fix_code(self, llm, code: str, error: str) -> str:
        prompt = (
            f"This Talon talent plugin has an error:\n\n"
            f"Error: {error}\n\n"
            f"```python\n{code}\n```\n\n"
            "Fix the error and return the complete corrected file. Return ONLY Python code."
        )
        try:
            raw = llm.generate(prompt, max_length=2048, temperature=0.0)
            fixed = self._extract_code_block(raw)
            return fixed if fixed else code
        except Exception:
            return code

    # ── Validation ──

    @staticmethod
    def _validate_code(code: str) -> tuple[bool, str]:
        if not code or not code.strip():
            return False, "Empty code"

        # Must have a BaseTalent subclass
        if "BaseTalent" not in code:
            return False, "No BaseTalent subclass found"
        if "def execute(" not in code:
            return False, "No execute() method found"

        # Forbidden patterns
        for p in _FORBIDDEN_PATTERNS:
            if p in code:
                return False, f"Forbidden pattern: {p!r}"

        # Check for shell=True in subprocess calls (but allow subprocess without shell)
        if "shell=True" in code and "subprocess" in code:
            return False, "subprocess with shell=True is not allowed"

        # Syntax check via py_compile
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                tmp = f.name
            py_compile.compile(tmp, doraise=True)
        except py_compile.PyCompileError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, str(e)
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

        return True, ""

    # ── Helpers ──

    @staticmethod
    def _extract_code_block(raw: str) -> str:
        text = raw.strip()
        # Prefer explicit fenced block
        m = re.search(r'```(?:python)?\n?(.*?)```', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # If it starts with import or from or a comment, treat as raw code
        if text.startswith(("import ", "from ", "#", '"""', "'")):
            return text
        # Find first code-like line
        lines = text.split("\n")
        for i, line in enumerate(lines):
            s = line.strip()
            if s and re.match(r'^(import |from |#|class |def |""")', s):
                return "\n".join(lines[i:]).strip()
        return text

    @staticmethod
    def _extract_name_from_code(code: str) -> str | None:
        m = re.search(r'name\s*=\s*["\']([a-z_][a-z0-9_]*)["\']', code)
        return m.group(1) if m else None

    @staticmethod
    def _extract_examples_from_code(code: str) -> list[str]:
        m = re.search(r'examples\s*=\s*\[(.*?)\]', code, re.DOTALL)
        if m:
            return re.findall(r'["\']([^"\']+)["\']', m.group(1))
        return []

    @staticmethod
    def _update_name_in_code(code: str, new_name: str) -> str:
        code = re.sub(r'(name\s*=\s*["\'])[a-z_][a-z0-9_]*(["\'])',
                       rf'\g<1>{new_name}\g<2>', code, count=1)
        return code

    @staticmethod
    def _safe_name(raw: str, description: str) -> str:
        name = raw.strip().lower().replace(" ", "_").replace("-", "_")
        if name and re.match(r'^[a-z][a-z0-9_]*$', name):
            return name
        fallback = re.sub(r'[^a-z0-9_]', '_', description.lower()[:20]).strip('_')
        return fallback or "custom_talent"

    def _clear_pending(self):
        self._pending_code = None
        self._pending_name = None
        self._pending_description = None

    @staticmethod
    def _fail(msg: str) -> dict:
        return {"success": False, "response": msg, "actions_taken": [], "spoken": False}
