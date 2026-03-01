import json
import re
import time
import subprocess
import pyautogui
from talents.base import BaseTalent


class DesktopControlTalent(BaseTalent):
    name = "desktop_control"
    description = "Control desktop applications via keyboard and mouse, or describe what is on screen using vision"
    keywords = ["open", "click", "type", "press", "close", "start", "launch", "run",
                "change", "make", "set"]
    examples = [
        "open Chrome",
        "launch the calculator",
        "type hello world in notepad",
        "press ctrl+c",
        "what's on my screen right now",
        "what's in notepad",
        "read the text on screen",
        "describe what you see",
    ]
    priority = 40

    VISION_KEYWORDS = ["screen", "see", "show", "what", "find", "read", "window", "where"]

    # Phrases that indicate the user is ASKING about visual content,
    # not requesting a desktop action.  These trigger the vision-only path.
    _VISION_QUERY_PATTERNS = [
        r"\bwhat.{0,10}(?:on|in|at)\b.*\b(?:screen|display|monitor|desktop)\b",
        r"\bwhat.{0,10}(?:see|showing|visible|displayed|happening)\b",
        r"\bwhat.{0,20}(?:notepad|chrome|browser|window|editor|app|application)\b",
        r"\bdescribe\b.*\b(?:screen|window|display|what)\b",
        r"\bread\b.*\b(?:screen|text|window|page|display)\b",
        r"\bwhat.{0,6}(?:is|does|do)\b.*\b(?:say|read|show)\b",
        r"\blook\b.*\bat\b",
        r"\bcan you (?:see|read)\b",
        r"\btell me what\b",
        r"\bwhat.{0,6}(?:text|content|message)\b",
    ]

    APP_COMMANDS = {
        "calculator": "calc.exe",
        "notepad": "notepad.exe",
        "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "google chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "firefox": r"C:\Program Files\Mozilla Firefox\firefox.exe",
        "edge": "start msedge",
        "microsoft edge": "start msedge",
        "explorer": "explorer.exe",
        "paint": "mspaint.exe",
        "word": "winword.exe",
        "microsoft word": "winword.exe",
        "excel": "excel.exe",
        "microsoft excel": "excel.exe",
        "powerpoint": "powerpnt.exe",
        "microsoft powerpoint": "powerpnt.exe",
        "outlook": "outlook.exe"
    }

    def get_config_schema(self) -> dict:
        return {
            "fields": [
                {"key": "pyautogui_pause", "label": "Action Pause (s)",
                 "type": "float", "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1},
                {"key": "app_launch_delay", "label": "App Launch Delay (s)",
                 "type": "float", "default": 2.0, "min": 0.0, "max": 30.0, "step": 0.5},
            ]
        }

    def update_config(self, config: dict) -> None:
        """Apply updated automation settings."""
        super().update_config(config)
        if "pyautogui_pause" in config:
            pyautogui.PAUSE = config["pyautogui_pause"]
        if "app_launch_delay" in config:
            self.app_launch_delay = config["app_launch_delay"]

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    def initialize(self, config: dict) -> None:
        """Configure pyautogui from settings"""
        desktop_config = config.get("desktop", {})
        pyautogui.PAUSE = desktop_config.get("pyautogui_pause", 0.5)
        pyautogui.FAILSAFE = desktop_config.get("pyautogui_failsafe", True)
        self.action_delay = desktop_config.get("action_delay", 0.5)
        self.app_launch_delay = desktop_config.get("app_launch_delay", 2)

    def execute(self, command: str, context: dict) -> dict:
        llm = context["llm"]
        vision = context["vision"]
        memory_context = context.get("memory_context", "")
        speak_response = context.get("speak_response", True)
        voice = context.get("voice")

        # ── Branch: vision query vs. desktop action ──────────────
        if self._is_vision_query(command):
            return self._handle_vision_query(
                command, llm, vision, memory_context, speak_response, voice)
        else:
            return self._handle_desktop_action(
                command, llm, vision, memory_context, speak_response, voice)

    # ── Vision query path ────────────────────────────────────────

    def _handle_vision_query(self, command, llm, vision,
                             memory_context, speak_response, voice):
        """Use the screenshot + LLM to DESCRIBE what's on screen.

        No keyboard/mouse actions — purely visual analysis.
        """
        print("   [desktop_control] Vision query detected — capturing screenshot")
        screenshot_b64 = vision.capture_screenshot()

        if not screenshot_b64:
            return {
                "success": False,
                "response": "I couldn't capture a screenshot.",
                "actions_taken": [],
                "spoken": False,
            }

        prompt = (
            f"The user asked: \"{command}\"\n\n"
            "You are looking at a screenshot of their desktop. "
            "Answer the user's question by describing what you see. "
            "Be specific and helpful — mention application names, text content, "
            "UI elements, and anything relevant to their question. "
            "If you can read text in the screenshot, include it in your answer."
        )
        if memory_context:
            prompt = f"{memory_context}\n{prompt}"

        response = llm.generate(
            prompt,
            use_vision=True,
            screenshot_b64=screenshot_b64,
        )

        description = (response or "").strip()
        if not description:
            description = "I captured a screenshot but couldn't describe it."

        # Speak the description
        if speak_response and voice:
            voice.speak(description)
            spoken = True
        else:
            print(f"\n{description}")
            spoken = False

        return {
            "success": True,
            "response": description,
            "actions_taken": [{"action": "vision_query", "result": "screenshot analysed"}],
            "spoken": spoken,
        }

    # ── Desktop action path ──────────────────────────────────────

    def _handle_desktop_action(self, command, llm, vision,
                               memory_context, speak_response, voice):
        """Generate and execute keyboard/mouse actions via LLM JSON."""

        # Some action commands still benefit from a screenshot for context
        # (e.g. "click the save button") — supply one if vision keywords present
        needs_vision = any(kw in command for kw in self.VISION_KEYWORDS)
        screenshot_b64 = None
        if needs_vision:
            screenshot_b64 = vision.capture_screenshot()

        # Build action-request prompt
        prompt = f"User wants to: {command}\n\nGenerate the appropriate desktop actions."
        if memory_context:
            prompt = f"{memory_context}{prompt}"
        prompt = self._append_action_schema(prompt)

        response = llm.generate(
            prompt, use_vision=needs_vision, screenshot_b64=screenshot_b64)

        # Parse JSON
        action_plan = self._parse_action_json(response)
        if action_plan is None:
            return {
                "success": False,
                "response": "I'm not sure how to do that.",
                "actions_taken": [],
                "spoken": False
            }

        explanation = action_plan.get("explanation", "Executing...")

        # Speak explanation before executing
        if speak_response and voice:
            voice.speak(explanation)
        else:
            print(f"\n{explanation}")

        # Execute actions
        actions = action_plan.get("actions", [])
        results = []
        all_successful = True
        clipboard_text = None

        for action in actions:
            result = self._execute_single_action(action)
            success = not result.startswith("Error")
            results.append({"action": action, "result": result, "success": success})
            if not success:
                all_successful = False
            # Capture clipboard content from read_clipboard action
            if action.get("action") == "read_clipboard" and success:
                clipboard_text = result.replace("Clipboard: ", "", 1)
            print(f"  -> {result}")
            time.sleep(self.action_delay)

        # Build response — include clipboard content if we read it
        if clipboard_text:
            response_text = f"Here's what I found:\n\n{clipboard_text}"
        else:
            response_text = "Done!" if all_successful else "Some actions failed."

        return {
            "success": all_successful,
            "response": response_text,
            "actions_taken": results,
            "spoken": True if not clipboard_text else False,
        }

    def _append_action_schema(self, prompt):
        """Append the action JSON schema instructions"""
        prompt += """\n\nYou can control the desktop. Respond with a JSON object containing:
1. "explanation": Brief explanation of what you'll do (keep it SHORT, 1 sentence max)
2. "actions": Array of action objects to execute

Available actions:
- {"action": "open_application", "application": "calculator/notepad/chrome/etc"}
- {"action": "open_url", "url": "<full URL to open in default browser>"}
- {"action": "click", "x": <number>, "y": <number>}
- {"action": "type", "text": "<text to type>"}
- {"action": "press_key", "key": "<key name like 'enter', 'tab', etc>"}
- {"action": "hotkey", "keys": ["ctrl", "c"]}
- {"action": "read_clipboard"} — read and return the current clipboard text to the user
- {"action": "screenshot"} — silently capture the full screen (no interactive dialog); returns the saved file path

Example — reading text from an application:
{
  "explanation": "Reading the text from Notepad",
  "actions": [
    {"action": "hotkey", "keys": ["ctrl", "a"]},
    {"action": "hotkey", "keys": ["ctrl", "c"]},
    {"action": "read_clipboard"}
  ]
}

Example — opening a website:
{
  "explanation": "Opening CNN in Chrome",
  "actions": [
    {"action": "open_url", "url": "https://www.cnn.com"}
  ]
}

Example — calculator:
{
  "explanation": "Opening calculator and calculating 2 plus 7",
  "actions": [
    {"action": "open_application", "application": "calculator"},
    {"action": "type", "text": "2"},
    {"action": "press_key", "key": "plus"},
    {"action": "type", "text": "7"},
    {"action": "press_key", "key": "enter"}
  ]
}

Example — typing text in notepad:
{
  "explanation": "Opening notepad and writing a message",
  "actions": [
    {"action": "open_application", "application": "notepad"},
    {"action": "type", "text": "Hello World! This is a test."}
  ]
}

Respond ONLY with valid JSON, no additional text."""
        return prompt

    def _parse_action_json(self, response):
        """Clean and parse JSON from LLM response"""
        try:
            response_clean = response.strip()

            if "```" in response_clean:
                parts = response_clean.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{") and part.endswith("}"):
                        response_clean = part
                        break

            if not response_clean.startswith("{"):
                json_match = re.search(r'\{.*\}', response_clean, re.DOTALL)
                if json_match:
                    response_clean = json_match.group(0)

            return json.loads(response_clean)
        except (json.JSONDecodeError, AttributeError):
            return None

    def _execute_single_action(self, action_data):
        """Execute a single desktop action"""
        try:
            action_type = action_data.get("action")

            if action_type == "open_application":
                app_name = action_data.get("application")
                print(f"   -> Opening {app_name}...")

                cmd = self.APP_COMMANDS.get(app_name.lower(), app_name)
                subprocess.Popen(cmd, shell=True)
                time.sleep(self.app_launch_delay)
                return f"Opened {app_name}"

            elif action_type == "open_url":
                url = action_data.get("url", "")
                if not url.startswith(("http://", "https://")):
                    url = f"https://{url}"
                print(f"   -> Opening URL: {url}")
                import webbrowser
                webbrowser.open(url)
                time.sleep(self.app_launch_delay)
                return f"Opened {url}"

            elif action_type == "click":
                x = action_data.get("x")
                y = action_data.get("y")
                pyautogui.click(x, y)
                return f"Clicked at ({x}, {y})"

            elif action_type == "type":
                text = action_data.get("text")
                target = action_data.get("target", "")

                # Calculator mode: char-by-char with operator key mapping
                if target == "calculator" or self._is_calculator_text(text):
                    for char in text:
                        if char == '+':
                            pyautogui.press('add')
                        elif char == '-':
                            pyautogui.press('subtract')
                        elif char == '*':
                            pyautogui.press('multiply')
                        elif char == '/':
                            pyautogui.press('divide')
                        elif char == '=':
                            pyautogui.press('equals')
                        elif char == ' ':
                            pass  # skip spaces for calculator input
                        else:
                            pyautogui.write(char, interval=0.05)
                        time.sleep(0.1)
                else:
                    # Normal text mode: paste via clipboard for reliability
                    # pyautogui.write() drops spaces and non-ASCII chars
                    import pyperclip
                    old_clipboard = None
                    try:
                        old_clipboard = pyperclip.paste()
                    except Exception:
                        pass
                    pyperclip.copy(text)
                    pyautogui.hotkey('ctrl', 'v')
                    time.sleep(0.2)
                    # Restore old clipboard
                    if old_clipboard is not None:
                        try:
                            pyperclip.copy(old_clipboard)
                        except Exception:
                            pass

                return f"Typed: {text}"

            elif action_type == "press_key":
                key = action_data.get("key")
                key_map = {
                    'plus': 'add',
                    'minus': 'subtract',
                    'multiply': 'multiply',
                    'divide': 'divide',
                    'equals': 'equals'
                }
                actual_key = key_map.get(key.lower(), key)
                pyautogui.press(actual_key)
                return f"Pressed: {key}"

            elif action_type == "hotkey":
                keys = action_data.get("keys", [])
                # Win+L cannot be sent reliably via pyautogui (Windows blocks
                # simulated Windows-key presses for security hotkeys).
                # Use the LockWorkStation() API directly instead.
                if {k.lower() for k in keys} == {"winleft", "l"}:
                    import ctypes
                    ctypes.windll.user32.LockWorkStation()
                    return "Locked workstation"
                pyautogui.hotkey(*keys)
                return f"Hotkey: {'+'.join(keys)}"

            elif action_type == "read_clipboard":
                import pyperclip
                time.sleep(0.3)  # brief wait for clipboard to populate
                text = pyperclip.paste()
                if text:
                    return f"Clipboard: {text}"
                return "Clipboard: (empty)"

            elif action_type == "screenshot":
                # Silent full-screen capture — no interactive dialog.
                # PIL.ImageGrab.grab() works on Windows without any UI.
                from PIL import ImageGrab
                import os, tempfile
                img = ImageGrab.grab()
                path = os.path.join(tempfile.gettempdir(), "talon_screenshot.png")
                img.save(path, "PNG")
                return f"Screenshot: {path}"

            else:
                return f"Unknown action: {action_type}"

        except Exception as e:
            return f"Error: {str(e)}"

    def _is_vision_query(self, command: str) -> bool:
        """Decide whether *command* is a question about the screen (vision path)
        or an action request (keyboard/mouse path).

        Returns True  → use vision to describe what's on screen
        Returns False → generate desktop actions (open, type, click, etc.)
        """
        cmd = command.lower().strip()

        # ── Definite action verbs → always action path ───────────
        _ACTION_VERBS = (
            "open", "launch", "start", "run", "close", "quit", "exit",
            "type", "write", "press", "click", "drag", "scroll",
            "copy", "paste", "cut", "undo", "redo", "save",
            "switch", "minimize", "maximize", "resize", "move",
        )
        # "read" is intentionally NOT here — "read the text on screen"
        # is a vision query, not a desktop action.
        first_word = cmd.split()[0] if cmd else ""
        if first_word in _ACTION_VERBS:
            return False

        # ── Regex patterns for vision queries ────────────────────
        for pattern in self._VISION_QUERY_PATTERNS:
            if re.search(pattern, cmd):
                return True

        # ── Fallback heuristic: question words + vision keywords ─
        # If the command starts with a question word and contains a
        # vision keyword, treat it as a vision query.
        _QUESTION_STARTERS = ("what", "where", "how", "can you see",
                              "tell me", "describe", "is there")
        has_question = any(cmd.startswith(q) for q in _QUESTION_STARTERS)
        has_vision_kw = any(kw in cmd for kw in self.VISION_KEYWORDS)
        if has_question and has_vision_kw:
            return True

        return False

    @staticmethod
    def _is_calculator_text(text):
        """Detect if text looks like calculator input (digits + operators only)."""
        return bool(text) and all(c in '0123456789+-*/=. ' for c in text)
