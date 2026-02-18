import json
import re
import time
import subprocess
import pyautogui
from talents.base import BaseTalent


class DesktopControlTalent(BaseTalent):
    name = "desktop_control"
    description = "Control desktop applications via keyboard and mouse"
    keywords = ["open", "click", "type", "press", "close", "start", "launch", "run",
                "change", "make", "set"]
    examples = [
        "open Chrome",
        "launch the calculator",
        "type hello world in notepad",
        "press ctrl+c",
    ]
    priority = 40

    VISION_KEYWORDS = ["screen", "see", "show", "what", "find", "read", "window", "where"]

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

        # Determine if vision is needed
        needs_vision = any(kw in command for kw in self.VISION_KEYWORDS)

        screenshot_b64 = None
        if needs_vision:
            screenshot_b64 = vision.capture_screenshot()

        # Build action-request prompt
        prompt = f"User wants to: {command}\n\nGenerate the appropriate desktop actions."
        if memory_context:
            prompt = f"{memory_context}{prompt}"
        prompt = self._append_action_schema(prompt)

        response = llm.generate(prompt, use_vision=needs_vision, screenshot_b64=screenshot_b64)

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

        for action in actions:
            result = self._execute_single_action(action)
            success = not result.startswith("Error")
            results.append({"action": action, "result": result, "success": success})
            if not success:
                all_successful = False
            print(f"  -> {result}")
            time.sleep(self.action_delay)

        return {
            "success": all_successful,
            "response": "Done!" if all_successful else "Some actions failed.",
            "actions_taken": results,
            "spoken": True
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
                keys = action_data.get("keys")
                pyautogui.hotkey(*keys)
                return f"Hotkey: {'+'.join(keys)}"

            else:
                return f"Unknown action: {action_type}"

        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def _is_calculator_text(text):
        """Detect if text looks like calculator input (digits + operators only)."""
        return bool(text) and all(c in '0123456789+-*/=. ' for c in text)
