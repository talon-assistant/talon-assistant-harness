import json
import re
import os
from talents.base import BaseTalent
from phue import Bridge


class HueLightsTalent(BaseTalent):
    name = "hue_lights"
    description = "Control Philips Hue smart lights"
    keywords = ["turn", "lights", "brighten", "dim", "light", "brightness", "color"]
    examples = [
        "turn on the lights",
        "dim the lights to 50 percent",
        "change the lights to blue",
        "turn off bedroom lights",
    ]
    priority = 70

    def __init__(self):
        super().__init__()
        self.bridge = None
        self.light_groups = {}
        self.color_map = {}
        self.hue_enabled = False

    def initialize(self, config: dict) -> None:
        """Connect to Hue bridge using hue_config.json"""
        try:
            hue_config = self._load_hue_config()
            self.bridge = Bridge(hue_config["bridge_ip"])
            self.bridge.connect()

            all_lights = self.bridge.get_light_objects('name')

            for group_name, light_names in hue_config.get("light_groups", {}).items():
                self.light_groups[group_name] = [
                    all_lights[name] for name in light_names if name in all_lights
                ]

            self.color_map = hue_config.get("colors", {})
            total_lights = sum(len(lights) for lights in self.light_groups.values())
            print(f"   Loaded {total_lights} lights")
            self.hue_enabled = True

            # Populate managed_lights in _config so the GUI shows them
            managed = []
            for light_names in hue_config.get("light_groups", {}).values():
                for ln in light_names:
                    if ln not in managed:
                        managed.append(ln)
            self._config["managed_lights"] = managed

        except Exception as e:
            print(f"   Hue unavailable: {e}")
            self.hue_enabled = False

    def _load_hue_config(self):
        """Load hue_config.json"""
        config_path = os.path.join("config", "hue_config.json")
        with open(config_path, 'r') as f:
            return json.load(f)

    def _save_hue_config(self, hue_config):
        """Save hue_config.json"""
        config_path = os.path.join("config", "hue_config.json")
        with open(config_path, 'w') as f:
            json.dump(hue_config, f, indent=2)

    @property
    def routing_available(self) -> bool:
        return self.hue_enabled

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    def execute(self, command: str, context: dict) -> dict:
        llm = context["llm"]
        memory_context = context.get("memory_context", "")
        voice = context.get("voice")
        speak_response = context.get("speak_response", True)

        # Ask LLM to generate hue action JSON
        prompt = self._build_hue_prompt(command, memory_context)
        response = llm.generate(prompt)

        # Parse action JSON
        action_plan = self._parse_action_json(response)
        if action_plan is None:
            return {
                "success": False,
                "response": "Could not understand light command.",
                "actions_taken": [],
                "spoken": False
            }

        explanation = action_plan.get("explanation", "Adjusting lights...")
        if speak_response and voice:
            voice.speak(explanation)

        actions = action_plan.get("actions", [])
        results = []
        all_successful = True

        for action in actions:
            result = self._control_hue(action)
            success = not result.startswith("Error") and not result.startswith("Unknown")
            results.append({"action": action, "result": result, "success": success})
            if not success:
                all_successful = False

        return {
            "success": all_successful,
            "response": "Done!" if all_successful else "Some light commands failed.",
            "actions_taken": results,
            "spoken": True
        }

    def _build_hue_prompt(self, command, memory_context=""):
        """Build prompt for LLM to generate hue action JSON"""
        available_colors = ", ".join(self.color_map.keys()) if self.color_map else "red, green, blue, yellow, purple, pink, orange, cyan, magenta, white"

        prompt = f"User wants to: {command}\n\nGenerate the appropriate light controls."
        if memory_context:
            prompt = f"{memory_context}{prompt}"

        prompt += f"""\n\nYou can control smart lights. Respond with a JSON object containing:
1. "explanation": Brief explanation of what you'll do (keep it SHORT, 1 sentence max)
2. "actions": Array of action objects to execute

Available actions:
- {{"action": "hue_light", "hue_action": "on"}}
- {{"action": "hue_light", "hue_action": "off"}}
- {{"action": "hue_light", "hue_action": "brighten"}}
- {{"action": "hue_light", "hue_action": "dim"}}
- {{"action": "hue_light", "hue_action": "brightness", "level": 0-254}}
- {{"action": "hue_light", "hue_action": "color", "color": "{available_colors}"}}

Example response:
{{
  "explanation": "Turning on the lights and setting them to blue",
  "actions": [
    {{"action": "hue_light", "hue_action": "on"}},
    {{"action": "hue_light", "hue_action": "color", "color": "blue"}}
  ]
}}

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

    def _control_hue(self, action_data):
        """Control Philips Hue lights"""
        if not self.hue_enabled:
            return "Hue lights not available"

        try:
            hue_action = action_data.get("hue_action")

            # Get all lights from all groups
            all_lights = []
            for lights in self.light_groups.values():
                all_lights.extend(lights)

            if not all_lights:
                return "No lights found"

            if hue_action == "on":
                for light in all_lights:
                    light.on = True
                return "Turned on lights"

            elif hue_action == "off":
                for light in all_lights:
                    light.on = False
                return "Turned off lights"

            elif hue_action == "brighten":
                for light in all_lights:
                    current = light.brightness
                    light.brightness = min(254, current + 50)
                return "Brightened lights"

            elif hue_action == "dim":
                for light in all_lights:
                    current = light.brightness
                    light.brightness = max(0, current - 50)
                return "Dimmed lights"

            elif hue_action == "brightness":
                level = action_data.get("level", 254)
                for light in all_lights:
                    light.brightness = int(level)
                return f"Set lights to {level}"

            elif hue_action == "color":
                color = action_data.get("color", "white")
                color_lower = color.lower()

                if color_lower in self.color_map:
                    xy = self.color_map[color_lower]
                    for light in all_lights:
                        light.xy = xy
                    return f"Changed to {color}"
                else:
                    return f"Unknown color: {color}"

            else:
                return f"Unknown Hue action: {hue_action}"

        except Exception as e:
            return f"Hue error: {str(e)}"

    # ── Config schema & update ─────────────────────────────────────

    def get_config_schema(self) -> dict:
        return {
            "fields": [
                {"key": "bridge_ip", "label": "Bridge IP Address",
                 "type": "string", "default": "192.168.1.131"},
                {"key": "managed_lights", "label": "Managed Lights (exact names)",
                 "type": "list", "default": []},
            ]
        }

    def update_config(self, config: dict) -> None:
        """Re-connect to bridge if IP changed; sync light list to hue_config.json."""
        old_ip = self._config.get("bridge_ip", "")
        super().update_config(config)
        new_ip = config.get("bridge_ip", "")

        reconnect = (new_ip and new_ip != old_ip)

        if reconnect:
            try:
                self.bridge = Bridge(new_ip)
                self.bridge.connect()
                self.hue_enabled = True
                print(f"   [Hue] Reconnected to bridge at {new_ip}")
            except Exception as e:
                print(f"   [Hue] Failed to reconnect: {e}")
                self.hue_enabled = False
                return

        # Sync the managed_lights list into hue_config.json and rebuild groups
        managed = config.get("managed_lights", None)
        if managed is not None:
            self._sync_lights(managed, new_ip or old_ip)

    def _sync_lights(self, managed_light_names, bridge_ip):
        """Write managed_lights into hue_config.json and rebuild in-memory groups.

        Takes the flat list of light names the user configured in the GUI and
        puts them all in the 'default' group. Preserves existing colors.
        """
        try:
            hue_config = self._load_hue_config()
        except (FileNotFoundError, json.JSONDecodeError):
            hue_config = {}

        # Update bridge IP
        if bridge_ip:
            hue_config["bridge_ip"] = bridge_ip

        # Update light_groups — put all managed lights in a "default" group.
        # Preserve any other groups the user may have manually created.
        if managed_light_names:
            hue_config.setdefault("light_groups", {})
            hue_config["light_groups"]["default"] = managed_light_names
        elif "default" in hue_config.get("light_groups", {}):
            # User cleared the list — remove the default group
            del hue_config["light_groups"]["default"]

        self._save_hue_config(hue_config)

        # Rebuild in-memory groups from the updated config
        if self.hue_enabled and self.bridge:
            try:
                all_lights = self.bridge.get_light_objects('name')
                self.light_groups = {}
                for group_name, light_names in hue_config.get("light_groups", {}).items():
                    self.light_groups[group_name] = [
                        all_lights[name] for name in light_names if name in all_lights
                    ]
                total = sum(len(lg) for lg in self.light_groups.values())
                print(f"   [Hue] Synced lights — {total} active across {len(self.light_groups)} group(s)")
            except Exception as e:
                print(f"   [Hue] Error rebuilding light groups: {e}")

    def get_available_lights(self):
        """Return all light names the bridge knows about.

        Useful for a future 'discover lights' button in the config dialog.
        """
        if not self.hue_enabled or not self.bridge:
            return []
        try:
            all_lights = self.bridge.get_light_objects('name')
            return sorted(all_lights.keys())
        except Exception as e:
            print(f"   [Hue] Error listing lights: {e}")
            return []
