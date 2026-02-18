import os
import json
import importlib
import inspect
import pkgutil
import threading
from pathlib import Path

from core.memory import MemorySystem
from core.llm_client import LLMClient
from core.vision import VisionSystem
from core.voice import VoiceSystem
from core.credential_store import CredentialStore
from talents.base import BaseTalent


class TalonAssistant:
    _ROUTING_SYSTEM_PROMPT = (
        "You are a command router for a desktop assistant. "
        "Given a user command, respond with ONLY the name of the handler "
        "that should process it.\n"
        "Respond with a single word — the handler name, nothing else.\n\n"
        "Available handlers:\n{talent_roster}\n"
        "conversation — General chat, questions, greetings, opinions, "
        "or anything that doesn't clearly fit a specific handler above.\n\n"
        "Rules:\n"
        "- Choose the MOST SPECIFIC handler that matches the user's intent.\n"
        "- If the command mentions a task list or todo list, choose todo.\n"
        "- If the command is general knowledge or chitchat, choose conversation.\n"
        "- Respond with ONLY the handler name. No punctuation, no explanation."
    )

    _CONVERSATION = object()  # Sentinel: LLM explicitly chose conversation

    def __init__(self, config_dir="config"):
        print("=" * 50)
        print("INITIALIZING TALON ASSISTANT")
        print("=" * 50)

        # 1. Load configuration
        self.config = self._load_config(config_dir)
        self.talents_config = self._load_talents_config(config_dir)

        # 2. Initialize core services
        print("\n[1/5] Testing KoboldCpp connection...")
        self.llm = LLMClient(self.config)
        self.llm.test_connection()

        print("[2/5] Initializing Memory + RAG...")
        memory_config = self.config["memory"]
        self.memory = MemorySystem(
            db_path=memory_config["db_path"],
            chroma_path=memory_config["chroma_path"],
            embedding_model=memory_config["embedding_model"]
        )

        print("[3/5] Initializing Vision...")
        self.vision = VisionSystem()
        print("   Vision ready!")

        print("[4/5] Loading Voice System...")
        self.voice = VoiceSystem(self.config, command_callback=self.process_command)

        # 3. Discover and load talents
        print("[5/5] Loading Talents...")
        self.talents: list[BaseTalent] = []
        self.credential_store = CredentialStore()
        self._discover_talents()

        # 3b. Legacy migration first (old keyring entries -> new service name)
        self._migrate_legacy_credentials()

        # 3c. Scrub any plaintext secrets from talents.json into keyring
        self._scrub_plaintext_secrets(config_dir)

        # 3d. Inject keyring secrets into talent configs for runtime use
        self._inject_secrets()

        # 4. Notification callback (set by bridge for talents that need it)
        self.notify_callback = None

        # 5. LLM routing prompt cache (rebuilt when talents change)
        self._routing_prompt_cache = None

        # 6. Thread safety
        self.command_lock = threading.Lock()

        print("\n" + "=" * 50)
        print("TALON READY")
        print("=" * 50 + "\n")

    def _load_config(self, config_dir):
        """Load settings.json"""
        config_path = os.path.join(config_dir, "settings.json")
        with open(config_path, 'r') as f:
            return json.load(f)

    def _load_talents_config(self, config_dir):
        """Load talents.json"""
        config_path = os.path.join(config_dir, "talents.json")
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _discover_talents(self):
        """Auto-discover talent classes in talents/ and talents/user/ directories"""
        talent_packages = [
            ("talents", Path("talents")),
            ("talents.user", Path("talents/user"))
        ]

        for package_name, talent_dir in talent_packages:
            if not talent_dir.exists():
                continue

            for module_info in pkgutil.iter_modules([str(talent_dir)]):
                if module_info.name in ("base", "__init__"):
                    continue

                try:
                    module = importlib.import_module(f"{package_name}.{module_info.name}")

                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (inspect.isclass(attr)
                                and issubclass(attr, BaseTalent)
                                and attr is not BaseTalent):

                            talent_instance = attr()

                            # Check if enabled in talents.json
                            talent_cfg = self.talents_config.get(talent_instance.name, {})
                            if not talent_cfg.get("enabled", True):
                                talent_instance.enabled = False
                                print(f"   [Talents] Loaded (disabled): {talent_instance.name}")
                            else:
                                print(f"   [Talents] Loaded: {talent_instance.name} (priority: {talent_instance.priority})")

                            # Initialize with full config
                            talent_instance.initialize(self.config)

                            # Load per-talent config from talents.json
                            per_talent_cfg = talent_cfg.get("config", {})
                            if per_talent_cfg:
                                talent_instance.update_config(per_talent_cfg)

                            self.talents.append(talent_instance)

                except Exception as e:
                    print(f"   [Talents] Error loading {module_info.name}: {e}")

        # Sort by priority for display order (sidebar) and keyword fallback
        self.talents.sort(key=lambda t: t.priority, reverse=True)

    def _inject_secrets(self):
        """Fill empty password fields from the OS keyring so talents
        have real credentials at runtime without storing them on disk."""
        for talent in self.talents:
            schema = talent.get_config_schema() or {}
            password_keys = [
                f["key"] for f in schema.get("fields", [])
                if f.get("type") == "password"
            ]
            if not password_keys:
                continue

            cfg = dict(talent.talent_config)
            changed = False
            for key in password_keys:
                current = cfg.get(key, "")
                if not current:
                    secret = self.credential_store.get_secret(talent.name, key)
                    if secret:
                        cfg[key] = secret
                        changed = True
            if changed:
                talent.update_config(cfg)
                print(f"   [Credentials] Injected keyring secrets for: {talent.name}")

    def _scrub_plaintext_secrets(self, config_dir):
        """One-time migration: move any plaintext passwords already in
        talents.json into the keyring and replace with empty strings.

        Only scrubs if the keyring write actually succeeds — never
        deletes plaintext without confirming it's safely stored.
        """
        if not self.credential_store.available:
            print("   [Credentials] Keyring unavailable, skipping plaintext scrub")
            return

        config_path = os.path.join(config_dir, "talents.json")
        try:
            with open(config_path, 'r') as f:
                talents_cfg = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return

        dirty = False
        for talent in self.talents:
            schema = talent.get_config_schema() or {}
            password_keys = [
                f["key"] for f in schema.get("fields", [])
                if f.get("type") == "password"
            ]
            if not password_keys:
                continue

            tcfg = talents_cfg.get(talent.name, {}).get("config", {})
            for key in password_keys:
                value = tcfg.get(key, "")
                if value:
                    # Only scrub if keyring write succeeds
                    stored = self.credential_store.store_secret(
                        talent.name, key, value)
                    if stored:
                        tcfg[key] = ""
                        dirty = True
                        print(f"   [Credentials] Scrubbed plaintext {talent.name}.{key}")
                    else:
                        print(f"   [Credentials] Keyring write failed, "
                              f"keeping plaintext for {talent.name}.{key}")

        if dirty:
            with open(config_path, 'w') as f:
                json.dump(talents_cfg, f, indent=2)

    def _migrate_legacy_credentials(self):
        """Migrate legacy email keyring entries to the new service name."""
        for talent in self.talents:
            if talent.name == "email":
                username = talent.talent_config.get("username", "")
                self.credential_store.migrate_legacy_email(username)
                break

    def _build_context(self, command, speak_response):
        """Build the context dict passed to talent.execute()"""
        memory_context = self.memory.get_relevant_context(command)
        ctx = {
            "llm": self.llm,
            "memory": self.memory,
            "vision": self.vision,
            "voice": self.voice,
            "config": self.config,
            "memory_context": memory_context,
            "speak_response": speak_response,
        }
        if self.notify_callback:
            ctx["notify"] = self.notify_callback
        return ctx

    def _find_talent(self, command):
        """Route command to the best talent using LLM intent classification.
        Falls back to keyword matching only if the LLM is unreachable."""
        result = self._route_with_llm(command)
        if result is self._CONVERSATION:
            return None                          # LLM chose conversation
        if result is not None:
            return result                        # Valid talent
        # LLM unreachable — degraded keyword fallback
        print("   [Router] WARNING: LLM unavailable, using keyword fallback")
        return self._find_talent_by_keywords(command)

    def _find_talent_by_keywords(self, command):
        """Keyword-based fallback: first talent whose can_handle() returns True."""
        for talent in self.talents:
            if talent.enabled and talent.can_handle(command):
                return talent
        return None

    def _route_with_llm(self, command):
        """Ask the LLM which talent should handle this command."""
        try:
            prompt = self._build_routing_prompt()
            response = self.llm.generate(
                f"Route this command: {command}",
                system_prompt=prompt,
                temperature=0.1,
                max_length=20,
            )
            if not response or response.startswith("Error:"):
                return None
            talent_name = response.strip().split()[0].strip(".,!\"'").lower()
            print(f"   [LLM Router] -> {talent_name}")
            if talent_name == "conversation":
                return self._CONVERSATION
            return self._get_talent_by_name(talent_name)
        except Exception as e:
            print(f"   [LLM Router] Error: {e}")
            return None

    def _build_routing_prompt(self):
        """Build (and cache) the system prompt listing all enabled talents."""
        if self._routing_prompt_cache:
            return self._routing_prompt_cache
        lines = []
        for talent in self.talents:
            if not talent.enabled:
                continue
            if not talent.routing_available:
                continue
            if talent.examples:
                examples_str = "; ".join(talent.examples[:5])
                lines.append(
                    f"{talent.name} — {talent.description} "
                    f"(examples: {examples_str})")
            else:
                kws = ", ".join(talent.keywords[:5])
                lines.append(
                    f"{talent.name} — {talent.description} "
                    f"(e.g. {kws})")
        roster = "\n".join(lines)
        self._routing_prompt_cache = self._ROUTING_SYSTEM_PROMPT.format(
            talent_roster=roster)
        return self._routing_prompt_cache

    def _get_talent_by_name(self, name):
        """Look up an enabled talent by name."""
        for talent in self.talents:
            if talent.name == name and talent.enabled:
                return talent
        return None

    def invalidate_routing_cache(self):
        """Clear the cached LLM routing prompt."""
        self._routing_prompt_cache = None

    def _detect_repeat_request(self, command):
        """Detect if user wants to repeat last action"""
        repeat_keywords = ['again', 'repeat', 'do that', 'same thing', 'one more time']
        return any(kw in command for kw in repeat_keywords)

    def _handle_repeat(self, speak_response):
        """Handle a repeat-last-action request"""
        last_action = self.memory.get_last_successful_action()
        if last_action:
            cmd_text, action_json, result = last_action
            print(f"Repeating last action: {cmd_text}")

            try:
                actions = json.loads(action_json)

                # Find the right talent to re-execute
                for talent in self.talents:
                    if hasattr(talent, '_execute_single_action'):
                        result = talent._execute_single_action(actions)
                        print(f"  -> {result}")
                        break
                    elif hasattr(talent, '_control_hue') and actions.get("action") == "hue_light":
                        result = talent._control_hue(actions)
                        print(f"  -> {result}")
                        break

                if speak_response:
                    self.voice.speak("Done!")
                else:
                    print("\nDone!")
            except Exception as e:
                print(f"Error repeating action: {e}")
                if speak_response:
                    self.voice.speak("Sorry, I couldn't repeat that.")
        else:
            if speak_response:
                self.voice.speak("I don't have a previous action to repeat.")
            else:
                print("No previous action found.")

    def _detect_preference(self, command, response=""):
        """Detect if command contains a preference to remember"""
        preference_keywords = ['prefer', 'like', 'favorite', 'always', 'usually', 'remember']
        if any(kw in command for kw in preference_keywords):
            self.memory.store_preference(command, category="general")
            return True
        return False

    def _handle_conversation(self, command, context, speak_response):
        """Handle commands that no talent matched -- general conversation"""
        # Only trigger vision for phrases that *clearly* ask about the screen.
        # Single words like "what" or "find" are too generic and cause false
        # positives (e.g. "what are the top stories" → screenshot of desktop).
        vision_phrases = [
            "on my screen", "on the screen", "on screen",
            "what do you see", "what can you see", "what's on",
            "look at my", "look at the", "read my screen",
            "read the screen", "this window", "current window",
            "what window", "which window", "screenshot",
            "what am i looking at", "what is this",
            "inside of notepad", "inside notepad", "in notepad",
            "text inside", "what's in the",
            "inside of", "displayed on", "showing on",
        ]
        needs_vision = any(phrase in command.lower() for phrase in vision_phrases)

        screenshot_b64 = None
        if needs_vision:
            prompt = f"User command: {command}\n\nAnalyze the screenshot and respond briefly. Keep your response concise (2-3 sentences max)."
            screenshot_b64 = self.vision.capture_screenshot()
        else:
            prompt = f"{command}\n\nRespond briefly and conversationally (2-3 sentences max)."

        memory_context = context.get("memory_context", "")
        if memory_context:
            prompt = f"{memory_context}{prompt}"

        response = self.llm.generate(prompt, use_vision=needs_vision, screenshot_b64=screenshot_b64)

        self.memory.log_command(command, success=True, response=response)
        self._detect_preference(command, response)

        if speak_response:
            self.voice.speak(response)
        else:
            print(f"\nTalon: {response}")

        return response

    def process_command(self, command, speak_response=True):
        """Central command processing pipeline.

        Returns:
            dict with keys: response (str), talent (str), success (bool)
            or None if command was too short.
        """
        with self.command_lock:
            print(f"\n{'=' * 50}")
            print(f"COMMAND: {command}")
            print(f"{'=' * 50}\n")

            if not command or len(command.strip()) < 1:
                return None

            # Step 1: Repeat detection
            if self._detect_repeat_request(command):
                self._handle_repeat(speak_response)
                return {"response": "Done!", "talent": "", "success": True}

            # Step 2: Build context
            context = self._build_context(command, speak_response)

            # Step 3: Route to talent
            talent = self._find_talent(command)

            if talent:
                print(f"   [Routing] -> {talent.name}")
                result = talent.execute(command, context)

                # Step 4: Log to memory
                command_id = self.memory.log_command(
                    command,
                    success=result["success"],
                    response=result.get("response", "")
                )
                for action_info in result.get("actions_taken", []):
                    self.memory.log_action(
                        command_id,
                        action_info.get("action", {}),
                        action_info.get("result", ""),
                        action_info.get("success", True)
                    )

                # Step 5: Store successful pattern
                if result["success"] and result.get("actions_taken"):
                    actions_list = [a.get("action", {}) for a in result["actions_taken"]]
                    self.memory.store_successful_pattern(command, actions_list)

                # Step 6: Speak response if talent didn't already
                if not result.get("spoken", False):
                    response_text = result.get("response", "")
                    if speak_response:
                        self.voice.speak(response_text)
                    else:
                        print(f"\nTalon: {response_text}")

                # Step 7: Preference detection
                self._detect_preference(command, result.get("response", ""))

                print(f"\n{'=' * 50}\n")
                return {
                    "response": result.get("response", ""),
                    "talent": talent.name,
                    "success": result.get("success", True)
                }

            else:
                # No talent matched -- conversational fallback
                response = self._handle_conversation(command, context, speak_response)

                print(f"\n{'=' * 50}\n")
                return {
                    "response": response or "",
                    "talent": "",
                    "success": True
                }

    def text_interface(self):
        """Text-based command interface"""
        print("\n" + "=" * 50)
        print("TALON - TEXT INTERFACE")
        print("=" * 50)
        print("Type commands or 'exit' to quit\n")

        while True:
            try:
                command = input("You: ").strip()

                if not command:
                    continue

                if command.lower() in ['exit', 'quit', 'bye']:
                    print("Goodbye!")
                    break

                if command.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  exit   - Exit Talon")
                    print("  help   - Show this message")
                    print("  again  - Repeat last action")
                    print("\nLoaded talents:")
                    for talent in self.talents:
                        print(f"  {talent.name}: {talent.description}")
                        print(f"    Keywords: {', '.join(talent.keywords)}")
                    print("\nMemory: Say 'I prefer...' to store preferences\n")
                    continue

                self.process_command(command, speak_response=False)

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}\n")

    def run(self, mode="voice"):
        """Start Talon"""
        try:
            if mode == "both":
                print("\n" + "=" * 50)
                print("DUAL MODE: Voice + Text")
                print("=" * 50)
                print("Voice: Use wake word")
                print("Text: Type in this window\n")

                voice_thread = threading.Thread(target=self.voice.listen_for_wake_word, daemon=True)
                voice_thread.start()

                self.text_interface()

            elif mode == "text":
                self.text_interface()
            else:
                self.voice.listen_for_wake_word()

        except KeyboardInterrupt:
            print("\n\nShutting down. Goodbye!")
