import os
import json
import shutil
import inspect
import importlib
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from gui.workers import CommandWorker, TTSWorker, VoiceListenWorker
from talents.base import BaseTalent
from core.credential_store import CredentialStore


class AssistantBridge(QObject):
    """Bridge between TalonAssistant (blocking, print-based) and Qt GUI (signal-based).

    Manages worker threads for all blocking operations and provides a clean
    signal-based API for the GUI to consume.
    """

    # Initialization signals
    init_complete = pyqtSignal()
    init_error = pyqtSignal(str)

    # Command processing signals
    command_started = pyqtSignal(str)
    command_response = pyqtSignal(str, str)     # (command, response)
    command_error = pyqtSignal(str, str)        # (command, error_message)
    talent_activated = pyqtSignal(str)          # talent name

    # Voice signals
    voice_status = pyqtSignal(str)              # listening/recording/transcribing/off
    audio_level = pyqtSignal(float)
    wake_word_detected = pyqtSignal()
    voice_command = pyqtSignal(str)             # transcribed command from voice

    # TTS signals
    tts_started = pyqtSignal()
    tts_finished = pyqtSignal()
    tts_stopped = pyqtSignal()  # emitted when TTS is interrupted early

    # System status
    llm_status = pyqtSignal(bool)
    activity = pyqtSignal(str)                  # idle/processing/speaking/initializing

    # Talent management
    talents_loaded = pyqtSignal(list)           # list of talent info dicts
    talent_toggled = pyqtSignal(str, bool)

    # Settings
    settings_changed = pyqtSignal(dict)

    # Notifications (for system tray)
    notification_requested = pyqtSignal(str, str)  # title, message

    # Reminder alert (dismissable dialog)
    reminder_fired = pyqtSignal(str, str, str, int)  # reminder_id, title, message, default_snooze_minutes

    # LLM server status
    server_status = pyqtSignal(str)  # stopped/starting/running/error

    def __init__(self, config_dir="config"):
        super().__init__()
        self.config_dir = config_dir
        self.assistant = None
        self._command_worker = None
        self._tts_worker = None
        self._voice_worker = None
        self._tts_enabled = True
        self.credential_store = CredentialStore()
        self.server_manager = None  # LLMServerManager, set by main.py

    def set_assistant(self, assistant):
        """Accept a pre-built TalonAssistant (loaded on the main thread).

        CTranslate2 / faster-whisper segfaults when WhisperModel is created
        inside a QThread on Windows, so the assistant must be built on the
        main thread before the event loop starts.
        """
        self.assistant = assistant

        # Wire notification callback so talents can push toast/tray notifications
        self.assistant.notify_callback = self._talent_notify

        # Re-wire any ReminderTalent timers loaded at startup (before bridge existed)
        for talent in self.assistant.talents:
            if hasattr(talent, 'rewire_notify'):
                talent.rewire_notify(self._reminder_alert)

        # Check LLM status
        try:
            connected = self.assistant.llm.test_connection()
            self.llm_status.emit(connected)
        except Exception:
            self.llm_status.emit(False)

        # Build talent info list for sidebar
        self._emit_full_talent_list()

        self.activity.emit("idle")
        self.init_complete.emit()

    def set_server_manager(self, manager):
        """Accept an LLMServerManager for lifecycle management."""
        self.server_manager = manager
        if manager:
            manager.on_status_changed = self._on_server_status_changed

    def _on_server_status_changed(self, status):
        """Relay server status changes to the GUI."""
        self.server_status.emit(status)

    @pyqtSlot(str)
    def submit_command(self, command):
        """Submit a command for processing in a background thread."""
        if self.assistant is None:
            self.command_error.emit(command, "Assistant not initialized yet")
            return
        if self._command_worker and self._command_worker.isRunning():
            self.command_error.emit(command, "Already processing a command")
            return

        # Auto-stop any in-progress TTS when user sends a new command
        self.stop_speaking()

        self.command_started.emit(command)
        self.activity.emit("processing")

        self._command_worker = CommandWorker(self.assistant, command)
        self._command_worker.response_ready.connect(self._on_command_done)
        self._command_worker.error.connect(self._on_command_error)
        self._command_worker.start()

    def _on_command_done(self, command, response, talent_name, success):
        """Called when process_command() completes."""
        if talent_name:
            self.talent_activated.emit(talent_name)

        self.command_response.emit(command, response)
        self.notification_requested.emit("Talon", response or "Done!")
        self.activity.emit("idle")

        # Optionally speak the response
        if self._tts_enabled and response and not response.startswith("Error"):
            self._speak(response)

    def _on_command_error(self, command, error_msg):
        """Called if process_command() raises an exception."""
        self.command_error.emit(command, error_msg)
        self.activity.emit("idle")

    def _speak(self, text):
        """Run TTS in a background thread."""
        if self.assistant is None:
            return
        self._tts_worker = TTSWorker(self.assistant.voice, text)
        self._tts_worker.started_speaking.connect(self.tts_started.emit)
        self._tts_worker.finished_speaking.connect(self._on_tts_done)
        self._tts_worker.stopped_early.connect(self._on_tts_stopped)
        self._tts_worker.error.connect(lambda e: print(f"TTS Error: {e}"))
        self.activity.emit("speaking")
        self._tts_worker.start()

    def _on_tts_done(self):
        self.tts_finished.emit()
        self.activity.emit("idle")

    def _on_tts_stopped(self):
        self.tts_stopped.emit()
        self.tts_finished.emit()
        self.activity.emit("idle")

    def stop_speaking(self):
        """Interrupt in-progress TTS playback."""
        if self._tts_worker and self._tts_worker.isRunning():
            self._tts_worker.request_stop()

    @pyqtSlot(bool)
    def toggle_voice(self, enabled):
        """Start or stop the voice listening loop."""
        if enabled:
            if self.assistant is None:
                self.voice_status.emit("off")
                return
            self._voice_worker = VoiceListenWorker(self.assistant.voice)
            self._voice_worker.audio_level.connect(self.audio_level.emit)
            self._voice_worker.wake_word_detected.connect(self.wake_word_detected.emit)
            self._voice_worker.command_transcribed.connect(self._on_voice_command)
            self._voice_worker.status_changed.connect(self.voice_status.emit)
            self._voice_worker.heard_text.connect(
                lambda t: print(f"   [Voice] Heard: {t}"))
            self._voice_worker.error.connect(
                lambda e: print(f"   [Voice] Error: {e}"))
            self._voice_worker.start()
        else:
            if self._voice_worker:
                self._voice_worker.stop()
                self._voice_worker.wait(5000)
                self._voice_worker = None
            self.voice_status.emit("off")
            self.audio_level.emit(0.0)

    def _on_voice_command(self, command_text):
        """Voice command transcribed — submit through normal pipeline."""
        self.voice_command.emit(command_text)
        self.submit_command(command_text)

    @pyqtSlot(str, bool)
    def toggle_talent(self, talent_name, enabled):
        """Enable or disable a talent and persist to talents.json."""
        if self.assistant is None:
            return

        # Update in-memory state
        for talent in self.assistant.talents:
            if talent.name == talent_name:
                talent.enabled = enabled
                break

        # Rebuild LLM routing prompt (talent roster changed)
        self.assistant.invalidate_routing_cache()

        # Persist to config/talents.json
        config_path = os.path.join(self.config_dir, "talents.json")
        try:
            with open(config_path, 'r') as f:
                talents_cfg = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            talents_cfg = {}

        if talent_name not in talents_cfg:
            talents_cfg[talent_name] = {}
        talents_cfg[talent_name]["enabled"] = enabled

        with open(config_path, 'w') as f:
            json.dump(talents_cfg, f, indent=2)

        self.talent_toggled.emit(talent_name, enabled)

    @pyqtSlot(bool)
    def set_tts_enabled(self, enabled):
        """Toggle TTS on/off."""
        self._tts_enabled = enabled

    @pyqtSlot(dict)
    def update_settings(self, new_settings):
        """Apply new settings to the running assistant where safe."""
        if self.assistant is None:
            return

        # Hot-swap LLM parameters (safe to change at runtime)
        if "llm" in new_settings:
            llm_cfg = new_settings["llm"]
            self.assistant.config["llm"] = llm_cfg
            self.assistant.llm.endpoint = llm_cfg.get(
                "endpoint", self.assistant.llm.endpoint)
            self.assistant.llm.max_length = llm_cfg.get(
                "max_length", self.assistant.llm.max_length)
            self.assistant.llm.temperature = llm_cfg.get(
                "temperature", self.assistant.llm.temperature)
            self.assistant.llm.top_p = llm_cfg.get(
                "top_p", self.assistant.llm.top_p)
            self.assistant.llm.rep_pen = llm_cfg.get(
                "rep_pen", self.assistant.llm.rep_pen)
            self.assistant.llm.timeout = llm_cfg.get(
                "timeout", self.assistant.llm.timeout)
            self.assistant.llm.api_format = llm_cfg.get(
                "api_format", self.assistant.llm.api_format)

        # Hot-swap voice settings
        if "voice" in new_settings:
            voice_cfg = new_settings["voice"]
            self.assistant.config["voice"] = voice_cfg
            self.assistant.voice.tts_voice = voice_cfg.get(
                "tts_voice", self.assistant.voice.tts_voice)
            self.assistant.voice.wake_words = voice_cfg.get(
                "wake_words", self.assistant.voice.wake_words)

        # Hot-swap audio thresholds
        if "audio" in new_settings:
            audio_cfg = new_settings["audio"]
            self.assistant.config["audio"] = audio_cfg

        # Hot-swap desktop settings
        if "desktop" in new_settings:
            self.assistant.config["desktop"] = new_settings["desktop"]

        # Update full config reference
        self.assistant.config.update(new_settings)
        self.settings_changed.emit(new_settings)
        self.activity.emit("idle")

    def import_talent(self, source_filepath):
        """Copy talent file to talents/user/, dynamically import, register."""
        if self.assistant is None:
            raise RuntimeError("Assistant not initialized")

        filename = os.path.basename(source_filepath)
        dest = os.path.join("talents", "user", filename)
        if os.path.abspath(source_filepath) != os.path.abspath(dest):
            shutil.copy2(source_filepath, dest)

        module_name = filename.replace('.py', '')
        module = importlib.import_module(f"talents.user.{module_name}")

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (inspect.isclass(attr)
                    and issubclass(attr, BaseTalent)
                    and attr is not BaseTalent):
                instance = attr()
                instance.initialize(self.assistant.config)
                self.assistant.talents.append(instance)
                self.assistant.talents.sort(
                    key=lambda t: t.priority, reverse=True)

                # Rebuild LLM routing prompt (new talent added)
                self.assistant.invalidate_routing_cache()

                # Persist to talents.json
                config_path = os.path.join(self.config_dir, "talents.json")
                try:
                    with open(config_path, 'r') as f:
                        talents_cfg = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    talents_cfg = {}
                talents_cfg[instance.name] = {"enabled": True}
                with open(config_path, 'w') as f:
                    json.dump(talents_cfg, f, indent=2)

                self._emit_full_talent_list()
                return {
                    "name": instance.name,
                    "description": instance.description,
                    "enabled": True,
                    "keywords": instance.keywords,
                    "priority": instance.priority
                }

        raise ValueError(f"No BaseTalent subclass found in {filename}")

    def get_talent(self, talent_name):
        """Return the talent instance by name, or None."""
        if self.assistant is None:
            return None
        for talent in self.assistant.talents:
            if talent.name == talent_name:
                return talent
        return None

    @pyqtSlot(str, dict)
    def update_talent_config(self, talent_name, config):
        """Update a talent's per-talent config at runtime and persist.

        Password-type fields are intercepted: the real value is stored in
        the OS keyring and an empty string is written to talents.json.
        """
        talent = self.get_talent(talent_name)
        if talent is None:
            return

        # Identify password fields from the talent's schema
        schema = talent.get_config_schema() or {}
        password_keys = {
            f["key"] for f in schema.get("fields", [])
            if f.get("type") == "password"
        }

        # Separate secrets from the config that gets written to disk
        disk_config = dict(config)
        for key in password_keys:
            value = config.get(key, "")
            if value:
                # Store real secret in keyring
                self.credential_store.store_secret(talent_name, key, value)
                # Write empty string to disk so plaintext never hits JSON
                disk_config[key] = ""

        # Update in-memory with real values (talent needs them at runtime)
        talent.update_config(config)

        # Rebuild LLM routing prompt (routing_available may have changed)
        if self.assistant:
            self.assistant.invalidate_routing_cache()

        # Persist sanitised config to config/talents.json
        config_path = os.path.join(self.config_dir, "talents.json")
        try:
            with open(config_path, 'r') as f:
                talents_cfg = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            talents_cfg = {}

        if talent_name not in talents_cfg:
            talents_cfg[talent_name] = {}
        talents_cfg[talent_name]["config"] = disk_config

        with open(config_path, 'w') as f:
            json.dump(talents_cfg, f, indent=2)

    def uninstall_talent(self, talent_name):
        """Remove a talent from the running assistant and talents.json.

        The file removal is handled by MarketplaceClient.uninstall_talent().
        This method handles the in-memory and config cleanup.
        """
        if self.assistant is None:
            return

        # Remove from in-memory talent list
        self.assistant.talents = [
            t for t in self.assistant.talents if t.name != talent_name
        ]

        # Remove from talents.json
        config_path = os.path.join(self.config_dir, "talents.json")
        try:
            with open(config_path, 'r') as f:
                talents_cfg = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            talents_cfg = {}

        if talent_name in talents_cfg:
            del talents_cfg[talent_name]
            with open(config_path, 'w') as f:
                json.dump(talents_cfg, f, indent=2)

        # Refresh sidebar
        self._emit_full_talent_list()
        print(f"   [Bridge] Uninstalled talent: {talent_name}")

    def _emit_full_talent_list(self):
        """Re-emit talents_loaded with current full list (including has_config)."""
        talent_list = []
        for talent in self.assistant.talents:
            schema = talent.get_config_schema()
            talent_list.append({
                "name": talent.name,
                "description": talent.description,
                "enabled": talent.enabled,
                "keywords": talent.keywords,
                "priority": talent.priority,
                "has_config": bool(schema and schema.get("fields")),
            })
        self.talents_loaded.emit(talent_list)

    def _talent_notify(self, title, message):
        """Callback given to talents via context['notify'].

        Thread-safe: emits a Qt signal that the system tray picks up.
        """
        self.notification_requested.emit(title, message)

    def _reminder_alert(self, reminder_id, title, message, default_snooze_minutes=5):
        """Callback for ReminderTalent — shows a dismissable alert dialog.

        Thread-safe: emits a Qt signal that MainWindow picks up on the GUI thread.
        """
        self.reminder_fired.emit(reminder_id, title, message, default_snooze_minutes)

    def snooze_reminder(self, reminder_id, message, seconds):
        """Re-arm a snoozed reminder via the ReminderTalent."""
        for talent in self.assistant.talents:
            if talent.name == "reminder" and hasattr(talent, 'snooze_reminder'):
                talent.snooze_reminder(reminder_id, message, seconds)
                break

    def cleanup(self):
        """Stop all workers before app exit."""
        try:
            if self._voice_worker:
                self._voice_worker.stop()
                if not self._voice_worker.wait(3000):
                    self._voice_worker.terminate()
                    self._voice_worker.wait(1000)
        except RuntimeError:
            pass

        try:
            if self._command_worker and self._command_worker.isRunning():
                if not self._command_worker.wait(3000):
                    self._command_worker.terminate()
                    self._command_worker.wait(1000)
        except RuntimeError:
            pass

        try:
            if self._tts_worker and self._tts_worker.isRunning():
                if not self._tts_worker.wait(3000):
                    self._tts_worker.terminate()
                    self._tts_worker.wait(1000)
        except RuntimeError:
            pass

        # Stop built-in LLM server if running
        try:
            if self.server_manager and self.server_manager.is_running():
                self.server_manager.stop()
        except Exception:
            pass
