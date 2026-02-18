import os
import json
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QSplitter,
                             QMenuBar, QStatusBar, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QActionGroup, QShortcut, QKeySequence

from gui.assistant_bridge import AssistantBridge
from gui.widgets.chat_view import ChatView
from gui.widgets.text_input import TextInput
from gui.widgets.voice_panel import VoicePanel
from gui.widgets.talent_sidebar import TalentSidebar
from gui.widgets.status_bar import StatusBarWidget
from gui.widgets.activity_log import ActivityLog


class MainWindow(QMainWindow):
    """Talon Assistant main window.

    Layout:
        +-------------------+---------------------------+
        | Talent Sidebar    | Chat View                 |
        |                   |                           |
        |                   +---------------------------+
        |                   | Text Input  [Send]        |
        +-------------------+---------------------------+
        | [Mic] [Level Bar] Voice Status                |
        +-----------------------------------------------+
        | Activity Log (collapsible)                    |
        +-----------------------------------------------+
        | Status Bar                                    |
        +-----------------------------------------------+
    """

    def __init__(self, bridge, theme_manager=None, config_dir="config"):
        super().__init__()
        self.bridge = bridge
        self.theme_manager = theme_manager
        self.config_dir = config_dir
        self.system_tray = None  # Set externally after construction
        self._minimize_to_tray = True
        self.setWindowTitle("Talon Assistant")
        self.setMinimumSize(900, 600)
        self.resize(1200, 800)

        # Lazy imports for optional components
        self.chat_store = None

        self._setup_menubar()
        self._setup_central_widget()
        self._setup_status_bar()
        self._connect_signals()

        # Sync theme radio buttons with saved preference
        if self.theme_manager:
            current = self.theme_manager.current_theme
            self.dark_theme_action.setChecked(current == "dark")
            self.light_theme_action.setChecked(current == "light")

        # Show initialization message
        self.chat_view.add_system_message("Initializing Talon... Please wait.")

    def _setup_menubar(self):
        menubar = self.menuBar()

        # ── File menu ──────────────────────────────────────────
        file_menu = menubar.addMenu("File")

        save_action = QAction("Save Conversation", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_conversation)
        file_menu.addAction(save_action)

        load_action = QAction("Load Conversation...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._load_conversation)
        file_menu.addAction(load_action)

        export_action = QAction("Export Conversation...", self)
        export_action.setShortcut("Ctrl+Shift+E")
        export_action.triggered.connect(self._export_conversation)
        file_menu.addAction(export_action)

        clear_action = QAction("Clear Chat", self)
        clear_action.triggered.connect(self._clear_chat)
        file_menu.addAction(clear_action)

        file_menu.addSeparator()

        import_talent_action = QAction("Import Talent...", self)
        import_talent_action.triggered.connect(self._open_import_talent_dialog)
        file_menu.addAction(import_talent_action)

        marketplace_action = QAction("Talent Marketplace...", self)
        marketplace_action.setShortcut("Ctrl+M")
        marketplace_action.triggered.connect(self._open_marketplace)
        file_menu.addAction(marketplace_action)

        file_menu.addSeparator()

        settings_action = QAction("Settings...", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self._open_settings)
        file_menu.addAction(settings_action)

        llm_server_action = QAction("LLM Server...", self)
        llm_server_action.triggered.connect(self._open_llm_setup)
        file_menu.addAction(llm_server_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self._force_exit)
        file_menu.addAction(exit_action)

        # ── View menu ─────────────────────────────────────────
        view_menu = menubar.addMenu("View")

        self.toggle_log_action = QAction("Toggle Activity Log", self)
        self.toggle_log_action.setShortcut("Ctrl+L")
        self.toggle_log_action.triggered.connect(self._toggle_activity_log)
        view_menu.addAction(self.toggle_log_action)

        self.toggle_sidebar_action = QAction("Toggle Sidebar", self)
        self.toggle_sidebar_action.setShortcut("Ctrl+B")
        self.toggle_sidebar_action.triggered.connect(self._toggle_sidebar)
        view_menu.addAction(self.toggle_sidebar_action)

        view_menu.addSeparator()

        # Appearance submenu
        appearance_menu = view_menu.addMenu("Appearance")

        theme_group = QActionGroup(self)
        self.dark_theme_action = QAction("Dark Theme", self)
        self.dark_theme_action.setCheckable(True)
        self.dark_theme_action.setChecked(True)
        self.dark_theme_action.triggered.connect(lambda: self._set_theme("dark"))
        theme_group.addAction(self.dark_theme_action)
        appearance_menu.addAction(self.dark_theme_action)

        self.light_theme_action = QAction("Light Theme", self)
        self.light_theme_action.setCheckable(True)
        self.light_theme_action.triggered.connect(lambda: self._set_theme("light"))
        theme_group.addAction(self.light_theme_action)
        appearance_menu.addAction(self.light_theme_action)

        appearance_menu.addSeparator()

        inc_font_action = QAction("Increase Font Size", self)
        inc_font_action.setShortcut("Ctrl+=")
        inc_font_action.triggered.connect(self._increase_font)
        appearance_menu.addAction(inc_font_action)

        dec_font_action = QAction("Decrease Font Size", self)
        dec_font_action.setShortcut("Ctrl+-")
        dec_font_action.triggered.connect(self._decrease_font)
        appearance_menu.addAction(dec_font_action)

        reset_font_action = QAction("Reset Font Size", self)
        reset_font_action.setShortcut("Ctrl+0")
        reset_font_action.triggered.connect(self._reset_font)
        appearance_menu.addAction(reset_font_action)

        # ── Help menu ─────────────────────────────────────────
        help_menu = menubar.addMenu("Help")

        help_topics_action = QAction("Help Topics...", self)
        help_topics_action.setShortcut("Ctrl+F1")
        help_topics_action.triggered.connect(self._open_help)
        help_menu.addAction(help_topics_action)

        shortcuts_action = QAction("Keyboard Shortcuts...", self)
        shortcuts_action.triggered.connect(self._open_shortcuts_help)
        help_menu.addAction(shortcuts_action)

        help_menu.addSeparator()

        about_action = QAction("About Talon", self)
        about_action.triggered.connect(self._open_about)
        help_menu.addAction(about_action)

    def _setup_central_widget(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Main vertical splitter: top area | activity log
        self.v_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top area: horizontal splitter: sidebar | chat area
        self.h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Talent sidebar
        self.talent_sidebar = TalentSidebar()
        self.h_splitter.addWidget(self.talent_sidebar)

        # Chat area (chat view + text input + voice panel)
        chat_container = QWidget()
        chat_layout = QVBoxLayout(chat_container)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)

        self.chat_view = ChatView()
        chat_layout.addWidget(self.chat_view)

        self.text_input = TextInput()
        chat_layout.addWidget(self.text_input)

        self.voice_panel = VoicePanel()
        chat_layout.addWidget(self.voice_panel)

        self.h_splitter.addWidget(chat_container)
        self.h_splitter.setSizes([220, 800])
        self.h_splitter.setStretchFactor(0, 0)  # Sidebar doesn't stretch
        self.h_splitter.setStretchFactor(1, 1)  # Chat area stretches

        self.v_splitter.addWidget(self.h_splitter)

        # Activity log
        self.activity_log = ActivityLog()
        self.v_splitter.addWidget(self.activity_log)
        self.v_splitter.setSizes([500, 150])
        self.v_splitter.setCollapsible(1, True)

        main_layout.addWidget(self.v_splitter)

    def _setup_status_bar(self):
        self.status_bar_widget = StatusBarWidget()
        status_bar = QStatusBar()
        status_bar.addPermanentWidget(self.status_bar_widget, 1)
        self.setStatusBar(status_bar)

    def _connect_signals(self):
        """Wire all signals between bridge and widgets."""

        # Text input -> bridge
        self.text_input.command_submitted.connect(self.bridge.submit_command)

        # Bridge init signals
        self.bridge.init_complete.connect(self._on_init_complete)
        self.bridge.init_error.connect(self._on_init_error)

        # Bridge command signals -> chat view
        self.bridge.command_started.connect(self._on_command_started)
        self.bridge.command_response.connect(self._on_command_response)
        self.bridge.command_error.connect(self._on_command_error)

        # Bridge -> talent sidebar
        self.bridge.talents_loaded.connect(self.talent_sidebar.populate_talents)
        self.bridge.talent_activated.connect(self.talent_sidebar.highlight_talent)

        # Talent sidebar -> bridge
        self.talent_sidebar.talent_toggled.connect(self.bridge.toggle_talent)

        # Voice panel -> bridge
        self.voice_panel.voice_toggled.connect(self.bridge.toggle_voice)
        self.voice_panel.tts_toggled.connect(self.bridge.set_tts_enabled)

        # Bridge -> voice panel
        self.bridge.voice_status.connect(self.voice_panel.update_status)
        self.bridge.audio_level.connect(self.voice_panel.update_level)
        self.bridge.wake_word_detected.connect(self.voice_panel.flash_wake_indicator)

        # Bridge voice command -> chat view (show as user message)
        self.bridge.voice_command.connect(
            lambda cmd: self.chat_view.add_user_message(f"[voice] {cmd}"))

        # Bridge -> status bar
        self.bridge.activity.connect(self.status_bar_widget.set_activity)
        self.bridge.llm_status.connect(self.status_bar_widget.set_llm_status)
        self.bridge.voice_status.connect(self.status_bar_widget.set_voice_status)
        self.bridge.server_status.connect(self.status_bar_widget.set_server_status)

        # Set server mode for status bar display
        if self.bridge.server_manager:
            self.status_bar_widget.set_server_mode("builtin")

        # Bridge TTS signals
        self.bridge.tts_started.connect(
            lambda: self.activity_log.append_entry("TTS", "Speaking..."))
        self.bridge.tts_finished.connect(
            lambda: self.activity_log.append_entry("TTS", "Finished speaking"))
        self.bridge.tts_stopped.connect(
            lambda: self.activity_log.append_entry("TTS", "Stopped by user"))

        # TTS start/stop -> voice panel stop button visibility
        self.bridge.tts_started.connect(self.voice_panel.on_tts_started)
        self.bridge.tts_finished.connect(self.voice_panel.on_tts_finished)

        # Voice panel stop button -> bridge
        self.voice_panel.stop_requested.connect(self.bridge.stop_speaking)

        # Escape key shortcut to stop TTS
        self._stop_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._stop_shortcut.activated.connect(self.bridge.stop_speaking)

        # Talent sidebar import + configure + marketplace
        self.talent_sidebar.import_requested.connect(
            self._open_import_talent_dialog)
        self.talent_sidebar.configure_requested.connect(
            self._open_talent_config)
        self.talent_sidebar.marketplace_requested.connect(
            self._open_marketplace)

        # Notifications (only show when window hidden)
        self.bridge.notification_requested.connect(self._on_notification)

        # Reminder alert (dismissable dialog)
        self.bridge.reminder_fired.connect(self._on_reminder_fired)

    def _on_init_complete(self):
        """Called when TalonAssistant is fully initialized."""
        self.chat_view.add_system_message("Talon is ready! Type a command or toggle voice mode.")
        self.text_input.set_enabled(True)
        self.text_input.input_field.setFocus()

    def _on_init_error(self, error_msg):
        """Called if initialization fails."""
        self.chat_view.add_error_message(f"Initialization failed: {error_msg}")
        self.text_input.set_enabled(False)

    def _on_command_started(self, command):
        """Command submitted — show user bubble and disable input."""
        self.chat_view.add_user_message(command)
        self.text_input.set_enabled(False)

    def _on_command_response(self, command, response):
        """Command completed — show response and re-enable input."""
        if response:
            self.chat_view.add_assistant_message(response)
        else:
            self.chat_view.add_assistant_message("Done!")
        self.text_input.set_enabled(True)

    def _on_command_error(self, command, error_msg):
        """Command failed — show error and re-enable input."""
        self.chat_view.add_error_message(error_msg)
        self.text_input.set_enabled(True)

    def _toggle_activity_log(self):
        self.activity_log._toggle()

    def _toggle_sidebar(self):
        self.talent_sidebar.setVisible(not self.talent_sidebar.isVisible())

    # ── Settings ─────────────────────────────────────────────

    def _open_settings(self):
        from gui.dialogs.settings_dialog import SettingsDialog
        config_path = os.path.join(self.config_dir, "settings.json")
        try:
            with open(config_path, 'r') as f:
                current = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            current = {}

        dialog = SettingsDialog(current, config_path, self)
        dialog.settings_saved.connect(self.bridge.update_settings)
        dialog.exec()

    # ── LLM Server ────────────────────────────────────────────

    def _open_llm_setup(self):
        """Open the LLM Server configuration dialog."""
        from gui.dialogs.llm_setup_dialog import LLMSetupDialog

        config_path = os.path.join(self.config_dir, "settings.json")
        try:
            with open(config_path, 'r') as f:
                full_settings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            full_settings = {}

        llm_config = full_settings.get("llm", {})
        server_config = full_settings.get("llm_server", {})

        # Create server manager on-the-fly if one doesn't exist yet
        # (user started in external mode but wants to configure builtin)
        if self.bridge.server_manager is None:
            from core.llm_server import LLMServerManager
            manager = LLMServerManager(server_config)
            self.bridge.set_server_manager(manager)

        dialog = LLMSetupDialog(
            llm_config, server_config,
            server_manager=self.bridge.server_manager,
            config_path=config_path,
            parent=self,
        )
        dialog.settings_saved.connect(self._on_llm_settings_saved)
        dialog.exec()

    def _on_llm_settings_saved(self, combined):
        """Apply LLM settings changes at runtime."""
        if "llm" in combined:
            self.bridge.update_settings(combined)
            # Also hot-swap api_format on the LLM client
            if self.bridge.assistant:
                llm = self.bridge.assistant.llm
                llm.api_format = combined["llm"].get(
                    "api_format", llm.api_format)

        if "llm_server" in combined and self.bridge.server_manager:
            self.bridge.server_manager.update_config(combined["llm_server"])

        # Re-test LLM connection with new settings
        if self.bridge.assistant:
            try:
                connected = self.bridge.assistant.llm.test_connection()
                self.bridge.llm_status.emit(connected)
            except Exception:
                self.bridge.llm_status.emit(False)

    # ── Theme / Appearance ───────────────────────────────────

    def _set_theme(self, name):
        if self.theme_manager:
            self.theme_manager.set_theme(name)
            # Update radio buttons
            self.dark_theme_action.setChecked(name == "dark")
            self.light_theme_action.setChecked(name == "light")

    def _increase_font(self):
        if self.theme_manager:
            self.theme_manager.set_font_size(
                self.theme_manager.font_size + 1)

    def _decrease_font(self):
        if self.theme_manager:
            self.theme_manager.set_font_size(
                self.theme_manager.font_size - 1)

    def _reset_font(self):
        if self.theme_manager:
            self.theme_manager.set_font_size(
                self.theme_manager.DEFAULT_FONT_SIZE)

    # ── Chat History / Export ────────────────────────────────

    def _get_chat_store(self):
        if self.chat_store is None:
            from core.chat_store import ChatStore
            self.chat_store = ChatStore()
        return self.chat_store

    def _save_conversation(self):
        messages = self.chat_view.get_messages()
        if not messages:
            return
        store = self._get_chat_store()
        store.save_conversation(messages)
        self.chat_view.add_system_message("Conversation saved.")

    def _load_conversation(self):
        from gui.dialogs.chat_history_dialog import ChatHistoryDialog
        store = self._get_chat_store()
        dialog = ChatHistoryDialog(store, self)
        if dialog.exec():
            filepath = dialog.selected_filepath
            if filepath:
                messages = store.load_conversation(filepath)
                self.chat_view.load_messages(messages)

    def _export_conversation(self):
        messages = self.chat_view.get_messages()
        if not messages:
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Conversation", "",
            "Text Files (*.txt);;Markdown (*.md);;JSON (*.json)")
        if not filepath:
            return
        store = self._get_chat_store()
        if filepath.endswith('.md'):
            store.export_as_markdown(messages, filepath)
        elif filepath.endswith('.json'):
            store.save_conversation(messages, filepath)
        else:
            store.export_as_text(messages, filepath)
        self.chat_view.add_system_message(f"Exported to {os.path.basename(filepath)}")

    def _clear_chat(self):
        self.chat_view.clear_messages()

    # ── Import Talent ────────────────────────────────────────

    def _open_import_talent_dialog(self):
        from gui.dialogs.import_talent_dialog import ImportTalentDialog
        dialog = ImportTalentDialog(self)
        if dialog.exec():
            source_path = dialog.selected_filepath
            if source_path:
                try:
                    self.bridge.import_talent(source_path)
                    self.chat_view.add_system_message(
                        f"Talent imported from {os.path.basename(source_path)}")
                except Exception as e:
                    self.chat_view.add_error_message(
                        f"Failed to import talent: {e}")

    # ── Marketplace ─────────────────────────────────────────

    def _open_marketplace(self):
        """Open the talent marketplace dialog."""
        from core.marketplace import MarketplaceClient
        from gui.dialogs.marketplace_dialog import MarketplaceDialog

        client = MarketplaceClient()
        installed = client.get_installed_talent_names()

        # Also include built-in talent names so they show as "installed"
        if self.bridge.assistant:
            for t in self.bridge.assistant.talents:
                installed.add(t.name)

        dialog = MarketplaceDialog(client, installed_names=installed, parent=self)
        dialog.talent_installed.connect(self._on_marketplace_install)
        dialog.talent_removed.connect(self._on_marketplace_remove)
        dialog.exec()

    def _on_marketplace_install(self, filepath):
        """Called when the marketplace downloads a talent file."""
        try:
            self.bridge.import_talent(filepath)
            name = os.path.basename(filepath).replace('.py', '')
            self.chat_view.add_system_message(
                f"Installed talent from marketplace: {name}")
        except Exception as e:
            self.chat_view.add_error_message(
                f"Failed to load marketplace talent: {e}")

    def _on_marketplace_remove(self, talent_name):
        """Called when the marketplace removes a talent file."""
        self.bridge.uninstall_talent(talent_name)
        self.chat_view.add_system_message(
            f"Removed talent: {talent_name}")

    # ── Talent Config ─────────────────────────────────────────

    def _open_talent_config(self, talent_name):
        """Open the per-talent configuration dialog."""
        from gui.dialogs.talent_config_dialog import TalentConfigDialog
        talent = self.bridge.get_talent(talent_name)
        if talent is None:
            return
        schema = talent.get_config_schema()
        if not schema or not schema.get("fields"):
            self.chat_view.add_system_message(
                f"'{talent_name}' has no configurable settings.")
            return
        current_config = talent.talent_config
        dialog = TalentConfigDialog(
            talent_name, talent.name, schema, current_config, self,
            credential_store=self.bridge.credential_store)
        dialog.config_saved.connect(self.bridge.update_talent_config)
        dialog.exec()

    # ── Help / About ─────────────────────────────────────────

    def _open_help(self):
        from gui.dialogs.help_dialog import HelpDialog
        dialog = HelpDialog(parent=self)
        dialog.exec()

    def _open_shortcuts_help(self):
        from gui.dialogs.help_dialog import HelpDialog
        dialog = HelpDialog(parent=self, initial_topic="Keyboard Shortcuts")
        dialog.exec()

    def _open_about(self):
        from gui.dialogs.about_dialog import AboutDialog
        dialog = AboutDialog(parent=self)
        dialog.exec()

    # ── Notifications ────────────────────────────────────────

    def _on_notification(self, title, message):
        if self.system_tray and not self.isVisible():
            self.system_tray.show_notification(title, message)

    # ── Reminder alert (dismissable dialog) ────────────────────

    def _on_reminder_fired(self, reminder_id, title, message, default_snooze):
        """Show a dismissable reminder alert dialog."""
        from gui.dialogs.reminder_alert_dialog import ReminderAlertDialog

        dialog = ReminderAlertDialog(
            reminder_id, message, default_snooze, parent=self
        )
        dialog.dismissed.connect(self._on_reminder_dismissed)
        dialog.snoozed.connect(self._on_reminder_snoozed)

        # Restore window if minimized to tray
        if not self.isVisible():
            self.show()
            self.raise_()
            self.activateWindow()

        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_reminder_dismissed(self, reminder_id):
        """Reminder was dismissed — log to chat."""
        self.chat_view.add_system_message("Reminder dismissed.")

    def _on_reminder_snoozed(self, reminder_id, message, snooze_seconds):
        """Reminder was snoozed — re-arm via bridge."""
        self.bridge.snooze_reminder(reminder_id, message, snooze_seconds)
        minutes = snooze_seconds // 60
        self.chat_view.add_system_message(
            f"Reminder snoozed for {minutes} minute(s).")

    # ── Exit / Close ─────────────────────────────────────────

    def _force_exit(self):
        """Bypass tray minimize and actually quit."""
        self._minimize_to_tray = False
        self.close()

    def closeEvent(self, event):
        """Minimize to tray or clean up and close."""
        if (self._minimize_to_tray
                and self.system_tray
                and self.system_tray.tray_icon.isVisible()):
            event.ignore()
            self.hide()
            self.system_tray.show_notification(
                "Talon", "Minimized to system tray. Double-click to restore.")
        else:
            if self.system_tray:
                self.system_tray.cleanup()
            self.bridge.cleanup()
            event.accept()
