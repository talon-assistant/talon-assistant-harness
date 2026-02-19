import json
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QWidget, QFormLayout, QLineEdit, QSpinBox,
                             QDoubleSpinBox, QCheckBox, QComboBox, QLabel,
                             QPushButton, QListWidget, QListWidgetItem,
                             QInputDialog)
from PyQt6.QtCore import pyqtSignal, Qt


class ListEditor(QWidget):
    """Editable list with Add/Remove buttons for string lists."""

    def __init__(self, items=None):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(120)
        if items:
            for item in items:
                self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_item)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_item)
        up_btn = QPushButton("\u25b2")  # ▲
        up_btn.setFixedWidth(28)
        up_btn.setToolTip("Move selected item up")
        up_btn.clicked.connect(self._move_up)
        down_btn = QPushButton("\u25bc")  # ▼
        down_btn.setFixedWidth(28)
        down_btn.setToolTip("Move selected item down")
        down_btn.clicked.connect(self._move_down)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addWidget(up_btn)
        btn_layout.addWidget(down_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _add_item(self):
        text, ok = QInputDialog.getText(self, "Add Item", "Value:")
        if ok and text.strip():
            self.list_widget.addItem(text.strip())

    def _remove_item(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.list_widget.takeItem(row)

    def _move_up(self):
        row = self.list_widget.currentRow()
        if row > 0:
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row - 1, item)
            self.list_widget.setCurrentRow(row - 1)

    def _move_down(self):
        row = self.list_widget.currentRow()
        if 0 <= row < self.list_widget.count() - 1:
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row + 1, item)
            self.list_widget.setCurrentRow(row + 1)

    def get_items(self):
        return [self.list_widget.item(i).text()
                for i in range(self.list_widget.count())]


class SettingsDialog(QDialog):
    """Tabbed settings editor for config/settings.json."""

    settings_saved = pyqtSignal(dict)

    # Keys that require an app restart to take effect
    RESTART_REQUIRED = {
        "whisper.model_size", "whisper.preferred_device",
        "whisper.preferred_compute_type", "whisper.fallback_device",
        "whisper.fallback_compute_type",
        "memory.embedding_model", "memory.db_path", "memory.chroma_path",
        "audio.sample_rate",
    }

    def __init__(self, current_settings, config_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(600, 500)
        self.resize(700, 550)
        self._config_path = config_path
        self._original = current_settings
        self._fields = {}  # dotted key -> widget

        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_llm_tab(), "LLM")
        self.tabs.addTab(self._build_audio_tab(), "Audio")
        self.tabs.addTab(self._build_voice_tab(), "Voice")
        self.tabs.addTab(self._build_whisper_tab(), "Whisper")
        self.tabs.addTab(self._build_memory_tab(), "Memory")
        self.tabs.addTab(self._build_desktop_tab(), "Desktop")
        layout.addWidget(self.tabs)

        # Warning label
        self.restart_label = QLabel("")
        self.restart_label.setObjectName("restart_warning")
        self.restart_label.setVisible(False)
        layout.addWidget(self.restart_label)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._on_save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    # ── Tab builders ─────────────────────────────────────────────

    def _build_llm_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        llm = self._original.get("llm", {})

        self._add_line("llm.endpoint", form, "API Endpoint",
                       llm.get("endpoint", ""))
        self._add_combo("llm.api_format", form, "API Format",
                        ["koboldcpp", "llamacpp", "openai"],
                        llm.get("api_format", "koboldcpp"))
        self._add_spin("llm.max_length", form, "Max Length",
                       llm.get("max_length", 512), 1, 8192)
        self._add_dspin("llm.temperature", form, "Temperature",
                        llm.get("temperature", 0.7), 0.0, 2.0, 0.05)
        self._add_dspin("llm.top_p", form, "Top P",
                        llm.get("top_p", 0.9), 0.0, 1.0, 0.05)
        self._add_dspin("llm.rep_pen", form, "Repetition Penalty",
                        llm.get("rep_pen", 1.1), 1.0, 3.0, 0.05)
        self._add_spin("llm.timeout", form, "Timeout (s)",
                       llm.get("timeout", 120), 5, 600)

        stop = llm.get("stop_sequences", [])
        self._add_list("llm.stop_sequences", form, "Stop Sequences", stop)

        # Prompt template fields
        pt = llm.get("prompt_template", {})
        self._add_line("llm.prompt_template.user_prefix", form,
                       "User Prefix", pt.get("user_prefix", ""))
        self._add_line("llm.prompt_template.user_suffix", form,
                       "User Suffix", pt.get("user_suffix", ""))
        self._add_line("llm.prompt_template.assistant_prefix", form,
                       "Assistant Prefix", pt.get("assistant_prefix", ""))
        self._add_line("llm.prompt_template.vision_prefix", form,
                       "Vision Prefix", pt.get("vision_prefix", ""))

        return w

    def _build_audio_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        audio = self._original.get("audio", {})

        self._add_spin("audio.sample_rate", form, "Sample Rate (Hz)",
                       audio.get("sample_rate", 16000), 8000, 48000)
        self._add_spin("audio.chunk_duration", form, "Chunk Duration (s)",
                       audio.get("chunk_duration", 3), 1, 30)
        self._add_spin("audio.command_duration", form, "Command Duration (s)",
                       audio.get("command_duration", 5), 1, 30)
        self._add_spin("audio.energy_threshold", form, "Energy Threshold",
                       audio.get("energy_threshold", 100), 0, 10000)
        self._add_spin("audio.variance_threshold", form, "Variance Threshold",
                       audio.get("variance_threshold", 1000), 0, 100000)

        noise = audio.get("noise_words", [])
        self._add_list("audio.noise_words", form, "Noise Words", noise)

        return w

    def _build_voice_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        voice = self._original.get("voice", {})

        self._add_line("voice.tts_voice", form, "TTS Voice",
                       voice.get("tts_voice", "en-US-AriaNeural"))

        ww = voice.get("wake_words", [])
        self._add_list("voice.wake_words", form, "Wake Words", ww)

        return w

    def _build_whisper_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        whisper = self._original.get("whisper", {})

        self._add_combo("whisper.model_size", form, "Model Size",
                        ["tiny", "base", "small", "medium", "large-v2"],
                        whisper.get("model_size", "small"))
        self._add_combo("whisper.preferred_device", form, "Preferred Device",
                        ["cuda", "cpu"],
                        whisper.get("preferred_device", "cuda"))
        self._add_combo("whisper.preferred_compute_type", form,
                        "Preferred Compute Type",
                        ["float16", "float32", "int8"],
                        whisper.get("preferred_compute_type", "float16"))
        self._add_combo("whisper.fallback_device", form, "Fallback Device",
                        ["cpu", "cuda"],
                        whisper.get("fallback_device", "cpu"))
        self._add_combo("whisper.fallback_compute_type", form,
                        "Fallback Compute Type",
                        ["int8", "float32", "float16"],
                        whisper.get("fallback_compute_type", "int8"))
        self._add_line("whisper.language", form, "Language",
                       whisper.get("language", "en"))

        note = QLabel("Changes to Whisper settings require an app restart.")
        note.setObjectName("restart_warning")
        form.addRow(note)

        return w

    def _build_memory_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        mem = self._original.get("memory", {})

        self._add_line("memory.db_path", form, "SQLite DB Path",
                       mem.get("db_path", "data/talon_memory.db"))
        self._add_line("memory.chroma_path", form, "ChromaDB Path",
                       mem.get("chroma_path", "data/chroma_db"))
        self._add_line("memory.embedding_model", form, "Embedding Model",
                       mem.get("embedding_model", "all-MiniLM-L6-v2"))

        docs = self._original.get("documents", {})
        self._add_line("documents.directory", form, "Documents Directory",
                       docs.get("directory", "documents"))

        note = QLabel("Changes to memory/embedding settings require an app restart.")
        note.setObjectName("restart_warning")
        form.addRow(note)

        return w

    def _build_desktop_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        desk = self._original.get("desktop", {})

        self._add_dspin("desktop.pyautogui_pause", form, "PyAutoGUI Pause (s)",
                        desk.get("pyautogui_pause", 0.5), 0.0, 5.0, 0.1)
        self._add_check("desktop.pyautogui_failsafe", form, "Failsafe Enabled",
                        desk.get("pyautogui_failsafe", True))
        self._add_dspin("desktop.action_delay", form, "Action Delay (s)",
                        desk.get("action_delay", 0.5), 0.0, 10.0, 0.1)
        self._add_dspin("desktop.app_launch_delay", form, "App Launch Delay (s)",
                        desk.get("app_launch_delay", 2.0), 0.0, 30.0, 0.5)

        return w

    # ── Field helpers ────────────────────────────────────────────

    def _add_line(self, key, form, label, value):
        field = QLineEdit(str(value))
        self._fields[key] = ("line", field)
        form.addRow(label, field)

    def _add_spin(self, key, form, label, value, mn, mx):
        field = QSpinBox()
        field.setRange(mn, mx)
        field.setValue(int(value))
        self._fields[key] = ("spin", field)
        form.addRow(label, field)

    def _add_dspin(self, key, form, label, value, mn, mx, step=0.1):
        field = QDoubleSpinBox()
        field.setRange(mn, mx)
        field.setSingleStep(step)
        field.setDecimals(2)
        field.setValue(float(value))
        self._fields[key] = ("dspin", field)
        form.addRow(label, field)

    def _add_check(self, key, form, label, value):
        field = QCheckBox()
        field.setChecked(bool(value))
        self._fields[key] = ("check", field)
        form.addRow(label, field)

    def _add_combo(self, key, form, label, options, value):
        field = QComboBox()
        field.addItems(options)
        idx = field.findText(str(value))
        if idx >= 0:
            field.setCurrentIndex(idx)
        self._fields[key] = ("combo", field)
        form.addRow(label, field)

    def _add_list(self, key, form, label, items):
        field = ListEditor(items)
        self._fields[key] = ("list", field)
        form.addRow(label, field)

    # ── Collect & save ───────────────────────────────────────────

    def _collect_values(self):
        """Read all widget values back into a settings dict."""
        result = {}
        for dotted_key, (kind, widget) in self._fields.items():
            parts = dotted_key.split(".")
            d = result
            for p in parts[:-1]:
                d = d.setdefault(p, {})

            if kind == "line":
                d[parts[-1]] = widget.text()
            elif kind == "spin":
                d[parts[-1]] = widget.value()
            elif kind == "dspin":
                d[parts[-1]] = round(widget.value(), 2)
            elif kind == "check":
                d[parts[-1]] = widget.isChecked()
            elif kind == "combo":
                d[parts[-1]] = widget.currentText()
            elif kind == "list":
                d[parts[-1]] = widget.get_items()

        return result

    def _changed_restart_keys(self, new_settings):
        """Return set of restart-required keys whose values changed."""
        changed = set()
        for dotted_key in self.RESTART_REQUIRED:
            parts = dotted_key.split(".")
            old_val = self._original
            new_val = new_settings
            for p in parts:
                old_val = old_val.get(p, {}) if isinstance(old_val, dict) else None
                new_val = new_val.get(p, {}) if isinstance(new_val, dict) else None
            if old_val != new_val:
                changed.add(dotted_key)
        return changed

    def _on_save(self):
        new_settings = self._collect_values()

        # Check for restart-required changes
        changed = self._changed_restart_keys(new_settings)
        if changed:
            self.restart_label.setText(
                "Some changes require an app restart to take effect.")
            self.restart_label.setVisible(True)

        # Persist to disk
        with open(self._config_path, 'w') as f:
            json.dump(new_settings, f, indent=2)

        self.settings_saved.emit(new_settings)
        self.accept()
