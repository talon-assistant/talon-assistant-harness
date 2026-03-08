import json
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QWidget, QFormLayout, QLineEdit, QSpinBox,
                             QDoubleSpinBox, QCheckBox, QComboBox, QLabel,
                             QPushButton, QListWidget, QListWidgetItem,
                             QInputDialog, QScrollArea, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView)
from PyQt6.QtCore import pyqtSignal, Qt


class ListEditor(QWidget):
    """Editable list with Add/Remove buttons for string lists."""

    def __init__(self, items=None):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.list_widget = QListWidget()
        self.list_widget.setMinimumHeight(60)
        self.list_widget.setMaximumHeight(120)
        if items:
            for item in items:
                self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)
        for label, slot in [("Add", self._add_item),
                            ("Remove", self._remove_item),
                            ("Up", self._move_up),
                            ("Down", self._move_down)]:
            btn = QPushButton(label)
            btn.setFixedHeight(28)
            btn.clicked.connect(slot)
            btn_layout.addWidget(btn)
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
        if row < 0 and self.list_widget.count() > 0:
            # Nothing selected — select the first item so user knows to pick one
            self.list_widget.setCurrentRow(0)
            return
        if row > 0:
            text = self.list_widget.item(row).text()
            self.list_widget.takeItem(row)
            self.list_widget.insertItem(row - 1, QListWidgetItem(text))
            self.list_widget.setCurrentRow(row - 1)

    def _move_down(self):
        row = self.list_widget.currentRow()
        if row < 0 and self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)
            return
        if 0 <= row < self.list_widget.count() - 1:
            text = self.list_widget.item(row).text()
            self.list_widget.takeItem(row)
            self.list_widget.insertItem(row + 1, QListWidgetItem(text))
            self.list_widget.setCurrentRow(row + 1)

    def get_items(self):
        return [self.list_widget.item(i).text()
                for i in range(self.list_widget.count())]


class AddFeedDialog(QDialog):
    """Small dialog to enter a new RSS feed entry."""

    CATEGORIES = ["General", "Technology", "Infosec", "Finance", "Science", "Sports"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add RSS Feed")
        self.setMinimumWidth(400)

        layout = QFormLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 12)

        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. Reuters Top News")
        self._url = QLineEdit()
        self._url.setPlaceholderText("https://feeds.example.com/rss.xml")
        self._category = QComboBox()
        self._category.addItems(self.CATEGORIES)

        layout.addRow("Name", self._name)
        layout.addRow("Feed URL", self._url)
        layout.addRow("Category", self._category)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("Add")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._validate)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addRow(btn_row)

    def _validate(self):
        if self._url.text().strip():
            self.accept()

    def get_feed(self) -> dict:
        url = self._url.text().strip()
        return {
            "name":     self._name.text().strip() or url,
            "url":      url,
            "category": self._category.currentText(),
            "enabled":  True,
        }


class FeedTableEditor(QWidget):
    """Table widget for managing RSS feed entries (enabled, name, category, URL)."""

    def __init__(self, feeds=None, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["✓", "Name", "Category", "URL"])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 32)
        self.table.setMinimumHeight(220)
        self.table.setMaximumHeight(340)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked |
            QAbstractItemView.EditTrigger.SelectedClicked)
        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)
        for label, slot in [("Add Feed", self._add_feed),
                             ("Remove", self._remove_feed)]:
            btn = QPushButton(label)
            btn.setFixedHeight(28)
            btn.clicked.connect(slot)
            btn_layout.addWidget(btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        for feed in (feeds or []):
            self._insert_row(feed)

    def _insert_row(self, feed: dict):
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Col 0 — enabled checkbox (centred)
        cb = QCheckBox()
        cb.setChecked(feed.get("enabled", True))
        cell = QWidget()
        cell_layout = QHBoxLayout(cell)
        cell_layout.addWidget(cb)
        cell_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cell_layout.setContentsMargins(0, 0, 0, 0)
        self.table.setCellWidget(row, 0, cell)

        # Cols 1-3
        for col, key in enumerate(("name", "category", "url"), start=1):
            item = QTableWidgetItem(feed.get(key, ""))
            self.table.setItem(row, col, item)

    def _add_feed(self):
        dlg = AddFeedDialog(parent=self)
        if dlg.exec():
            self._insert_row(dlg.get_feed())

    def _remove_feed(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def get_feeds(self) -> list[dict]:
        feeds = []
        for row in range(self.table.rowCount()):
            cell = self.table.cellWidget(row, 0)
            enabled = True
            if cell:
                cb = cell.findChild(QCheckBox)
                enabled = cb.isChecked() if cb else True

            def _text(col):
                item = self.table.item(row, col)
                return item.text().strip() if item else ""

            url = _text(3)
            if url:
                feeds.append({
                    "name":     _text(1) or url,
                    "category": _text(2) or "General",
                    "url":      url,
                    "enabled":  enabled,
                })
        return feeds


class AddTaskDialog(QDialog):
    """Dialog to create or edit a single scheduled task."""

    DAYS_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    DAYS_FULL  = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

    def __init__(self, task=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Scheduled Task" if task is None else "Edit Task")
        self.setMinimumWidth(440)

        layout = QFormLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 12)

        self._command = QLineEdit(task.get("command", "") if task else "")
        self._command.setPlaceholderText("e.g. generate morning news digest")

        self._time = QLineEdit(task.get("time", "07:00") if task else "07:00")
        self._time.setPlaceholderText("HH:MM  (24-hour)")
        self._time.setMaximumWidth(80)

        # Days — row of labelled checkboxes
        days_widget = QWidget()
        days_layout = QHBoxLayout(days_widget)
        days_layout.setContentsMargins(0, 0, 0, 0)
        days_layout.setSpacing(6)
        current_days = {d.lower()[:3] for d in task.get("days", self.DAYS_FULL)} \
            if task else set(self.DAYS_FULL)
        self._day_checks: list[tuple[str, QCheckBox]] = []
        for short, full in zip(self.DAYS_SHORT, self.DAYS_FULL):
            cb = QCheckBox(short)
            cb.setChecked(full in current_days)
            self._day_checks.append((full, cb))
            days_layout.addWidget(cb)
        days_layout.addStretch()

        self._enabled = QCheckBox()
        self._enabled.setChecked(task.get("enabled", True) if task else True)

        layout.addRow("Command", self._command)
        layout.addRow("Time (HH:MM)", self._time)
        layout.addRow("Days", days_widget)
        layout.addRow("Enabled", self._enabled)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("Save" if task else "Add")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._validate)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addRow(btn_row)

    def _validate(self):
        if self._command.text().strip():
            self.accept()

    def get_task(self) -> dict:
        days = [full for full, cb in self._day_checks if cb.isChecked()]
        return {
            "command": self._command.text().strip(),
            "time":    self._time.text().strip(),
            "days":    days or self.DAYS_FULL,
            "enabled": self._enabled.isChecked(),
        }


class SchedulerTableEditor(QWidget):
    """Table for managing scheduled Talon tasks."""

    DAYS_FULL = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

    def __init__(self, schedule=None, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["✓", "Command", "Time", "Days"])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(0, 32)
        self.table.setColumnWidth(2, 64)
        self.table.setMinimumHeight(100)
        self.table.setMaximumHeight(220)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(self._edit_task)
        layout.addWidget(self.table)

        lbl = QLabel("Double-click a row to edit.  Time uses 24-hour format (HH:MM).")
        lbl.setObjectName("settings_hint")
        layout.addWidget(lbl)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)
        for label, slot in [("Add Task", self._add_task),
                             ("Edit",     self._edit_task),
                             ("Remove",   self._remove_task)]:
            btn = QPushButton(label)
            btn.setFixedHeight(28)
            btn.clicked.connect(slot)
            btn_layout.addWidget(btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        for task in (schedule or []):
            self._insert_row(task)

    # ── helpers ───────────────────────────────────────────────────

    def _days_label(self, days: list) -> str:
        ds = {d.lower()[:3] for d in days}
        if ds == set(self.DAYS_FULL):
            return "Every day"
        if ds == {"mon", "tue", "wed", "thu", "fri"}:
            return "Weekdays"
        if ds == {"sat", "sun"}:
            return "Weekend"
        order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        return " ".join(d.capitalize() for d in order if d in ds)

    def _insert_row(self, task: dict, at: int = None):
        row = self.table.rowCount() if at is None else at
        self.table.insertRow(row)

        # Col 0 — enabled checkbox (centred)
        cb = QCheckBox()
        cb.setChecked(task.get("enabled", True))
        cell = QWidget()
        cl = QHBoxLayout(cell)
        cl.addWidget(cb)
        cl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cl.setContentsMargins(0, 0, 0, 0)
        self.table.setCellWidget(row, 0, cell)

        cmd_item = QTableWidgetItem(task.get("command", ""))
        cmd_item.setData(Qt.ItemDataRole.UserRole, task)   # stash full task
        self.table.setItem(row, 1, cmd_item)
        self.table.setItem(row, 2, QTableWidgetItem(task.get("time", "")))
        self.table.setItem(row, 3, QTableWidgetItem(
            self._days_label(task.get("days", self.DAYS_FULL))))

    def _current_task_at(self, row: int) -> dict:
        """Reconstruct the task dict from the current row state."""
        cmd_item = self.table.item(row, 1)
        stored   = dict(cmd_item.data(Qt.ItemDataRole.UserRole) or {}) if cmd_item else {}
        cell     = self.table.cellWidget(row, 0)
        cb       = cell.findChild(QCheckBox) if cell else None
        stored["enabled"] = cb.isChecked() if cb else stored.get("enabled", True)
        if cmd_item:
            stored["command"] = cmd_item.text()
        t_item = self.table.item(row, 2)
        if t_item:
            stored["time"] = t_item.text()
        return stored

    # ── slots ─────────────────────────────────────────────────────

    def _add_task(self):
        dlg = AddTaskDialog(parent=self)
        if dlg.exec():
            self._insert_row(dlg.get_task())

    def _edit_task(self):
        row = self.table.currentRow()
        if row < 0:
            return
        dlg = AddTaskDialog(task=self._current_task_at(row), parent=self)
        if dlg.exec():
            self.table.removeRow(row)
            self._insert_row(dlg.get_task(), at=row)
            self.table.setCurrentCell(row, 1)

    def _remove_task(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def get_schedule(self) -> list[dict]:
        schedule = []
        for row in range(self.table.rowCount()):
            task = self._current_task_at(row)
            if task.get("command"):
                schedule.append(task)
        return schedule


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
        self.tabs.addTab(self._build_scheduler_tab(), "Scheduler")
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

    def _scrollable(self, inner):
        """Wrap a widget in a QScrollArea so tabs with many rows stay usable."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        return scroll

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

        return self._scrollable(w)

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

        return self._scrollable(w)

    def _build_voice_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        voice = self._original.get("voice", {})

        self._add_line("voice.tts_voice", form, "TTS Voice",
                       voice.get("tts_voice", "en-US-AriaNeural"))

        ww = voice.get("wake_words", [])
        self._add_list("voice.wake_words", form, "Wake Words", ww)

        return self._scrollable(w)

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

    def _build_scheduler_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(8)

        lbl = QLabel(
            "Scheduled tasks run automatically at the configured time.\n"
            "The scheduler checks every 20 seconds; each task fires at most once per day."
        )
        lbl.setWordWrap(True)
        lbl.setObjectName("settings_hint")
        layout.addWidget(lbl)

        schedule = self._original.get("scheduler", [])
        # Filter out internal _note keys left from example JSON
        clean = [t for t in schedule if isinstance(t, dict) and "command" in t]
        editor = SchedulerTableEditor(clean)
        self._fields["scheduler"] = ("schedule", editor)
        layout.addWidget(editor)
        layout.addStretch()
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
            elif kind == "schedule":
                # Scheduler is a top-level list, not a nested key
                result["scheduler"] = widget.get_schedule()

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
