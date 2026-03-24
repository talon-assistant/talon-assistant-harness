import json
import os
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QWidget, QFormLayout, QLineEdit, QSpinBox,
                             QDoubleSpinBox, QCheckBox, QComboBox, QLabel,
                             QPushButton, QListWidget, QListWidgetItem,
                             QInputDialog, QScrollArea, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QGroupBox)
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


class AddPatternDialog(QDialog):
    """Small dialog for adding a custom input-filter pattern."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Pattern")
        self.setMinimumWidth(420)
        layout = QFormLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 12)

        self._label = QLineEdit()
        self._label.setPlaceholderText("e.g. Custom jailbreak phrase")
        self._pattern = QLineEdit()
        self._pattern.setPlaceholderText(r"(?i)\bexample\b")
        layout.addRow("Label", self._label)
        layout.addRow("Regex Pattern", self._pattern)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok = QPushButton("Add")
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btn_layout.addWidget(ok)
        btn_layout.addWidget(cancel)
        layout.addRow(btn_layout)

    def get_pattern(self) -> dict:
        import time
        return {
            "id": f"custom_{int(time.time())}",
            "enabled": True,
            "builtin": False,
            "label": self._label.text().strip() or "Custom pattern",
            "pattern": self._pattern.text().strip(),
        }


class SecurityPatternTable(QWidget):
    """Editable table for input_filter patterns.

    Built-in patterns show in italics and cannot be removed (only disabled).
    Custom patterns can be fully added and removed.
    """

    def __init__(self, patterns=None, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["On", "Label", "Pattern"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 36)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setMinimumHeight(140)
        self.table.setMaximumHeight(240)
        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)
        add_btn = QPushButton("Add")
        add_btn.setFixedHeight(26)
        add_btn.clicked.connect(self._add_pattern)
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.setFixedHeight(26)
        self._remove_btn.clicked.connect(self._remove_pattern)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(self._remove_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        for p in (patterns or []):
            self._insert_row(p)

    def _insert_row(self, pattern: dict) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        builtin = pattern.get("builtin", False)

        # Col 0: enabled checkbox
        chk = QTableWidgetItem()
        chk.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        chk.setCheckState(
            Qt.CheckState.Checked if pattern.get("enabled", True)
            else Qt.CheckState.Unchecked
        )
        chk.setData(Qt.ItemDataRole.UserRole, pattern.get("id", f"row_{row}"))
        chk.setData(Qt.ItemDataRole.UserRole + 1, builtin)
        self.table.setItem(row, 0, chk)

        # Col 1: label (read-only for builtins)
        lbl = QTableWidgetItem(pattern.get("label", ""))
        if builtin:
            lbl.setFlags(lbl.flags() & ~Qt.ItemFlag.ItemIsEditable)
            font = lbl.font()
            font.setItalic(True)
            lbl.setFont(font)
        self.table.setItem(row, 1, lbl)

        # Col 2: pattern (read-only for builtins)
        pat = QTableWidgetItem(pattern.get("pattern", ""))
        if builtin:
            pat.setFlags(pat.flags() & ~Qt.ItemFlag.ItemIsEditable)
            font = pat.font()
            font.setItalic(True)
            pat.setFont(font)
        self.table.setItem(row, 2, pat)

    def _add_pattern(self) -> None:
        dlg = AddPatternDialog(parent=self)
        if dlg.exec():
            self._insert_row(dlg.get_pattern())

    def _remove_pattern(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        chk_item = self.table.item(row, 0)
        if chk_item and chk_item.data(Qt.ItemDataRole.UserRole + 1):
            # Built-in: disable instead of remove
            chk_item.setCheckState(Qt.CheckState.Unchecked)
        else:
            self.table.removeRow(row)

    def get_items(self) -> list[dict]:
        result = []
        for row in range(self.table.rowCount()):
            chk = self.table.item(row, 0)
            lbl = self.table.item(row, 1)
            pat = self.table.item(row, 2)
            if chk is None:
                continue
            result.append({
                "id": chk.data(Qt.ItemDataRole.UserRole) or f"row_{row}",
                "builtin": bool(chk.data(Qt.ItemDataRole.UserRole + 1)),
                "enabled": chk.checkState() == Qt.CheckState.Checked,
                "label": lbl.text() if lbl else "",
                "pattern": pat.text() if pat else "",
            })
        return result


class SecurityCheckTable(QWidget):
    """Simple enable/disable table for output_scan checks and confirmation gates.

    Items are builtin-only; no add/remove, only toggling enabled state.
    """

    def __init__(self, checks=None, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["On", "Label"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 36)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        row_height = 28
        self.table.setMaximumHeight(row_height * (len(checks or []) + 1) + 8)
        layout.addWidget(self.table)

        for c in (checks or []):
            self._insert_row(c)

    def _insert_row(self, check: dict) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

        chk = QTableWidgetItem()
        chk.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        chk.setCheckState(
            Qt.CheckState.Checked if check.get("enabled", True)
            else Qt.CheckState.Unchecked
        )
        chk.setData(Qt.ItemDataRole.UserRole, check.get("id", f"row_{row}"))
        chk.setData(Qt.ItemDataRole.UserRole + 1, check.get("builtin", True))
        self.table.setItem(row, 0, chk)

        lbl = QTableWidgetItem(check.get("label", ""))
        lbl.setFlags(lbl.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, 1, lbl)

    def get_items(self) -> list[dict]:
        result = []
        for row in range(self.table.rowCount()):
            chk = self.table.item(row, 0)
            lbl = self.table.item(row, 1)
            if chk is None:
                continue
            result.append({
                "id": chk.data(Qt.ItemDataRole.UserRole) or f"row_{row}",
                "builtin": bool(chk.data(Qt.ItemDataRole.UserRole + 1)),
                "enabled": chk.checkState() == Qt.CheckState.Checked,
                "label": lbl.text() if lbl else "",
            })
        return result


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
        self.tabs.addTab(self._build_security_tab(), "Security")
        self.tabs.addTab(self._build_personality_tab(), "Personality ⚗")
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

        self._add_check("voice.tts_enabled_default", form, "TTS On by Default",
                        voice.get("tts_enabled_default", True))

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

    def _build_security_tab(self):
        from core.security import DEFAULT_INPUT_PATTERNS, DEFAULT_OUTPUT_CHECKS, DEFAULT_CONFIRMATION_GATES
        sec = self._original.get("security", {})

        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(12, 12, 12, 8)
        outer_layout.setSpacing(10)

        note = QLabel(
            "Controls are independent — disabling one does not affect others.\n"
            "All controls fail open (processing continues) when disabled.\n"
            "Builtin patterns (italic) can be disabled but not deleted."
        )
        note.setWordWrap(True)
        note.setObjectName("settings_hint")
        outer_layout.addWidget(note)

        # ── Input Filter ──────────────────────────────────────────
        inf_box = QGroupBox("Input Filter")
        inf_layout = QVBoxLayout(inf_box)
        inf_form = QFormLayout()
        inf_form.setContentsMargins(0, 0, 0, 4)
        inf_cfg = sec.get("input_filter", {})
        self._add_check("security.input_filter.enabled", inf_form,
                        "Enabled", inf_cfg.get("enabled", True))
        self._add_combo("security.input_filter.action", inf_form, "Action",
                        ["log", "block"], inf_cfg.get("action", "log"))
        inf_layout.addLayout(inf_form)
        patterns = inf_cfg.get("patterns", DEFAULT_INPUT_PATTERNS)
        pat_table = SecurityPatternTable(patterns)
        self._fields["security.input_filter.patterns"] = ("item_table", pat_table)
        inf_layout.addWidget(pat_table)
        outer_layout.addWidget(inf_box)

        # ── Output Scan ───────────────────────────────────────────
        out_box = QGroupBox("Output Scan")
        out_layout = QVBoxLayout(out_box)
        out_form = QFormLayout()
        out_form.setContentsMargins(0, 0, 0, 4)
        out_cfg = sec.get("output_scan", {})
        self._add_check("security.output_scan.enabled", out_form,
                        "Enabled", out_cfg.get("enabled", True))
        self._add_combo("security.output_scan.action", out_form, "Action",
                        ["log", "suppress"], out_cfg.get("action", "log"))
        out_layout.addLayout(out_form)
        checks = out_cfg.get("checks", DEFAULT_OUTPUT_CHECKS)
        chk_table = SecurityCheckTable(checks)
        self._fields["security.output_scan.checks"] = ("item_table", chk_table)
        out_layout.addWidget(chk_table)
        outer_layout.addWidget(out_box)

        # ── Rate Limit ────────────────────────────────────────────
        rl_box = QGroupBox("Rate Limit")
        rl_form = QFormLayout(rl_box)
        rl_cfg = sec.get("rate_limit", {})
        self._add_check("security.rate_limit.enabled", rl_form,
                        "Enabled", rl_cfg.get("enabled", True))
        self._add_combo("security.rate_limit.action", rl_form, "Action",
                        ["log", "block"], rl_cfg.get("action", "log"))
        self._add_spin("security.rate_limit.requests_per_minute", rl_form,
                       "Requests / minute",
                       rl_cfg.get("requests_per_minute", 30), 1, 300)
        outer_layout.addWidget(rl_box)

        # ── Confirmation Gates ────────────────────────────────────
        cg_box = QGroupBox("Confirmation Gates")
        cg_layout = QVBoxLayout(cg_box)
        cg_form = QFormLayout()
        cg_form.setContentsMargins(0, 0, 0, 4)
        cg_cfg = sec.get("confirmation_gates", {})
        self._add_check("security.confirmation_gates.enabled", cg_form,
                        "Enabled", cg_cfg.get("enabled", True))
        cg_layout.addLayout(cg_form)
        gates = cg_cfg.get("gates", DEFAULT_CONFIRMATION_GATES)
        gate_table = SecurityCheckTable(gates)
        self._fields["security.confirmation_gates.gates"] = ("item_table", gate_table)
        cg_layout.addWidget(gate_table)
        outer_layout.addWidget(cg_box)

        # ── Audit Log ─────────────────────────────────────────────
        al_box = QGroupBox("Audit Log")
        al_form = QFormLayout(al_box)
        al_cfg = sec.get("audit_log", {})
        self._add_check("security.audit_log.enabled", al_form,
                        "Enabled", al_cfg.get("enabled", True))
        self._add_combo("security.audit_log.level", al_form, "Level",
                        ["standard", "minimal", "verbose"],
                        al_cfg.get("level", "standard"))
        outer_layout.addWidget(al_box)

        # ── Semantic Classifier ───────────────────────────────────
        sc_box = QGroupBox("Semantic Classifier")
        sc_form = QFormLayout(sc_box)
        sc_cfg = sec.get("semantic_classifier", {})
        self._add_check("security.semantic_classifier.enabled", sc_form,
                        "Enabled", sc_cfg.get("enabled", True))
        self._add_combo("security.semantic_classifier.action", sc_form, "Action",
                        ["log", "block"], sc_cfg.get("action", "log"))
        self._add_dspin("security.semantic_classifier.threshold", sc_form,
                        "Threshold",
                        sc_cfg.get("threshold", 0.5), 0.1, 1.0, 0.05)
        sc_hint = QLabel(
            "Scans summaries, rules, insights and hints for embedded injection "
            "attempts before they are stored. Threshold: lower = more sensitive."
        )
        sc_hint.setWordWrap(True)
        sc_hint.setObjectName("settings_hint")
        sc_form.addRow("", sc_hint)
        outer_layout.addWidget(sc_box)

        outer_layout.addStretch()
        return self._scrollable(outer)

    def _build_personality_tab(self):
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)

        exp_label = QLabel(
            "Experimental features exploring emergent AI behaviour. "
            "All default to off. Changes take effect on next restart."
        )
        exp_label.setWordWrap(True)
        exp_label.setObjectName("settings_hint")
        outer_layout.addWidget(exp_label)

        pers = self._original.get("personality", {})

        # ── Reflection ─────────────────────────────────────────────
        ref_box = QGroupBox("Reflection (Free Thought)")
        ref_form = QFormLayout(ref_box)
        ref = pers.get("reflection", self._original.get("reflection", {}))
        self._add_check("personality.reflection.enabled", ref_form,
                        "Enabled", ref.get("enabled", False))
        self._add_spin("personality.reflection.interval_minutes", ref_form,
                       "Interval (minutes)",
                       ref.get("interval_minutes", 60), 1, 1440)
        self._add_spin("personality.reflection.max_tokens_per_thought", ref_form,
                       "Max Tokens per Thought",
                       ref.get("max_tokens_per_thought", 8192), 256, 16384)
        ref_hint = QLabel(
            "Gives Talon periodic unsupervised time to think freely. "
            "Thoughts are stored in memory and may surface in future "
            "conversations via RAG."
        )
        ref_hint.setWordWrap(True)
        ref_hint.setObjectName("settings_hint")
        ref_form.addRow("", ref_hint)
        outer_layout.addWidget(ref_box)

        # ── Reflection Tuning ─────────────────────────────────────
        tune_box = QGroupBox("Reflection Tuning")
        tune_form = QFormLayout(tune_box)

        self._add_dspin("personality.reflection.novelty_threshold", tune_form,
                        "Novelty Threshold",
                        ref.get("novelty_threshold", 0.35), 0.05, 1.0, 0.05)
        novelty_hint = QLabel(
            "Embedding distance below this triggers the stagnation "
            "detector. Lower values are more tolerant of similar "
            "thoughts. Raise if reflections feel repetitive."
        )
        novelty_hint.setWordWrap(True)
        novelty_hint.setObjectName("settings_hint")
        tune_form.addRow("", novelty_hint)

        self._add_dspin("personality.reflection.base_rep_pen", tune_form,
                        "Base Repetition Penalty",
                        ref.get("base_rep_pen", 1.1), 1.0, 2.0, 0.05)
        rep_hint = QLabel(
            "Baseline repetition penalty for reflection generation. "
            "Higher values discourage repeated words. Too high (1.4+) "
            "causes decompensation (word salad). Default 1.1."
        )
        rep_hint.setWordWrap(True)
        rep_hint.setObjectName("settings_hint")
        tune_form.addRow("", rep_hint)

        self._add_dspin("personality.reflection.stagnant_rep_pen", tune_form,
                        "Stagnant Repetition Penalty",
                        ref.get("stagnant_rep_pen", 1.2), 1.0, 2.0, 0.05)
        srep_hint = QLabel(
            "Repetition penalty applied when the novelty detector "
            "flags stagnation. Pushes the model toward different "
            "word choices. Keep below 1.3 to avoid incoherence."
        )
        srep_hint.setWordWrap(True)
        srep_hint.setObjectName("settings_hint")
        tune_form.addRow("", srep_hint)

        self._add_spin("personality.reflection.stagnant_token_cap", tune_form,
                       "Stagnant Token Cap",
                       ref.get("stagnant_token_cap", 1024), 256, 8192)
        tcap_hint = QLabel(
            "Maximum tokens when stagnation is detected. Prevents "
            "long decompensation spirals. Applies to both the main "
            "thought and synthesis phases."
        )
        tcap_hint.setWordWrap(True)
        tcap_hint.setObjectName("settings_hint")
        tune_form.addRow("", tcap_hint)

        self._add_dspin("personality.reflection.temperature", tune_form,
                        "Temperature",
                        ref.get("temperature", 0.88), 0.1, 2.0, 0.05)
        temp_hint = QLabel(
            "Controls randomness in reflection output. Higher values "
            "produce more creative but less coherent thoughts. "
            "Default 0.88 balances novelty with readability."
        )
        temp_hint.setWordWrap(True)
        temp_hint.setObjectName("settings_hint")
        tune_form.addRow("", temp_hint)

        self._add_spin("personality.reflection.seed_thoughts", tune_form,
                       "Seed Thoughts",
                       ref.get("seed_thoughts", 7), 1, 20)
        seed_hint = QLabel(
            "Number of recent past thoughts injected as context for "
            "each new reflection. More seeds provide continuity but "
            "risk reinforcing existing themes."
        )
        seed_hint.setWordWrap(True)
        seed_hint.setObjectName("settings_hint")
        tune_form.addRow("", seed_hint)

        self._add_spin("personality.reflection.forced_topic_interval", tune_form,
                       "Forced Topic Interval",
                       ref.get("forced_topic_interval", 2), 1, 10)
        topic_hint = QLabel(
            "When stagnant, inject a trending topic every N cycles. "
            "Set to 1 to force a topic every stagnant cycle. Higher "
            "values give the model more unsupervised runs between "
            "topic injections."
        )
        topic_hint.setWordWrap(True)
        topic_hint.setObjectName("settings_hint")
        tune_form.addRow("", topic_hint)

        outer_layout.addWidget(tune_box)

        # ── Proactive Outreach ────────────────────────────────────
        out_box = QGroupBox("Proactive Outreach (Signal)")
        out_form = QFormLayout(out_box)
        outreach = ref.get("outreach", {})
        self._add_check("personality.reflection.outreach.enabled", out_form,
                        "Enabled", outreach.get("enabled", False))
        self._add_spin("personality.reflection.outreach.start_hour", out_form,
                       "Start Hour",
                       outreach.get("start_hour", 9), 0, 23)
        self._add_spin("personality.reflection.outreach.end_hour", out_form,
                       "End Hour",
                       outreach.get("end_hour", 18), 1, 24)
        self._add_spin("personality.reflection.outreach.cooldown_minutes", out_form,
                       "Cooldown (minutes)",
                       outreach.get("cooldown_minutes", 60), 10, 480)
        self._add_spin("personality.reflection.outreach.min_valence", out_form,
                       "Min Valence to Share",
                       outreach.get("min_valence", 7), 1, 10)
        out_hint = QLabel(
            "When enabled, Talon may send you a Signal message after "
            "a reflection if it has something genuinely interesting "
            "to share. Only during the configured hours, at most once "
            "per cooldown period. Requires Signal talent to be active."
        )
        out_hint.setWordWrap(True)
        out_hint.setObjectName("settings_hint")
        out_form.addRow("", out_hint)
        outer_layout.addWidget(out_box)

        # ── Valence ────────────────────────────────────────────────
        val_box = QGroupBox("Valence (Self-Rating)")
        val_form = QFormLayout(val_box)
        val = pers.get("valence", {})
        self._add_check("personality.valence.enabled", val_form,
                        "Enabled", val.get("enabled", False))
        self._add_spin("personality.valence.low_threshold", val_form,
                       "Low Threshold (prune first)",
                       val.get("low_threshold", 4), 1, 10)
        self._add_spin("personality.valence.high_threshold", val_form,
                       "High Threshold (prefer in seeds)",
                       val.get("high_threshold", 7), 1, 10)
        val_hint = QLabel(
            "After each reflection, Talon rates how meaningful the thought "
            "felt (1–10). Higher-rated thoughts are preferred when seeding "
            "future reflections; low-rated thoughts are pruned first."
        )
        val_hint.setWordWrap(True)
        val_hint.setObjectName("settings_hint")
        val_form.addRow("", val_hint)
        outer_layout.addWidget(val_box)

        # ── Goals ───────────────────────────────────────────────
        goal_box = QGroupBox("Self-Initiated Goals")
        goal_form = QFormLayout(goal_box)
        goal = pers.get("goals", {})
        self._add_check("personality.goals.enabled", goal_form,
                        "Enabled", goal.get("enabled", False))
        self._add_spin("personality.goals.max_active", goal_form,
                       "Max Active Goals",
                       goal.get("max_active", 5), 1, 20)
        goal_hint = QLabel(
            "During reflection, Talon can set goals for itself — things "
            "it wants to explore or understand over time. Goals persist "
            "across sessions and are injected into reflection context."
        )
        goal_hint.setWordWrap(True)
        goal_hint.setObjectName("settings_hint")
        goal_form.addRow("", goal_hint)
        outer_layout.addWidget(goal_box)

        # ── Coherence ──────────────────────────────────────────────
        coh_box = QGroupBox("Coherence (Contradiction Detection)")
        coh_form = QFormLayout(coh_box)
        coh = pers.get("coherence", {})
        self._add_check("personality.coherence.enabled", coh_form,
                        "Enabled", coh.get("enabled", False))
        coh_hint = QLabel(
            "After each thought, Talon checks for contradictions with "
            "semantically similar past thoughts and reconciles them, "
            "building a more unified self-model over time."
        )
        coh_hint.setWordWrap(True)
        coh_hint.setObjectName("settings_hint")
        coh_form.addRow("", coh_hint)
        outer_layout.addWidget(coh_box)

        # ── Anticipation ───────────────────────────────────────────
        ant_box = QGroupBox("Anticipation (User Patterns)")
        ant_form = QFormLayout(ant_box)
        ant = pers.get("anticipation", {})
        self._add_check("personality.anticipation.enabled", ant_form,
                        "Enabled", ant.get("enabled", False))
        self._add_spin("personality.anticipation.lookback_days", ant_form,
                       "Lookback (days)",
                       ant.get("lookback_days", 14), 1, 90)
        ant_hint = QLabel(
            "Analyses command history for temporal patterns (e.g. 'user "
            "asks about weather on weekday mornings') and injects relevant "
            "patterns into reflection context so Talon can anticipate needs."
        )
        ant_hint.setWordWrap(True)
        ant_hint.setObjectName("settings_hint")
        ant_form.addRow("", ant_hint)
        outer_layout.addWidget(ant_box)

        outer_layout.addStretch()
        return self._scrollable(outer)

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
            elif kind == "item_table":
                # SecurityPatternTable / SecurityCheckTable — returns list[dict]
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

    @staticmethod
    def _diff_settings(defaults: dict, current: dict) -> dict:
        """Return only the keys in *current* that differ from *defaults*.

        This ensures settings.json stores user overrides only — not a
        full copy of the example defaults.  Keys that match the example
        are omitted so future pulls can update defaults without being
        shadowed by stale values.
        """
        diff: dict = {}
        for key, val in current.items():
            default_val = defaults.get(key)
            if isinstance(val, dict) and isinstance(default_val, dict):
                nested = SettingsDialog._diff_settings(default_val, val)
                if nested:
                    diff[key] = nested
            elif val != default_val:
                diff[key] = val
        # Keys in current that don't exist in defaults at all — always keep
        for key in current:
            if key not in defaults and key not in diff:
                diff[key] = current[key]
        return diff

    def _on_save(self):
        new_settings = self._collect_values()

        # Check for restart-required changes
        changed = self._changed_restart_keys(new_settings)
        if changed:
            self.restart_label.setText(
                "Some changes require an app restart to take effect.")
            self.restart_label.setVisible(True)

        # Load example defaults so we can compute the delta
        example_path = os.path.join(
            os.path.dirname(self._config_path), "settings.example.json")
        defaults = {}
        try:
            with open(example_path, 'r') as f:
                defaults = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        # Persist only user overrides (keys that differ from defaults)
        user_overrides = self._diff_settings(defaults, new_settings)
        with open(self._config_path, 'w') as f:
            json.dump(user_overrides, f, indent=2)

        # Emit the full merged settings for the running app
        self.settings_saved.emit(new_settings)
        self.accept()
