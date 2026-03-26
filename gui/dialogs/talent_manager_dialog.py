"""Talent Manager — view, edit, test, and delete user-created talents."""

import os

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QTextEdit, QSplitter,
    QMessageBox, QWidget, QLineEdit,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont


class TalentManagerDialog(QDialog):
    """Dialog for managing user-created talents.

    Lists all talents in talents/user/, lets the user view/edit source,
    enable/disable, test with a sample command, or delete.
    """

    talent_changed = pyqtSignal()  # Emitted when talents are modified

    def __init__(self, assistant, bridge, parent=None):
        super().__init__(parent)
        self._assistant = assistant
        self._bridge = bridge
        self._current_file = None
        self._user_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "talents", "user",
        )

        self.setWindowTitle("Talent Manager")
        self.setObjectName("talent_manager_dialog")
        self.setMinimumSize(800, 550)
        self.setModal(False)  # Non-modal so user can interact with Talon
        self._setup_ui()
        self._refresh_list()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # Splitter: talent list on left, editor on right
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left panel: talent list ─────────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("User Talents"))

        self._talent_list = QListWidget()
        self._talent_list.setObjectName("talent_manager_list")
        self._talent_list.currentItemChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self._talent_list)

        # List action buttons
        list_btns = QHBoxLayout()
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._on_delete)
        list_btns.addWidget(self._delete_btn)

        self._toggle_btn = QPushButton("Disable")
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.clicked.connect(self._on_toggle)
        list_btns.addWidget(self._toggle_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_list)
        list_btns.addWidget(refresh_btn)

        left_layout.addLayout(list_btns)
        splitter.addWidget(left)

        # ── Right panel: source editor + test ───────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Info bar
        self._info_label = QLabel("Select a talent to view its source.")
        self._info_label.setWordWrap(True)
        self._info_label.setObjectName("talent_manager_info")
        right_layout.addWidget(self._info_label)

        # Source editor
        self._editor = QTextEdit()
        self._editor.setObjectName("talent_manager_editor")
        mono = QFont("Consolas", 10)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._editor.setFont(mono)
        self._editor.setPlaceholderText("Talent source code will appear here.")
        right_layout.addWidget(self._editor, stretch=1)

        # Editor buttons
        editor_btns = QHBoxLayout()

        self._save_btn = QPushButton("Save Changes")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        editor_btns.addWidget(self._save_btn)

        self._reload_btn = QPushButton("Reload Talent")
        self._reload_btn.setEnabled(False)
        self._reload_btn.setToolTip(
            "Save changes and hot-reload this talent without restarting"
        )
        self._reload_btn.clicked.connect(self._on_save_and_reload)
        editor_btns.addWidget(self._reload_btn)

        editor_btns.addStretch()
        right_layout.addLayout(editor_btns)

        # Test section
        test_label = QLabel("Test command:")
        right_layout.addWidget(test_label)

        test_row = QHBoxLayout()
        self._test_input = QLineEdit()
        self._test_input.setObjectName("talent_manager_test_input")
        self._test_input.setPlaceholderText("Type a command to test this talent...")
        self._test_input.returnPressed.connect(self._on_test)
        test_row.addWidget(self._test_input)

        self._test_btn = QPushButton("Test")
        self._test_btn.setEnabled(False)
        self._test_btn.clicked.connect(self._on_test)
        test_row.addWidget(self._test_btn)

        right_layout.addLayout(test_row)

        # Test output
        self._test_output = QTextEdit()
        self._test_output.setObjectName("talent_manager_test_output")
        self._test_output.setReadOnly(True)
        self._test_output.setMaximumHeight(120)
        self._test_output.setFont(mono)
        self._test_output.setPlaceholderText("Test results will appear here.")
        right_layout.addWidget(self._test_output)

        splitter.addWidget(right)
        splitter.setSizes([250, 550])

        layout.addWidget(splitter)

    # ── List management ────────────────────────────────────────────────────

    def _refresh_list(self):
        self._talent_list.clear()
        self._current_file = None

        if not os.path.isdir(self._user_dir):
            return

        for fname in sorted(os.listdir(self._user_dir)):
            if not fname.endswith(".py") or fname.startswith("__"):
                continue

            talent_name = fname[:-3]  # strip .py

            # Check if talent is loaded and get its status
            talent_obj = None
            for t in self._assistant.talents:
                if t.name == talent_name:
                    talent_obj = t
                    break

            status = ""
            if talent_obj:
                status = "enabled" if talent_obj.enabled else "disabled"
            else:
                status = "not loaded"

            item = QListWidgetItem(f"{talent_name}  [{status}]")
            item.setData(Qt.ItemDataRole.UserRole, {
                "name": talent_name,
                "file": os.path.join(self._user_dir, fname),
                "loaded": talent_obj is not None,
                "enabled": talent_obj.enabled if talent_obj else False,
                "talent": talent_obj,
            })
            self._talent_list.addItem(item)

    def _on_selection_changed(self, current, previous):
        if current is None:
            self._editor.clear()
            self._info_label.setText("Select a talent to view its source.")
            self._delete_btn.setEnabled(False)
            self._toggle_btn.setEnabled(False)
            self._save_btn.setEnabled(False)
            self._reload_btn.setEnabled(False)
            self._test_btn.setEnabled(False)
            self._current_file = None
            return

        data = current.data(Qt.ItemDataRole.UserRole)
        self._current_file = data["file"]
        talent = data.get("talent")

        # Update info
        info_parts = [f"Name: {data['name']}"]
        if talent:
            info_parts.append(f"Description: {talent.description}")
            info_parts.append(f"Priority: {talent.priority}")
            info_parts.append(f"Keywords: {', '.join(talent.keywords[:5])}")
        info_parts.append(f"File: {data['file']}")
        self._info_label.setText("\n".join(info_parts))

        # Load source
        try:
            with open(data["file"], "r", encoding="utf-8") as f:
                self._editor.setPlainText(f.read())
        except Exception as e:
            self._editor.setPlainText(f"# Error reading file: {e}")

        # Update buttons
        self._delete_btn.setEnabled(True)
        self._toggle_btn.setEnabled(data["loaded"])
        self._toggle_btn.setText(
            "Disable" if data["enabled"] else "Enable"
        )
        self._save_btn.setEnabled(True)
        self._reload_btn.setEnabled(True)
        self._test_btn.setEnabled(True)

        # Pre-fill test input with first example if available
        if talent and talent.examples:
            self._test_input.setPlaceholderText(
                f'e.g. "{talent.examples[0]}"'
            )

    # ── Actions ────────────────────────────────────────────────────────────

    def _on_delete(self):
        current = self._talent_list.currentItem()
        if not current:
            return
        data = current.data(Qt.ItemDataRole.UserRole)
        name = data["name"]

        reply = QMessageBox.question(
            self, "Delete Talent",
            f"Delete talent '{name}'?\n\nThis will remove the file and "
            "unload the talent. This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Unload from assistant
        if data["loaded"]:
            self._bridge.uninstall_talent(name)

        # Delete file
        try:
            os.remove(data["file"])
            print(f"   [TalentManager] Deleted {data['file']}")
        except Exception as e:
            print(f"   [TalentManager] Delete failed: {e}")

        self._refresh_list()
        self._editor.clear()
        self._info_label.setText("Talent deleted.")
        self.talent_changed.emit()

    def _on_toggle(self):
        current = self._talent_list.currentItem()
        if not current:
            return
        data = current.data(Qt.ItemDataRole.UserRole)
        if not data["loaded"]:
            return

        new_state = not data["enabled"]
        self._bridge.toggle_talent(data["name"], new_state)
        self._refresh_list()
        self.talent_changed.emit()

    def _on_save(self):
        if not self._current_file:
            return
        try:
            code = self._editor.toPlainText()
            with open(self._current_file, "w", encoding="utf-8") as f:
                f.write(code)
            self._info_label.setText(
                self._info_label.text().split("\n")[0]
                + "\n** Saved successfully. **"
            )
        except Exception as e:
            self._info_label.setText(f"Save failed: {e}")

    def _on_save_and_reload(self):
        if not self._current_file:
            return

        # Save first
        self._on_save()

        # Hot-reload
        if hasattr(self._assistant, "load_user_talent"):
            result = self._assistant.load_user_talent(self._current_file)
            if result.get("success"):
                self._info_label.setText(
                    f"Saved and reloaded '{result['name']}' successfully."
                )
                self._bridge._emit_full_talent_list()
                self._refresh_list()
                self.talent_changed.emit()
            else:
                self._info_label.setText(
                    f"Saved but reload failed: {result.get('error', 'unknown')}"
                )
        else:
            self._info_label.setText("Saved. Restart Talon to reload.")

    def _on_test(self):
        command = self._test_input.text().strip()
        if not command:
            return

        current = self._talent_list.currentItem()
        if not current:
            return
        data = current.data(Qt.ItemDataRole.UserRole)
        talent = data.get("talent")

        self._test_output.clear()

        if not talent:
            self._test_output.setPlainText("Talent not loaded. Save and reload first.")
            return

        if not talent.enabled:
            self._test_output.setPlainText("Talent is disabled. Enable it first.")
            return

        # Check if talent would handle this command
        can_handle = talent.can_handle(command)
        self._test_output.append(f"can_handle('{command}'): {can_handle}\n")

        if not can_handle:
            self._test_output.append(
                "This talent would NOT match this command. "
                "Try one of its keywords or examples."
            )
            return

        # Execute the talent directly
        try:
            context = {
                "llm": self._assistant.llm,
                "memory": self._assistant.memory,
                "vision": getattr(self._assistant, "vision", None),
                "voice": getattr(self._assistant, "voice", None),
                "config": self._assistant.config,
                "assistant": self._assistant,
                "speak_response": False,
                "command_source": "test",
            }
            result = talent.execute(command, context)
            self._test_output.append(f"Success: {result.get('success')}")
            self._test_output.append(f"Response:\n{result.get('response', '')}")
        except Exception as e:
            self._test_output.append(f"EXCEPTION: {e}")
