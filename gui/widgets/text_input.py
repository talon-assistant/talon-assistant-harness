import os
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLineEdit,
    QPushButton, QLabel, QFileDialog,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff"}


class TextInput(QWidget):
    """Text input bar with send button, file attachment, and command history.

    Signals:
        command_submitted(str, list): Emitted with (command_text, attachment_paths).
            attachment_paths is an empty list when no files are attached.
    """

    command_submitted = pyqtSignal(str, list)

    def __init__(self):
        super().__init__()
        self._history: list[str] = []
        self._history_index = -1
        self._pending_attachments: list[str] = []

        # Accept drag-and-drop of image files onto the widget.
        self.setAcceptDrops(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        # ── Row 1: attach button + input field + send button ─────────────
        row = QHBoxLayout()
        row.setContentsMargins(8, 4, 8, 4)

        self._attach_btn = QPushButton("📎")
        self._attach_btn.setFixedWidth(32)
        self._attach_btn.setToolTip("Attach image(s)")
        self._attach_btn.clicked.connect(self._open_file_dialog)
        row.addWidget(self._attach_btn)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a command...")
        self.input_field.returnPressed.connect(self._submit)
        self.input_field.installEventFilter(self)
        row.addWidget(self.input_field)

        self.send_button = QPushButton("Send")
        self.send_button.setFixedWidth(70)
        self.send_button.clicked.connect(self._submit)
        row.addWidget(self.send_button)

        outer.addLayout(row)

        # ── Row 2: attachment strip (hidden until files are attached) ─────
        strip_row = QHBoxLayout()
        strip_row.setContentsMargins(8, 0, 8, 4)

        self._attach_label = QLabel()
        self._attach_label.setStyleSheet("color: #888; font-size: 11px;")
        strip_row.addWidget(self._attach_label)

        self._clear_btn = QPushButton("✕")
        self._clear_btn.setFixedWidth(22)
        self._clear_btn.setFixedHeight(18)
        self._clear_btn.setStyleSheet(
            "font-size: 10px; padding: 0; border: none; color: #888;")
        self._clear_btn.setToolTip("Remove all attachments")
        self._clear_btn.clicked.connect(self._clear_attachments)
        strip_row.addWidget(self._clear_btn)
        strip_row.addStretch()

        self._attach_strip = QWidget()
        self._attach_strip.setLayout(strip_row)
        self._attach_strip.setVisible(False)
        outer.addWidget(self._attach_strip)

    # ── Attachment management ─────────────────────────────────────────────

    def _open_file_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Attach image(s)",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.gif *.tiff)",
        )
        if paths:
            self._add_attachments(paths)

    def _add_attachments(self, paths: list):
        for p in paths:
            if p not in self._pending_attachments:
                self._pending_attachments.append(p)
        self._update_strip()

    def _clear_attachments(self):
        self._pending_attachments.clear()
        self._update_strip()

    def _update_strip(self):
        if self._pending_attachments:
            names = ", ".join(os.path.basename(p) for p in self._pending_attachments)
            self._attach_label.setText(f"\U0001f4ce {names}")
            self._attach_strip.setVisible(True)
        else:
            self._attach_label.setText("")
            self._attach_strip.setVisible(False)

    # ── Submission ────────────────────────────────────────────────────────

    def _submit(self):
        text = self.input_field.text().strip()
        attachments = list(self._pending_attachments)

        # Require either text or attachments (or both).
        if not text and not attachments:
            return

        if text:
            self._history.append(text)
            self._history_index = len(self._history)

        self.command_submitted.emit(text, attachments)
        self.input_field.clear()
        self._clear_attachments()

    # ── Drag-and-drop ─────────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(
                os.path.splitext(u.toLocalFile())[1].lower() in _IMAGE_EXTS
                for u in urls
            ):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        paths = [
            u.toLocalFile()
            for u in event.mimeData().urls()
            if os.path.splitext(u.toLocalFile())[1].lower() in _IMAGE_EXTS
        ]
        if paths:
            self._add_attachments(paths)
            event.acceptProposedAction()

    # ── Misc ──────────────────────────────────────────────────────────────

    def set_enabled(self, enabled: bool):
        """Enable/disable input during command processing."""
        self.input_field.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
        self._attach_btn.setEnabled(enabled)
        if enabled:
            self.input_field.setFocus()

    def eventFilter(self, obj, event):
        """Handle Up/Down arrow keys for command history."""
        if obj == self.input_field and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key.Key_Up and self._history:
                self._history_index = max(0, self._history_index - 1)
                self.input_field.setText(self._history[self._history_index])
                return True
            elif event.key() == Qt.Key.Key_Down and self._history:
                self._history_index = min(
                    len(self._history), self._history_index + 1)
                if self._history_index < len(self._history):
                    self.input_field.setText(
                        self._history[self._history_index])
                else:
                    self.input_field.clear()
                return True
        return super().eventFilter(obj, event)
