"""Dialog for reviewing and managing Talon's free-thought reflections."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QTextBrowser,
    QPushButton, QLabel, QMessageBox,
)
from PyQt6.QtCore import Qt


class FreeThoughtsDialog(QDialog):
    """Browse, read, and delete Talon's stored free thoughts."""

    def __init__(self, memory, parent=None):
        super().__init__(parent)
        self._memory = memory
        self._thoughts: list[dict] = []

        self.setWindowTitle("Talon's Thoughts")
        self.setMinimumSize(720, 480)
        self.resize(860, 560)

        layout = QVBoxLayout(self)

        # ── header ───────────────────────────────────────────────
        header = QLabel(
            "Free thoughts generated during Talon's unsupervised reflection time."
        )
        header.setStyleSheet("color: #888; font-size: 11px; padding: 4px 0;")
        layout.addWidget(header)

        # ── splitter: list | preview ─────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_selection)
        splitter.addWidget(self._list)

        self._preview = QTextBrowser()
        self._preview.setReadOnly(True)
        self._preview.setStyleSheet("font-size: 13px; line-height: 1.5;")
        splitter.addWidget(self._preview)

        splitter.setSizes([280, 560])
        layout.addWidget(splitter)

        # ── buttons ──────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(self._delete_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._on_clear_all)
        btn_row.addWidget(clear_btn)

        btn_row.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)

        self._populate()

    # ── internal ─────────────────────────────────────────────────

    def _populate(self):
        self._list.clear()
        self._thoughts = self._memory.get_free_thoughts()

        for t in self._thoughts:
            ts      = t["timestamp"][:19].replace("T", "  ")
            preview = t["text"][:60].replace("\n", " ")
            if len(t["text"]) > 60:
                preview += "…"
            item = QListWidgetItem(f"{ts}\n  {preview}")
            item.setData(Qt.ItemDataRole.UserRole, t["id"])
            self._list.addItem(item)

        count = len(self._thoughts)
        self.setWindowTitle(
            f"Talon's Thoughts  ({count} thought{'s' if count != 1 else ''})"
        )

        if not self._thoughts:
            self._preview.setPlainText(
                "No free thoughts stored yet.\n\n"
                "Enable the reflection loop in Settings to let Talon think freely."
            )

    def _on_selection(self, current, _previous):
        if current is None:
            self._preview.clear()
            self._delete_btn.setEnabled(False)
            return

        doc_id = current.data(Qt.ItemDataRole.UserRole)
        thought = next((t for t in self._thoughts if t["id"] == doc_id), None)
        if thought:
            ts = thought["timestamp"][:19].replace("T", " at ")
            self._preview.setHtml(
                f"<p style='color:#666; font-size:11px; margin-bottom:12px;'>"
                f"Generated {ts}</p>"
                f"<p style='white-space: pre-wrap;'>{thought['text']}</p>"
            )
        self._delete_btn.setEnabled(True)

    def _on_delete(self):
        current = self._list.currentItem()
        if current is None:
            return
        doc_id = current.data(Qt.ItemDataRole.UserRole)
        reply = QMessageBox.question(
            self, "Delete Thought",
            "Delete this thought permanently?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._memory.delete_free_thought(doc_id)
            self._preview.clear()
            self._populate()

    def _on_clear_all(self):
        if not self._thoughts:
            return
        reply = QMessageBox.question(
            self, "Clear All Thoughts",
            f"Permanently delete all {len(self._thoughts)} stored thoughts?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            n = self._memory.clear_free_thoughts()
            self._preview.clear()
            self._populate()
            self.setWindowTitle(f"Talon's Thoughts  (deleted {n})")
