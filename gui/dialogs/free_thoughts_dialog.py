"""Dialog for reviewing and managing Talon's free-thought reflections and goals."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget, QWidget,
    QListWidget, QListWidgetItem, QTextBrowser, QTreeWidget, QTreeWidgetItem,
    QPushButton, QLabel, QMessageBox, QHeaderView,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor


class FreeThoughtsDialog(QDialog):
    """Browse, read, and delete Talon's stored free thoughts and goals."""

    def __init__(self, memory, parent=None):
        super().__init__(parent)
        self._memory = memory
        self._thoughts: list[dict] = []

        self.setWindowTitle("Talon's Thoughts & Goals")
        self.setMinimumSize(720, 480)
        self.resize(860, 560)

        layout = QVBoxLayout(self)

        # ── tabs ────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        self._tabs.addTab(self._build_thoughts_tab(), "Thoughts")
        self._tabs.addTab(self._build_goals_tab(), "Goals")

        # ── close button (shared) ──────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._populate_thoughts()
        self._populate_goals()

    # ================================================================
    #  THOUGHTS TAB
    # ================================================================

    def _build_thoughts_tab(self) -> QWidget:
        tab = QWidget()
        vbox = QVBoxLayout(tab)

        header = QLabel(
            "Free thoughts generated during Talon's unsupervised reflection time."
        )
        header.setStyleSheet("color: #888; font-size: 11px; padding: 4px 0;")
        vbox.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_thought_selection)
        splitter.addWidget(self._list)

        self._preview = QTextBrowser()
        self._preview.setReadOnly(True)
        self._preview.setStyleSheet("font-size: 13px; line-height: 1.5;")
        splitter.addWidget(self._preview)

        splitter.setSizes([280, 560])
        vbox.addWidget(splitter)

        btn_row = QHBoxLayout()

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._on_delete_thought)
        btn_row.addWidget(self._delete_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._on_clear_all_thoughts)
        btn_row.addWidget(clear_btn)

        btn_row.addStretch()
        vbox.addLayout(btn_row)
        return tab

    # ── thoughts internals ─────────────────────────────────────────

    def _populate_thoughts(self):
        self._list.clear()
        self._thoughts = self._memory.get_free_thoughts()

        for t in self._thoughts:
            ts      = t["timestamp"][:19].replace("T", "  ")
            preview = t["text"][:60].replace("\n", " ")
            if len(t["text"]) > 60:
                preview += "…"
            v = t.get("valence")
            v_tag = f"  [{v}/10]" if v is not None else ""
            item = QListWidgetItem(f"{ts}{v_tag}\n  {preview}")
            item.setData(Qt.ItemDataRole.UserRole, t["id"])
            self._list.addItem(item)

        count = len(self._thoughts)
        self._tabs.setTabText(
            0, f"Thoughts ({count})" if count else "Thoughts"
        )

        if not self._thoughts:
            self._preview.setPlainText(
                "No free thoughts stored yet.\n\n"
                "Enable the reflection loop in Settings to let Talon think freely."
            )

    def _on_thought_selection(self, current, _previous):
        if current is None:
            self._preview.clear()
            self._delete_btn.setEnabled(False)
            return

        doc_id = current.data(Qt.ItemDataRole.UserRole)
        thought = next((t for t in self._thoughts if t["id"] == doc_id), None)
        if thought:
            ts = thought["timestamp"][:19].replace("T", " at ")
            v = thought.get("valence")
            v_html = (f" &nbsp;·&nbsp; Valence: <b>{v}/10</b>"
                      if v is not None else "")
            self._preview.setHtml(
                f"<p style='color:#666; font-size:11px; margin-bottom:12px;'>"
                f"Generated {ts}{v_html}</p>"
                f"<p style='white-space: pre-wrap;'>{thought['text']}</p>"
            )
        self._delete_btn.setEnabled(True)

    def _on_delete_thought(self):
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
            self._populate_thoughts()

    def _on_clear_all_thoughts(self):
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
            self._populate_thoughts()

    # ================================================================
    #  GOALS TAB
    # ================================================================

    _STATUS_COLORS = {
        "active":    None,                          # default text color
        "completed": QColor(100, 200, 120),         # green-ish
        "abandoned": QColor(130, 130, 130),         # gray
    }

    def _build_goals_tab(self) -> QWidget:
        tab = QWidget()
        vbox = QVBoxLayout(tab)

        header = QLabel(
            "Goals Talon set for itself during reflection."
        )
        header.setStyleSheet("color: #888; font-size: 11px; padding: 4px 0;")
        vbox.addWidget(header)

        self._goals_tree = QTreeWidget()
        self._goals_tree.setHeaderLabels(
            ["ID", "Goal", "Status", "Created", "Last Progress"]
        )
        self._goals_tree.setRootIsDecorated(False)
        self._goals_tree.setAlternatingRowColors(True)
        self._goals_tree.setSelectionMode(
            QTreeWidget.SelectionMode.ExtendedSelection
        )
        self._goals_tree.setStyleSheet(
            "QTreeWidget { font-size: 12px; }"
            "QHeaderView::section { font-size: 11px; }"
        )

        hdr = self._goals_tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        vbox.addWidget(self._goals_tree)

        btn_row = QHBoxLayout()

        abandon_btn = QPushButton("Abandon Selected")
        abandon_btn.clicked.connect(self._on_abandon_goals)
        btn_row.addWidget(abandon_btn)

        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self._on_delete_goals)
        btn_row.addWidget(delete_btn)

        btn_row.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._populate_goals)
        btn_row.addWidget(refresh_btn)

        vbox.addLayout(btn_row)
        return tab

    # ── goals internals ────────────────────────────────────────────

    def _populate_goals(self):
        self._goals_tree.clear()
        goals = self._memory.get_all_goals()

        for g in goals:
            # Truncate goal text for the table
            short = g["text"][:80].replace("\n", " ")
            if len(g["text"]) > 80:
                short += "…"

            created = (g["created_at"] or "")[:16].replace("T", " ")

            # Last progress note: grab final non-empty line
            progress = (g["progress"] or "").strip()
            last_note = ""
            if progress:
                lines = [ln for ln in progress.split("\n") if ln.strip()]
                if lines:
                    last_note = lines[-1][:80]

            item = QTreeWidgetItem([
                str(g["id"]),
                short,
                g["status"],
                created,
                last_note,
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, g["id"])

            color = self._STATUS_COLORS.get(g["status"])
            if color:
                for col in range(5):
                    item.setForeground(col, color)

            self._goals_tree.addTopLevelItem(item)

        count = len(goals)
        self._tabs.setTabText(
            1, f"Goals ({count})" if count else "Goals"
        )

    def _selected_goal_ids(self) -> list[int]:
        return [
            item.data(0, Qt.ItemDataRole.UserRole)
            for item in self._goals_tree.selectedItems()
        ]

    def _on_abandon_goals(self):
        ids = self._selected_goal_ids()
        if not ids:
            return
        reply = QMessageBox.question(
            self, "Abandon Goals",
            f"Mark {len(ids)} selected goal(s) as abandoned?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            for gid in ids:
                self._memory.complete_goal(gid, status="abandoned")
            self._populate_goals()

    def _on_delete_goals(self):
        ids = self._selected_goal_ids()
        if not ids:
            return
        reply = QMessageBox.question(
            self, "Delete Goals",
            f"Permanently delete {len(ids)} selected goal(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            for gid in ids:
                self._memory.delete_goal(gid)
            self._populate_goals()
