from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QTextEdit, QPushButton, QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont


class TaskAssistDialog(QDialog):
    """Modal dialog for reviewing, revising, and accepting a Task Assist draft.

    The task_assist talent produces a first draft and returns it as
    ``result["pending_task_assist"]``.  The bridge emits
    ``task_assist_requested``, and MainWindow creates this dialog.

    The user can:
      - Edit the draft directly in the text box
      - Request a revision with additional instructions (loops back to LLM)
      - Accept — copies final text to clipboard
      - Decline — dismisses without action

    Signals:
        accepted_text(str)  — emitted with final text when user clicks Accept
        declined()          — emitted when user clicks Decline
    """

    accepted_text = pyqtSignal(str)
    declined = pyqtSignal()

    def __init__(self, task: str, draft: str, screenshot_b64: str | None,
                 llm_client, parent=None):
        super().__init__(parent)
        self._task = task
        self._screenshot_b64 = screenshot_b64
        self._llm = llm_client
        self._worker = None
        self.setWindowTitle("Talon Task Assist")
        self.setObjectName("task_assist_dialog")
        self.setMinimumSize(620, 520)
        self.setModal(True)
        self._setup_ui(task, draft)

    def _setup_ui(self, task: str, draft: str):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 16, 16, 16)

        # Task label
        task_label = QLabel(f"Task: {task}")
        task_label.setWordWrap(True)
        task_label.setObjectName("task_assist_task_label")
        layout.addWidget(task_label)

        # Draft output
        layout.addWidget(QLabel("Draft:"))
        self._draft_edit = QTextEdit()
        self._draft_edit.setObjectName("task_assist_draft")
        self._draft_edit.setPlainText(draft)
        mono = QFont("Consolas", 11)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._draft_edit.setFont(mono)
        self._draft_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._draft_edit, stretch=1)

        # Revision instructions
        layout.addWidget(QLabel("Revision instructions (optional):"))
        self._revision_input = QLineEdit()
        self._revision_input.setObjectName("task_assist_revision")
        self._revision_input.setPlaceholderText(
            "e.g. make it shorter, add bullet points, fix the tone...")
        self._revision_input.returnPressed.connect(self._on_revise)
        layout.addWidget(self._revision_input)

        # Status label
        self._status_label = QLabel("")
        self._status_label.setObjectName("task_assist_status")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

        # Buttons
        btn_row = QHBoxLayout()

        self._decline_btn = QPushButton("Decline")
        self._decline_btn.setObjectName("task_assist_decline")
        self._decline_btn.clicked.connect(self._on_decline)
        btn_row.addWidget(self._decline_btn)

        btn_row.addStretch()

        self._revise_btn = QPushButton("Revise")
        self._revise_btn.setObjectName("task_assist_revise")
        self._revise_btn.clicked.connect(self._on_revise)
        btn_row.addWidget(self._revise_btn)

        self._accept_btn = QPushButton("Accept")
        self._accept_btn.setObjectName("task_assist_accept")
        self._accept_btn.setDefault(True)
        self._accept_btn.clicked.connect(self._on_accept)
        btn_row.addWidget(self._accept_btn)

        layout.addLayout(btn_row)

    # ── slots ─────────────────────────────────────────────────────────────────

    def _on_accept(self):
        import pyperclip
        text = self._draft_edit.toPlainText()
        try:
            pyperclip.copy(text)
        except Exception:
            pass
        self.accepted_text.emit(text)
        self.accept()

    def _on_decline(self):
        self.declined.emit()
        self.reject()

    def _on_revise(self):
        instruction = self._revision_input.text().strip()
        if not instruction:
            self._revision_input.setFocus()
            return

        from gui.workers import TaskAssistReviseWorker
        self._set_busy(True)
        self._worker = TaskAssistReviseWorker(
            llm_client=self._llm,
            task=self._task,
            current_draft=self._draft_edit.toPlainText(),
            instruction=instruction,
            screenshot_b64=self._screenshot_b64,
        )
        self._worker.revised.connect(self._on_revised)
        self._worker.error.connect(self._on_revise_error)
        self._worker.start()

    def _on_revised(self, new_draft: str):
        self._draft_edit.setPlainText(new_draft)
        self._revision_input.clear()
        self._set_busy(False)

    def _on_revise_error(self, error_msg: str):
        self._status_label.setText(f"Revision failed: {error_msg}")
        self._set_busy(False)

    def _set_busy(self, busy: bool):
        self._accept_btn.setEnabled(not busy)
        self._revise_btn.setEnabled(not busy)
        self._decline_btn.setEnabled(not busy)
        self._revision_input.setEnabled(not busy)
        self._status_label.setText("Generating revised draft..." if busy else "")
