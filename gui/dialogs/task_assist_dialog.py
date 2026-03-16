from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QTextEdit, QPushButton, QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont, QPixmap


class TaskAssistPreDialog(QDialog):
    """Small 'what do you need help with?' dialog shown before generating a draft.

    Displayed immediately after hotkey/menu trigger so the user can describe
    their task before the LLM runs.  Shows a thumbnail of the captured
    screenshot (if any) so the user can confirm the right window was grabbed.

    Signals:
        confirmed(task, screenshot_b64) — user clicked OK with a task description
    """

    confirmed = pyqtSignal(str, str)   # task_text, screenshot_b64 (may be "")

    def __init__(self, screenshot_b64: str = "", parent=None):
        super().__init__(parent)
        self._screenshot_b64 = screenshot_b64
        self.setWindowTitle("Talon Task Assist")
        self.setObjectName("task_assist_pre_dialog")
        self.setMinimumWidth(480)
        self.setModal(True)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # Screenshot thumbnail
        if self._screenshot_b64:
            try:
                import base64
                from PyQt6.QtCore import QByteArray
                raw = base64.b64decode(self._screenshot_b64)
                pixmap = QPixmap()
                pixmap.loadFromData(QByteArray(raw))
                if not pixmap.isNull():
                    thumb = pixmap.scaledToWidth(
                        440, Qt.TransformationMode.SmoothTransformation)
                    img_label = QLabel()
                    img_label.setPixmap(thumb)
                    img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    img_label.setObjectName("task_assist_thumb")
                    ctx_label = QLabel("Captured screen context:")
                    ctx_label.setObjectName("task_assist_ctx_label")
                    layout.addWidget(ctx_label)
                    layout.addWidget(img_label)
            except Exception:
                pass
        else:
            no_ctx = QLabel("No screen context captured — you can attach a file after the draft is generated.")
            no_ctx.setWordWrap(True)
            no_ctx.setObjectName("task_assist_no_ctx")
            layout.addWidget(no_ctx)

        # Task input
        task_label = QLabel("What do you need help with?")
        task_label.setObjectName("task_assist_pre_label")
        layout.addWidget(task_label)

        self._task_input = QLineEdit()
        self._task_input.setObjectName("task_assist_pre_input")
        self._task_input.setPlaceholderText(
            "e.g. improve the summary section of my resume")
        self._task_input.returnPressed.connect(self._on_ok)
        layout.addWidget(self._task_input)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        ok_btn = QPushButton("Generate draft")
        ok_btn.setDefault(True)
        ok_btn.setObjectName("task_assist_pre_ok")
        ok_btn.clicked.connect(self._on_ok)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        self._task_input.setFocus()

    def _on_ok(self):
        task = self._task_input.text().strip()
        if not task:
            self._task_input.setFocus()
            return
        self.confirmed.emit(task, self._screenshot_b64)
        self.accept()


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

        self._attach_btn = QPushButton("Attach file...")
        self._attach_btn.setObjectName("task_assist_attach")
        self._attach_btn.setToolTip("Load a file and regenerate the draft with its content")
        self._attach_btn.clicked.connect(self._on_attach_file)
        btn_row.addWidget(self._attach_btn)

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

    def _on_attach_file(self):
        """Open a file browser and regenerate the draft with the selected file's content."""
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Attach file for Task Assist",
            "",
            "Documents (*.txt *.md *.py *.js *.ts *.html *.css *.json *.yaml *.yml "
            "*.xml *.csv *.docx);;All files (*)"
        )
        if not path:
            return

        file_text = self._read_file(path)
        if not file_text:
            self._status_label.setText("Could not read that file.")
            return

        import os
        filename = os.path.basename(path)
        instruction = f"Use the attached file '{filename}' as context:\n\n{file_text[:4000]}"
        self._set_busy(True)
        from gui.workers import TaskAssistReviseWorker
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

    def _read_file(self, path: str) -> str:
        """Read a file and return its text content."""
        try:
            if path.lower().endswith(".docx"):
                try:
                    from docx import Document
                    doc = Document(path)
                    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                except ImportError:
                    return ""
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""

    def _set_busy(self, busy: bool):
        self._accept_btn.setEnabled(not busy)
        self._revise_btn.setEnabled(not busy)
        self._decline_btn.setEnabled(not busy)
        self._attach_btn.setEnabled(not busy)
        self._revision_input.setEnabled(not busy)
        self._status_label.setText("Generating revised draft..." if busy else "")
