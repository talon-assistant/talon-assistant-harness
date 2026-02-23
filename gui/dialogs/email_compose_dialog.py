from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QTextEdit, QPushButton,
)
from PyQt6.QtCore import pyqtSignal


class EmailComposeDialog(QDialog):
    """Modal dialog for reviewing and editing a drafted email before sending.

    The email talent composes a draft (via LLM) and returns it as
    ``result["pending_email"]``.  The bridge detects this and emits
    ``compose_requested``, which causes MainWindow to create and exec this
    dialog.  The user can edit To / Subject / Body, then click Send or Cancel.

    Signals:
        compose_complete(dict)  — emitted on Send with {to, subject, body,
                                  reply_to_uid}
        compose_cancelled()     — emitted on Cancel
    """

    compose_complete  = pyqtSignal(dict)
    compose_cancelled = pyqtSignal()

    def __init__(self, draft: dict, parent=None):
        super().__init__(parent)
        self._reply_to_uid = draft.get("reply_to_uid", "")
        self.setWindowTitle("Review Email Draft")
        self.setObjectName("email_compose_dialog")
        self.setMinimumSize(540, 460)
        self.setModal(True)
        self._setup_ui(draft)

    def _setup_ui(self, draft: dict):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 16, 16, 16)

        # To
        layout.addWidget(QLabel("To:"))
        self._to = QLineEdit(draft.get("to", ""))
        self._to.setObjectName("email_compose_to")
        self._to.setPlaceholderText("recipient@example.com")
        layout.addWidget(self._to)

        # Subject
        layout.addWidget(QLabel("Subject:"))
        self._subject = QLineEdit(draft.get("subject", ""))
        self._subject.setObjectName("email_compose_subject")
        layout.addWidget(self._subject)

        # Body
        layout.addWidget(QLabel("Body:"))
        self._body = QTextEdit()
        self._body.setObjectName("email_compose_body")
        self._body.setPlainText(draft.get("body", ""))
        self._body.setMinimumHeight(200)
        layout.addWidget(self._body)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("email_compose_cancel")
        cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(cancel_btn)

        send_btn = QPushButton("Send")
        send_btn.setObjectName("email_compose_send")
        send_btn.setDefault(True)
        send_btn.clicked.connect(self._on_send)
        btn_row.addWidget(send_btn)

        layout.addLayout(btn_row)

    # ── slots ────────────────────────────────────────────────────────────────

    def _on_send(self):
        to = self._to.text().strip()
        if not to:
            self._to.setFocus()
            return   # don't send with empty To
        self.compose_complete.emit({
            "to":           to,
            "subject":      self._subject.text().strip(),
            "body":         self._body.toPlainText(),
            "reply_to_uid": self._reply_to_uid,
        })
        self.accept()

    def _on_cancel(self):
        self.compose_cancelled.emit()
        self.reject()
