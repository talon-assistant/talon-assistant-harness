"""ReminderAlertDialog — always-on-top dismissable reminder popup.

When a reminder fires, this dialog appears on top of all windows and
cannot be closed with the X button or Alt+F4.  The user must click
**Dismiss** or **Snooze** to make it go away.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal


class ReminderAlertDialog(QDialog):
    """Always-on-top modal dialog shown when a reminder fires."""

    dismissed = pyqtSignal(str)              # reminder_id
    snoozed = pyqtSignal(str, str, int)      # reminder_id, message, snooze_seconds

    def __init__(self, reminder_id, message, default_snooze_minutes=5, parent=None):
        super().__init__(parent)
        self._reminder_id = reminder_id
        self._message = message
        self._default_snooze = default_snooze_minutes

        self.setWindowTitle("Talon Reminder")
        self.setObjectName("reminder_alert_dialog")
        self.setMinimumSize(400, 220)
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
            # Deliberately omit WindowCloseButtonHint to remove X button
        )
        self.setModal(False)  # Non-blocking — user can still interact with app

        self._setup_ui()

    # ── UI ─────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        # Header
        header = QLabel("\U0001f514  Reminder")
        header.setObjectName("reminder_alert_header")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Message (prominent)
        msg_label = QLabel(self._message)
        msg_label.setObjectName("reminder_alert_message")
        msg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label)

        layout.addSpacing(8)

        # ── Snooze row ──────────────────────────────────────────
        snooze_layout = QHBoxLayout()
        snooze_layout.addStretch()

        self._snooze_combo = QComboBox()
        self._snooze_combo.setObjectName("reminder_alert_snooze_combo")
        self._snooze_combo.addItems([
            "5 minutes", "10 minutes", "15 minutes",
            "30 minutes", "1 hour", "2 hours",
        ])
        snooze_layout.addWidget(self._snooze_combo)

        snooze_btn = QPushButton("Snooze")
        snooze_btn.setObjectName("reminder_alert_snooze_btn")
        snooze_btn.clicked.connect(self._on_snooze)
        snooze_layout.addWidget(snooze_btn)

        snooze_layout.addStretch()
        layout.addLayout(snooze_layout)

        # ── Dismiss button (primary action) ─────────────────────
        dismiss_layout = QHBoxLayout()
        dismiss_layout.addStretch()

        dismiss_btn = QPushButton("Dismiss")
        dismiss_btn.setObjectName("reminder_alert_dismiss_btn")
        dismiss_btn.setFixedWidth(120)
        dismiss_btn.clicked.connect(self._on_dismiss)
        dismiss_layout.addWidget(dismiss_btn)

        dismiss_layout.addStretch()
        layout.addLayout(dismiss_layout)

    # ── Handlers ───────────────────────────────────────────────────

    def _on_dismiss(self):
        self.dismissed.emit(self._reminder_id)
        self.accept()

    def _on_snooze(self):
        snooze_text = self._snooze_combo.currentText()
        seconds = self._parse_snooze(snooze_text)
        self.snoozed.emit(self._reminder_id, self._message, seconds)
        self.accept()

    @staticmethod
    def _parse_snooze(text):
        """Convert '5 minutes', '1 hour' etc. to seconds."""
        parts = text.split()
        amount = int(parts[0])
        unit = parts[1]
        if "hour" in unit:
            return amount * 3600
        return amount * 60

    # ── Block window-manager close ─────────────────────────────────

    def closeEvent(self, event):
        """Prevent X / Alt+F4 — user must click Dismiss or Snooze."""
        event.ignore()
