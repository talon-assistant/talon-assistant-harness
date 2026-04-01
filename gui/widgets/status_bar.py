from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel


def _restyle(widget):
    """Force Qt to re-evaluate object-name stylesheet rules.

    Avoids style().unpolish/polish which can cause heap corruption
    on PyQt6 + Python 3.10 (GIL released during C++ destructor).
    """
    widget.setStyleSheet(widget.styleSheet())


class StatusBarWidget(QWidget):
    """Custom status bar content with colored status indicators."""

    def __init__(self):
        super().__init__()
        self._server_mode = "external"  # "external" or "builtin"
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)

        # LLM status
        self.llm_indicator = QLabel("LLM: Checking...")
        self.llm_indicator.setObjectName("status_llm_unknown")
        layout.addWidget(self.llm_indicator)

        layout.addWidget(self._separator())

        # Voice status
        self.voice_indicator = QLabel("Voice: Off")
        self.voice_indicator.setObjectName("status_voice_off")
        layout.addWidget(self.voice_indicator)

        layout.addWidget(self._separator())

        # Activity
        self.activity_label = QLabel("Activity: Idle")
        self.activity_label.setObjectName("status_idle")
        layout.addWidget(self.activity_label)

        layout.addStretch()

    def _separator(self):
        sep = QLabel("|")
        sep.setObjectName("status_separator")
        return sep

    def set_server_mode(self, mode):
        """Set the LLM server mode for status display ('builtin' or 'external')."""
        self._server_mode = mode

    def set_llm_status(self, connected):
        prefix = "Built-in" if self._server_mode == "builtin" else "External"
        if connected:
            text = f"LLM: {prefix} (Connected)"
            obj_name = "status_llm_connected"
        else:
            text = f"LLM: {prefix} (Disconnected)"
            obj_name = "status_llm_disconnected"
        self.llm_indicator.setText(text)
        self.llm_indicator.setObjectName(obj_name)
        _restyle(self.llm_indicator)

    def set_server_status(self, status):
        """Update the LLM indicator for built-in server status changes."""
        status_map = {
            "starting": ("LLM: Built-in (Starting...)", "status_llm_unknown"),
            "running": ("LLM: Built-in (Running)", "status_llm_connected"),
            "stopped": ("LLM: Built-in (Stopped)", "status_llm_disconnected"),
            "error": ("LLM: Built-in (Error)", "status_llm_disconnected"),
        }
        text, obj_name = status_map.get(
            status, (f"LLM: Built-in ({status})", "status_llm_unknown"))
        self.llm_indicator.setText(text)
        self.llm_indicator.setObjectName(obj_name)
        _restyle(self.llm_indicator)

    def set_voice_status(self, status):
        self.voice_indicator.setText(f"Voice: {status.capitalize()}")
        obj_name = f"status_voice_{status}"
        self.voice_indicator.setObjectName(obj_name)
        _restyle(self.voice_indicator)

    def set_activity(self, activity):
        self.activity_label.setText(f"Activity: {activity.capitalize()}")
        obj_name = f"status_{activity}"
        self.activity_label.setObjectName(obj_name)
        _restyle(self.activity_label)
