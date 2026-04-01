from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QProgressBar, QLabel
from PyQt6.QtCore import pyqtSignal, QTimer


class VoicePanel(QWidget):
    """Voice controls: mic toggle, TTS toggle, stop button, audio level bar, status indicator."""

    voice_toggled = pyqtSignal(bool)
    tts_toggled = pyqtSignal(bool)
    stop_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setObjectName("voice_panel")
        self.setFixedHeight(44)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        # Mic toggle button
        self.mic_button = QPushButton("Mic Off")
        self.mic_button.setCheckable(True)
        self.mic_button.setFixedWidth(100)
        self.mic_button.toggled.connect(self._on_mic_toggle)
        layout.addWidget(self.mic_button)

        # TTS toggle button
        self.tts_button = QPushButton("TTS On")
        self.tts_button.setCheckable(True)
        self.tts_button.setChecked(True)
        self.tts_button.setFixedWidth(100)
        self.tts_button.toggled.connect(self._on_tts_toggle)
        layout.addWidget(self.tts_button)

        # Stop speaking button (hidden by default, shown during TTS playback)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stop_speaking_btn")
        self.stop_button.setFixedWidth(70)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.stop_button.setVisible(False)
        layout.addWidget(self.stop_button)

        # Audio level bar
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedHeight(16)
        layout.addWidget(self.level_bar)

        # Status label
        self.status_label = QLabel("Voice: Off")
        self.status_label.setFixedWidth(220)
        layout.addWidget(self.status_label)

        # Wake word indicator dot
        self.wake_indicator = QLabel("")
        self.wake_indicator.setFixedSize(16, 16)
        self.wake_indicator.setObjectName("wake_indicator_off")
        layout.addWidget(self.wake_indicator)

    def _on_mic_toggle(self, checked):
        self.mic_button.setText("Mic On" if checked else "Mic Off")
        self.voice_toggled.emit(checked)
        if not checked:
            self.level_bar.setValue(0)
            self.status_label.setText("Voice: Off")

    def _on_tts_toggle(self, checked):
        self.tts_button.setText("TTS On" if checked else "TTS Off")
        self.tts_toggled.emit(checked)

    def update_status(self, status):
        """Update the voice status label."""
        status_map = {
            "listening": "Listening for wake word...",
            "recording_command": "Recording command...",
            "transcribing": "Transcribing...",
            "off": "Voice: Off"
        }
        self.status_label.setText(status_map.get(status, status))

    def update_level(self, level):
        """Update the audio level bar (0.0 to 1.0)."""
        self.level_bar.setValue(int(level * 100))

    @staticmethod
    def _restyle(widget):
        widget.setStyleSheet(widget.styleSheet())

    def flash_wake_indicator(self):
        """Briefly turn indicator green when wake word detected."""
        self.wake_indicator.setObjectName("wake_indicator_on")
        self._restyle(self.wake_indicator)
        QTimer.singleShot(2000, self._reset_wake_indicator)

    def _reset_wake_indicator(self):
        self.wake_indicator.setObjectName("wake_indicator_off")
        self._restyle(self.wake_indicator)

    def on_tts_started(self):
        """Show the stop button while Talon is speaking."""
        self.stop_button.setVisible(True)

    def on_tts_finished(self):
        """Hide the stop button when TTS ends (normal or interrupted)."""
        self.stop_button.setVisible(False)
