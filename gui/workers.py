import threading
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class InitWorker(QThread):
    """Initializes TalonAssistant off the GUI thread.

    TalonAssistant.__init__() loads Whisper, ChromaDB, SentenceTransformer,
    connects to Hue, tests LLM — total 10-30 seconds.
    """

    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, config_dir="config"):
        super().__init__()
        self.config_dir = config_dir

    def run(self):
        try:
            from core.assistant import TalonAssistant
            assistant = TalonAssistant(config_dir=self.config_dir)
            self.finished.emit(assistant)
        except Exception as e:
            self.error.emit(str(e))


class CommandWorker(QThread):
    """Runs process_command() off the GUI thread.

    Calls with speak_response=False so TTS is handled separately by the bridge.
    """

    response_ready = pyqtSignal(str, str, str, bool, dict)  # command, response, talent_name, success, result
    error = pyqtSignal(str, str)  # command, error_message

    def __init__(self, assistant, command):
        super().__init__()
        self.assistant = assistant
        self.command = command

    def run(self):
        try:
            result = self.assistant.process_command(self.command, speak_response=False)
            if result and isinstance(result, dict):
                self.response_ready.emit(
                    self.command,
                    result.get("response", ""),
                    result.get("talent", ""),
                    result.get("success", True),
                    result,
                )
            elif result is None:
                # Command was too short / empty
                self.response_ready.emit(
                    self.command, "Command too short to process.", "", False, {}
                )
            else:
                # Unexpected return type
                self.response_ready.emit(self.command, str(result), "", True, {})
        except Exception as e:
            self.error.emit(self.command, str(e))


class TTSWorker(QThread):
    """Runs text-to-speech off the GUI thread.

    voice.speak() uses asyncio.run() internally, which is safe from a QThread.
    Supports mid-speech interruption via request_stop().
    """

    started_speaking = pyqtSignal()
    finished_speaking = pyqtSignal()
    stopped_early = pyqtSignal()  # emitted when TTS was interrupted
    error = pyqtSignal(str)

    def __init__(self, voice_system, text):
        super().__init__()
        self.voice = voice_system
        self.text = text

    def run(self):
        try:
            self.started_speaking.emit()
            completed = self.voice.speak(self.text)
            if completed:
                self.finished_speaking.emit()
            else:
                self.stopped_early.emit()
        except Exception as e:
            self.error.emit(str(e))

    def request_stop(self):
        """Ask the voice system to stop playback immediately."""
        self.voice.stop_speaking()


class VoiceListenWorker(QThread):
    """Controllable wake word listener that reimplements the loop from VoiceSystem.

    Does NOT call voice.listen_for_wake_word() because that has no stop mechanism.
    Instead, reimplements the loop body with a thread-safe _running flag.
    """

    wake_word_detected = pyqtSignal()
    command_transcribed = pyqtSignal(str)
    audio_level = pyqtSignal(float)
    heard_text = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, voice_system):
        super().__init__()
        self.voice = voice_system
        self._lock = threading.Lock()
        self._running = False

    @property
    def running(self):
        with self._lock:
            return self._running

    @running.setter
    def running(self, value):
        with self._lock:
            self._running = value

    def run(self):
        self.running = True
        self.status_changed.emit("listening")

        while self.running:
            try:
                # Record audio chunk (blocking for chunk_duration seconds)
                audio = self.voice.record_audio(self.voice.chunk_duration)

                if not self.running:
                    break

                # Check energy/variance thresholds
                audio_energy = float(np.abs(audio).mean())
                audio_variance = float(np.var(audio))

                # Emit normalized audio level (0.0-1.0)
                normalized_level = min(1.0, audio_energy / 500.0)
                self.audio_level.emit(normalized_level)

                if audio_energy < self.voice.energy_threshold:
                    continue
                if audio_variance < self.voice.variance_threshold:
                    continue

                # Transcribe chunk
                self.status_changed.emit("transcribing")
                text = self.voice.transcribe_audio(audio)

                if text.strip() in self.voice.noise_words:
                    self.status_changed.emit("listening")
                    continue

                if text and len(text) > 1:
                    self.heard_text.emit(text)

                # Check for wake word
                if any(ww in text for ww in self.voice.wake_words):
                    self.wake_word_detected.emit()
                    self.status_changed.emit("recording_command")

                    # Acknowledge wake word — speak "Ready" so user knows to give command
                    try:
                        self.voice.speak("Ready")
                    except Exception:
                        pass

                    # Record command audio
                    command_audio = self.voice.record_audio(self.voice.command_duration)

                    if not self.running:
                        break

                    self.status_changed.emit("transcribing")
                    command_text = self.voice.transcribe_audio(command_audio)

                    if command_text and len(command_text) > 1:
                        self.command_transcribed.emit(command_text)

                self.status_changed.emit("listening")

            except Exception as e:
                if not self.running:
                    break
                self.error.emit(str(e))

    def stop(self):
        """Stop the listening loop. May take up to chunk_duration to finish."""
        self.running = False


class DownloadWorker(QThread):
    """Downloads llama.cpp release in a background thread.

    Used by LLMSetupDialog to avoid blocking the GUI during large downloads.
    """

    progress = pyqtSignal(int, int)     # bytes_downloaded, total_bytes
    finished = pyqtSignal(str)          # path to extracted binary
    error = pyqtSignal(str)
    status = pyqtSignal(str)            # status message for UI

    def __init__(self, server_manager):
        super().__init__()
        self.server_manager = server_manager

    def run(self):
        try:
            self.server_manager.download_server(
                progress_cb=self._on_progress,
                status_cb=self._on_status,
            )
            exe_path = str(self.server_manager.server_exe_path)
            self.finished.emit(exe_path)
        except Exception as e:
            self.error.emit(str(e))

    def _on_progress(self, downloaded, total):
        self.progress.emit(downloaded, total)

    def _on_status(self, message):
        self.status.emit(message)


class ServerStartWorker(QThread):
    """Starts the llama-server and waits for it to become healthy.

    Runs in background so the GUI stays responsive during model loading.
    """

    ready = pyqtSignal()
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, server_manager):
        super().__init__()
        self.server_manager = server_manager

    def run(self):
        self.status.emit("Starting server...")

        # Wire callbacks for this run
        original_ready = self.server_manager.on_ready
        original_error = self.server_manager.on_error

        ready_event = threading.Event()
        error_msg = [None]

        def on_ready():
            ready_event.set()

        def on_error(msg):
            error_msg[0] = msg
            ready_event.set()

        self.server_manager.on_ready = on_ready
        self.server_manager.on_error = on_error

        try:
            self.server_manager.start()

            # Wait for either ready or error (max 130s to exceed health poll timeout)
            ready_event.wait(timeout=130)

            if error_msg[0]:
                self.error.emit(error_msg[0])
            elif self.server_manager.status == "running":
                self.ready.emit()
            else:
                self.error.emit("Server failed to start (unknown reason).")
        except Exception as e:
            self.error.emit(str(e))
        finally:
            # Restore original callbacks
            self.server_manager.on_ready = original_ready
            self.server_manager.on_error = original_error


class EmailSendWorker(QThread):
    """Sends a composed/edited email draft via SMTP off the GUI thread.

    Receives a draft dict {to, subject, body, reply_to_uid} from the
    EmailComposeDialog and dispatches it using the email talent's SMTP methods.
    """

    finished = pyqtSignal(str, str)   # to_addr, subject
    error    = pyqtSignal(str)

    def __init__(self, bridge, draft: dict):
        super().__init__()
        self._bridge = bridge
        self._draft  = draft

    def run(self):
        try:
            talent = self._bridge.get_talent("email")
            if talent is None:
                self.error.emit("Email talent not loaded.")
                return
            reply_uid = self._draft.get("reply_to_uid", "")
            to_addr   = self._draft.get("to", "")
            subject   = self._draft.get("subject", "")
            body      = self._draft.get("body", "")
            if reply_uid:
                talent._send_smtp_reply(
                    to_addr=to_addr,
                    subject=subject,
                    body=body,
                    reply_uid=reply_uid,
                )
            else:
                talent._send_smtp(
                    to_addr=to_addr,
                    subject=subject,
                    body=body,
                )
            self.finished.emit(to_addr, subject)
        except Exception as e:
            self.error.emit(str(e))
