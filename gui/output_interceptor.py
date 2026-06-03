from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, Qt, QMetaObject, Q_ARG


class OutputInterceptor(QObject):
    """Replaces sys.stdout to capture print() output as Qt signals.

    Thread-safe: write() can be called from any thread (worker threads call
    print() freely).  The signal is emitted via QMetaObject.invokeMethod
    with Qt.ConnectionType.QueuedConnection so the slot always runs on the
    GUI thread regardless of which thread called write().

    Dual-writes to both the signal and the original stdout so terminal
    output still works during development.
    """

    text_written = pyqtSignal(str)

    def __init__(self, original_stdout):
        super().__init__()
        self._original = original_stdout

    def write(self, text):
        # Always write to original stdout (thread-safe in CPython)
        if self._original:
            try:
                self._original.write(text)
            except Exception:
                pass

        # Only emit signal for non-empty text
        if text and text.strip():
            try:
                # Use QueuedConnection to safely deliver from any thread
                QMetaObject.invokeMethod(
                    self,
                    "_emit_text",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(str, str(text))
                )
            except RuntimeError:
                # QObject may already be destroyed during shutdown
                pass

    @pyqtSlot(str)
    def _emit_text(self, text):
        """Slot that runs on the GUI thread to emit the signal safely."""
        try:
            self.text_written.emit(text)
        except RuntimeError:
            pass

    def flush(self):
        if self._original:
            try:
                self._original.flush()
            except Exception:
                pass
