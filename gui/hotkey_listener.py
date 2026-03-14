"""Global hotkey listener using pynput.

Runs pynput's GlobalHotKeys in a daemon thread.  The ``triggered`` signal
safely crosses into the Qt main thread via Qt's automatic queued-connection
mechanism — no manual thread marshalling needed.

Usage
-----
    listener = HotkeyListener("ctrl+shift+t", parent=self)
    listener.triggered.connect(self._on_hotkey)
    listener.start()
    # ... on cleanup:
    listener.stop()
"""

from PyQt6.QtCore import QObject, pyqtSignal


def _to_pynput(hotkey: str) -> str:
    """Convert 'ctrl+shift+t' → '<ctrl>+<shift>+t' for pynput."""
    modifiers = {"ctrl", "shift", "alt", "cmd", "meta", "super"}
    parts = hotkey.lower().split("+")
    result = []
    for part in parts:
        part = part.strip()
        result.append(f"<{part}>" if part in modifiers else part)
    return "+".join(result)


class HotkeyListener(QObject):
    """System-wide hotkey listener.

    Emits ``triggered`` when the registered hotkey is pressed from any
    application.  Uses pynput — no admin rights required on Windows.
    """

    triggered = pyqtSignal()

    def __init__(self, hotkey_str: str, parent=None):
        super().__init__(parent)
        self._raw = hotkey_str
        self._pynput_str = _to_pynput(hotkey_str)
        self._listener = None

    def start(self):
        try:
            from pynput import keyboard as _kb
            self._listener = _kb.GlobalHotKeys({
                self._pynput_str: self._on_activate,
            })
            self._listener.daemon = True
            self._listener.start()
            print(f"   [Hotkey] Task Assist hotkey registered: {self._pynput_str}")
        except ImportError:
            print("   [Hotkey] pynput not installed — global hotkey unavailable.")
        except Exception as e:
            print(f"   [Hotkey] Failed to register '{self._pynput_str}': {e}")

    def stop(self):
        if self._listener:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    def _on_activate(self):
        # Called from pynput's thread — Qt queues this signal safely.
        self.triggered.emit()
