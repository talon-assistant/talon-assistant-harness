"""Global hotkey listener using pynput.

Uses a thread-safe queue + QTimer to safely cross from pynput's background
thread into the Qt main thread — more reliable than direct cross-thread
signal emission in PyQt6.

The listener captures a screenshot immediately in the pynput callback
(before any window switch occurs) and passes it through the signal so
Task Assist sees the user's active window rather than Talon.

Usage
-----
    listener = HotkeyListener("ctrl+shift+t", parent=self)
    listener.triggered.connect(self._on_hotkey)   # slot receives screenshot_b64 or ""
    listener.start()
    # ... on cleanup:
    listener.stop()
"""

import queue

from PyQt6.QtCore import QObject, QTimer, pyqtSignal


def _to_pynput(hotkey: str) -> str:
    """Convert 'ctrl+shift+t' → '<ctrl>+<shift>+t' for pynput."""
    modifiers = {"ctrl", "shift", "alt", "cmd", "meta", "super"}
    parts = hotkey.lower().split("+")
    result = []
    for part in parts:
        part = part.strip()
        result.append(f"<{part}>" if part in modifiers else part)
    return "+".join(result)


def _capture_screenshot_sync() -> str:
    """Capture the active window right now and return base64 PNG (or empty string).

    Uses win32gui to get the foreground window's bounding rect, then grabs
    just that region with all_screens=True so secondary monitors are included.
    Falls back to full primary-monitor grab if win32gui is unavailable.
    """
    try:
        from PIL import ImageGrab
        import base64, io
        try:
            import win32gui
            hwnd = win32gui.GetForegroundWindow()
            rect = win32gui.GetWindowRect(hwnd)   # (left, top, right, bottom)
            # Clamp to sane dimensions in case rect is degenerate
            left, top, right, bot = rect
            if right - left > 50 and bot - top > 50:
                img = ImageGrab.grab(bbox=rect, all_screens=True)
            else:
                img = ImageGrab.grab(all_screens=True)
        except Exception:
            # win32gui unavailable — fall back to primary monitor
            img = ImageGrab.grab()
        # Cap at 1280px longest side
        img.thumbnail((1280, 1280))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


class HotkeyListener(QObject):
    """System-wide hotkey listener.

    Emits ``triggered(screenshot_b64)`` when the registered hotkey is pressed
    from any application.  The screenshot is captured immediately in the
    pynput thread (before any window switch) and delivered to Qt via a
    thread-safe queue + QTimer poll.
    """

    # Passes the pre-captured screenshot (or "" if capture failed)
    triggered = pyqtSignal(str)

    def __init__(self, hotkey_str: str, parent=None):
        super().__init__(parent)
        self._raw = hotkey_str
        self._pynput_str = _to_pynput(hotkey_str)
        self._listener = None
        self._queue: queue.Queue = queue.Queue()

        # Poll the queue every 50 ms on the Qt main thread
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._drain_queue)

    def start(self):
        try:
            from pynput import keyboard as _kb
            self._listener = _kb.GlobalHotKeys({
                self._pynput_str: self._on_activate,
            })
            self._listener.daemon = True
            self._listener.start()
            self._poll_timer.start()
            print(f"   [Hotkey] Task Assist hotkey registered: {self._pynput_str}")
        except ImportError:
            print("   [Hotkey] pynput not installed — global hotkey unavailable.")
        except Exception as e:
            print(f"   [Hotkey] Failed to register '{self._pynput_str}': {e}")

    def stop(self):
        self._poll_timer.stop()
        if self._listener:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    def _on_activate(self):
        """Called from pynput's thread — capture screenshot immediately then enqueue."""
        screenshot_b64 = _capture_screenshot_sync()
        self._queue.put(screenshot_b64)

    def _drain_queue(self):
        """Called from Qt main thread every 50 ms — emit queued triggers."""
        while not self._queue.empty():
            try:
                screenshot_b64 = self._queue.get_nowait()
                self.triggered.emit(screenshot_b64)
            except queue.Empty:
                break
