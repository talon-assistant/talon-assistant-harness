"""Global hotkey listener using Win32 RegisterHotKey.

Uses RegisterHotKey + GetMessage instead of pynput WH_KEYBOARD_LL hooks.
pynput hooks require a live Win32 message pump and cause access violations
/ heap corruption when the process exits while the hook thread is running.

The listener captures a screenshot immediately when the hotkey fires
(before any window switch occurs) and passes it through the signal so
Task Assist sees the user's active window rather than Talon.

Usage
-----
    listener = HotkeyListener("ctrl+alt+j", parent=self)
    listener.triggered.connect(self._on_hotkey)   # slot receives screenshot_b64, window_info
    listener.start()
    # ... on cleanup:
    listener.stop()
"""

import ctypes
import ctypes.wintypes
import queue
import threading

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

# ── Window info capture ────────────────────────────────────────────────────────


def _capture_window_info_sync() -> dict:
    """Capture the foreground window's title and process name.

    Returns dict with 'app_title' and 'process_name' keys, or empty dict
    on failure.  Called from the hotkey thread alongside screenshot capture.
    """
    try:
        import win32gui
        import win32process
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        process_name = ""
        try:
            import psutil
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process_name = psutil.Process(pid).name()
        except Exception:
            pass
        return {"app_title": title, "process_name": process_name}
    except Exception:
        return {}


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


# ── Hotkey string → Win32 VK/MOD mapping ──────────────────────────────────────

_MOD_MAP = {
    "ctrl": 0x0002,    # MOD_CONTROL
    "control": 0x0002,
    "shift": 0x0004,   # MOD_SHIFT
    "alt": 0x0001,     # MOD_ALT
    "win": 0x0008,     # MOD_WIN
    "super": 0x0008,
    "meta": 0x0008,
    "cmd": 0x0008,
}

_VK_MAP = {
    "space": 0x20, "enter": 0x0D, "return": 0x0D, "tab": 0x09,
    "backspace": 0x08, "delete": 0x2E, "esc": 0x1B, "escape": 0x1B,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "home": 0x24, "end": 0x23, "insert": 0x2D,
    "page_up": 0x21, "pageup": 0x21, "page_down": 0x22, "pagedown": 0x22,
    **{f"f{i}": 0x6F + i for i in range(1, 13)},
}


def _parse_hotkey(hotkey_str: str):
    """Parse 'ctrl+alt+j' → (combined_modifiers, vk_code)."""
    parts = [p.strip().lower() for p in hotkey_str.split("+")]
    modifiers = 0
    vk = 0
    for part in parts:
        if part in _MOD_MAP:
            modifiers |= _MOD_MAP[part]
        elif part in _VK_MAP:
            vk = _VK_MAP[part]
        elif len(part) == 1 and part.isalnum():
            vk = ord(part.upper())
        else:
            print(f"   [Hotkey] Unknown key: {part!r}")
    return modifiers, vk


class HotkeyListener(QObject):
    """System-wide hotkey listener using Win32 RegisterHotKey.

    Emits ``triggered(screenshot_b64, window_info)`` when the registered
    hotkey is pressed from any application.  The screenshot is captured
    immediately in the hotkey thread (before any window switch) and
    delivered to Qt via a thread-safe queue + QTimer poll.
    """

    # Passes the pre-captured screenshot (or "") and window info dict
    triggered = pyqtSignal(str, dict)

    _HOTKEY_ID = 2  # Avoid collision with system_tray's HOTKEY_ID=1

    def __init__(self, hotkey_str: str, parent=None):
        super().__init__(parent)
        self._raw = hotkey_str
        self._modifiers, self._vk = _parse_hotkey(hotkey_str)
        self._listener_thread: threading.Thread | None = None
        self._listener_thread_id: int | None = None
        self._queue: queue.Queue = queue.Queue()

        # Poll the queue every 50 ms on the Qt main thread
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._drain_queue)

    def start(self):
        if not self._vk:
            print(f"   [Hotkey] Could not parse hotkey: {self._raw!r}")
            return

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        modifiers = self._modifiers
        vk = self._vk
        hotkey_id = self._HOTKEY_ID
        q = self._queue
        WM_HOTKEY = 0x0312

        id_holder: list[int | None] = [None]
        ready = threading.Event()

        def _loop():
            id_holder[0] = kernel32.GetCurrentThreadId()
            ready.set()
            if not user32.RegisterHotKey(None, hotkey_id, modifiers, vk):
                print(f"   [Hotkey] RegisterHotKey failed for {self._raw!r} "
                      f"(may already be in use).")
                return
            msg = ctypes.wintypes.MSG()
            try:
                while user32.GetMessageA(ctypes.byref(msg), None, 0, 0) != 0:
                    if msg.message == WM_HOTKEY:
                        screenshot_b64 = _capture_screenshot_sync()
                        window_info = _capture_window_info_sync()
                        q.put((screenshot_b64, window_info))
            finally:
                user32.UnregisterHotKey(None, hotkey_id)

        t = threading.Thread(target=_loop, daemon=True, name="task-assist-hotkey")
        t.start()
        ready.wait(timeout=2.0)

        self._listener_thread = t
        self._listener_thread_id = id_holder[0]
        self._poll_timer.start()
        print(f"   [Hotkey] Task Assist hotkey registered: {self._raw}")

    def stop(self):
        self._poll_timer.stop()
        # Send WM_QUIT to the hotkey thread's message loop
        if self._listener_thread_id is not None:
            try:
                ctypes.windll.user32.PostThreadMessageW(
                    self._listener_thread_id, 0x0012, 0, 0)  # WM_QUIT
            except Exception:
                pass
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=2.0)
            self._listener_thread = None
            self._listener_thread_id = None

    def _drain_queue(self):
        """Called from Qt main thread every 50 ms — emit queued triggers."""
        while not self._queue.empty():
            try:
                screenshot_b64, window_info = self._queue.get_nowait()
                self.triggered.emit(screenshot_b64, window_info)
            except queue.Empty:
                break
