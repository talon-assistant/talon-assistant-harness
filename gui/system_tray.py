import ctypes
import ctypes.wintypes
import threading

from PyQt6.QtWidgets import QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QAction
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QMetaObject, Qt

import logging
log = logging.getLogger(__name__)


def _create_default_icon():
    """Create a simple 64x64 blue circle 'J' icon programmatically."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#89b4fa"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(2, 2, 60, 60)
    painter.setPen(QColor("#1e1e2e"))
    painter.setFont(QFont("Segoe UI", 32, QFont.Weight.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "J")
    painter.end()
    return QIcon(pixmap)


class SystemTrayManager(QObject):
    """Manages system tray icon, context menu, notifications, and global hotkeys."""

    show_requested = pyqtSignal()
    hide_requested = pyqtSignal()
    exit_requested = pyqtSignal()

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self._window = window
        self._setup_tray_icon()
        self._setup_hotkey()

    def _setup_tray_icon(self):
        """Create QSystemTrayIcon with context menu."""
        self.tray_icon = QSystemTrayIcon(self._window)
        self.tray_icon.setIcon(_create_default_icon())
        self.tray_icon.setToolTip("Talon Assistant")

        menu = QMenu()
        show_action = QAction("Show Talon", menu)
        show_action.triggered.connect(self._show_window)
        menu.addAction(show_action)

        hide_action = QAction("Hide to Tray", menu)
        hide_action.triggered.connect(self._hide_window)
        menu.addAction(hide_action)

        menu.addSeparator()

        exit_action = QAction("Exit", menu)
        exit_action.triggered.connect(self.exit_requested.emit)
        menu.addAction(exit_action)

        self.tray_icon.setContextMenu(menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

    def _on_tray_activated(self, reason):
        """Double-click on tray icon shows/hides window."""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            if self._window.isVisible():
                self._hide_window()
            else:
                self._show_window()

    def _show_window(self):
        self._window.show()
        self._window.raise_()
        self._window.activateWindow()
        self.show_requested.emit()

    def _hide_window(self):
        self._window.hide()
        self.hide_requested.emit()

    def _setup_hotkey(self):
        """Register global hotkey Ctrl+Shift+J via Win32 RegisterHotKey API.

        Uses RegisterHotKey + GetMessage instead of pynput WH_KEYBOARD_LL hooks.
        Hooks require a live Win32 message pump in their thread and crash with an
        access violation when the process exits while the hook thread is still
        running. RegisterHotKey posts WM_HOTKEY to a simple message loop and
        tears down cleanly via PostThreadMessageW(WM_QUIT).
        """
        self._hotkey_thread: threading.Thread | None = None
        self._hotkey_thread_id: int | None = None
        try:
            user32   = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            MOD_CONTROL = 0x0002
            MOD_SHIFT   = 0x0004
            VK_J        = 0x4A      # Virtual-key code for 'J'
            WM_HOTKEY   = 0x0312
            HOTKEY_ID   = 1

            _id_holder: list[int | None] = [None]
            _ready = threading.Event()

            def _hotkey_loop() -> None:
                _id_holder[0] = kernel32.GetCurrentThreadId()
                _ready.set()
                if not user32.RegisterHotKey(
                        None, HOTKEY_ID, MOD_CONTROL | MOD_SHIFT, VK_J):
                    log.error("[SystemTray] RegisterHotKey failed "
                          "(Ctrl+Shift+J may already be in use).")
                    return
                msg = ctypes.wintypes.MSG()
                try:
                    while user32.GetMessageA(ctypes.byref(msg), None, 0, 0) != 0:
                        if msg.message == WM_HOTKEY:
                            QMetaObject.invokeMethod(
                                self, "_toggle_window",
                                Qt.ConnectionType.QueuedConnection)
                finally:
                    user32.UnregisterHotKey(None, HOTKEY_ID)

            t = threading.Thread(
                target=_hotkey_loop, daemon=True, name="hotkey-listener")
            t.start()
            _ready.wait(timeout=2.0)

            self._hotkey_thread    = t
            self._hotkey_thread_id = _id_holder[0]
            self._kernel32         = kernel32
            log.info("[SystemTray] Global hotkey Ctrl+Shift+J registered.")
        except Exception as e:
            log.error(f"[SystemTray] Global hotkey setup failed: {e}")

    @pyqtSlot()
    def _toggle_window(self):
        if self._window.isVisible():
            self._hide_window()
        else:
            self._show_window()

    def show_notification(self, title, message, duration_ms=5000):
        """Show a desktop notification via the system tray icon."""
        if self.tray_icon.supportsMessages():
            self.tray_icon.showMessage(
                title, message,
                QSystemTrayIcon.MessageIcon.Information,
                duration_ms)

    def cleanup(self):
        """Stop hotkey listener, hide tray icon."""
        if self._hotkey_thread_id:
            try:
                # Post WM_QUIT to the GetMessage loop so it exits cleanly and
                # calls UnregisterHotKey before the thread dies.
                WM_QUIT = 0x0012
                self._kernel32.PostThreadMessageW(
                    self._hotkey_thread_id, WM_QUIT, 0, 0)
            except Exception:
                pass
        self.tray_icon.hide()
