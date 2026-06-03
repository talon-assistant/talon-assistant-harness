import os
import re
import json
from PyQt6.QtCore import QObject, pyqtSignal

from core.config import update_settings


class ThemeManager(QObject):
    """Manages theme switching and font size scaling.

    Loads QSS files, applies font-size overrides via regex,
    and persists preference to settings.json['appearance'].
    """

    theme_changed = pyqtSignal()

    THEMES = {
        "dark": "gui/styles/dark_theme.qss",
        "light": "gui/styles/light_theme.qss",
    }
    DEFAULT_FONT_SIZE = 13
    MIN_FONT_SIZE = 10
    MAX_FONT_SIZE = 20

    def __init__(self, app, config_dir="config"):
        super().__init__()
        self._app = app
        self._config_dir = config_dir
        self._current_theme = "dark"
        self._font_size = self.DEFAULT_FONT_SIZE
        self._base_font_size = self.DEFAULT_FONT_SIZE
        self._load_preference()
        self._apply_stylesheet()

    def _load_preference(self):
        """Read appearance section from settings.json if present."""
        config_path = os.path.join(self._config_dir, "settings.json")
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            appearance = cfg.get("appearance", {})
            self._current_theme = appearance.get("theme", "dark")
            self._font_size = appearance.get("font_size", self.DEFAULT_FONT_SIZE)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def set_theme(self, theme_name):
        """Switch to the named theme and apply."""
        if theme_name not in self.THEMES:
            return
        self._current_theme = theme_name
        self._apply_stylesheet()
        self.save_preference()
        self.theme_changed.emit()

    def set_font_size(self, size):
        """Change font size and re-apply stylesheet."""
        size = max(self.MIN_FONT_SIZE, min(self.MAX_FONT_SIZE, size))
        self._font_size = size
        self._apply_stylesheet()
        self.save_preference()
        self.theme_changed.emit()

    def _apply_stylesheet(self):
        """Load QSS file, apply font scaling, set on app."""
        qss_path = self.THEMES.get(self._current_theme)
        if not qss_path or not os.path.exists(qss_path):
            # Fall back to dark theme
            qss_path = self.THEMES["dark"]
        if not os.path.exists(qss_path):
            return

        with open(qss_path, 'r') as f:
            qss = f.read()

        # Scale font-size declarations relative to base
        if self._font_size != self._base_font_size:
            scale = self._font_size / self._base_font_size

            def _scale_font(match):
                original = int(match.group(1))
                scaled = max(8, int(original * scale))
                return f"font-size: {scaled}px"

            qss = re.sub(r'font-size:\s*(\d+)px', _scale_font, qss)

        self._app.setStyleSheet(qss)

    def save_preference(self):
        """Persist current theme + font_size to settings.json."""
        config_path = os.path.join(self._config_dir, "settings.json")
        update_settings(config_path, {
            "appearance": {
                "theme": self._current_theme,
                "font_size": self._font_size,
            }
        })

    @property
    def current_theme(self):
        return self._current_theme

    @property
    def font_size(self):
        return self._font_size
