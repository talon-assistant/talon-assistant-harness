from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QCheckBox, QScrollArea, QFrame, QPushButton)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer


class TalentItem(QFrame):
    """Single talent entry with checkbox, gear icon, and info."""

    toggled = pyqtSignal(str, bool)
    configure_requested = pyqtSignal(str)

    def __init__(self, talent_info):
        super().__init__()
        self.talent_name = talent_info["name"]
        self._has_config = talent_info.get("has_config", False)
        self.setObjectName("talent_item")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        # Top row: checkbox + gear icon
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(4)

        self.checkbox = QCheckBox(talent_info["name"])
        self.checkbox.setChecked(talent_info.get("enabled", True))
        self.checkbox.toggled.connect(
            lambda checked: self.toggled.emit(self.talent_name, checked))
        top_row.addWidget(self.checkbox)

        top_row.addStretch()

        # Gear button — only visible if talent has config
        self.gear_btn = QPushButton("\u2699")
        self.gear_btn.setObjectName("talent_gear_btn")
        self.gear_btn.setFixedSize(24, 24)
        self.gear_btn.setToolTip(f"Configure {talent_info['name']}")
        self.gear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.gear_btn.clicked.connect(
            lambda: self.configure_requested.emit(self.talent_name))
        self.gear_btn.setVisible(self._has_config)
        top_row.addWidget(self.gear_btn)

        layout.addLayout(top_row)

        # Description
        desc = QLabel(talent_info.get("description", ""))
        desc.setWordWrap(True)
        desc.setObjectName("talent_description")
        layout.addWidget(desc)

        # Keywords
        kw_text = ", ".join(talent_info.get("keywords", []))
        if kw_text:
            kw_label = QLabel(f"Keywords: {kw_text}")
            kw_label.setObjectName("talent_keywords")
            kw_label.setWordWrap(True)
            layout.addWidget(kw_label)

    def set_highlighted(self, highlighted):
        """Visual feedback when this talent handles a command."""
        self.setObjectName("talent_item_active" if highlighted else "talent_item")
        # setStyleSheet("") forces Qt to re-evaluate the object-name-based
        # stylesheet rules without calling style().unpolish/polish, which can
        # trigger heap corruption on PyQt6 + Python 3.10 (GIL release during
        # C++ destructor).
        self.setStyleSheet(self.styleSheet())


class TalentSidebar(QScrollArea):
    """Sidebar listing all talents with enable/disable toggles."""

    talent_toggled = pyqtSignal(str, bool)
    import_requested = pyqtSignal()
    configure_requested = pyqtSignal(str)
    marketplace_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFixedWidth(220)
        self._items = {}
        self._highlight_generation = 0  # Prevents stale timer callbacks

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)

        header = QLabel("Talents")
        header.setObjectName("sidebar_header")
        self._layout.addWidget(header)

        self._layout.addStretch()

        # Marketplace button
        self.marketplace_button = QPushButton("Browse Marketplace")
        self.marketplace_button.setObjectName("marketplace_sidebar_btn")
        self.marketplace_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.marketplace_button.clicked.connect(self.marketplace_requested.emit)
        self._layout.addWidget(self.marketplace_button)

        # Import button at the bottom
        self.import_button = QPushButton("+ Import Talent")
        self.import_button.setObjectName("import_talent_button")
        self.import_button.clicked.connect(self.import_requested.emit)
        self._layout.addWidget(self.import_button)

        self.setWidget(container)

    def populate_talents(self, talent_list):
        """Populate with talent info dicts: {name, description, enabled, keywords, has_config}."""
        # Invalidate any pending highlight timers
        self._highlight_generation += 1

        # Remove talent items (keep header at 0, stretch, marketplace_button, import_button)
        # Layout order: header, [talent items...], stretch, marketplace_button, import_button
        # Remove everything between header (index 0) and the stretch
        while self._layout.count() > 4:  # header + stretch + marketplace + import
            item = self._layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        self._items.clear()

        # Remove the stretch (now at index 1), add items, re-add stretch
        self._layout.takeAt(1)

        for info in talent_list:
            item = TalentItem(info)
            item.toggled.connect(self.talent_toggled.emit)
            item.configure_requested.connect(self.configure_requested.emit)
            self._items[info["name"]] = item
            # Insert before the marketplace button (which is near the end)
            self._layout.insertWidget(self._layout.count() - 2, item)

        # Re-add stretch before marketplace button
        self._layout.insertStretch(self._layout.count() - 2)

    def highlight_talent(self, talent_name):
        """Highlight which talent handled the last command (auto-clear after 3s).

        Deferred to the next event-loop tick via QTimer.singleShot(0) to
        avoid access violations when called during signal delivery from a
        QThread — gives Qt time to finish processing pending events.
        """
        QTimer.singleShot(0, lambda: self._do_highlight(talent_name))

    def _do_highlight(self, talent_name):
        try:
            self._highlight_generation += 1
            gen = self._highlight_generation

            for name, item in list(self._items.items()):
                try:
                    item.set_highlighted(name == talent_name)
                except RuntimeError:
                    # C++ object deleted — skip stale widget
                    pass

            QTimer.singleShot(3000, lambda g=gen: self._clear_highlights(g))
        except Exception as e:
            print(f"   [Sidebar] highlight error: {e}")

    def _clear_highlights(self, generation):
        """Clear highlights only if no newer highlight has been requested."""
        if generation != self._highlight_generation:
            return
        for item in list(self._items.values()):
            try:
                item.set_highlighted(False)
            except RuntimeError:
                # Widget already deleted
                pass
