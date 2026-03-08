"""Dynamic per-talent configuration dialog.

Builds a form from the talent's get_config_schema() output, using the
same widget patterns as SettingsDialog.
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
                             QComboBox, QLabel, QPushButton)
from PyQt6.QtCore import pyqtSignal, Qt


class TalentConfigDialog(QDialog):
    """Config dialog for a single talent, built dynamically from schema."""

    config_saved = pyqtSignal(str, dict)  # (talent_name, config_dict)

    def __init__(self, talent_name, display_name, schema, current_config,
                 parent=None, credential_store=None):
        super().__init__(parent)
        self.talent_name = talent_name
        self._fields = {}  # key -> (kind, widget)
        self._schema = schema
        self._current = current_config or {}
        self._credential_store = credential_store

        self.setWindowTitle(f"Configure \u2014 {display_name}")
        self.setMinimumWidth(420)
        self.resize(480, 0)  # Height will auto-fit

        layout = QVBoxLayout(self)

        # Header
        header = QLabel(f"<b>{display_name}</b> settings")
        header.setObjectName("talent_config_header")
        layout.addWidget(header)

        # Form
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setSpacing(10)

        fields = schema.get("fields", [])
        for field in fields:
            self._add_field(form, field)

        layout.addLayout(form)

        # No-config fallback (shouldn't happen but be safe)
        if not fields:
            layout.addWidget(QLabel("This talent has no configurable settings."))

        layout.addSpacing(12)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        save_btn = QPushButton("Save")
        save_btn.setObjectName("talent_config_save_btn")
        save_btn.clicked.connect(self._on_save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _add_field(self, form, field):
        """Add a single config field to the form."""
        key = field["key"]
        label = field.get("label", key)
        ftype = field.get("type", "string")
        default = field.get("default", "")
        value = self._current.get(key, default)

        if ftype == "string":
            widget = QLineEdit(str(value))
            self._fields[key] = ("line", widget)
            form.addRow(label, widget)

        elif ftype == "password":
            # Check if a keyring secret exists for this field
            has_keyring_secret = False
            if self._credential_store and not value:
                has_keyring_secret = self._credential_store.has_secret(
                    self.talent_name, key)

            widget = QLineEdit(str(value))
            widget.setEchoMode(QLineEdit.EchoMode.Password)

            if has_keyring_secret and not value:
                widget.setPlaceholderText("(stored in system keyring)")

            # Add show/hide toggle
            container = QHBoxLayout()
            container.setContentsMargins(0, 0, 0, 0)
            container.addWidget(widget)
            toggle_btn = QPushButton("\U0001f441")  # eye emoji
            toggle_btn.setFixedWidth(32)
            toggle_btn.setCheckable(True)
            toggle_btn.toggled.connect(
                lambda checked, w=widget: w.setEchoMode(
                    QLineEdit.EchoMode.Normal if checked
                    else QLineEdit.EchoMode.Password))
            container.addWidget(toggle_btn)
            self._fields[key] = ("password", widget)
            form.addRow(label, container)

        elif ftype == "int":
            widget = QSpinBox()
            widget.setRange(
                field.get("min", 0),
                field.get("max", 99999))
            widget.setValue(int(value))
            self._fields[key] = ("spin", widget)
            form.addRow(label, widget)

        elif ftype == "float":
            widget = QDoubleSpinBox()
            widget.setRange(
                field.get("min", 0.0),
                field.get("max", 99999.0))
            widget.setSingleStep(field.get("step", 0.1))
            widget.setDecimals(2)
            widget.setValue(float(value))
            self._fields[key] = ("dspin", widget)
            form.addRow(label, widget)

        elif ftype == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(value))
            self._fields[key] = ("check", widget)
            form.addRow(label, widget)

        elif ftype == "choice":
            widget = QComboBox()
            choices = field.get("choices", [])
            widget.addItems(choices)
            idx = widget.findText(str(value))
            if idx >= 0:
                widget.setCurrentIndex(idx)
            self._fields[key] = ("combo", widget)
            form.addRow(label, widget)

        elif ftype == "list":
            from gui.dialogs.settings_dialog import ListEditor
            items = value if isinstance(value, list) else []
            widget = ListEditor(items)
            self._fields[key] = ("list", widget)
            form.addRow(label, widget)

        elif ftype == "feed_table":
            from gui.dialogs.settings_dialog import FeedTableEditor
            feeds = value if isinstance(value, list) else []
            widget = FeedTableEditor(feeds)
            # Feed table is wide — expand dialog to accommodate
            self.setMinimumWidth(680)
            self.resize(720, 0)
            self._fields[key] = ("feed_table", widget)
            form.addRow(label, widget)

    def _collect_values(self):
        """Read all widget values into a flat config dict.

        For password fields: if the user left the field empty and a keyring
        secret already exists, we pull the real value so the bridge doesn't
        interpret an empty string as "delete the credential".
        """
        result = {}
        for key, (kind, widget) in self._fields.items():
            if kind == "line":
                result[key] = widget.text()
            elif kind == "password":
                text = widget.text()
                if not text and self._credential_store:
                    # Preserve existing keyring secret when field left blank
                    existing = self._credential_store.get_secret(
                        self.talent_name, key)
                    result[key] = existing if existing else ""
                else:
                    result[key] = text
            elif kind == "spin":
                result[key] = widget.value()
            elif kind == "dspin":
                result[key] = round(widget.value(), 2)
            elif kind == "check":
                result[key] = widget.isChecked()
            elif kind == "combo":
                result[key] = widget.currentText()
            elif kind == "list":
                result[key] = widget.get_items()
            elif kind == "feed_table":
                result[key] = widget.get_feeds()
        return result

    def _on_save(self):
        config = self._collect_values()
        self.config_saved.emit(self.talent_name, config)
        self.accept()
