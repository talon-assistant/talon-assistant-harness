"""GUI editor and redacted audit viewer for Talon's capability broker."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.capabilities import (
    AUDIT_EVENTS,
    DEFAULT_POLICIES,
    POLICY_SOURCES,
    CapabilityBroker,
)
from core.capability_manifest import coverage_counts
from core.config import update_settings


CAPABILITY_LABELS = {
    "external_send": "External sends",
    "rule_write": "Rule changes",
    "destructive_file_ops": "Destructive file operations",
    "mcp_write": "Mutating MCP tools",
    "desktop_control": "Desktop control",
    "plugin_install": "Talent install / removal",
    "local_data_write": "Local data changes",
    "device_control": "Device and media control",
    "clipboard_write": "Clipboard changes",
    "process_execution": "Process execution",
    "credential_write": "Credential changes",
    "external_account_write": "External account changes",
}

AUDIT_CAPABILITY_LABELS = {
    **CAPABILITY_LABELS,
    "talent_sandbox": "Talent sandbox",
}

SOURCE_LABELS = {
    "local": "Local",
    "signal": "Signal",
    "hermes": "Hermes",
    "scheduler": "Scheduler",
    "reflection": "Reflection",
    "default": "Other",
}

DECISIONS = ("allow", "confirm", "deny")


class CapabilityPolicyEditor(QWidget):
    """Editable matrix of capability decisions by command source."""

    def __init__(self, config: dict | None = None, parent=None):
        super().__init__(parent)
        self._config = deepcopy(config) if isinstance(config, dict) else {}

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Choose what each command source may do. Confirm means approval "
            "must be returned through the same source that requested the action."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        options = QHBoxLayout()
        self.enabled_check = QCheckBox("Enable capability enforcement")
        self.enabled_check.setChecked(self._config.get("enabled", True))
        options.addWidget(self.enabled_check)

        options.addStretch()
        options.addWidget(QLabel("Approval timeout:"))
        self.ttl_spin = QSpinBox()
        self.ttl_spin.setRange(30, 3600)
        self.ttl_spin.setSuffix(" seconds")
        self.ttl_spin.setValue(self._bounded_ttl(
            self._config.get("pending_ttl_seconds", 300)
        ))
        options.addWidget(self.ttl_spin)

        options.addWidget(QLabel("Unknown capabilities:"))
        self.default_combo = self._decision_combo(
            self._config.get("default_decision", "confirm")
        )
        options.addWidget(self.default_combo)
        layout.addLayout(options)

        self.table = QTableWidget(len(CAPABILITY_LABELS), len(POLICY_SOURCES) + 1)
        self.table.setHorizontalHeaderLabels(
            ["Capability"] + [SOURCE_LABELS[source] for source in POLICY_SOURCES]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAccessibleName("Capability policy matrix")

        policies = self._config.get("policies") or {}
        if not isinstance(policies, dict):
            policies = {}
        self._combos: dict[tuple[str, str], QComboBox] = {}
        for row, (capability, label) in enumerate(CAPABILITY_LABELS.items()):
            name_item = QTableWidgetItem(label)
            name_item.setData(Qt.ItemDataRole.UserRole, capability)
            name_item.setToolTip(capability)
            self.table.setItem(row, 0, name_item)

            override = policies.get(capability, {})
            for column, source in enumerate(POLICY_SOURCES, start=1):
                if isinstance(override, str):
                    decision = override
                elif isinstance(override, dict) and source in override:
                    decision = override[source]
                else:
                    decision = DEFAULT_POLICIES[capability][source]
                combo = self._decision_combo(decision)
                combo.setAccessibleName(f"{label}, {SOURCE_LABELS[source]}")
                combo.currentTextChanged.connect(self._update_warning)
                self._combos[(capability, source)] = combo
                self.table.setCellWidget(row, column, combo)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, len(POLICY_SOURCES) + 1):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.resizeRowsToContents()
        layout.addWidget(self.table)

        footer = QHBoxLayout()
        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setObjectName("capability_policy_warning")
        footer.addWidget(self.warning_label, 1)
        self.restore_button = QPushButton("Restore Safe Defaults")
        self.restore_button.clicked.connect(self.restore_safe_defaults)
        footer.addWidget(self.restore_button)
        layout.addLayout(footer)

        self.enabled_check.toggled.connect(self._update_warning)
        self.default_combo.currentTextChanged.connect(self._update_warning)
        self._update_warning()

    @staticmethod
    def _decision_combo(value: str) -> QComboBox:
        combo = QComboBox()
        combo.addItems(DECISIONS)
        value = str(value).lower()
        combo.setCurrentText(value if value in DECISIONS else "confirm")
        return combo

    @staticmethod
    def _bounded_ttl(value) -> int:
        try:
            return max(30, min(3600, int(value)))
        except (TypeError, ValueError):
            return 300

    def policy_config(self) -> dict:
        policies = {}
        for capability in CAPABILITY_LABELS:
            policies[capability] = {
                source: self._combos[(capability, source)].currentText()
                for source in POLICY_SOURCES
            }
        return {
            "enabled": self.enabled_check.isChecked(),
            "pending_ttl_seconds": self.ttl_spin.value(),
            "default_decision": self.default_combo.currentText(),
            "policies": policies,
        }

    def restore_safe_defaults(self, checked=False, *, ask: bool = True) -> None:
        del checked
        if ask:
            answer = QMessageBox.question(
                self,
                "Restore safe defaults",
                "Replace all displayed capability decisions with Talon's safe defaults?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.enabled_check.setChecked(True)
        self.ttl_spin.setValue(300)
        self.default_combo.setCurrentText("confirm")
        for capability, policy in DEFAULT_POLICIES.items():
            for source, decision in policy.items():
                self._combos[(capability, source)].setCurrentText(decision)
        self._update_warning()

    def _update_warning(self, _value=None) -> None:
        if not self.enabled_check.isChecked():
            self.warning_label.setText(
                "Warning: enforcement is disabled; all capability requests are allowed."
            )
            return

        unattended = []
        remote = []
        fallback = []
        local_side_effects = []
        for capability, label in CAPABILITY_LABELS.items():
            for source in ("scheduler", "reflection"):
                if self._combos[(capability, source)].currentText() == "allow":
                    unattended.append(f"{label} from {SOURCE_LABELS[source]}")
            for source in ("signal", "hermes"):
                if self._combos[(capability, source)].currentText() == "allow":
                    remote.append(f"{label} from {SOURCE_LABELS[source]}")
            if self._combos[(capability, "default")].currentText() == "allow":
                fallback.append(label)
            if (DEFAULT_POLICIES[capability]["local"] != "allow"
                    and self._combos[(capability, "local")].currentText() == "allow"):
                local_side_effects.append(label)

        messages = []
        if unattended:
            messages.append(
                f"{len(unattended)} unattended route(s) can act without approval."
            )
        if remote:
            messages.append(f"{len(remote)} remote route(s) bypass approval.")
        if fallback:
            messages.append(f"{len(fallback)} source fallback(s) bypass approval.")
        if local_side_effects:
            messages.append(
                f"{len(local_side_effects)} local side-effect route(s) bypass approval."
            )
        if self.default_combo.currentText() == "allow":
            messages.append("Unknown capabilities are allowed without approval.")
        self.warning_label.setText(
            "Warning: " + " ".join(messages)
            if messages else (
                "Safe posture: remote and unattended actions require approval "
                "or are denied."
            )
        )


class CapabilityAuditViewer(QWidget):
    """Paginated view of the broker's deliberately redacted audit records."""

    def __init__(self, broker: CapabilityBroker, parent=None):
        super().__init__(parent)
        self._broker = broker
        self._offset = 0
        self._total = 0

        layout = QVBoxLayout(self)
        privacy = QLabel(
            "Audit entries contain decisions, short summaries, and bounded errors only. "
            "Message bodies, credentials, file contents, and action metadata are not stored."
        )
        privacy.setWordWrap(True)
        layout.addWidget(privacy)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Capability:"))
        self.capability_filter = QComboBox()
        self.capability_filter.addItem("All", None)
        for capability, label in AUDIT_CAPABILITY_LABELS.items():
            self.capability_filter.addItem(label, capability)
        filters.addWidget(self.capability_filter)

        filters.addWidget(QLabel("Source:"))
        self.source_filter = QComboBox()
        self.source_filter.addItem("All", None)
        for source in POLICY_SOURCES[:-1]:
            self.source_filter.addItem(SOURCE_LABELS[source], source)
        filters.addWidget(self.source_filter)

        filters.addWidget(QLabel("Event:"))
        self.event_filter = QComboBox()
        self.event_filter.addItem("All", None)
        for event in AUDIT_EVENTS:
            self.event_filter.addItem(event.replace("_", " ").title(), event)
        filters.addWidget(self.event_filter)

        filters.addStretch()
        filters.addWidget(QLabel("Rows:"))
        self.page_size = QComboBox()
        for size in (25, 50, 100):
            self.page_size.addItem(str(size), size)
        self.page_size.setCurrentText("50")
        filters.addWidget(self.page_size)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        filters.addWidget(self.refresh_button)
        layout.addLayout(filters)

        columns = ("Time", "Capability", "Source", "Event", "Summary", "Request", "Error")
        self.table = QTableWidget(0, len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)
        self.table.setAccessibleName("Capability audit records")
        header = self.table.horizontalHeader()
        for column in (0, 1, 2, 3, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        navigation = QHBoxLayout()
        self.page_label = QLabel()
        navigation.addWidget(self.page_label)
        navigation.addStretch()
        self.previous_button = QPushButton("Previous")
        self.previous_button.clicked.connect(self._previous_page)
        navigation.addWidget(self.previous_button)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self._next_page)
        navigation.addWidget(self.next_button)
        layout.addLayout(navigation)

        for combo in (
            self.capability_filter, self.source_filter,
            self.event_filter, self.page_size,
        ):
            combo.currentIndexChanged.connect(self._filters_changed)
        self.refresh()

    def _filters(self) -> dict:
        return {
            "capability": self.capability_filter.currentData(),
            "source": self.source_filter.currentData(),
            "event": self.event_filter.currentData(),
        }

    def refresh(self, checked=False) -> None:
        del checked
        filters = self._filters()
        page_size = int(self.page_size.currentData())
        self._total = self._broker.count_audit(**filters)
        if self._total == 0:
            self._offset = 0
        elif self._offset >= self._total:
            self._offset = ((self._total - 1) // page_size) * page_size
        rows = self._broker.query_audit(
            **filters, limit=page_size, offset=self._offset
        )

        self.table.setRowCount(len(rows))
        for row_index, record in enumerate(rows):
            values = (
                self._format_time(record.get("timestamp")),
                AUDIT_CAPABILITY_LABELS.get(
                    record.get("capability"), record.get("capability", "")
                ),
                SOURCE_LABELS.get(record.get("source"), record.get("source", "")),
                str(record.get("event", "")).replace("_", " ").title(),
                record.get("summary", ""),
                record.get("request_id", ""),
                record.get("error", ""),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value or ""))
                item.setToolTip(str(value or ""))
                self.table.setItem(row_index, column, item)

        first = self._offset + 1 if rows else 0
        last = self._offset + len(rows)
        self.page_label.setText(f"Showing {first}–{last} of {self._total}")
        self.previous_button.setEnabled(self._offset > 0)
        self.next_button.setEnabled(self._offset + len(rows) < self._total)

    @staticmethod
    def _format_time(timestamp) -> str:
        try:
            return datetime.fromtimestamp(float(timestamp)).astimezone().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except (TypeError, ValueError, OSError):
            return ""

    def _filters_changed(self, _index) -> None:
        self._offset = 0
        self.refresh()

    def _previous_page(self) -> None:
        self._offset = max(0, self._offset - int(self.page_size.currentData()))
        self.refresh()

    def _next_page(self) -> None:
        page_size = int(self.page_size.currentData())
        if self._offset + page_size < self._total:
            self._offset += page_size
        self.refresh()


class CapabilityInventoryViewer(QWidget):
    """Coverage inventory plus a side-effect-free policy simulator."""

    def __init__(self, broker: CapabilityBroker, inventory_provider=None, parent=None):
        super().__init__(parent)
        self._broker = broker
        self._inventory_provider = inventory_provider

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Protected entries declare brokered side effects; read-only entries "
            "declare no privileged mutations. Undeclared talents are disabled "
            "and blocked at dispatch. Third-party talent code runs only in a "
            "fresh isolated worker."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        simulator = QHBoxLayout()
        simulator.addWidget(QLabel("Policy simulator — source:"))
        self.sim_source = QComboBox()
        for source in POLICY_SOURCES[:-1]:
            self.sim_source.addItem(SOURCE_LABELS[source], source)
        simulator.addWidget(self.sim_source)
        simulator.addWidget(QLabel("Capability:"))
        self.sim_capability = QComboBox()
        for capability, label in CAPABILITY_LABELS.items():
            self.sim_capability.addItem(label, capability)
        simulator.addWidget(self.sim_capability)
        self.sim_result = QLabel()
        self.sim_result.setObjectName("capability_simulation_result")
        simulator.addWidget(self.sim_result)
        simulator.addStretch()
        layout.addLayout(simulator)

        summary_row = QHBoxLayout()
        self.coverage_label = QLabel()
        summary_row.addWidget(self.coverage_label)
        summary_row.addStretch()
        refresh_button = QPushButton("Refresh Inventory")
        refresh_button.clicked.connect(self.refresh)
        summary_row.addWidget(refresh_button)
        layout.addLayout(summary_row)

        columns = (
            "Owner", "Type", "Access", "Capabilities", "Enforcement",
            "Sandbox", "Status", "Detail",
        )
        self.table = QTableWidget(0, len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setAccessibleName("Talent and MCP capability inventory")
        header = self.table.horizontalHeader()
        for column in (1, 2, 4, 5, 6):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        self.sim_source.currentIndexChanged.connect(self.update_simulation)
        self.sim_capability.currentIndexChanged.connect(self.update_simulation)
        self.update_simulation()
        self.refresh()

    def update_simulation(self, _index=None) -> None:
        capability = self.sim_capability.currentData()
        source = self.sim_source.currentData()
        decision = self._broker.effective_decision(capability, source)
        self.sim_result.setText(f"Decision: {decision.value.upper()}")
        self.sim_result.setToolTip(
            "Simulation only — no request is created and no action is executed."
        )

    def refresh(self, checked=False) -> None:
        del checked
        if callable(self._inventory_provider):
            try:
                items = list(self._inventory_provider())
            except Exception:
                items = []
        else:
            items = list(self._inventory_provider or [])

        counts = coverage_counts(items)
        self.coverage_label.setText(
            f"Coverage: {counts.get('protected', 0)} protected · "
            f"{counts.get('read_only', 0)} read-only · "
            f"{counts.get('undeclared', 0)} undeclared"
        )
        self.table.setRowCount(len(items))
        for row, item in enumerate(items):
            record = item.to_dict() if hasattr(item, "to_dict") else dict(item)
            capabilities = record.get("capabilities") or ()
            values = (
                record.get("owner", ""),
                str(record.get("owner_type", "")).upper(),
                str(record.get("access", "")).replace("_", " ").title(),
                ", ".join(capabilities),
                str(record.get("enforcement", "")).title(),
                str(record.get("sandbox", "")).replace("_", " ").title(),
                str(record.get("status", "")).replace("_", " ").title(),
                record.get("detail", ""),
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                cell.setToolTip(str(value))
                self.table.setItem(row, column, cell)


class CapabilityCenterDialog(QDialog):
    """Top-level Capability Center with policy and audit tabs."""

    settings_saved = pyqtSignal(dict)

    def __init__(
        self,
        current_settings: dict,
        config_path: str,
        broker: CapabilityBroker,
        parent=None,
        inventory_provider=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Capability Center")
        self.setMinimumSize(900, 560)
        self.resize(1080, 680)
        self._config_path = config_path

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.policy_editor = CapabilityPolicyEditor(
            current_settings.get("capabilities", {})
        )
        self.audit_viewer = CapabilityAuditViewer(broker)
        self.inventory_viewer = CapabilityInventoryViewer(
            broker, inventory_provider
        )
        self.tabs.addTab(self.policy_editor, "Policy Editor")
        self.tabs.addTab(self.inventory_viewer, "Inventory & Simulator")
        self.tabs.addTab(self.audit_viewer, "Audit Viewer")
        layout.addWidget(self.tabs)

        buttons = QHBoxLayout()
        self.status_label = QLabel()
        buttons.addWidget(self.status_label)
        buttons.addStretch()
        save_button = QPushButton("Save Policy")
        save_button.clicked.connect(self._save_policy)
        buttons.addWidget(save_button)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    def _save_policy(self) -> None:
        try:
            merged = update_settings(
                self._config_path,
                {"capabilities": self.policy_editor.policy_config()},
                replace_sections=("capabilities",),
            )
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, "Policy save failed", str(exc))
            return
        self.settings_saved.emit(merged)
        self.status_label.setText("Policy saved and applied to new requests.")
        self.inventory_viewer.update_simulation()
        self.audit_viewer.refresh()
