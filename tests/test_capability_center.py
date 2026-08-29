"""Offscreen widget tests for the Capability Center."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from core.capabilities import CapabilityBroker, CapabilityDecision, DEFAULT_POLICIES
from core.capability_manifest import CapabilityInventoryItem
from gui.dialogs.capability_center_dialog import (
    CAPABILITY_LABELS,
    CapabilityAuditViewer,
    CapabilityCenterDialog,
    CapabilityInventoryViewer,
    CapabilityPolicyEditor,
)


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_policy_editor_expands_partial_config_and_restores_defaults(qt_app):
    assert set(CAPABILITY_LABELS) == set(DEFAULT_POLICIES)
    editor = CapabilityPolicyEditor({
        "pending_ttl_seconds": 90,
        "policies": {"external_send": {"signal": "deny"}},
    })

    config = editor.policy_config()
    assert config["pending_ttl_seconds"] == 90
    assert config["policies"]["external_send"]["signal"] == "deny"
    assert config["policies"]["external_send"]["scheduler"] == "deny"

    editor._combos[("desktop_control", "scheduler")].setCurrentText("allow")
    assert "unattended route" in editor.warning_label.text()
    editor.enabled_check.setChecked(False)
    assert "enforcement is disabled" in editor.warning_label.text()
    editor.enabled_check.setChecked(True)
    editor.default_combo.setCurrentText("allow")
    assert "Unknown capabilities are allowed" in editor.warning_label.text()

    editor.restore_safe_defaults(ask=False)
    restored = editor.policy_config()
    assert restored["enabled"] is True
    assert restored["pending_ttl_seconds"] == 300
    assert restored["policies"]["desktop_control"]["local"] == "allow"
    assert restored["policies"]["desktop_control"]["scheduler"] == "deny"
    assert "Safe posture" in editor.warning_label.text()

    malformed = CapabilityPolicyEditor({
        "pending_ttl_seconds": "not a number", "policies": [],
    })
    assert malformed.policy_config()["pending_ttl_seconds"] == 300


def test_audit_viewer_filters_and_paginates_redacted_records(qt_app, tmp_path):
    broker = CapabilityBroker({
        "policies": {"external_send": {"local": "allow"}},
    }, db_path=str(tmp_path / "audit.db"))
    for index in range(30):
        broker.request(
            "external_send",
            source="local",
            summary=f"Send summary {index}",
            metadata={"body": f"secret body {index}"},
        )
    broker.request(
        "plugin_install", source="scheduler", summary="Install blocked talent"
    )
    broker.record_event(
        "talent_sandbox", source="signal", summary="Sandbox test",
        event="sandbox_completed",
    )

    viewer = CapabilityAuditViewer(broker)
    viewer.page_size.setCurrentText("25")
    assert viewer.table.rowCount() == 25
    assert viewer.page_label.text() == "Showing 1–25 of 32"
    assert viewer.next_button.isEnabled()

    viewer.next_button.click()
    assert viewer.table.rowCount() == 7
    assert viewer.page_label.text() == "Showing 26–32 of 32"

    viewer.source_filter.setCurrentIndex(viewer.source_filter.findData("scheduler"))
    assert viewer.table.rowCount() == 1
    assert viewer.table.item(0, 3).text() == "Denied"
    visible = " ".join(
        viewer.table.item(0, column).text()
        for column in range(viewer.table.columnCount())
    )
    assert "secret body" not in visible

    viewer.capability_filter.setCurrentIndex(
        viewer.capability_filter.findData("talent_sandbox")
    )
    viewer.source_filter.setCurrentIndex(0)
    assert viewer.table.rowCount() == 1
    assert viewer.table.item(0, 1).text() == "Talent sandbox"


def test_capability_center_saves_and_hot_reloads_policy(qt_app, tmp_path):
    config_path = tmp_path / "settings.json"
    config_path.write_text(
        json.dumps({
            "unrelated": {"keep": True},
            "capabilities": {
                "policies": {"external_send": {"hidden_source": "allow"}},
            },
        }),
        encoding="utf-8",
    )
    broker = CapabilityBroker()
    dialog = CapabilityCenterDialog({}, str(config_path), broker)
    dialog.settings_saved.connect(
        lambda settings: broker.reload(settings["capabilities"])
    )
    dialog.policy_editor._combos[
        ("external_send", "signal")
    ].setCurrentText("deny")

    dialog._save_policy()

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["unrelated"] == {"keep": True}
    assert saved["capabilities"]["policies"]["external_send"]["signal"] == "deny"
    assert "hidden_source" not in saved["capabilities"]["policies"]["external_send"]
    assert broker.effective_decision(
        "external_send", "signal"
    ) is CapabilityDecision.DENY
    assert "saved and applied" in dialog.status_label.text()


def test_inventory_viewer_reports_coverage_and_simulates_without_request(qt_app):
    broker = CapabilityBroker()
    inventory = [
        CapabilityInventoryItem(
            "email", "talent", "brokered", ("external_send",),
            "internal", "protected", "Action-aware internal checks",
        ),
        CapabilityInventoryItem(
            "weather", "talent", "read_only", (), "none", "read_only", "",
        ),
    ]
    viewer = CapabilityInventoryViewer(broker, inventory)
    assert viewer.table.rowCount() == 2
    assert viewer.table.columnCount() == 8
    assert viewer.table.horizontalHeaderItem(5).text() == "Sandbox"
    assert "1 protected" in viewer.coverage_label.text()
    assert "1 read-only" in viewer.coverage_label.text()

    viewer.sim_source.setCurrentIndex(viewer.sim_source.findData("scheduler"))
    viewer.sim_capability.setCurrentIndex(
        viewer.sim_capability.findData("external_send")
    )
    assert viewer.sim_result.text() == "Decision: DENY"
    assert broker.pending_count == 0
