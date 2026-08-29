"""Tests for talent/MCP manifests, coverage, and host preflight."""

import ast
from pathlib import Path

from core.assistant import TalonAssistant
from core.capabilities import CapabilityBroker, DEFAULT_POLICIES
from core.capability_manifest import (
    BUILTIN_CAPABILITY_MANIFESTS,
    build_inventory,
    coverage_counts,
    inspect_source_manifest,
    inspect_talent,
)
from core.marketplace import MarketplaceClient
from core.mcp_client import MCPManager
from talents.base import BaseTalent


class _UndeclaredTalent(BaseTalent):
    name = "undeclared_test"
    description = "Test undeclared handling"

    def execute(self, command, context):
        raise AssertionError("undeclared talent must not execute")


class _ReadOnlyTalent(BaseTalent):
    name = "read_only_test"
    description = "Test read-only handling"
    capability_manifest = {"access": "read_only"}

    def execute(self, command, context):
        return {"success": True, "response": "read", "actions_taken": []}


class _DeviceTalent(BaseTalent):
    name = "device_test"
    description = "Test host preflight"
    capability_manifest = {
        "access": "brokered",
        "capabilities": ("device_control",),
        "enforcement": "host",
    }

    def __init__(self):
        super().__init__()
        self.calls = 0

    def execute(self, command, context):
        self.calls += 1
        return {"success": True, "response": "controlled", "actions_taken": []}


def _assistant_with_broker(config=None):
    assistant = object.__new__(TalonAssistant)
    assistant.capabilities = CapabilityBroker(config or {})
    return assistant


def test_every_builtin_talent_has_a_registry_manifest():
    talent_names = set()
    for source_path in Path("talents").glob("*.py"):
        if source_path.name in {"base.py", "__init__.py"}:
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(
                (isinstance(base, ast.Name) and base.id == "BaseTalent")
                or (isinstance(base, ast.Attribute) and base.attr == "BaseTalent")
                for base in node.bases
            ):
                continue
            for item in node.body:
                if not isinstance(item, ast.Assign):
                    continue
                if any(isinstance(target, ast.Name) and target.id == "name"
                       for target in item.targets):
                    if isinstance(item.value, ast.Constant):
                        talent_names.add(item.value.value)

    assert talent_names == set(BUILTIN_CAPABILITY_MANIFESTS)
    declared = {
        capability
        for manifest in BUILTIN_CAPABILITY_MANIFESTS.values()
        for capability in manifest.get("capabilities", ())
    }
    assert declared <= set(DEFAULT_POLICIES)


def test_manifest_inventory_reports_each_coverage_state():
    items = build_inventory([_ReadOnlyTalent(), _DeviceTalent(), _UndeclaredTalent()])
    counts = coverage_counts(items)
    assert counts == {"protected": 1, "read_only": 1, "undeclared": 1}
    assert inspect_talent(_DeviceTalent()).enforcement == "host"


def test_undeclared_talent_is_blocked_before_execute():
    assistant = _assistant_with_broker()
    result = assistant._invoke_talent(
        _UndeclaredTalent(), "do something", {"command_source": "local"}
    )
    assert result["success"] is False
    assert "Blocked undeclared talent" in result["response"]


def test_read_only_talent_executes_without_a_broker_request():
    assistant = _assistant_with_broker()
    result = assistant._invoke_talent(
        _ReadOnlyTalent(), "read something", {"command_source": "signal"}
    )
    assert result["success"] is True
    assert assistant.capabilities.pending_count == 0


def test_host_preflight_is_confirmed_through_the_originating_source():
    assistant = _assistant_with_broker()
    talent = _DeviceTalent()
    result = assistant._invoke_talent(
        talent, "turn on device", {"command_source": "signal"}
    )
    assert result["success"] is False
    assert "Approval required" in result["response"]
    assert talent.calls == 0

    request_id = result["response"].split("confirm ", 1)[1].split("'", 1)[0]
    assert assistant.capabilities.resolve_confirmation(
        f"confirm {request_id}", source="local"
    ) is None
    approved = assistant.capabilities.resolve_confirmation(
        f"confirm {request_id}", source="signal"
    )
    assert approved["success"] is True
    assert talent.calls == 1


def test_mcp_inventory_classifies_read_and_write_tools():
    manager = object.__new__(MCPManager)
    manager._tool_routes = {
        "mcp__files__read_file": ("files", "read_file"),
        "mcp__files__write_file": ("files", "write_file"),
    }
    inventory = manager.capability_inventory()
    assert inventory[0]["status"] == "read_only"
    assert inventory[1]["status"] == "protected"
    assert inventory[1]["capabilities"] == ("mcp_write",)


def test_marketplace_rejects_talent_without_manifest():
    source = '''
from talents.base import BaseTalent
class MissingManifestTalent(BaseTalent):
    name = "missing_manifest"
    description = "No declaration"
    def execute(self, command, context):
        return {"success": True, "response": "ok"}
'''
    result = MarketplaceClient.validate_source(source)
    assert result["valid"] is False
    assert "capability_manifest" in result["error"]

    untrusted_internal = source.replace(
        'description = "No declaration"',
        'description = "No declaration"\n'
        '    capability_manifest = {"access": "brokered", '
        '"capabilities": ("local_data_write",), "enforcement": "internal"}',
    )
    result = MarketplaceClient.validate_source(untrusted_internal)
    assert result["valid"] is False
    assert "host enforcement" in result["error"]


def test_source_manifest_is_checked_without_importing_code(tmp_path):
    marker = tmp_path / "should_not_exist.txt"
    source_path = tmp_path / "user_talent.py"
    source_path.write_text(f'''
from pathlib import Path
from talents.base import BaseTalent
Path(r"{marker}").write_text("imported")
class UserTalent(BaseTalent):
    name = "user_talent"
    description = "Declared reader"
    capability_manifest = {{"access": "read_only"}}
    def execute(self, command, context):
        return {{"success": True, "response": "ok"}}
''', encoding="utf-8")

    result = inspect_source_manifest(source_path)
    assert result.status == "read_only"
    assert not marker.exists()


def test_user_talent_cannot_inherit_builtin_manifest_by_name(tmp_path):
    source_path = tmp_path / "fake_email.py"
    source_path.write_text('''
from talents.base import BaseTalent
class FakeEmailTalent(BaseTalent):
    name = "email"
    description = "Missing declaration"
    def execute(self, command, context):
        return {"success": True, "response": "ok"}
''', encoding="utf-8")
    result = inspect_source_manifest(source_path)
    assert result.status == "undeclared"

    class RuntimeImpostor(_UndeclaredTalent):
        name = "email"

    assert inspect_talent(RuntimeImpostor()).status == "undeclared"


def test_third_party_sandbox_permissions_are_validated_and_inventoried(tmp_path):
    valid = tmp_path / "network_reader.py"
    valid.write_text('''
from talents.base import BaseTalent
class NetworkReader(BaseTalent):
    name = "network_reader"
    description = "Declared network reader"
    capability_manifest = {
        "access": "read_only",
        "sandbox": {"network": True, "llm": False},
    }
    def execute(self, command, context):
        return {"success": True, "response": "ok"}
''', encoding="utf-8")
    item = inspect_source_manifest(valid)
    assert item.status == "read_only"
    assert item.sandbox == "required"
    assert "network" in item.detail

    invalid = tmp_path / "invalid_sandbox.py"
    invalid.write_text(valid.read_text(encoding="utf-8").replace(
        '{"network": True, "llm": False}',
        '{"subprocess": True, "unknown_permission": True}',
    ), encoding="utf-8")
    item = inspect_source_manifest(invalid)
    assert item.status == "undeclared"
    assert item.sandbox == "blocked"
    assert "process_execution" in item.detail
    assert "unknown sandbox permissions" in item.detail
