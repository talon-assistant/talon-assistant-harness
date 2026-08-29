"""Tests for centralized capability policy, confirmation, and auditing."""

import sqlite3
from unittest.mock import MagicMock

import pytest

from core.capabilities import CapabilityBroker, CapabilityDecision
from core.marketplace import MarketplaceClient
from talents.desktop_control import DesktopControlTalent
from talents.email_talent import EmailTalent
from talents.file_organizer import FileOrganizerTalent
from talents.rules import RulesTalent
from talents.signal_remote import SignalRemoteTalent


_VALID_TALENT_SOURCE = '''
from talents.base import BaseTalent

class ExampleTalent(BaseTalent):
    name = "example"
    description = "Example talent"
    keywords = ["example"]
    capability_manifest = {"access": "read_only"}

    def can_handle(self, command):
        return False

    def execute(self, command, context):
        return {"success": True, "response": "ok"}
'''


def test_default_external_send_requires_confirmation():
    broker = CapabilityBroker()
    auth = broker.request(
        "external_send", source="local", summary="Send a test email")
    assert auth.decision is CapabilityDecision.CONFIRM
    assert broker.pending_count == 1
    assert auth.request.request_id in broker.confirmation_message(auth)


def test_unattended_scheduler_send_is_denied():
    broker = CapabilityBroker()
    auth = broker.request(
        "external_send", source="scheduler", summary="Send daily report")
    assert auth.decision is CapabilityDecision.DENY
    assert broker.pending_count == 0


def test_config_can_allow_a_capability():
    broker = CapabilityBroker({
        "policies": {"external_send": {"local": "allow"}},
    })
    auth = broker.request(
        "external_send", source="local", summary="Send a test email")
    assert auth.allowed


def test_partial_policy_keeps_safe_defaults_for_other_sources():
    broker = CapabilityBroker({
        "policies": {"external_send": {"local": "allow"}},
    })
    auth = broker.request(
        "external_send", source="scheduler", summary="Send scheduled email")
    assert auth.decision is CapabilityDecision.DENY


def test_authorization_metadata_is_immutable():
    broker = CapabilityBroker()
    auth = broker.request(
        "plugin_install", source="local", summary="Install example",
        metadata={"filename": "example.py"})
    with pytest.raises(TypeError):
        auth.request.metadata["filename"] = "different.py"


def test_confirmation_runs_executor_once():
    broker = CapabilityBroker()
    executor = MagicMock(return_value={
        "response": "sent", "success": True, "actions_taken": []})
    auth = broker.request(
        "external_send", source="signal", summary="Send email",
        executor=executor)

    result = broker.resolve_confirmation(
        f"confirm {auth.request.request_id}", source="signal")
    assert result["success"] is True
    assert result["response"] == "sent"
    executor.assert_called_once_with()
    assert broker.resolve_confirmation(
        f"confirm {auth.request.request_id}", source="signal") is None


def test_confirmation_cannot_cross_command_sources():
    broker = CapabilityBroker()
    auth = broker.request(
        "external_send", source="signal", summary="Send email",
        executor=lambda: "sent")
    assert broker.resolve_confirmation(
        f"confirm {auth.request.request_id}", source="hermes") is None
    assert broker.pending_count == 1


def test_cancel_discards_pending_executor():
    broker = CapabilityBroker()
    executor = MagicMock()
    auth = broker.request(
        "rule_write", source="local", summary="Delete rule",
        executor=executor)
    result = broker.resolve_confirmation(
        f"cancel {auth.request.request_id}", source="local")
    assert result["success"] is True
    assert "Cancelled" in result["response"]
    executor.assert_not_called()
    assert broker.pending_count == 0


def test_multiple_pending_requests_require_an_id():
    broker = CapabilityBroker()
    first = broker.request(
        "external_send", source="local", summary="First")
    broker.request("external_send", source="local", summary="Second")
    assert broker.resolve_confirmation("confirm", source="local") is None
    assert broker.resolve_confirmation(
        f"confirm {first.request.request_id}", source="local") is not None


def test_audit_does_not_persist_metadata(tmp_path):
    db_path = str(tmp_path / "audit.db")
    broker = CapabilityBroker(db_path=db_path)
    broker.request(
        "external_send",
        source="local",
        summary="Send email to person@example.com",
        metadata={"body": "TOP SECRET BODY"},
    )
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT summary, event FROM capability_audit").fetchone()
        columns = [r[1] for r in conn.execute(
            "PRAGMA table_info(capability_audit)").fetchall()]
    assert row == ("Send email to person@example.com", "confirmation_required")
    assert "metadata" not in columns
    assert "TOP SECRET BODY" not in repr(row)


def test_effective_decision_does_not_create_a_request():
    broker = CapabilityBroker({
        "policies": {"external_send": {"signal": "deny"}},
    })

    assert broker.effective_decision(
        "external_send", "signal") is CapabilityDecision.DENY
    assert broker.pending_count == 0


def test_audit_query_filters_counts_and_paginates(tmp_path):
    db_path = str(tmp_path / "audit.db")
    broker = CapabilityBroker({
        "policies": {"external_send": {"local": "allow"}},
    }, db_path=db_path)
    broker.request("external_send", source="local", summary="First send")
    broker.request("external_send", source="local", summary="Second send")
    broker.request("plugin_install", source="scheduler", summary="Install talent")

    assert broker.count_audit() == 3
    assert broker.count_audit(
        capability="external_send", source="local", event="allowed") == 2

    first_page = broker.query_audit(
        capability="external_send", event="allowed", limit=1)
    second_page = broker.query_audit(
        capability="external_send", event="allowed", limit=1, offset=1)
    assert first_page[0]["summary"] == "Second send"
    assert second_page[0]["summary"] == "First send"
    assert set(first_page[0]) == {
        "id", "timestamp", "request_id", "capability", "source",
        "summary", "event", "error",
    }
    assert broker.query_audit(source="signal") == []


def test_audit_query_is_empty_without_durable_database():
    broker = CapabilityBroker()
    broker.request("external_send", source="local", summary="Not persisted")
    assert broker.query_audit() == []
    assert broker.count_audit() == 0


def test_signal_email_is_pending_until_confirmed():
    broker = CapabilityBroker()
    talent = EmailTalent()
    talent._config = {"imap_server": "imap.example.com",
                      "username": "me@example.com"}
    talent._send_smtp = MagicMock()

    result = talent._handle_send(
        "send the status",
        {"llm": None, "command_source": "signal", "capabilities": broker},
        typed={"to": "you@example.com", "subject": "Status", "body": "Done"},
    )
    assert result["success"] is False
    assert "Approval required" in result["response"]
    assert "Draft email:" in result["response"]
    assert "To: you@example.com" in result["response"]
    assert "Subject: Status" in result["response"]
    assert "Done" in result["response"]
    assert "pending_email" not in result
    talent._send_smtp.assert_not_called()

    request_id = result["response"].split("confirm ", 1)[1].split("'", 1)[0]
    approved = broker.resolve_confirmation(
        f"confirm {request_id}", source="signal")
    assert approved["success"] is True
    talent._send_smtp.assert_called_once()


def test_signal_email_delete_is_pending_until_remote_confirmation():
    broker = CapabilityBroker()
    talent = EmailTalent()
    talent._delete_now = MagicMock(return_value={
        "success": True, "response": "deleted", "actions_taken": []})

    result = talent._handle_delete(
        "delete email 3",
        {"command_source": "signal", "capabilities": broker},
    )
    assert result["success"] is False
    assert "Approval required" in result["response"]
    talent._delete_now.assert_not_called()

    request_id = result["response"].split("confirm ", 1)[1].split("'", 1)[0]
    approved = broker.resolve_confirmation(
        f"confirm {request_id}", source="signal")
    assert approved["success"] is True
    talent._delete_now.assert_called_once_with("delete email 3")


def test_scheduler_email_move_is_denied_before_imap_mutation():
    broker = CapabilityBroker()
    talent = EmailTalent()
    talent._move_now = MagicMock()

    result = talent._handle_move(
        "move email 2 to Archive",
        {"command_source": "scheduler", "capabilities": broker},
    )
    assert result["success"] is False
    assert "Blocked by capability policy" in result["response"]
    talent._move_now.assert_not_called()


def test_rule_delete_is_pending_until_confirmed():
    broker = CapabilityBroker()
    memory = MagicMock()
    memory.delete_rule.return_value = True
    talent = RulesTalent()
    context = {
        "memory": memory,
        "command_source": "local",
        "capabilities": broker,
    }
    result = talent.execute("delete rule 7", context)
    assert result["success"] is False
    memory.delete_rule.assert_not_called()

    request_id = result["response"].split("confirm ", 1)[1].split("'", 1)[0]
    approved = broker.resolve_confirmation(
        f"confirm {request_id}", source="local")
    assert approved["success"] is True
    memory.delete_rule.assert_called_once_with(7)


def test_signal_accepts_unprefixed_reply_for_pending_approval():
    broker = CapabilityBroker()
    executor = MagicMock(return_value={
        "response": "Email sent.", "success": True, "actions_taken": []})
    auth = broker.request(
        "external_send", source="signal", summary="Send status email",
        executor=executor)

    assistant = MagicMock()
    assistant.capabilities = broker
    assistant.process_command.side_effect = lambda command, **kwargs: (
        broker.resolve_confirmation(command, source=kwargs["command_source"])
    )

    talent = SignalRemoteTalent()
    talent._assistant = assistant
    talent._config = {
        "command_prefix": "talon: ",
        "account_number": "+15551234567",
        "max_response_chars": 1000,
    }
    talent._send_reply = MagicMock()

    talent._handle_envelope({
        "envelope": {
            "source": "+15551234567",
            "syncMessage": {"sentMessage": {
                "destination": "+15551234567",
                "message": f"confirm {auth.request.request_id}",
            }},
        },
    })

    assistant.process_command.assert_called_once()
    assert assistant.process_command.call_args.args[0] == (
        f"confirm {auth.request.request_id}")
    assert assistant.process_command.call_args.kwargs["command_source"] == "signal"
    executor.assert_called_once_with()
    talent._send_reply.assert_called_once_with(
        "+15551234567", "Email sent.", attachments=[])


def test_signal_still_ignores_other_unprefixed_messages_while_pending():
    broker = CapabilityBroker()
    broker.request("external_send", source="signal", summary="Send email")
    assistant = MagicMock()
    assistant.capabilities = broker

    talent = SignalRemoteTalent()
    talent._assistant = assistant
    talent._config = {
        "command_prefix": "talon: ",
        "account_number": "+15551234567",
    }

    talent._handle_envelope({
        "envelope": {
            "source": "+15551234567",
            "syncMessage": {"sentMessage": {
                "destination": "+15551234567",
                "message": "send it now",
            }},
        },
    })

    assistant.process_command.assert_not_called()


def test_signal_file_organize_waits_for_remote_confirmation(tmp_path):
    (tmp_path / "report.pdf").write_text("report", encoding="utf-8")
    broker = CapabilityBroker()
    talent = FileOrganizerTalent()

    result = talent._preview_organize(
        str(tmp_path),
        {"capabilities": broker, "command_source": "signal"},
    )

    assert result["success"] is False
    assert "Approval required" in result["response"]
    assert (tmp_path / "report.pdf").exists()
    assert talent._pending_organize is None

    request_id = result["response"].split("confirm ", 1)[1].split("'", 1)[0]
    approved = broker.resolve_confirmation(
        f"confirm {request_id}", source="signal")
    assert approved["success"] is True
    assert not (tmp_path / "report.pdf").exists()
    assert (tmp_path / "documents" / "report.pdf").exists()


def test_scheduler_file_organize_is_denied_without_changes(tmp_path):
    (tmp_path / "report.pdf").write_text("report", encoding="utf-8")
    broker = CapabilityBroker()
    talent = FileOrganizerTalent()

    result = talent._preview_organize(
        str(tmp_path),
        {"capabilities": broker, "command_source": "scheduler"},
    )

    assert result["success"] is False
    assert "Blocked by capability policy" in result["response"]
    assert (tmp_path / "report.pdf").exists()
    assert not (tmp_path / "documents").exists()


def test_signal_desktop_action_executes_only_after_confirmation():
    broker = CapabilityBroker()
    talent = DesktopControlTalent()
    talent.action_delay = 0
    talent._execute_single_action = MagicMock(return_value="Pressed: enter")
    llm = MagicMock()
    llm.generate.return_value = (
        '{"explanation":"Press Enter",'
        '"actions":[{"action":"press","key":"enter"}]}'
    )
    vision = MagicMock()

    result = talent._handle_desktop_action(
        "press enter", llm, vision, "", False, None,
        {"capabilities": broker, "command_source": "signal"},
    )

    assert result["success"] is False
    talent._execute_single_action.assert_not_called()
    request_id = result["response"].split("confirm ", 1)[1].split("'", 1)[0]
    approved = broker.resolve_confirmation(
        f"confirm {request_id}", source="signal")
    assert approved["success"] is True
    talent._execute_single_action.assert_called_once()


def test_scheduler_desktop_action_is_denied():
    broker = CapabilityBroker()
    talent = DesktopControlTalent()
    talent.action_delay = 0
    talent._execute_single_action = MagicMock(return_value="Pressed: enter")
    llm = MagicMock()
    llm.generate.return_value = (
        '{"explanation":"Press Enter",'
        '"actions":[{"action":"press","key":"enter"}]}'
    )

    result = talent._handle_desktop_action(
        "press enter", llm, MagicMock(), "", False, None,
        {"capabilities": broker, "command_source": "scheduler"},
    )

    assert result["success"] is False
    assert "Blocked by capability policy" in result["response"]
    talent._execute_single_action.assert_not_called()


def test_local_destructive_desktop_request_requires_confirmation():
    broker = CapabilityBroker()
    talent = DesktopControlTalent()
    talent.action_delay = 0
    talent._execute_single_action = MagicMock(return_value="Pressed: delete")
    llm = MagicMock()
    llm.generate.return_value = (
        '{"explanation":"Delete the file",'
        '"actions":[{"action":"press_key","key":"delete"}]}'
    )

    result = talent._handle_desktop_action(
        "delete the selected file", llm, MagicMock(), "", False, None,
        {"capabilities": broker, "command_source": "local"},
    )

    assert result["success"] is False
    assert "Approval required" in result["response"]
    talent._execute_single_action.assert_not_called()


def test_marketplace_install_requires_and_consumes_approval(
        tmp_path, monkeypatch):
    monkeypatch.setattr("core.marketplace._user_talents_dir",
                        lambda: str(tmp_path))
    broker = CapabilityBroker()
    client = MarketplaceClient(capabilities=broker)

    missing = client.commit_install("example.py", _VALID_TALENT_SOURCE)
    assert missing["success"] is False
    assert not (tmp_path / "example.py").exists()

    authorization = client.request_plugin_change(
        "install", "example", "example.py")
    installed = client.commit_install(
        "example.py", _VALID_TALENT_SOURCE, authorization=authorization)
    assert installed["success"] is True
    assert (tmp_path / "example.py").exists()

    replay = client.commit_install(
        "example.py", _VALID_TALENT_SOURCE, authorization=authorization)
    assert replay["success"] is False


def test_marketplace_scheduler_install_is_denied(tmp_path, monkeypatch):
    monkeypatch.setattr("core.marketplace._user_talents_dir",
                        lambda: str(tmp_path))
    broker = CapabilityBroker()
    client = MarketplaceClient(
        capabilities=broker, command_source="scheduler")
    authorization = client.request_plugin_change(
        "install", "example", "example.py")

    result = client.commit_install(
        "example.py", _VALID_TALENT_SOURCE, authorization=authorization)

    assert result["success"] is False
    assert "Blocked by capability policy" in result["error"]
    assert not (tmp_path / "example.py").exists()


def test_marketplace_uninstall_requires_approval(tmp_path, monkeypatch):
    monkeypatch.setattr("core.marketplace._user_talents_dir",
                        lambda: str(tmp_path))
    talent_path = tmp_path / "example.py"
    talent_path.write_text(_VALID_TALENT_SOURCE, encoding="utf-8")
    broker = CapabilityBroker()
    client = MarketplaceClient(capabilities=broker)

    missing = client.uninstall_talent("example")
    assert missing["success"] is False
    assert talent_path.exists()

    authorization = client.request_plugin_change("remove", "example")
    removed = client.uninstall_talent(
        "example", authorization=authorization)
    assert removed["success"] is True
    assert not talent_path.exists()


def test_marketplace_approval_cannot_authorize_another_target(
        tmp_path, monkeypatch):
    monkeypatch.setattr("core.marketplace._user_talents_dir",
                        lambda: str(tmp_path))
    broker = CapabilityBroker()
    client = MarketplaceClient(capabilities=broker)
    authorization = client.request_plugin_change(
        "install", "example", "example.py")

    result = client.commit_install(
        "different.py", _VALID_TALENT_SOURCE,
        authorization=authorization)

    assert result["success"] is False
    assert "does not match this target" in result["error"]
    assert not (tmp_path / "different.py").exists()
