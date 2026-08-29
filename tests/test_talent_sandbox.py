import textwrap
from pathlib import Path

from core.assistant import TalonAssistant
from core.capabilities import CapabilityBroker
from core.talent_sandbox import (
    SandboxedTalentProxy,
    parse_sandboxed_talent,
    run_sandboxed_talent,
)


def _write_talent(tmp_path: Path, body: str, name="sandbox_test") -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _proxy(path: Path) -> SandboxedTalentProxy:
    return SandboxedTalentProxy(parse_sandboxed_talent(path))


def _config(tmp_path: Path, **overrides) -> dict:
    values = {
        "enabled": True,
        "timeout_seconds": 5,
        "memory_limit_mb": 256,
        "output_limit_bytes": 16384,
        "max_host_calls": 4,
        "base_dir": str(tmp_path / "workers"),
    }
    values.update(overrides)
    return {"talent_sandbox": values}


class _Broker:
    def __init__(self):
        self.events = []

    def record_event(self, capability, **kwargs):
        self.events.append((capability, kwargs))


def test_proxy_parsing_does_not_execute_user_source(tmp_path):
    marker = tmp_path / "imported.txt"
    path = _write_talent(tmp_path, f"""
        from pathlib import Path
        from talents.base import BaseTalent
        Path({str(marker)!r}).write_text("imported")

        class StaticOnlyTalent(BaseTalent):
            name = "static_only"
            description = "Parsed without import"
            keywords = ["static"]
            examples = ["run static"]
            priority = 44
            capability_manifest = {{"access": "read_only"}}

            def get_config_schema(self):
                return {{"fields": [{{"key": "style", "type": "string"}}]}}

            def execute(self, command, context):
                return {{"success": True, "response": "ok"}}
    """)

    proxy = _proxy(path)

    assert proxy.name == "static_only"
    assert proxy.get_config_schema()["fields"][0]["key"] == "style"
    assert not marker.exists()


def test_dynamic_registration_never_imports_user_source(tmp_path):
    marker = tmp_path / "host_imported.txt"
    path = _write_talent(tmp_path, f"""
        from pathlib import Path
        from talents.base import BaseTalent
        Path({str(marker)!r}).write_text("host import")

        class RegisteredTalent(BaseTalent):
            name = "registered"
            description = "Safely registered"
            keywords = ["registered"]
            examples = ["run registered"]
            capability_manifest = {{"access": "read_only"}}
            def execute(self, command, context):
                return {{"success": True, "response": "ok"}}
    """)
    assistant = object.__new__(TalonAssistant)
    assistant.config_dir = str(tmp_path)
    assistant.config = {}
    assistant.talents_config = {}
    assistant.talents = []
    assistant._blocked_talent_manifests = []

    result = assistant.load_user_talent(str(path))

    assert result["success"] is True
    assert result["sandboxed"] is True
    assert isinstance(assistant.talents[0], SandboxedTalentProxy)
    assert not marker.exists()


def test_worker_gets_minimal_context_private_workdir_and_no_secrets(tmp_path):
    path = _write_talent(tmp_path, """
        from pathlib import Path
        from talents.base import BaseTalent

        class MinimalContextTalent(BaseTalent):
            name = "minimal_context"
            description = "Checks the worker context"
            keywords = ["minimal"]
            examples = ["run minimal"]
            capability_manifest = {"access": "read_only"}

            def execute(self, command, context):
                work = Path(context["config"]["talent_sandbox"]["work_dir"])
                (work / "created.txt").write_text("private", encoding="utf-8")
                exposed = sorted(context)
                has_secret = "api_key" in self.talent_config
                return {
                    "success": True,
                    "response": f"keys={','.join(exposed)} secret={has_secret}",
                    "actions_taken": [],
                    "spoken": True,
                }
    """)
    proxy = _proxy(path)
    proxy.update_config({"api_key": "do-not-send", "display_name": "safe"})
    broker = _Broker()

    result = run_sandboxed_talent(
        proxy,
        "run minimal",
        {
            "command_source": "signal",
            "llm": object(),
            "assistant": object(),
            "memory": object(),
            "config": {"also_secret": "no"},
        },
        app_config=_config(tmp_path),
        broker=broker,
    )

    assert result["success"] is True
    assert "assistant" not in result["response"]
    assert "memory" not in result["response"]
    assert "secret=False" in result["response"]
    assert result["spoken"] is False
    assert (tmp_path / "workers" / "minimal_context" / "created.txt").read_text() == "private"
    assert [event[1]["event"] for event in broker.events] == [
        "sandbox_started", "sandbox_completed"
    ]
    assert all(event[1]["source"] == "signal" for event in broker.events)


def test_worker_denies_filesystem_escape(tmp_path):
    outside = tmp_path / "outside.txt"
    path = _write_talent(tmp_path, f"""
        from pathlib import Path
        from talents.base import BaseTalent

        class EscapeTalent(BaseTalent):
            name = "escape"
            description = "Attempts a write escape"
            keywords = ["escape"]
            examples = ["run escape"]
            capability_manifest = {{"access": "read_only"}}

            def execute(self, command, context):
                Path({str(outside)!r}).write_text("escaped", encoding="utf-8")
                return {{"success": True, "response": "escaped"}}
    """)
    broker = _Broker()

    result = run_sandboxed_talent(
        _proxy(path), "escape", {"command_source": "local"},
        app_config=_config(tmp_path), broker=broker,
    )

    assert result["success"] is False
    assert "Sandbox denied" in result["response"]
    assert not outside.exists()
    assert broker.events[-1][1]["event"] == "sandbox_denied"


def test_worker_denies_network_by_default(tmp_path):
    path = _write_talent(tmp_path, """
        import socket
        from talents.base import BaseTalent

        class NetworkTalent(BaseTalent):
            name = "network_test"
            description = "Attempts network access"
            keywords = ["network"]
            examples = ["run network"]
            capability_manifest = {"access": "read_only"}

            def execute(self, command, context):
                socket.socket()
                return {"success": True, "response": "network opened"}
    """)

    result = run_sandboxed_talent(
        _proxy(path), "network", {"command_source": "local"},
        app_config=_config(tmp_path),
    )

    assert result["success"] is False
    assert "network access is not declared" in result["response"]


def test_worker_mediates_llm_calls(tmp_path):
    path = _write_talent(tmp_path, """
        from talents.base import BaseTalent

        class LlmTalent(BaseTalent):
            name = "llm_test"
            description = "Uses the mediated LLM"
            keywords = ["llm"]
            examples = ["run llm"]
            capability_manifest = {"access": "read_only", "sandbox": {"llm": True}}

            def execute(self, command, context):
                answer = context["llm"].generate(
                    "sandbox prompt", max_length=99999, temperature=99
                )
                return {"success": True, "response": answer}
    """)

    class Llm:
        def __init__(self):
            self.call = None

        def generate(self, prompt, **kwargs):
            self.call = (prompt, kwargs)
            return "mediated"

    llm = Llm()
    result = run_sandboxed_talent(
        _proxy(path), "llm", {"command_source": "local", "llm": llm},
        app_config=_config(tmp_path),
    )

    assert result["success"] is True
    assert result["response"] == "mediated"
    assert llm.call == (
        "sandbox prompt", {"max_length": 2048, "temperature": 2.0}
    )


def test_worker_timeout_is_hard_and_audited(tmp_path):
    path = _write_talent(tmp_path, """
        from talents.base import BaseTalent

        class LoopTalent(BaseTalent):
            name = "loop_test"
            description = "Loops forever"
            keywords = ["loop"]
            examples = ["run loop"]
            capability_manifest = {"access": "read_only"}

            def execute(self, command, context):
                while True:
                    pass
    """)
    broker = _Broker()

    result = run_sandboxed_talent(
        _proxy(path), "loop", {"command_source": "scheduler"},
        app_config=_config(tmp_path, timeout_seconds=1), broker=broker,
    )

    assert result["success"] is False
    assert "timed out" in result["response"]
    assert broker.events[-1][1]["event"] == "sandbox_timeout"


def test_assistant_keeps_remote_preflight_outside_sandbox(tmp_path):
    path = _write_talent(tmp_path, """
        from talents.base import BaseTalent

        class ControlledTalent(BaseTalent):
            name = "controlled_test"
            description = "A brokered sandbox talent"
            keywords = ["controlled"]
            examples = ["run controlled"]
            capability_manifest = {
                "access": "brokered",
                "capabilities": ("device_control",),
                "enforcement": "host",
            }

            def execute(self, command, context):
                return {"success": True, "response": "ran in sandbox"}
    """)
    assistant = object.__new__(TalonAssistant)
    assistant.config = _config(tmp_path)
    assistant.capabilities = CapabilityBroker()
    talent = _proxy(path)

    pending = assistant._invoke_talent(
        talent, "run controlled", {"command_source": "signal"}
    )

    assert pending["success"] is False
    assert "Approval required" in pending["response"]
    request_id = pending["response"].split("confirm ", 1)[1].split("'", 1)[0]
    assert assistant.capabilities.resolve_confirmation(
        f"confirm {request_id}", source="local"
    ) is None
    result = assistant.capabilities.resolve_confirmation(
        f"confirm {request_id}", source="signal"
    )
    assert result["success"] is True
    assert result["sandboxed"] is True
