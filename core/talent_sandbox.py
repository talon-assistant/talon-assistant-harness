"""Third-party talent proxy, worker protocol, and process containment.

User talent source is parsed into a proxy without importing it in Talon's host
process. Execution happens in a fresh isolated Python worker with a minimal JSON
context. Python audit hooks provide default-deny filesystem/network/process
rules inside the worker; Windows Job Objects and timeouts provide resource and
process-tree containment. Audit hooks are defense in depth, not a replacement
for a future AppContainer/restricted-token boundary.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.capability_manifest import inspect_source_manifest
from talents.base import BaseTalent


log = logging.getLogger(__name__)

_PROTOCOL_VERSION = 1
_MAX_PROTOCOL_LINE = 1_048_576
_SECRET_KEY_RE = re.compile(
    r"(?:password|passwd|secret|token|api[_-]?key|private[_-]?key|credential)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SandboxedTalentSpec:
    source_path: str
    class_name: str
    name: str
    description: str
    keywords: tuple[str, ...]
    examples: tuple[str, ...]
    priority: int
    required_packages: tuple[str, ...]
    required_config: tuple[str, ...]
    required_env: tuple[str, ...]
    tool_parameters: dict[str, Any] | None
    tool_required: tuple[str, ...] | None
    config_schema: dict[str, Any]
    capability_manifest: dict[str, Any]


def _literal_assignments(class_node: ast.ClassDef) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in class_node.body:
        if not isinstance(item, ast.Assign):
            continue
        names = [target.id for target in item.targets if isinstance(target, ast.Name)]
        if not names:
            continue
        try:
            value = ast.literal_eval(item.value)
        except (ValueError, TypeError):
            continue
        for name in names:
            values[name] = value
    return values


def parse_sandboxed_talent(path: str | Path) -> SandboxedTalentSpec:
    """Parse the first top-level BaseTalent subclass without importing it."""
    source_path = Path(path).resolve()
    manifest_item = inspect_source_manifest(source_path)
    if manifest_item.status == "undeclared":
        raise ValueError(manifest_item.detail)
    try:
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"), filename=str(source_path)
        )
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise ValueError(f"Source cannot be parsed: {exc}") from exc

    class_node = None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if any(
            (isinstance(base, ast.Name) and base.id == "BaseTalent")
            or (isinstance(base, ast.Attribute) and base.attr == "BaseTalent")
            for base in node.bases
        ):
            class_node = node
            break
    if class_node is None:  # inspect_source_manifest normally catches this.
        raise ValueError("No top-level BaseTalent subclass")

    values = _literal_assignments(class_node)
    name = str(values.get("name") or source_path.stem).strip()
    description = str(values.get("description") or name).strip()
    if not name or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,63}", name):
        raise ValueError("Talent name must be a simple identifier")
    manifest = values.get("capability_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Missing literal capability_manifest")

    tool_parameters = values.get("tool_parameters")
    if not isinstance(tool_parameters, dict):
        tool_parameters = None
    tool_required = values.get("tool_required")
    if not isinstance(tool_required, (tuple, list)):
        tool_required = None
    config_schema = manifest.get("config_schema")
    if not isinstance(config_schema, dict):
        config_schema = {}
        for item in class_node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if item.name != "get_config_schema":
                continue
            for statement in item.body:
                if not isinstance(statement, ast.Return):
                    continue
                try:
                    literal = ast.literal_eval(statement.value)
                except (ValueError, TypeError):
                    continue
                if isinstance(literal, dict):
                    config_schema = literal
                    break

    return SandboxedTalentSpec(
        source_path=str(source_path),
        class_name=class_node.name,
        name=name,
        description=description,
        keywords=tuple(str(value) for value in (values.get("keywords") or ())),
        examples=tuple(str(value) for value in (values.get("examples") or ())),
        priority=max(0, min(100, int(values.get("priority", 50)))),
        required_packages=tuple(
            str(value) for value in (values.get("required_packages") or ())
        ),
        required_config=tuple(
            str(value) for value in (values.get("required_config") or ())
        ),
        required_env=tuple(
            str(value) for value in (values.get("required_env") or ())
        ),
        tool_parameters=tool_parameters,
        tool_required=(
            tuple(str(value) for value in tool_required)
            if tool_required is not None else None
        ),
        config_schema=config_schema,
        capability_manifest=manifest,
    )


class SandboxedTalentProxy(BaseTalent):
    """Non-executing host representation of a third-party talent source file."""

    def __init__(self, spec: SandboxedTalentSpec):
        super().__init__()
        self._sandbox_spec = spec
        self._source_path = spec.source_path
        self._source_class_name = spec.class_name
        self._source_manifest = spec.capability_manifest
        self.name = spec.name
        self.description = spec.description
        self.keywords = list(spec.keywords)
        self.examples = list(spec.examples)
        self.priority = spec.priority
        self.required_packages = list(spec.required_packages)
        self.required_config = list(spec.required_config)
        self.required_env = list(spec.required_env)
        if spec.tool_parameters is not None:
            self.tool_parameters = spec.tool_parameters
        if spec.tool_required is not None:
            self.tool_required = list(spec.tool_required)

    def execute(self, command: str, context: dict) -> dict:
        return {
            "success": False,
            "response": "Sandboxed talents cannot execute in Talon's host process.",
            "actions_taken": [],
            "spoken": False,
        }

    def get_config_schema(self) -> dict:
        return self._sandbox_spec.config_schema


def is_sandboxed_talent(talent) -> bool:
    return isinstance(talent, SandboxedTalentProxy)


def _json_safe(value, *, depth: int = 0):
    if depth > 6:
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth=depth + 1) for item in value[:100]]
    if isinstance(value, dict):
        return {
            str(key)[:100]: _json_safe(item, depth=depth + 1)
            for key, item in list(value.items())[:100]
            if not _SECRET_KEY_RE.search(str(key))
        }
    return str(value)[:1000]


def _sanitized_talent_config(config: dict) -> dict:
    return {
        str(key): _json_safe(value)
        for key, value in (config or {}).items()
        if not _SECRET_KEY_RE.search(str(key))
    }


def _bounded_number(value, default, minimum, maximum, cast):
    try:
        number = cast(value)
    except (TypeError, ValueError, OverflowError):
        number = cast(default)
    return max(minimum, min(number, maximum))


def _declared_roots(values, *, repo_root: Path) -> list[str]:
    roots = []
    for value in values or ():
        raw = os.path.expandvars(os.path.expanduser(str(value)))
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        try:
            roots.append(str(candidate.resolve()))
        except OSError:
            continue
    return roots[:16]


class _WindowsJob:
    """Best-effort Job Object: kill descendants and cap per-process memory."""

    def __init__(self, process, memory_mb: int):
        self._handle = None
        if os.name != "nt":
            return
        try:
            import ctypes
            from ctypes import wintypes

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_ulonglong),
                    ("WriteOperationCount", ctypes.c_ulonglong),
                    ("OtherOperationCount", ctypes.c_ulonglong),
                    ("ReadTransferCount", ctypes.c_ulonglong),
                    ("WriteTransferCount", ctypes.c_ulonglong),
                    ("OtherTransferCount", ctypes.c_ulonglong),
                ]

            class BASIC_LIMITS(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class EXTENDED_LIMITS(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", BASIC_LIMITS),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            kernel32.CreateJobObjectW.restype = wintypes.HANDLE
            kernel32.SetInformationJobObject.argtypes = [
                wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
            ]
            kernel32.SetInformationJobObject.restype = wintypes.BOOL
            kernel32.AssignProcessToJobObject.argtypes = [
                wintypes.HANDLE, wintypes.HANDLE,
            ]
            kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                return
            limits = EXTENDED_LIMITS()
            limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x100
            limits.ProcessMemoryLimit = int(memory_mb) * 1024 * 1024
            ok = kernel32.SetInformationJobObject(
                handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
            )
            if not ok or not kernel32.AssignProcessToJobObject(
                handle, wintypes.HANDLE(process._handle)
            ):
                kernel32.CloseHandle(handle)
                return
            self._handle = handle
            self._kernel32 = kernel32
        except Exception as exc:  # pragma: no cover - platform best effort
            log.warning("[Sandbox] Windows Job Object unavailable: %s", exc)

    def close(self):
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def _terminate_process_tree(process) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=5,
            )
        else:  # pragma: no cover - Windows is Talon's primary platform
            os.killpg(process.pid, signal.SIGKILL)
    except Exception:
        process.kill()


def _audit(broker, source: str, owner: str, event: str, error: str = "") -> None:
    if broker is None:
        return
    broker.record_event(
        "talent_sandbox",
        source=source,
        summary=f"Sandbox {owner}",
        event=event,
        error=error,
    )


def run_sandboxed_talent(
    talent: SandboxedTalentProxy,
    command: str,
    context: dict,
    *,
    app_config: dict,
    broker=None,
) -> dict:
    """Execute a third-party proxy through the JSON worker protocol."""
    sandbox_cfg = app_config.get("talent_sandbox", {}) or {}
    if not isinstance(sandbox_cfg, dict):
        sandbox_cfg = {}
    if not sandbox_cfg.get("enabled", True):
        _audit(
            broker,
            str(context.get("command_source", "local")),
            talent.name,
            "sandbox_denied",
            "sandbox execution disabled",
        )
        return {
            "success": False,
            "response": "Third-party talent execution is disabled by sandbox policy.",
            "actions_taken": [],
            "spoken": False,
        }

    spec = talent._sandbox_spec
    source = str(context.get("command_source", "local"))
    timeout = _bounded_number(
        sandbox_cfg.get("timeout_seconds", 30), 30, 1.0, 120.0, float
    )
    memory_mb = _bounded_number(
        sandbox_cfg.get("memory_limit_mb", 512), 512, 64, 2048, int
    )
    output_limit = _bounded_number(
        sandbox_cfg.get("output_limit_bytes", 131072),
        131072, 4096, 1_048_576, int,
    )
    repo_root = Path(__file__).resolve().parent.parent
    base_dir = Path(str(sandbox_cfg.get("base_dir", "data/talent_sandboxes")))
    if not base_dir.is_absolute():
        base_dir = repo_root / base_dir
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", talent.name)[:64] or "talent"
    work_dir = (base_dir / safe_name).resolve()
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _audit(broker, source, talent.name, "sandbox_failed", str(exc))
        return _failure("Talent sandbox private directory is unavailable.")

    sandbox_manifest = spec.capability_manifest.get("sandbox") or {}
    if not isinstance(sandbox_manifest, dict):
        sandbox_manifest = {}
    capabilities = set(spec.capability_manifest.get("capabilities") or ())
    allow_writes = bool(capabilities & {
        "local_data_write", "destructive_file_ops", "plugin_install",
        "credential_write", "external_account_write",
    })
    allow_subprocess = (
        bool(sandbox_manifest.get("subprocess"))
        and "process_execution" in capabilities
    )
    payload = {
        "version": _PROTOCOL_VERSION,
        "plugin_path": spec.source_path,
        "class_name": spec.class_name,
        "command": str(command)[:20_000],
        "talent_config": _sanitized_talent_config(talent.talent_config),
        "context": {
            "command_source": source,
            "tool_args": _json_safe(context.get("tool_args", {})),
            "speak_response": False,
        },
        "permissions": {
            "network": bool(sandbox_manifest.get("network", False)),
            "subprocess": allow_subprocess,
            "llm": bool(sandbox_manifest.get("llm", True)),
            "read_roots": _declared_roots(
                sandbox_manifest.get("filesystem_read"), repo_root=repo_root
            ),
            "write_roots": (
                _declared_roots(
                    sandbox_manifest.get("filesystem_write"), repo_root=repo_root
                ) if allow_writes else []
            ),
        },
        "limits": {
            "memory_mb": memory_mb,
            "output_bytes": output_limit,
            "host_calls": _bounded_number(
                sandbox_cfg.get("max_host_calls", 8), 8, 0, 32, int
            ),
        },
        "work_dir": str(work_dir),
        "repo_root": str(repo_root),
    }

    worker_path = repo_root / "core" / "talent_sandbox_worker.py"
    creationflags = 0
    popen_kwargs = {}
    if os.name == "nt":
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    else:  # pragma: no cover - Windows is Talon's primary platform
        popen_kwargs["start_new_session"] = True
    env = {
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TALON_SANDBOX": "1",
        "TEMP": str(work_dir),
        "TMP": str(work_dir),
    }
    for key in (
        "SYSTEMROOT", "WINDIR", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
    ):
        if os.environ.get(key):
            env[key] = os.environ[key]

    _audit(broker, source, talent.name, "sandbox_started")
    try:
        process = subprocess.Popen(
            [sys.executable, "-I", "-u", str(worker_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(work_dir),
            env=env,
            creationflags=creationflags,
            bufsize=1,
            **popen_kwargs,
        )
    except Exception as exc:
        _audit(broker, source, talent.name, "sandbox_failed", str(exc))
        log.warning("[Sandbox] could not start %s: %s", talent.name, exc)
        return _failure("Talent sandbox could not be started.")
    job = _WindowsJob(process, memory_mb)
    messages: queue.Queue[tuple[str, str]] = queue.Queue()
    stderr_parts: list[str] = []

    def _read_stream(stream, kind):
        try:
            for line in iter(stream.readline, ""):
                messages.put((kind, line))
        finally:
            messages.put((f"{kind}_eof", ""))

    threading.Thread(
        target=_read_stream, args=(process.stdout, "stdout"), daemon=True
    ).start()
    threading.Thread(
        target=_read_stream, args=(process.stderr, "stderr"), daemon=True
    ).start()

    try:
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()
        deadline = time.monotonic() + timeout
        host_calls = 0
        result = None
        while time.monotonic() < deadline:
            try:
                kind, line = messages.get(
                    timeout=min(0.2, max(0.01, deadline - time.monotonic()))
                )
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            if kind == "stderr":
                remaining = output_limit - sum(len(part) for part in stderr_parts)
                if remaining > 0:
                    stderr_parts.append(line[:remaining])
                continue
            if kind != "stdout":
                continue
            if len(line.encode("utf-8", errors="replace")) > _MAX_PROTOCOL_LINE:
                raise RuntimeError("sandbox protocol output exceeded its limit")
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError("sandbox emitted invalid protocol data") from exc
            if message.get("type") == "host_call":
                host_calls += 1
                if host_calls > payload["limits"]["host_calls"]:
                    response = {"ok": False, "error": "host call limit exceeded"}
                else:
                    response = _handle_host_call(message, context, output_limit)
                process.stdin.write(json.dumps({
                    "type": "host_response",
                    "id": message.get("id"),
                    **response,
                }, ensure_ascii=False) + "\n")
                process.stdin.flush()
            elif message.get("type") == "result":
                result = message.get("result")
                break
            elif message.get("type") == "denied":
                detail = str(message.get("error", "sandbox policy denied access"))
                _audit(broker, source, talent.name, "sandbox_denied", detail)
                return _failure(f"Sandbox denied the operation: {detail}")
            elif message.get("type") == "failure":
                detail = str(message.get("error", "worker failed"))[:500]
                _audit(broker, source, talent.name, "sandbox_failed", detail)
                return _failure("Talent sandbox failed while running the plugin.")

        if result is None:
            if time.monotonic() >= deadline:
                _terminate_process_tree(process)
                _audit(broker, source, talent.name, "sandbox_timeout")
                return _failure(f"Talent sandbox timed out after {timeout:g} seconds.")
            detail = "".join(stderr_parts)[-1000:] or f"worker exited {process.poll()}"
            _audit(broker, source, talent.name, "sandbox_failed", detail)
            return _failure("Talent sandbox exited without a valid result.")

        normalized = _normalize_result(result, output_limit)
        _audit(broker, source, talent.name, "sandbox_completed")
        normalized["sandboxed"] = True
        return normalized
    except Exception as exc:
        _terminate_process_tree(process)
        _audit(broker, source, talent.name, "sandbox_failed", str(exc))
        log.warning("[Sandbox] %s failed: %s", talent.name, exc)
        return _failure(f"Talent sandbox failed: {exc}")
    finally:
        try:
            if process.stdin:
                process.stdin.close()
        except OSError:
            pass
        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(process)
        job.close()


def _handle_host_call(message: dict, context: dict, output_limit: int) -> dict:
    if message.get("method") != "llm.generate":
        return {"ok": False, "error": "host method is not allowed"}
    llm = context.get("llm")
    if llm is None:
        return {"ok": False, "error": "LLM is unavailable"}
    params = message.get("params") or {}
    prompt = str(params.get("prompt", ""))[:20_000]
    kwargs = params.get("kwargs") or {}
    safe_kwargs = {
        "max_length": max(1, min(int(kwargs.get("max_length", 512)), 2048)),
        "temperature": max(0.0, min(float(kwargs.get("temperature", 0.7)), 2.0)),
    }
    if "system_prompt" in kwargs:
        safe_kwargs["system_prompt"] = str(kwargs["system_prompt"])[:10_000]
    try:
        value = llm.generate(prompt, **safe_kwargs)
        return {"ok": True, "result": str(value)[:output_limit]}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:500]}


def _normalize_result(result, output_limit: int) -> dict:
    if not isinstance(result, dict):
        return _failure("Sandboxed talent returned an invalid result.")
    response = str(result.get("response", ""))
    encoded = response.encode("utf-8", errors="replace")[:output_limit]
    response = encoded.decode("utf-8", errors="ignore")
    actions = result.get("actions_taken")
    if not isinstance(actions, list):
        actions = []
    return {
        "success": bool(result.get("success", False)),
        "response": response,
        "actions_taken": _json_safe(actions[:100]),
        "spoken": False,
    }


def _failure(message: str) -> dict:
    return {
        "success": False,
        "response": message,
        "actions_taken": [],
        "spoken": False,
        "sandboxed": True,
    }
