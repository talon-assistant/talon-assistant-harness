"""One-shot worker for an untrusted third-party Talon talent.

The host launches this file with ``python -I`` and communicates exclusively
through newline-delimited JSON. Plugin stdout/stderr is captured so it cannot
forge protocol messages. This module is intentionally not imported by Talon.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import sysconfig
import traceback
import uuid
from pathlib import Path


_PROTOCOL = sys.stdout
_INPUT = sys.stdin


def _send(message: dict) -> None:
    _PROTOCOL.write(json.dumps(message, ensure_ascii=False) + "\n")
    _PROTOCOL.flush()


def _receive() -> dict:
    line = _INPUT.readline(1_048_577)
    if not line or len(line) > 1_048_576:
        raise RuntimeError("invalid sandbox request")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise RuntimeError("sandbox request must be an object")
    return value


class _BoundedText(io.TextIOBase):
    def __init__(self, limit: int):
        self.limit = max(0, limit)
        self.parts: list[str] = []
        self.size = 0

    def writable(self):
        return True

    def write(self, value):
        text = str(value)
        remaining = self.limit - self.size
        if remaining > 0:
            chunk = text[:remaining]
            self.parts.append(chunk)
            self.size += len(chunk)
        return len(text)

    def flush(self):
        return None


def _resolved(path) -> Path | None:
    if isinstance(path, int):
        return None
    try:
        return Path(os.fsdecode(path)).resolve(strict=False)
    except (OSError, TypeError, ValueError):
        return None


def _inside(path: Path, roots: tuple[Path, ...]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _root_tuple(values) -> tuple[Path, ...]:
    roots = []
    for value in values or ():
        path = _resolved(value)
        if path is not None:
            roots.append(path)
    return tuple(roots)


def _install_limits(memory_mb: int) -> None:
    if os.name == "nt":
        return
    try:  # pragma: no cover - Talon's primary platform is Windows
        import resource

        memory = max(64, memory_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except (ImportError, OSError, ValueError):
        pass


def _install_audit_hook(
    *,
    plugin_path: Path,
    work_dir: Path,
    permissions: dict,
) -> None:
    stdlib = _resolved(sysconfig.get_paths().get("stdlib", ""))
    purelib = _resolved(sysconfig.get_paths().get("purelib", ""))
    platlib = _resolved(sysconfig.get_paths().get("platlib", ""))
    read_roots = [work_dir]
    read_roots.extend(root for root in (stdlib, purelib, platlib) if root is not None)
    read_roots.extend(_root_tuple(permissions.get("read_roots")))
    write_roots = [work_dir]
    write_roots.extend(_root_tuple(permissions.get("write_roots")))
    read_roots_t = tuple(read_roots)
    write_roots_t = tuple(write_roots)
    allow_network = bool(permissions.get("network"))
    allow_subprocess = bool(permissions.get("subprocess"))

    mutation_events = {
        "os.remove", "os.rmdir", "os.mkdir", "os.rename", "os.replace",
        "os.chmod", "os.chown", "os.truncate", "os.symlink", "os.link",
        "shutil.copyfile", "shutil.copymode", "shutil.copystat",
    }
    read_events = {"os.listdir", "os.scandir", "os.chdir"}

    def audit(event, args):
        if event == "open" and args:
            path = _resolved(args[0])
            if path is None:
                return
            mode = args[1] if len(args) > 1 else "r"
            flags = args[2] if len(args) > 2 else 0
            writing = (
                isinstance(mode, str) and any(char in mode for char in "wax+")
            ) or (
                isinstance(flags, int)
                and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            )
            roots = write_roots_t if writing else read_roots_t
            if path != plugin_path and not _inside(path, roots):
                raise PermissionError(
                    f"filesystem {'write' if writing else 'read'} blocked: {path}"
                )
        elif event in mutation_events and args:
            for raw_path in args[:2]:
                path = _resolved(raw_path)
                if path is not None and not _inside(path, write_roots_t):
                    raise PermissionError(f"filesystem mutation blocked: {path}")
        elif event in read_events and args:
            path = _resolved(args[0])
            if path is not None and not _inside(path, read_roots_t):
                raise PermissionError(f"filesystem read blocked: {path}")
        elif event.startswith("socket.") and not allow_network:
            raise PermissionError("network access is not declared")
        elif event.startswith("winreg."):
            raise PermissionError("Windows registry access is not allowed")
        elif event.startswith("ctypes."):
            raise PermissionError("direct native-library access is not allowed")
        elif (
            event == "subprocess.Popen"
            or event == "os.system"
            or event == "os.startfile"
            or event == "os.fork"
            or event == "os.kill"
            or event.startswith("os.spawn")
            or event.startswith("os.exec")
        ) and not allow_subprocess:
            raise PermissionError("subprocess execution is not declared")

    sys.addaudithook(audit)


class _SandboxLLM:
    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.calls = 0

    def generate(self, prompt, *args, **kwargs):
        del args
        self.calls += 1
        if self.calls > self.max_calls:
            raise RuntimeError("sandbox host-call limit exceeded")
        call_id = uuid.uuid4().hex[:12]
        _send({
            "type": "host_call",
            "id": call_id,
            "method": "llm.generate",
            "params": {"prompt": str(prompt), "kwargs": kwargs},
        })
        response = _receive()
        if response.get("type") != "host_response" or response.get("id") != call_id:
            raise RuntimeError("invalid host response")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "host call failed")))
        return str(response.get("result", ""))


def _json_safe(value, *, depth=0):
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
        }
    if all(hasattr(value, key) for key in ("success", "response")):
        return {
            "success": bool(value.success),
            "response": str(value.response),
            "actions_taken": _json_safe(getattr(value, "actions_taken", [])),
            "spoken": False,
        }
    return str(value)[:1000]


def _run(request: dict) -> dict:
    if request.get("version") != 1:
        raise RuntimeError("unsupported sandbox protocol version")
    plugin_path = Path(str(request.get("plugin_path", ""))).resolve(strict=True)
    work_dir = Path(str(request.get("work_dir", ""))).resolve(strict=True)
    repo_root = Path(str(request.get("repo_root", ""))).resolve(strict=True)
    if plugin_path.suffix.lower() != ".py" or not plugin_path.is_file():
        raise RuntimeError("invalid plugin path")

    # Import the trusted base before installing restrictions. The plugin's
    # later ``from talents.base import BaseTalent`` resolves from this cache.
    sys.path.insert(0, str(repo_root))
    from talents.base import BaseTalent

    del BaseTalent
    sys.dont_write_bytecode = True
    os.chdir(work_dir)
    limits = request.get("limits") or {}
    _install_limits(int(limits.get("memory_mb", 512)))
    _install_audit_hook(
        plugin_path=plugin_path,
        work_dir=work_dir,
        permissions=request.get("permissions") or {},
    )

    capture = _BoundedText(int(limits.get("output_bytes", 131072)))
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        module_name = f"talon_sandbox_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, plugin_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not create plugin loader")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        class_name = str(request.get("class_name", ""))
        talent_class = getattr(module, class_name, None)
        if not isinstance(talent_class, type):
            raise RuntimeError("declared talent class was not found")
        talent = talent_class()
        talent.initialize({
            "talent_sandbox": {
                "active": True,
                "work_dir": str(work_dir),
            }
        })
        talent.update_config(request.get("talent_config") or {})

        permissions = request.get("permissions") or {}
        context = dict(request.get("context") or {})
        context["config"] = {
            "talent_sandbox": {
                "active": True,
                "work_dir": str(work_dir),
            }
        }
        context["llm"] = (
            _SandboxLLM(int(limits.get("host_calls", 8)))
            if permissions.get("llm", True) else None
        )
        context["speak_response"] = False
        result = talent.execute(str(request.get("command", "")), context)
    return _json_safe(result)


def main() -> int:
    try:
        request = _receive()
        result = _run(request)
        _send({"type": "result", "result": result})
        return 0
    except PermissionError as exc:
        _send({"type": "denied", "error": str(exc)[:500]})
        return 3
    except Exception as exc:
        # Tracebacks stay on the worker's bounded stderr channel and are never
        # returned to a remote requester or written to the capability audit.
        traceback.print_exc(file=sys.stderr)
        _send({"type": "failure", "error": f"{type(exc).__name__}: {exc}"[:500]})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
