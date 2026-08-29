"""core/config.py — Configuration utilities for Talon Assistant."""

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

# Serializes every settings.json write across the whole process. Writers live
# on different threads — the GUI dialogs and theme manager run on the Qt main
# thread, while the LoRA trainer writes llm_server.lora_path from its worker
# thread. Without this lock two concurrent read-modify-write cycles can clobber
# each other (last writer wins, the other section's change is lost).
_WRITE_LOCK = threading.RLock()


def config_directory(config_dir: str | os.PathLike | None = None) -> Path:
    """Return the runtime config directory, with an environment override."""
    value = config_dir or os.environ.get("TALON_CONFIG_DIR") or "config"
    return Path(value).expanduser()


def load_runtime_settings(
    config_dir: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Load example defaults merged with untracked user settings."""
    directory = config_directory(config_dir)

    def _read(name: str) -> dict[str, Any]:
        try:
            value = json.loads((directory / name).read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}

    return deep_merge(_read("settings.example.json"), _read("settings.json"))


def get_setting(
    dot_path: str,
    default=None,
    *,
    settings: dict[str, Any] | None = None,
    config_dir: str | os.PathLike | None = None,
):
    """Read a dot-separated value from runtime settings."""
    current: Any = settings if settings is not None else load_runtime_settings(config_dir)
    for part in dot_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return default if current in (None, "") else current


def resolve_configured_path(
    dot_path: str,
    default: str,
    *,
    settings: dict[str, Any] | None = None,
    config_dir: str | os.PathLike | None = None,
) -> Path:
    """Resolve a configurable path with ``~`` and environment expansion."""
    value = str(get_setting(
        dot_path, default, settings=settings, config_dir=config_dir
    ))
    return Path(os.path.expandvars(value)).expanduser()


def get_user_profile(
    *, settings: dict[str, Any] | None = None,
    config_dir: str | os.PathLike | None = None,
) -> dict[str, str]:
    """Return the public identity fields used in generated documents."""
    runtime = settings if settings is not None else load_runtime_settings(config_dir)
    profile = runtime.get("user_profile", {})
    if not isinstance(profile, dict):
        profile = {}
    return {
        key: str(profile.get(key, "") or "").strip()
        for key in ("display_name", "email", "phone", "location")
    }


def format_contact_line(profile: dict[str, str]) -> str:
    """Format only configured contact fields, without placeholder PII."""
    return " | ".join(
        value for value in (
            profile.get("email", ""), profile.get("phone", ""),
            profile.get("location", ""),
        ) if value
    )


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict.
    Keys present in base but missing from override keep their base value."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _atomic_write_json(path: str, data: dict) -> None:
    """Write *data* as indented JSON to *path* atomically.

    Writes to a temp file in the same directory, flushes and fsyncs it, then
    os.replace()s it over the target. A crash or a concurrent reader therefore
    never sees a half-written settings.json — the file is either the old
    content or the new content, never a truncated mix.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".settings-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def update_settings(
    path: str, changes: dict, *, replace_sections: tuple[str, ...] = ()
) -> dict:
    """Merge *changes* into the JSON settings file at *path* and persist it.

    Serialized process-wide via _WRITE_LOCK and crash-safe via atomic replace.
    The current on-disk file is re-read inside the lock, so each writer merges
    onto the freshest content rather than a stale snapshot — a write to one
    section never drops another section a different thread just wrote. A
    missing or corrupt file is treated as ``{}``. Top-level keys named in
    *replace_sections* are replaced as complete values after the merge; this is
    useful for editors that own an entire policy section and must not retain
    hidden stale keys. Returns the merged dict.
    """
    with _WRITE_LOCK:
        try:
            with open(path, "r", encoding="utf-8") as f:
                current = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            current = {}
        merged = deep_merge(current, changes)
        for section in replace_sections:
            if section in changes:
                merged[section] = changes[section]
        _atomic_write_json(path, merged)
        return merged
