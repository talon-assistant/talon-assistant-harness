"""core/config.py — Configuration utilities for Talon Assistant."""

import json
import os
import tempfile
import threading

# Serializes every settings.json write across the whole process. Writers live
# on different threads — the GUI dialogs and theme manager run on the Qt main
# thread, while the LoRA trainer writes llm_server.lora_path from its worker
# thread. Without this lock two concurrent read-modify-write cycles can clobber
# each other (last writer wins, the other section's change is lost).
_WRITE_LOCK = threading.RLock()


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


def update_settings(path: str, changes: dict) -> dict:
    """Merge *changes* into the JSON settings file at *path* and persist it.

    Serialized process-wide via _WRITE_LOCK and crash-safe via atomic replace.
    The current on-disk file is re-read inside the lock, so each writer merges
    onto the freshest content rather than a stale snapshot — a write to one
    section never drops another section a different thread just wrote. A
    missing or corrupt file is treated as ``{}``. Returns the merged dict.
    """
    with _WRITE_LOCK:
        try:
            with open(path, "r", encoding="utf-8") as f:
                current = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            current = {}
        merged = deep_merge(current, changes)
        _atomic_write_json(path, merged)
        return merged
