"""Rich persistent task scheduler for Talon.

Supports three task types:
  once      — fires once at an ISO datetime, then auto-deletes
  interval  — fires every N seconds
  cron      — fires on a cron expression (requires ``croniter`` package)

Tasks are persisted to ``config/scheduled_tasks.json`` so they survive
restarts.  Legacy ``settings.json`` "scheduler" entries (daily time-based
tasks) continue to work unchanged via the backward-compat loader.

Poll interval: 15 seconds.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# croniter is an optional dependency — cron tasks require it at task-creation
# time, not at import time, so the rest of the scheduler always works.
try:
    from croniter import croniter as _croniter  # type: ignore
    _HAS_CRONITER = True
except ImportError:
    _croniter = None
    _HAS_CRONITER = False


_TASKS_FILE = os.path.join("config", "scheduled_tasks.json")
_POLL_INTERVAL = 15   # seconds


class Scheduler:
    """Persistent multi-type task scheduler wired to the assistant."""

    def __init__(self) -> None:
        self._tasks: list[dict] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._assistant = None
        # Fired-key set prevents double-fire of legacy daily tasks
        self._legacy_fired: set[str] = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, legacy_schedule: list[dict], assistant) -> None:
        """Start the scheduler background thread.

        Args:
            legacy_schedule: List of task dicts from settings.json "scheduler"
                key (backward compat — daily time-based tasks).
            assistant: TalonAssistant instance whose process_command() to call.
        """
        self._assistant = assistant
        self._load_tasks()
        self._import_legacy(legacy_schedule)

        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="TalonScheduler"
        )
        self._thread.start()

        with self._lock:
            enabled = [t for t in self._tasks if t.get("enabled", True)]
        cron_status = "croniter available" if _HAS_CRONITER else "croniter not installed — cron tasks disabled"
        print(
            f"[Scheduler] Started — {len(enabled)} active task(s) ({cron_status})"
        )

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._running = False

    def create_task(
        self,
        command: str,
        task_type: str,
        *,
        at: str | None = None,
        interval_seconds: int | None = None,
        cron: str | None = None,
        output: str = "gui",
        label: str = "",
        enabled: bool = True,
    ) -> dict:
        """Create and persist a new scheduled task.

        Returns the newly created task dict (including its generated id).
        Raises ValueError for invalid arguments.
        """
        task_type = task_type.lower()
        if task_type not in ("once", "interval", "cron"):
            raise ValueError(f"Unknown task_type: {task_type!r}")

        if task_type == "once":
            if not at:
                raise ValueError("'at' (ISO datetime string) is required for once tasks")
            try:
                fire_dt = datetime.fromisoformat(at)
            except ValueError as exc:
                raise ValueError(f"Invalid ISO datetime for 'at': {at!r}") from exc
            naive_now = datetime.now()
            fire_naive = fire_dt.replace(tzinfo=None)
            if fire_naive < naive_now:
                raise ValueError(f"'at' datetime {at!r} is in the past")

        elif task_type == "interval":
            if not interval_seconds or interval_seconds <= 0:
                raise ValueError("'interval_seconds' must be a positive integer")

        elif task_type == "cron":
            if not cron:
                raise ValueError("'cron' expression is required for cron tasks")
            if not _HAS_CRONITER:
                raise ValueError(
                    "croniter package is not installed — install it with: pip install croniter"
                )
            try:
                _croniter(cron, datetime.now())
            except Exception as exc:
                raise ValueError(f"Invalid cron expression {cron!r}: {exc}") from exc

        task_id = str(uuid.uuid4())[:8]
        now_iso = datetime.now().isoformat()

        task: dict = {
            "id":        task_id,
            "command":   command,
            "task_type": task_type,
            "output":    output,
            "label":     label or command[:40],
            "enabled":   enabled,
            "created":   now_iso,
            "last_run":  None,
        }

        if task_type == "once":
            task["at"] = at

        elif task_type == "interval":
            task["interval_seconds"] = interval_seconds
            next_dt = datetime.now() + timedelta(seconds=interval_seconds)
            task["next_run"] = next_dt.isoformat()

        elif task_type == "cron":
            task["cron"] = cron
            itr = _croniter(cron, datetime.now())
            task["next_cron_run"] = itr.get_next(datetime).isoformat()

        with self._lock:
            self._tasks.append(task)
            self._save_tasks()

        print(f"[Scheduler] Created task {task_id!r}: {task['label']!r} ({task_type})")
        return dict(task)

    def cancel_task(self, task_id: str) -> bool:
        """Remove a task by id. Returns True if found and removed."""
        with self._lock:
            before = len(self._tasks)
            self._tasks = [t for t in self._tasks if t.get("id") != task_id]
            removed = len(self._tasks) < before
            if removed:
                self._save_tasks()
        if removed:
            print(f"[Scheduler] Cancelled task {task_id!r}")
        return removed

    def list_tasks(self) -> list[dict]:
        """Return a snapshot of all tasks (copies, not references)."""
        with self._lock:
            return [dict(t) for t in self._tasks]

    def set_enabled(self, task_id: str, enabled: bool) -> bool:
        """Enable or disable a task. Returns True if the task was found."""
        with self._lock:
            for task in self._tasks:
                if task.get("id") == task_id:
                    task["enabled"] = enabled
                    self._save_tasks()
                    return True
        return False

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            try:
                self._tick()
            except Exception as exc:
                print(f"[Scheduler] Unexpected error in tick: {exc}")
            time.sleep(_POLL_INTERVAL)

    def _tick(self) -> None:
        now = datetime.now()

        with self._lock:
            tasks_snapshot = list(self._tasks)

        to_fire: list[dict] = []
        once_done_ids: list[str] = []

        for task in tasks_snapshot:
            if not task.get("enabled", True):
                continue

            task_type = task.get("task_type", "legacy")

            # ── once ──────────────────────────────────────────────────────
            if task_type == "once":
                at_str = task.get("at", "")
                if not at_str:
                    continue
                try:
                    fire_dt = datetime.fromisoformat(at_str).replace(tzinfo=None)
                    if now >= fire_dt:
                        to_fire.append(task)
                        once_done_ids.append(task["id"])
                except ValueError:
                    continue

            # ── interval ──────────────────────────────────────────────────
            elif task_type == "interval":
                next_str = task.get("next_run", "")
                if not next_str:
                    continue
                try:
                    next_dt = datetime.fromisoformat(next_str)
                    if now >= next_dt:
                        to_fire.append(task)
                        interval = int(task.get("interval_seconds", 3600))
                        with self._lock:
                            for t in self._tasks:
                                if t.get("id") == task["id"]:
                                    t["next_run"] = (now + timedelta(seconds=interval)).isoformat()
                                    t["last_run"] = now.isoformat()
                                    break
                except ValueError:
                    continue

            # ── cron ──────────────────────────────────────────────────────
            elif task_type == "cron":
                if not _HAS_CRONITER:
                    continue
                next_str = task.get("next_cron_run", "")
                if not next_str:
                    continue
                try:
                    next_dt = datetime.fromisoformat(next_str)
                    if now >= next_dt:
                        to_fire.append(task)
                        itr = _croniter(task["cron"], now)
                        with self._lock:
                            for t in self._tasks:
                                if t.get("id") == task["id"]:
                                    t["next_cron_run"] = itr.get_next(datetime).isoformat()
                                    t["last_run"] = now.isoformat()
                                    break
                except (ValueError, Exception):
                    continue

            # ── legacy (settings.json daily-time tasks) ───────────────────
            elif task_type == "legacy":
                today      = now.date().isoformat()
                day_abbr   = now.strftime("%a").lower()
                time_str   = now.strftime("%H:%M")
                task_time  = task.get("time", "")
                task_days  = [
                    d.lower()[:3]
                    for d in task.get("days", ["mon", "tue", "wed", "thu", "fri", "sat", "sun"])
                ]
                if not task_time or task_time != time_str:
                    continue
                if day_abbr not in task_days:
                    continue
                fired_key = f"{today}:{task_time}:{task.get('command', '')}"
                if fired_key in self._legacy_fired:
                    continue
                self._legacy_fired.add(fired_key)
                to_fire.append(task)

        # Expire legacy fired-set entries from previous days
        today_prefix = now.date().isoformat() + ":"
        self._legacy_fired = {k for k in self._legacy_fired if k.startswith(today_prefix)}

        # Remove completed once-tasks and persist interval/cron updates
        if once_done_ids or to_fire:
            with self._lock:
                if once_done_ids:
                    self._tasks = [t for t in self._tasks if t.get("id") not in once_done_ids]
                self._save_tasks()

        # Fire each task in its own daemon thread
        for task in to_fire:
            command   = task.get("command", "").strip()

            # Skip if the target talent is disabled — avoids log spam
            if not self._is_command_talent_enabled(command):
                continue

            output    = task.get("output", "gui")
            speak_tts = output in ("tts", "both")
            label     = task.get("label", command)
            print(f"[Scheduler] Firing {task.get('task_type', '?')} task {task.get('id', '?')!r}: {label!r}")
            threading.Thread(
                target=self._run_command,
                args=(command, speak_tts),
                daemon=True,
                name=f"Scheduler-{task.get('id', 'task')}",
            ).start()

    def _is_command_talent_enabled(self, command: str) -> bool:
        """Check if the talent that handles this command is enabled.

        Only checks internal (non-routed) talents — LLM-routed talents
        are always allowed through since the router handles them.
        Returns True if no matching internal talent found (let it fire).
        """
        if not self._assistant:
            return True
        for talent in getattr(self._assistant, "talents", []):
            if not talent.routing_available and talent.can_handle(command):
                return talent.enabled
        return True

    def _run_command(self, command: str, speak_tts: bool) -> None:
        try:
            # Respect the global TTS toggle — if the user has muted the
            # assistant, scheduled tasks should not override that.
            global_tts = getattr(self._assistant, "tts_enabled", True)
            self._assistant.process_command(
                command, speak_response=(speak_tts and global_tts)
            )
        except Exception as exc:
            print(f"[Scheduler] Error running {command!r}: {exc}")

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_tasks(self) -> None:
        """Load persisted tasks from config/scheduled_tasks.json."""
        try:
            if os.path.exists(_TASKS_FILE):
                with open(_TASKS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tasks = data if isinstance(data, list) else []
                # Auto-initialize interval tasks with null/empty next_run
                # so they fire on the first tick after startup.
                now_iso = datetime.now().isoformat()
                for t in tasks:
                    if (t.get("task_type") == "interval"
                            and not t.get("next_run")):
                        t["next_run"] = now_iso

                with self._lock:
                    self._tasks = tasks
                print(f"[Scheduler] Loaded {len(tasks)} persisted task(s)")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[Scheduler] Could not load tasks file: {exc}")

    def _save_tasks(self) -> None:
        """Persist tasks to disk. Must be called while holding self._lock."""
        # Legacy tasks come from settings.json — don't persist them here
        persistent = [t for t in self._tasks if t.get("task_type") != "legacy"]
        try:
            os.makedirs(os.path.dirname(_TASKS_FILE) or ".", exist_ok=True)
            with open(_TASKS_FILE, "w", encoding="utf-8") as f:
                json.dump(persistent, f, indent=2)
        except OSError as exc:
            print(f"[Scheduler] Could not save tasks: {exc}")

    def _import_legacy(self, legacy_schedule: list[dict]) -> None:
        """Import settings.json "scheduler" entries as in-memory legacy tasks.

        These are NOT written to scheduled_tasks.json — they are re-read from
        settings.json on every startup, matching the original behaviour.
        """
        if not legacy_schedule:
            return
        count = 0
        with self._lock:
            existing_cmds = {
                t.get("command") for t in self._tasks if t.get("task_type") == "legacy"
            }
            for entry in legacy_schedule:
                cmd = entry.get("command", "").strip()
                if not cmd or cmd in existing_cmds:
                    continue
                task = dict(entry)
                task["task_type"] = "legacy"
                task.setdefault("id", f"legacy:{cmd[:20]}")
                self._tasks.append(task)
                count += 1
        if count:
            print(f"[Scheduler] Imported {count} legacy schedule entry/entries")
