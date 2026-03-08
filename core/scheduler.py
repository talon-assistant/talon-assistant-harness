"""Generic background task scheduler for Talon.

Reads a schedule list from settings.json:

  "scheduler": [
    {
      "command": "generate morning news digest",
      "time":    "07:00",
      "days":    ["mon","tue","wed","thu","fri","sat","sun"],
      "enabled": true
    }
  ]

Each matching command is passed to assistant.process_command() at the
scheduled time in its own daemon thread so the scheduler loop is never
blocked.  Commands fire at most once per day per (time, command) pair.
"""
from __future__ import annotations

import threading
import time
import datetime


class Scheduler:
    """Lightweight cron-style scheduler wired to the assistant."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._running = False
        self._schedule: list[dict] = []
        self._assistant = None
        # (date_iso, time_str, command) — prevents double-fire within the same day
        self._fired: set[tuple] = set()

    # ── public API ────────────────────────────────────────────────────────────

    def start(self, schedule: list[dict], assistant) -> None:
        """Start the scheduler background thread.

        Args:
            schedule:  List of task dicts from settings.json "scheduler" key.
            assistant: TalonAssistant instance whose process_command() to call.
        """
        if not schedule:
            return
        self._schedule  = schedule
        self._assistant = assistant
        self._running   = True
        self._thread    = threading.Thread(
            target=self._loop, daemon=True, name="TalonScheduler"
        )
        self._thread.start()
        enabled = [t for t in schedule if t.get("enabled", True)]
        tasks_str = ", ".join(
            f"{t.get('time','?')} → \"{t.get('command','?')}\"" for t in enabled
        )
        print(f"[Scheduler] Started — {len(enabled)} active task(s): {tasks_str}")

    def stop(self) -> None:
        self._running = False

    # ── internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            now       = datetime.datetime.now()
            today     = now.date().isoformat()
            day_abbr  = now.strftime("%a").lower()   # "mon", "tue", …
            time_str  = now.strftime("%H:%M")

            # Expire fired-set entries from previous days
            self._fired = {k for k in self._fired if k[0] == today}

            for task in self._schedule:
                if not task.get("enabled", True):
                    continue

                task_time = task.get("time", "")
                task_days = [
                    d.lower()[:3]
                    for d in task.get("days", ["mon","tue","wed","thu","fri","sat","sun"])
                ]
                command = task.get("command", "").strip()

                if not command or task_time != time_str:
                    continue
                if day_abbr not in task_days:
                    continue

                key = (today, task_time, command)
                if key in self._fired:
                    continue

                self._fired.add(key)
                print(f"[Scheduler] Firing: {command!r}")
                threading.Thread(
                    target=self._run_command,
                    args=(command,),
                    daemon=True,
                    name=f"Scheduler-{task_time}",
                ).start()

            time.sleep(20)   # check granularity: every 20 seconds

    def _run_command(self, command: str) -> None:
        try:
            self._assistant.process_command(command)
        except Exception as e:
            print(f"[Scheduler] Error running {command!r}: {e}")
