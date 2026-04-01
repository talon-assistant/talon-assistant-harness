"""SchedulerTalent — create, list, and cancel persistent scheduled tasks.

Handles recurring reminders, interval-based automation, cron expressions,
and one-time future commands.  All tasks persist across restarts in
config/scheduled_tasks.json.

Distinct from ReminderTalent (priority 65) which handles one-off pop-up
reminder alerts.  This talent handles *scheduled commands* that Talon will
execute at a future or recurring time.

Examples:
    "run the news digest every morning at 7am"
    "every 30 minutes check the weather and read it aloud"
    "list my scheduled tasks"
    "cancel the weather schedule"
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

from talents.base import BaseTalent
from core.llm_client import LLMError


class SchedulerTalent(BaseTalent):
    name = "scheduler"
    description = (
        "Create, list, and cancel persistent scheduled commands — "
        "recurring (interval/cron) or one-time future execution"
    )
    keywords = [
        "schedule", "scheduled", "recurring",
        "every hour", "every day", "every morning", "every night",
        "every minute", "run every", "execute every",
        "set up a recurring", "cron",
        "list schedules", "list scheduled tasks", "show scheduled tasks",
        "cancel schedule", "delete schedule", "remove schedule",
    ]
    examples = [
        "run the news digest every morning at 7am",
        "every 30 minutes check the weather and speak it aloud",
        "schedule a command to run daily at midnight",
        "list my scheduled tasks",
        "show scheduled tasks",
        "cancel the weather schedule",
        "set up a recurring task every hour",
        "run a command at 2026-04-01 09:00",
        "every 2 hours summarize the news",
        "delete all scheduled tasks",
    ]
    priority = 60   # Below ReminderTalent (65) — reminder handles one-off alerts

    # ── Extraction prompt ─────────────────────────────────────────────────────

    _EXTRACT_SYSTEM_PROMPT = """\
You are a scheduling-intent extractor for a desktop assistant.
Given the user's message, return a single JSON object.

Possible actions:
  "create"  — schedule a new recurring or future task
  "list"    — list current scheduled tasks
  "cancel"  — cancel / delete a scheduled task

For "create", fields:
  task_type:        "once" | "interval" | "cron"
  command:          the exact command string Talon should execute (not the scheduling instruction itself)
  at:               ISO datetime string for once tasks with an absolute time — null otherwise
  relative_seconds: integer seconds from now for once tasks like "in 30 minutes" — null otherwise
  interval_seconds: integer seconds between runs for interval tasks — null otherwise
  cron:             cron expression for cron tasks, e.g. "0 7 * * *" — null otherwise
  output:           "gui" | "tts" | "both" — use "tts" or "both" if user says "speak", "read aloud", "say it"; default "gui"
  label:            short human-readable label

For "cancel":
  cancel_id:    task id if given — null otherwise
  cancel_match: text to match against label or command — null if cancel_id given

For "list": no extra fields needed.

Current local time (ISO): {now}

Return ONLY valid JSON, no markdown, no explanation.

Examples:
  "every 30 minutes check the weather aloud" ->
    {{"action":"create","task_type":"interval","command":"check the weather","interval_seconds":1800,"output":"tts","label":"weather every 30 min","at":null,"relative_seconds":null,"cron":null,"cancel_id":null,"cancel_match":null}}

  "run the news digest every morning at 7am" ->
    {{"action":"create","task_type":"cron","command":"generate morning news digest","cron":"0 7 * * *","output":"gui","label":"morning news digest","at":null,"relative_seconds":null,"interval_seconds":null,"cancel_id":null,"cancel_match":null}}

  "in 45 minutes run a backup" ->
    {{"action":"create","task_type":"once","command":"run backup","relative_seconds":2700,"output":"gui","label":"backup in 45 min","at":null,"interval_seconds":null,"cron":null,"cancel_id":null,"cancel_match":null}}

  "list my scheduled tasks" ->
    {{"action":"list","task_type":null,"command":null,"at":null,"relative_seconds":null,"interval_seconds":null,"cron":null,"output":"gui","label":null,"cancel_id":null,"cancel_match":null}}

  "cancel the weather schedule" ->
    {{"action":"cancel","task_type":null,"command":null,"at":null,"relative_seconds":null,"interval_seconds":null,"cron":null,"output":"gui","label":null,"cancel_id":null,"cancel_match":"weather"}}
"""

    # ── Execute ───────────────────────────────────────────────────────────────

    def execute(self, command: str, context: dict) -> dict:
        llm       = context.get("llm")
        assistant = context.get("assistant")

        if not llm:
            return self._err("LLM not available.")

        scheduler = getattr(assistant, "scheduler", None)
        if scheduler is None:
            return self._err("Scheduler is not initialised.")

        intent = self._extract_intent(command, llm)
        if intent is None:
            return self._err(
                "I couldn't understand that scheduling request. "
                "Try something like 'every 30 minutes check the weather' "
                "or 'list my scheduled tasks'."
            )

        action = (intent.get("action") or "").lower()

        if action == "list":
            return self._handle_list(scheduler)
        elif action == "cancel":
            return self._handle_cancel(scheduler, intent)
        elif action == "create":
            return self._handle_create(scheduler, intent)
        else:
            return self._err(f"Unknown scheduling action: {action!r}")

    # ── Action handlers ───────────────────────────────────────────────────────

    def _handle_list(self, scheduler) -> dict:
        tasks = [t for t in scheduler.list_tasks() if t.get("task_type") != "legacy"]

        if not tasks:
            return self._ok(
                "You have no scheduled tasks set up.",
                action="schedule_list", count=0,
            )

        lines = [f"You have {len(tasks)} scheduled task(s):\n"]
        for t in tasks:
            status = "enabled" if t.get("enabled", True) else "DISABLED"
            ttype  = t.get("task_type", "?")
            label  = t.get("label") or t.get("command", "?")
            tid    = t.get("id", "?")

            if ttype == "once":
                timing = f"fires at {t.get('at', '?')[:16].replace('T', ' ')}"
            elif ttype == "interval":
                secs   = int(t.get("interval_seconds", 0))
                next_r = t.get("next_run", "?")[:16].replace("T", " ")
                timing = f"every {self._fmt_seconds(secs)}, next: {next_r}"
            elif ttype == "cron":
                next_r = t.get("next_cron_run", "?")[:16].replace("T", " ")
                timing = f"cron '{t.get('cron', '?')}', next: {next_r}"
            else:
                timing = "?"

            lines.append(f"  [{tid}] {label} — {timing} ({status})")

        return self._ok("\n".join(lines), action="schedule_list", count=len(tasks))

    def _handle_cancel(self, scheduler, intent: dict) -> dict:
        cancel_id    = (intent.get("cancel_id") or "").strip()
        cancel_match = (intent.get("cancel_match") or "").strip().lower()

        if cancel_id:
            removed = scheduler.cancel_task(cancel_id)
            if removed:
                return self._ok(
                    f"Cancelled scheduled task {cancel_id!r}.",
                    action="schedule_cancel", id=cancel_id,
                )
            return self._err(f"No scheduled task found with id {cancel_id!r}.")

        if cancel_match:
            tasks   = [t for t in scheduler.list_tasks() if t.get("task_type") != "legacy"]
            matched = [
                t for t in tasks
                if cancel_match in (t.get("label") or "").lower()
                or cancel_match in (t.get("command") or "").lower()
            ]
            if not matched:
                return self._err(f"No scheduled task found matching '{cancel_match}'.")

            labels = []
            for t in matched:
                scheduler.cancel_task(t["id"])
                labels.append(t.get("label") or t.get("command", t["id"]))

            labels_str = ", ".join(f'"{l}"' for l in labels)
            return self._ok(
                f"Cancelled: {labels_str}.",
                action="schedule_cancel", matched=labels,
            )

        return self._err(
            "Please tell me which scheduled task to cancel — "
            "by id or by name (e.g. 'cancel the weather schedule')."
        )

    def _handle_create(self, scheduler, intent: dict) -> dict:
        task_type = (intent.get("task_type") or "").lower()
        command   = (intent.get("command") or "").strip()
        output    = intent.get("output") or "gui"
        label     = (intent.get("label") or command[:40]).strip()

        if not command:
            return self._err(
                "I couldn't identify which command to schedule. "
                "Try: 'schedule \"check the weather\" every hour'."
            )

        if task_type not in ("once", "interval", "cron"):
            return self._err(
                f"Unknown task type {task_type!r}. Use 'once', 'interval', or 'cron'."
            )

        # ── once ──────────────────────────────────────────────────────────
        if task_type == "once":
            at_str   = (intent.get("at") or "").strip()
            rel_secs = intent.get("relative_seconds")

            if rel_secs and not at_str:
                fire_dt = datetime.now() + timedelta(seconds=int(rel_secs))
                at_str  = fire_dt.isoformat()
            elif not at_str:
                return self._err(
                    "For a one-time task I need to know when. "
                    "Try 'in 45 minutes' or give a date and time."
                )

            try:
                task = scheduler.create_task(
                    command, "once", at=at_str, output=output, label=label
                )
            except ValueError as exc:
                return self._err(str(exc))

            fire_display = at_str[:16].replace("T", " at ")
            return self._ok(
                f"Scheduled: \"{command}\" will run once on {fire_display}. "
                f"Task id: {task['id']}.",
                action="schedule_create", task=task,
            )

        # ── interval ──────────────────────────────────────────────────────
        elif task_type == "interval":
            interval_seconds = intent.get("interval_seconds")
            if not interval_seconds:
                return self._err(
                    "I need to know how often to run this. "
                    "Try 'every 30 minutes' or 'every 2 hours'."
                )
            try:
                task = scheduler.create_task(
                    command, "interval",
                    interval_seconds=int(interval_seconds),
                    output=output, label=label,
                )
            except ValueError as exc:
                return self._err(str(exc))

            freq = self._fmt_seconds(int(interval_seconds))
            return self._ok(
                f"Scheduled: \"{command}\" will run every {freq}. "
                f"Task id: {task['id']}.",
                action="schedule_create", task=task,
            )

        # ── cron ──────────────────────────────────────────────────────────
        else:
            cron_expr = (intent.get("cron") or "").strip()
            if not cron_expr:
                return self._err(
                    "I need a cron expression. "
                    "For example, '0 7 * * *' runs at 7 am every day."
                )
            try:
                task = scheduler.create_task(
                    command, "cron", cron=cron_expr, output=output, label=label
                )
            except ValueError as exc:
                return self._err(str(exc))

            return self._ok(
                f"Scheduled: \"{command}\" will run on cron '{cron_expr}'. "
                f"Task id: {task['id']}.",
                action="schedule_create", task=task,
            )

    # ── LLM extraction ────────────────────────────────────────────────────────

    def _extract_intent(self, command: str, llm) -> dict | None:
        now_str       = datetime.now().isoformat(timespec="seconds")
        system_prompt = self._EXTRACT_SYSTEM_PROMPT.format(now=now_str)

        try:
            raw = llm.generate(
                f"Extract scheduling intent:\n\n{command}",
                system_prompt=system_prompt,
                temperature=0.1,
                max_length=300,
            )
        except LLMError as exc:
            print(f"   [SchedulerTalent] LLM unavailable: {exc}")
            return None
        except Exception as exc:
            print(f"   [SchedulerTalent] LLM error: {exc}")
            return None

        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not match:
                print(f"   [SchedulerTalent] No JSON in LLM response: {raw[:200]}")
                return None
            return json.loads(match.group())
        except json.JSONDecodeError as exc:
            print(f"   [SchedulerTalent] JSON parse error: {exc} — raw: {raw[:200]}")
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_seconds(seconds: int) -> str:
        if seconds < 60:
            return f"{seconds} second{'s' if seconds != 1 else ''}"
        elif seconds < 3600:
            m = seconds // 60
            return f"{m} minute{'s' if m != 1 else ''}"
        elif seconds < 86400:
            h = seconds // 3600
            m = (seconds % 3600) // 60
            s = f"{h} hour{'s' if h != 1 else ''}"
            if m:
                s += f" {m} min"
            return s
        else:
            d = seconds // 86400
            return f"{d} day{'s' if d != 1 else ''}"

    @staticmethod
    def _ok(message: str, **meta) -> dict:
        actions = [{"action": meta.pop("action", "scheduler")} | meta] if meta else [{"action": "scheduler"}]
        return {
            "success":      True,
            "response":     message,
            "actions_taken": actions,
            "spoken":       False,
        }

    @staticmethod
    def _err(message: str) -> dict:
        return {
            "success":      False,
            "response":     message,
            "actions_taken": [],
            "spoken":       False,
        }
