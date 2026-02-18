"""ReminderTalent — set timers and reminders with natural language.

Supports relative ("in 10 minutes") and absolute ("at 3pm") time expressions.
Uses threading.Timer for countdown, plyer for Windows toast notifications,
and persists active reminders to data/reminders.json so they survive restarts.

Examples:
    "remind me in 10 minutes to check the oven"
    "set a timer for 30 minutes"
    "remind me at 3pm to call Bob"
    "list my reminders"
    "cancel the oven reminder"
"""

import os
import json
import re
import time
import threading
from datetime import datetime, timedelta
from talents.base import BaseTalent

# Try to import plyer for native notifications; fall back to print
try:
    from plyer import notification as plyer_notification
    _HAS_PLYER = True
except ImportError:
    _HAS_PLYER = False


def _data_dir():
    """Ensure data/ directory exists and return its path."""
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(d, exist_ok=True)
    return d


class ReminderTalent(BaseTalent):
    name = "reminder"
    description = "Set timers and reminders with natural language"
    keywords = [
        "remind", "reminder", "timer", "alarm", "schedule",
        "in minutes", "in seconds", "in hours",
        "minutes from now", "hours from now",
        "at pm", "at am",
    ]
    examples = [
        "remind me in 10 minutes to check the oven",
        "set a timer for 30 minutes",
        "remind me at 3pm to call Bob",
    ]
    priority = 65

    _REMINDER_PHRASES = [
        "remind", "reminder", "timer", "alarm", "schedule",
        "set a timer", "set timer", "set a reminder", "set reminder",
        "in minutes", "in seconds", "in hours",
        "minutes from now", "hours from now",
        "at pm", "at am",
        "list my reminders", "show reminders", "cancel reminder",
        "cancel the", "delete reminder", "remove reminder",
    ]

    _SYSTEM_PROMPT = (
        "You are a time-parsing assistant. "
        "Given the user's message, extract EXACTLY ONE JSON object with these keys:\n"
        '  "action": "set" | "list" | "cancel"\n'
        '  "seconds": integer (total seconds until reminder fires, for "set" only)\n'
        '  "message": string (what to remind about, for "set" only)\n'
        '  "cancel_query": string (search text to match for "cancel" only)\n'
        "\n"
        "For relative times: 'in 10 minutes' -> seconds=600. "
        "'in 2 hours' -> seconds=7200. 'in 30 seconds' -> seconds=30.\n"
        "For absolute times: calculate seconds from NOW until the target time today. "
        "If the time has already passed today, assume tomorrow.\n"
        "NOW is: {now}\n"
        "\n"
        "Return ONLY the JSON object, no markdown, no explanation."
    )

    _REMINDERS_FILE = os.path.join(_data_dir(), "reminders.json")

    def __init__(self):
        super().__init__()
        self._active_timers: dict[str, threading.Timer] = {}
        self._reminders: dict[str, dict] = {}  # id -> {message, fire_at, ...}
        self._lock = threading.Lock()
        self._notify_cb = None  # Stored by rewire_notify() for alert dialogs
        self._load_reminders()

    # ── Config schema ──────────────────────────────────────────────

    def get_config_schema(self) -> dict:
        return {
            "fields": [
                {"key": "notification_sound", "label": "Play Notification Sound",
                 "type": "bool", "default": True},
                {"key": "default_snooze_minutes", "label": "Default Snooze (minutes)",
                 "type": "int", "default": 5, "min": 1, "max": 60},
            ]
        }

    # ── Routing ────────────────────────────────────────────────────

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    # ── Execution ──────────────────────────────────────────────────

    def execute(self, command: str, context: dict) -> dict:
        cmd_lower = command.lower()

        # Quick-path: list reminders
        if "list" in cmd_lower and ("reminder" in cmd_lower or "timer" in cmd_lower):
            return self._list_reminders()

        # Quick-path: cancel reminder
        if ("cancel" in cmd_lower or "delete" in cmd_lower or "remove" in cmd_lower) \
                and ("reminder" in cmd_lower or "timer" in cmd_lower):
            return self._cancel_reminder_by_query(command, context)

        # Use LLM to parse the time expression
        parsed = self._parse_with_llm(command, context)
        if parsed is None:
            return {
                "success": False,
                "response": "Sorry, I couldn't understand that reminder request.",
                "actions_taken": [{"action": "reminder_parse_fail"}],
                "spoken": False,
            }

        action = parsed.get("action", "set")

        if action == "list":
            return self._list_reminders()
        elif action == "cancel":
            return self._cancel_reminder_by_query(
                parsed.get("cancel_query", command), context
            )
        elif action == "set":
            seconds = parsed.get("seconds", 0)
            message = parsed.get("message", "Reminder!")
            if seconds <= 0:
                return {
                    "success": False,
                    "response": "I couldn't figure out when to remind you. Try something like 'in 10 minutes'.",
                    "actions_taken": [{"action": "reminder_bad_time"}],
                    "spoken": False,
                }
            return self._set_reminder(seconds, message, context)
        else:
            return {
                "success": False,
                "response": "I didn't understand what you want me to do with reminders.",
                "actions_taken": [],
                "spoken": False,
            }

    # ── Set a reminder ─────────────────────────────────────────────

    def _set_reminder(self, seconds, message, context):
        reminder_id = f"rem_{int(time.time() * 1000)}"
        fire_at = (datetime.now() + timedelta(seconds=seconds)).isoformat()

        reminder = {
            "id": reminder_id,
            "message": message,
            "fire_at": fire_at,
            "seconds": seconds,
            "created": datetime.now().isoformat(),
        }

        with self._lock:
            self._reminders[reminder_id] = reminder
            self._save_reminders()

        # Start the timer (uses self._notify_cb set by rewire_notify)
        timer = threading.Timer(seconds, self._fire_reminder, args=(reminder_id,))
        timer.daemon = True
        timer.start()
        self._active_timers[reminder_id] = timer

        # Human-friendly time description
        time_desc = self._format_duration(seconds)

        print(f"   [Reminder] Set: '{message}' in {time_desc} (id={reminder_id})")

        return {
            "success": True,
            "response": f"Got it! I'll remind you {time_desc}: \"{message}\"",
            "actions_taken": [{"action": "reminder_set", "message": message,
                               "seconds": seconds, "id": reminder_id}],
            "spoken": False,
        }

    def _fire_reminder(self, reminder_id):
        """Called by threading.Timer when the reminder fires."""
        with self._lock:
            reminder = self._reminders.pop(reminder_id, None)
            self._save_reminders()

        if reminder_id in self._active_timers:
            del self._active_timers[reminder_id]

        if reminder is None:
            return

        message = reminder.get("message", "Reminder!")
        print(f"\n   [Reminder] FIRED: {message}")

        default_snooze = self._config.get("default_snooze_minutes", 5)

        # Push through the dismissable alert dialog (bridge → MainWindow)
        if self._notify_cb:
            try:
                self._notify_cb(reminder_id, "Talon Reminder", message, default_snooze)
            except Exception as e:
                print(f"   [Reminder] Alert callback error: {e}")
                self._plyer_fallback(message)
        else:
            # No callback available (headless / pre-bridge) — use OS toast
            self._plyer_fallback(message)

    def _plyer_fallback(self, message):
        """Fall back to OS toast when Qt alert dialog is unavailable."""
        if _HAS_PLYER:
            try:
                plyer_notification.notify(
                    title="Talon Reminder",
                    message=message,
                    app_name="Talon",
                    timeout=10,
                )
            except Exception as e:
                print(f"   [Reminder] plyer notification error: {e}")

    # ── List reminders ─────────────────────────────────────────────

    def _list_reminders(self):
        with self._lock:
            reminders = list(self._reminders.values())

        if not reminders:
            return {
                "success": True,
                "response": "You don't have any active reminders.",
                "actions_taken": [{"action": "reminder_list"}],
                "spoken": False,
            }

        lines = ["Here are your active reminders:\n"]
        now = datetime.now()
        for r in sorted(reminders, key=lambda x: x["fire_at"]):
            try:
                fire_dt = datetime.fromisoformat(r["fire_at"])
                remaining = fire_dt - now
                if remaining.total_seconds() > 0:
                    time_left = self._format_duration(int(remaining.total_seconds()))
                else:
                    time_left = "any moment now"
            except (ValueError, TypeError):
                time_left = "unknown"

            lines.append(f"• \"{r['message']}\" — fires in {time_left}")

        return {
            "success": True,
            "response": "\n".join(lines),
            "actions_taken": [{"action": "reminder_list"}],
            "spoken": False,
        }

    # ── Cancel reminder ────────────────────────────────────────────

    def _cancel_reminder_by_query(self, query, context):
        query_lower = query.lower()

        # Strip noise words to get the search term
        for noise in ["cancel", "delete", "remove", "the", "my", "reminder",
                       "timer", "about", "for", "called"]:
            query_lower = query_lower.replace(noise, "")
        search_term = query_lower.strip()

        with self._lock:
            matches = []
            for rid, r in self._reminders.items():
                if search_term and search_term in r["message"].lower():
                    matches.append(rid)

            if not matches and search_term:
                # Fuzzy: check if any word overlaps
                search_words = set(search_term.split())
                for rid, r in self._reminders.items():
                    msg_words = set(r["message"].lower().split())
                    if search_words & msg_words:
                        matches.append(rid)

            if not matches:
                return {
                    "success": False,
                    "response": f"I couldn't find a reminder matching '{search_term}'.",
                    "actions_taken": [{"action": "reminder_cancel_miss"}],
                    "spoken": False,
                }

            cancelled = []
            for rid in matches:
                reminder = self._reminders.pop(rid, None)
                if reminder:
                    cancelled.append(reminder["message"])
                # Cancel the thread timer
                timer = self._active_timers.pop(rid, None)
                if timer:
                    timer.cancel()

            self._save_reminders()

        msgs = ", ".join(f'"{m}"' for m in cancelled)
        return {
            "success": True,
            "response": f"Cancelled: {msgs}",
            "actions_taken": [{"action": "reminder_cancel", "cancelled": cancelled}],
            "spoken": False,
        }

    # ── LLM parsing ───────────────────────────────────────────────

    def _parse_with_llm(self, command, context):
        """Use the LLM to extract time and message from natural language."""
        # First try regex for common patterns (fast path, no LLM call)
        quick = self._quick_parse(command)
        if quick:
            return quick

        llm = context.get("llm")
        if not llm:
            return None

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_prompt = self._SYSTEM_PROMPT.replace("{now}", now_str)

        response = llm.generate(
            f"Parse this reminder request:\n\n{command}",
            system_prompt=system_prompt,
            temperature=0.1,
        )

        # Extract JSON from response
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        print(f"   [Reminder] LLM parse failed: {response[:200]}")
        return None

    def _quick_parse(self, command):
        """Regex-based fast parsing for common patterns."""
        cmd = command.lower()

        # "in X minutes/hours/seconds"
        match = re.search(
            r'in\s+(\d+)\s*(minute|min|hour|hr|second|sec)s?',
            cmd
        )
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            if unit.startswith("hour") or unit.startswith("hr"):
                seconds = amount * 3600
            elif unit.startswith("min"):
                seconds = amount * 60
            else:
                seconds = amount

            # Extract the message — look for "to ..." or "about ..." after the time
            message = "Reminder!"
            msg_match = re.search(r'(?:to|about|that)\s+(.+?)(?:\.|$)', cmd)
            if msg_match:
                message = msg_match.group(1).strip().rstrip(".")
            else:
                # Try text before "in X minutes"
                before = cmd.split("in " + match.group(0).split("in ")[-1])[0]
                before = re.sub(
                    r'^(remind me|set a reminder|set reminder|set a timer|set timer)\s*',
                    '', before
                ).strip()
                if before and len(before) > 3:
                    message = before

            return {"action": "set", "seconds": seconds, "message": message}

        # "X minute/hour timer"
        match = re.search(
            r'(\d+)\s*(minute|min|hour|hr|second|sec)s?\s*(timer|alarm)',
            cmd
        )
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            if unit.startswith("hour") or unit.startswith("hr"):
                seconds = amount * 3600
            elif unit.startswith("min"):
                seconds = amount * 60
            else:
                seconds = amount
            return {"action": "set", "seconds": seconds, "message": f"{amount} {unit} timer"}

        return None

    # ── Persistence ────────────────────────────────────────────────

    def _load_reminders(self):
        """Load persisted reminders from disk and re-arm those still in the future."""
        try:
            if os.path.exists(self._REMINDERS_FILE):
                with open(self._REMINDERS_FILE, 'r') as f:
                    saved = json.load(f)

                now = datetime.now()
                rearmed = 0
                for rid, r in saved.items():
                    try:
                        fire_dt = datetime.fromisoformat(r["fire_at"])
                        remaining = (fire_dt - now).total_seconds()
                        if remaining > 0:
                            self._reminders[rid] = r
                            # Re-arm timer (self._notify_cb is None until bridge calls rewire_notify)
                            timer = threading.Timer(remaining, self._fire_reminder, args=(rid,))
                            timer.daemon = True
                            timer.start()
                            self._active_timers[rid] = timer
                            rearmed += 1
                        # Expired reminders are silently discarded
                    except (ValueError, KeyError):
                        continue

                if rearmed:
                    print(f"   [Reminder] Re-armed {rearmed} saved reminder(s)")
        except Exception as e:
            print(f"   [Reminder] Error loading reminders: {e}")

    def _save_reminders(self):
        """Persist active reminders to disk (call while holding self._lock)."""
        try:
            with open(self._REMINDERS_FILE, 'w') as f:
                json.dump(self._reminders, f, indent=2)
        except Exception as e:
            print(f"   [Reminder] Error saving reminders: {e}")

    # ── Re-wire notify callback after bridge is ready ──────────────

    def initialize(self, config: dict) -> None:
        """Called after construction. Re-arms timers with the notify callback
        once the assistant has set it up (happens after __init__)."""
        pass

    def rewire_notify(self, notify_cb):
        """Store the alert callback and re-create timers to use it.

        Called by the bridge after set_assistant() so that reminders loaded
        at startup (before the bridge exists) can still send notifications.
        _fire_reminder() reads self._notify_cb at fire time.
        """
        self._notify_cb = notify_cb
        with self._lock:
            for rid in list(self._active_timers.keys()):
                old_timer = self._active_timers[rid]
                remaining = self._reminders.get(rid, {}).get("fire_at")
                if not remaining:
                    continue
                try:
                    fire_dt = datetime.fromisoformat(remaining)
                    secs = (fire_dt - datetime.now()).total_seconds()
                    if secs <= 0:
                        continue
                    old_timer.cancel()
                    new_timer = threading.Timer(secs, self._fire_reminder, args=(rid,))
                    new_timer.daemon = True
                    new_timer.start()
                    self._active_timers[rid] = new_timer
                except (ValueError, TypeError):
                    continue

    # ── Snooze ─────────────────────────────────────────────────────

    def snooze_reminder(self, reminder_id, message, seconds):
        """Re-arm a reminder after the user clicks Snooze in the alert dialog."""
        new_id = f"rem_{int(time.time() * 1000)}"
        fire_at = (datetime.now() + timedelta(seconds=seconds)).isoformat()
        reminder = {
            "id": new_id,
            "message": message,
            "fire_at": fire_at,
            "seconds": seconds,
            "created": datetime.now().isoformat(),
        }
        with self._lock:
            self._reminders[new_id] = reminder
            self._save_reminders()
        timer = threading.Timer(seconds, self._fire_reminder, args=(new_id,))
        timer.daemon = True
        timer.start()
        self._active_timers[new_id] = timer

        time_desc = self._format_duration(seconds)
        print(f"   [Reminder] Snoozed: '{message}' {time_desc} (id={new_id})")

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _format_duration(seconds):
        """Format seconds into human-readable duration string."""
        if seconds < 60:
            return f"in {seconds} second{'s' if seconds != 1 else ''}"
        elif seconds < 3600:
            mins = seconds // 60
            return f"in {mins} minute{'s' if mins != 1 else ''}"
        else:
            hours = seconds // 3600
            mins = (seconds % 3600) // 60
            parts = [f"{hours} hour{'s' if hours != 1 else ''}"]
            if mins:
                parts.append(f"{mins} minute{'s' if mins != 1 else ''}")
            return "in " + " and ".join(parts)
