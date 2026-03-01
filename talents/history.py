"""history.py — search past commands and responses from the SQLite command log.

Commands: "what did I ask yesterday?", "did I ever ask about lights?",
          "show my recent commands", "show me failed commands from today"
"""
from datetime import datetime, timedelta

from talents.base import BaseTalent


class HistoryTalent(BaseTalent):
    name = "history"
    description = "Search past commands and responses from the command history log"
    keywords = [
        "what did i ask", "did i ever ask", "did i ask",
        "command history", "show my history", "show me my commands",
        "when did i ask", "last time i asked", "what have i asked",
        "past commands", "history search", "what commands did i",
    ]
    examples = [
        "what did I ask you yesterday?",
        "did I ever ask about lights?",
        "show me my recent commands",
        "what did I ask about the weather last week?",
        "show me failed commands from today",
    ]
    priority = 43

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    # ── Date parsing ───────────────────────────────────────────────

    def _parse_date_range(self, command: str) -> tuple[str | None, str | None]:
        """Return (start_iso, end_iso) from natural-language date references.

        Handles: today, yesterday, this week, last week, this month,
        last Monday/Tuesday/… etc.  Returns (None, None) if no date found.
        """
        now = datetime.now()
        cmd = command.lower()

        if "today" in cmd:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return start.isoformat(), now.isoformat()

        if "yesterday" in cmd:
            y = now - timedelta(days=1)
            return (
                y.replace(hour=0, minute=0, second=0, microsecond=0).isoformat(),
                y.replace(hour=23, minute=59, second=59).isoformat(),
            )

        if "this week" in cmd or "past week" in cmd:
            return (now - timedelta(days=7)).isoformat(), now.isoformat()

        if "last week" in cmd:
            return (
                (now - timedelta(days=14)).isoformat(),
                (now - timedelta(days=7)).isoformat(),
            )

        if "this month" in cmd:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return start.isoformat(), now.isoformat()

        # "last Monday", "last Tuesday", etc.
        day_names = [
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday",
        ]
        for i, day in enumerate(day_names):
            if f"last {day}" in cmd:
                delta = (now.weekday() - i) % 7 or 7
                d = now - timedelta(days=delta)
                return (
                    d.replace(hour=0, minute=0, second=0, microsecond=0).isoformat(),
                    d.replace(hour=23, minute=59, second=59).isoformat(),
                )

        return None, None

    # ── Formatting ─────────────────────────────────────────────────

    def _format_results(
        self,
        results: list[dict],
        keyword: str | None,
        start_ts: str | None,
    ) -> str:
        if not results:
            parts = ["No matching commands found"]
            if keyword:
                parts[0] += f" for '{keyword}'"
            if start_ts:
                parts[0] += " in that time range"
            return parts[0] + "."

        header_parts = [f"Found {len(results)} command(s)"]
        if keyword:
            header_parts.append(f"matching '{keyword}'")
        lines = [", ".join(header_parts) + ":\n"]

        for r in results:
            try:
                ts = datetime.fromisoformat(r["timestamp"]).strftime("%Y-%m-%d %H:%M")
            except Exception:
                ts = r["timestamp"][:16]
            status = "✓" if r["success"] else "✗"
            cmd = r["command"][:80] + ("…" if len(r["command"]) > 80 else "")
            lines.append(f"  {status} {ts} — {cmd}")

        return "\n".join(lines)

    # ── Execute ────────────────────────────────────────────────────

    def execute(self, command: str, context: dict) -> dict:
        llm    = context["llm"]
        memory = context["memory"]

        # 1. Extract optional keyword/topic
        keyword = self._extract_arg(
            llm, command, "search topic or keyword",
            max_length=30, temperature=0.0,
        )
        if keyword and keyword.lower() in (
            "none", "anything", "commands", "history", "recent", "all",
        ):
            keyword = None

        # 2. Date range from pattern matching
        start_ts, end_ts = self._parse_date_range(command)

        # 3. Success filter
        cmd_lower = command.lower()
        success_filter = None
        if any(w in cmd_lower for w in ("failed", "failure", "didn't work", "did not work")):
            success_filter = False
        elif any(w in cmd_lower for w in ("successful", "worked", "succeeded")):
            success_filter = True

        # 4. Limit — bump to 30 for "all / everything / every"
        limit = 30 if any(w in cmd_lower for w in ("all", "every", "everything")) else 10

        # 5. Query
        results = memory.search_commands(
            keyword=keyword,
            start_ts=start_ts,
            end_ts=end_ts,
            success_filter=success_filter,
            limit=limit,
        )

        # 6. Format and return
        response = self._format_results(results, keyword, start_ts)
        return {"success": True, "response": response, "actions_taken": []}
