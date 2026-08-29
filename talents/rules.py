"""RulesTalent — list, delete, and manage behavioral rules.

Behavioral rules let the user teach Talon conditional behaviors:
    "whenever I say goodnight, turn off the lights"

Rule *creation* is handled implicitly by the conversation handler in
core/assistant.py (_detect_and_store_rule).  This talent only handles
management operations: listing, deleting, and toggling rules.

Examples:
    "list my rules"
    "show my rules"
    "delete rule number 3"
    "remove the goodnight rule"
    "disable rule 2"
    "enable rule 2"
"""

import re
from talents.base import BaseTalent


class RulesTalent(BaseTalent):
    name = "rules"
    description = "List or delete existing behavioral rules (NOT for creating new rules)"
    keywords = [
        "rules", "rule", "my rules", "list rules",
        "delete rule", "remove rule", "disable rule", "enable rule",
    ]
    examples = [
        "list my rules",
        "show my rules",
        "delete rule number 3",
        "remove the goodnight rule",
    ]
    priority = 41  # Below notes (45), above desktop (40)

    _LIST_PHRASES = [
        "list", "show", "view", "what are",
    ]

    _DELETE_PHRASES = [
        "delete", "remove", "cancel", "clear",
    ]

    _ENABLE_PHRASES = ["enable"]
    _DISABLE_PHRASES = ["disable"]

    def execute(self, command: str, context: dict) -> dict:
        memory = context["memory"]
        cmd_lower = command.lower()

        # Determine sub-action
        if any(p in cmd_lower for p in self._DELETE_PHRASES):
            return self._handle_delete(cmd_lower, memory, context)
        elif any(p in cmd_lower for p in self._DISABLE_PHRASES):
            return self._handle_toggle(cmd_lower, memory, context, enabled=False)
        elif any(p in cmd_lower for p in self._ENABLE_PHRASES):
            return self._handle_toggle(cmd_lower, memory, context, enabled=True)
        else:
            # Default: list rules
            return self._handle_list(memory)

    # ── List ──────────────────────────────────────────────────────

    def _handle_list(self, memory):
        rules = memory.list_rules()

        if not rules:
            return {
                "success": True,
                "response": "You don't have any behavioral rules set up yet. "
                            "You can create one by saying something like "
                            "\"whenever I say goodnight, turn off the lights\".",
                "actions_taken": [{"action": "rules_list"}],
                "spoken": False,
            }

        lines = ["Here are your behavioral rules:\n"]
        for r in rules:
            status = "" if r["enabled"] else " (disabled)"
            lines.append(
                f"  {r['id']}. \"{r['trigger_phrase']}\" → "
                f"{r['action_text']}{status}"
            )

        return {
            "success": True,
            "response": "\n".join(lines),
            "actions_taken": [{"action": "rules_list", "count": len(rules)}],
            "spoken": False,
        }

    # ── Delete ────────────────────────────────────────────────────

    def _handle_delete(self, cmd_lower, memory, context):
        # Try to extract a numeric rule ID
        match = re.search(r'(?:rule|#)\s*(\d+)', cmd_lower)
        if match:
            rule_id = int(match.group(1))
            return self._authorize_mutation(
                context,
                summary=f"Delete behavioral rule #{rule_id}",
                metadata={"operation": "delete_rule", "rule_id": rule_id},
                executor=lambda: self._delete_rule(memory, rule_id),
            )

        # No numeric ID — try matching by trigger text
        search_term = self._extract_search_term(cmd_lower)
        if search_term:
            rules = memory.list_rules()
            for r in rules:
                if search_term in r["trigger_phrase"].lower():
                    rule_id = r["id"]
                    return self._authorize_mutation(
                        context,
                        summary=(f"Delete behavioral rule #{rule_id}: "
                                 f"{r['trigger_phrase']!r}"),
                        metadata={"operation": "delete_rule",
                                  "rule_id": rule_id},
                        executor=lambda rid=rule_id: self._delete_rule(
                            memory, rid),
                    )

        return {
            "success": False,
            "response": "I couldn't find a matching rule to delete. "
                        "Try \"list my rules\" to see rule numbers.",
            "actions_taken": [{"action": "rules_delete_miss"}],
            "spoken": False,
        }

    # ── Toggle (enable / disable) ─────────────────────────────────

    def _handle_toggle(self, cmd_lower, memory, context, enabled):
        match = re.search(r'(?:rule|#)\s*(\d+)', cmd_lower)
        if not match:
            state = "enable" if enabled else "disable"
            return {
                "success": False,
                "response": f"Please specify a rule number to {state}. "
                            "Try \"list my rules\" first.",
                "actions_taken": [],
                "spoken": False,
            }

        rule_id = int(match.group(1))
        state = "Enable" if enabled else "Disable"
        return self._authorize_mutation(
            context,
            summary=f"{state} behavioral rule #{rule_id}",
            metadata={"operation": "toggle_rule", "rule_id": rule_id,
                      "enabled": enabled},
            executor=lambda: self._toggle_rule(memory, rule_id, enabled),
        )

    @staticmethod
    def _delete_rule(memory, rule_id):
        deleted = memory.delete_rule(rule_id)
        return {
            "success": bool(deleted),
            "response": (f"Deleted rule #{rule_id}." if deleted else
                         f"I couldn't find rule #{rule_id}."),
            "actions_taken": [{
                "action": "rules_delete" if deleted else "rules_delete_miss",
                "rule_id": rule_id,
            }],
            "spoken": False,
        }

    @staticmethod
    def _toggle_rule(memory, rule_id, enabled):
        toggled = memory.toggle_rule(rule_id, enabled)
        state = "Enabled" if enabled else "Disabled"
        return {
            "success": bool(toggled),
            "response": (f"{state} rule #{rule_id}." if toggled else
                         f"I couldn't find rule #{rule_id}."),
            "actions_taken": [{
                "action": "rules_toggle" if toggled else "rules_toggle_miss",
                "rule_id": rule_id,
                "enabled": enabled,
            }],
            "spoken": False,
        }

    @staticmethod
    def _authorize_mutation(context, *, summary, metadata, executor):
        broker = context.get("capabilities")
        if not broker:
            return executor()
        auth = broker.request(
            "rule_write",
            source=context.get("command_source", "local"),
            summary=summary,
            metadata=metadata,
            executor=executor,
        )
        if auth.allowed:
            try:
                result = executor()
                broker.record_outcome(auth.request,
                                      success=result.get("success", True))
                return result
            except Exception as exc:
                broker.record_outcome(auth.request, success=False,
                                      error=str(exc))
                raise
        if auth.confirmation_required:
            return {
                "success": False,
                "response": broker.confirmation_message(auth),
                "actions_taken": [],
                "spoken": False,
            }
        return {
            "success": False,
            "response": broker.denial_message(auth),
            "actions_taken": [],
            "spoken": False,
        }

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _extract_search_term(cmd_lower):
        """Strip noise words to get a search term for rule matching."""
        for noise in ["delete", "remove", "cancel", "clear", "the",
                       "my", "rule", "rules", "about", "for", "called"]:
            cmd_lower = cmd_lower.replace(noise, "")
        term = cmd_lower.strip()
        return term if len(term) > 1 else ""
