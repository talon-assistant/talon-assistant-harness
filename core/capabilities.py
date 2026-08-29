"""Central policy and approval broker for actions with side effects.

Talents and MCP servers describe *what* they intend to do; this module decides
whether the action may run, must be confirmed, or is denied for the originating
command source.  Pending approvals are process-local and short lived.  Durable
audit records intentionally contain only a short summary, never action payloads
such as email bodies, credentials, or file contents.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping

import logging

log = logging.getLogger(__name__)


class CapabilityDecision(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass(frozen=True)
class CapabilityRequest:
    request_id: str
    capability: str
    source: str
    summary: str
    metadata: Mapping[str, Any]
    created_at: float
    expires_at: float


@dataclass(frozen=True)
class Authorization:
    decision: CapabilityDecision
    request: CapabilityRequest
    reason: str

    @property
    def allowed(self) -> bool:
        return self.decision is CapabilityDecision.ALLOW

    @property
    def confirmation_required(self) -> bool:
        return self.decision is CapabilityDecision.CONFIRM


@dataclass
class _PendingAction:
    request: CapabilityRequest
    executor: Callable[[], Any] | None = None


DEFAULT_POLICIES: dict[str, dict[str, str]] = {
    "external_send": {
        "local": "confirm",
        "signal": "confirm",
        "hermes": "confirm",
        "scheduler": "deny",
        "reflection": "deny",
        "default": "confirm",
    },
    "rule_write": {
        "local": "confirm",
        "signal": "confirm",
        "hermes": "confirm",
        "scheduler": "deny",
        "reflection": "deny",
        "default": "confirm",
    },
    "destructive_file_ops": {
        "local": "confirm",
        "signal": "confirm",
        "hermes": "confirm",
        "scheduler": "deny",
        "reflection": "deny",
        "default": "confirm",
    },
    "mcp_write": {
        "local": "confirm",
        "signal": "confirm",
        "hermes": "confirm",
        "scheduler": "deny",
        "reflection": "deny",
        "default": "confirm",
    },
    "desktop_control": {
        "local": "allow",
        "signal": "confirm",
        "hermes": "confirm",
        "scheduler": "deny",
        "reflection": "deny",
        "default": "confirm",
    },
    "plugin_install": {
        "local": "confirm",
        "signal": "confirm",
        "hermes": "confirm",
        "scheduler": "deny",
        "reflection": "deny",
        "default": "confirm",
    },
    "local_data_write": {
        "local": "allow",
        "signal": "confirm",
        "hermes": "confirm",
        "scheduler": "deny",
        "reflection": "deny",
        "default": "confirm",
    },
    "device_control": {
        "local": "allow",
        "signal": "confirm",
        "hermes": "confirm",
        "scheduler": "deny",
        "reflection": "deny",
        "default": "confirm",
    },
    "clipboard_write": {
        "local": "allow",
        "signal": "deny",
        "hermes": "deny",
        "scheduler": "deny",
        "reflection": "deny",
        "default": "deny",
    },
    "process_execution": {
        "local": "confirm",
        "signal": "confirm",
        "hermes": "confirm",
        "scheduler": "deny",
        "reflection": "deny",
        "default": "confirm",
    },
    "credential_write": {
        "local": "confirm",
        "signal": "deny",
        "hermes": "deny",
        "scheduler": "deny",
        "reflection": "deny",
        "default": "deny",
    },
    "external_account_write": {
        "local": "confirm",
        "signal": "confirm",
        "hermes": "confirm",
        "scheduler": "deny",
        "reflection": "deny",
        "default": "confirm",
    },
}

POLICY_SOURCES: tuple[str, ...] = (
    "local", "signal", "hermes", "scheduler", "reflection", "default",
)

AUDIT_EVENTS: tuple[str, ...] = (
    "confirmation_required", "allowed", "denied", "confirmed",
    "cancelled", "expired", "executed", "failed",
    "sandbox_started", "sandbox_completed", "sandbox_denied",
    "sandbox_timeout", "sandbox_failed",
)

_APPROVE_RE = re.compile(
    r"^\s*(?:yes\s*,?\s*)?(?:confirm|approve|proceed|do it|go ahead)"
    r"(?:\s+(?P<id>[a-f0-9]{6,32}))?[.!]?\s*$",
    re.IGNORECASE,
)
_CANCEL_RE = re.compile(
    r"^\s*(?:cancel|deny|reject|abort|stop)"
    r"(?:\s+(?P<id>[a-f0-9]{6,32}))?[.!]?\s*$",
    re.IGNORECASE,
)


class CapabilityBroker:
    """Evaluate, queue, resolve, and audit privileged action requests."""

    def __init__(self, config: dict | None = None, db_path: str | None = None):
        self._config = config or {}
        self._db_path = db_path
        self._lock = threading.RLock()
        self._pending: dict[str, _PendingAction] = {}
        self._init_audit_table()

    def reload(self, config: dict | None) -> None:
        with self._lock:
            self._config = config or {}
            self._purge_expired_locked()

    def request(
        self,
        capability: str,
        *,
        source: str = "local",
        summary: str,
        metadata: Mapping[str, Any] | None = None,
        executor: Callable[[], Any] | None = None,
    ) -> Authorization:
        """Return a policy decision and queue the action when confirmation is required."""
        capability = (capability or "unknown").strip().lower()
        source = (source or "local").strip().lower()
        now = time.time()
        ttl = max(30, int(self._config.get("pending_ttl_seconds", 300)))
        request = CapabilityRequest(
            request_id=uuid.uuid4().hex[:12],
            capability=capability,
            source=source,
            summary=(summary or capability).strip()[:300],
            metadata=MappingProxyType(dict(metadata or {})),
            created_at=now,
            expires_at=now + ttl,
        )
        decision = self._decision_for(capability, source)
        reason = f"policy for {capability!r} from {source!r}: {decision.value}"

        if decision is CapabilityDecision.CONFIRM:
            with self._lock:
                self._purge_expired_locked()
                self._pending[request.request_id] = _PendingAction(request, executor)
            self._audit(request, "confirmation_required")
        elif decision is CapabilityDecision.DENY:
            self._audit(request, "denied")
        else:
            self._audit(request, "allowed")
        return Authorization(decision, request, reason)

    def confirmation_message(self, authorization: Authorization) -> str:
        req = authorization.request
        return (
            f"Approval required for: {req.summary}. "
            f"Reply 'confirm {req.request_id}' to proceed or "
            f"'cancel {req.request_id}' to abort."
        )

    def denial_message(self, authorization: Authorization) -> str:
        req = authorization.request
        return (
            f"Blocked by capability policy: {req.summary} "
            f"({req.capability} from {req.source})."
        )

    def resolve_confirmation(self, command: str, source: str = "local") -> dict | None:
        """Resolve a textual confirm/cancel command for this source.

        Returns ``None`` when *command* is not an approval response or no
        matching request exists.  Executors run after the pending entry has
        been removed so retries always require a new approval.
        """
        match = _APPROVE_RE.match(command or "")
        approved = True
        if not match:
            match = _CANCEL_RE.match(command or "")
            approved = False
        if not match:
            return None

        with self._lock:
            self._purge_expired_locked()
            pending = self._select_pending_locked(
                source=(source or "local").lower(),
                request_prefix=match.group("id"),
            )
            if pending is None:
                return None
            self._pending.pop(pending.request.request_id, None)

        req = pending.request
        if not approved:
            self._audit(req, "cancelled")
            return {
                "response": f"Cancelled: {req.summary}.",
                "talent": "capability_broker",
                "success": True,
                "actions_taken": [],
            }

        self._audit(req, "confirmed")
        if pending.executor is None:
            return {
                "response": f"Approved: {req.summary}.",
                "talent": "capability_broker",
                "success": True,
                "actions_taken": [],
                "approved_request": req,
            }
        return self._execute(req, pending.executor)

    def approve(self, request_id: str, *, source: str = "local") -> CapabilityRequest | None:
        """Approve a request from a non-text UI and return it to the caller."""
        with self._lock:
            self._purge_expired_locked()
            pending = self._select_pending_locked(source.lower(), request_id)
            if pending is None:
                return None
            self._pending.pop(pending.request.request_id, None)
        self._audit(pending.request, "confirmed")
        return pending.request

    def cancel(self, request_id: str, *, source: str = "local") -> bool:
        with self._lock:
            self._purge_expired_locked()
            pending = self._select_pending_locked(source.lower(), request_id)
            if pending is None:
                return False
            self._pending.pop(pending.request.request_id, None)
        self._audit(pending.request, "cancelled")
        return True

    def record_outcome(
        self,
        request: CapabilityRequest,
        *,
        success: bool,
        error: str = "",
    ) -> None:
        self._audit(request, "executed" if success else "failed", error=error)

    def record_event(
        self,
        capability: str,
        *,
        source: str,
        summary: str,
        event: str,
        error: str = "",
    ) -> None:
        """Record a bounded lifecycle event without creating an authorization."""
        now = time.time()
        request = CapabilityRequest(
            request_id=uuid.uuid4().hex[:12],
            capability=(capability or "system").strip().lower(),
            source=(source or "local").strip().lower(),
            summary=(summary or capability or "system event").strip()[:300],
            metadata=MappingProxyType({}),
            created_at=now,
            expires_at=now,
        )
        self._audit(request, (event or "event").strip().lower(), error=error)

    @property
    def pending_count(self) -> int:
        with self._lock:
            self._purge_expired_locked()
            return len(self._pending)

    def has_pending(self, source: str) -> bool:
        """Return whether *source* currently has an unexpired approval request."""
        with self._lock:
            self._purge_expired_locked()
            source = (source or "local").lower()
            return any(item.request.source == source
                       for item in self._pending.values())

    @staticmethod
    def is_confirmation_command(command: str) -> bool:
        """Return True only for the broker's narrow approve/cancel grammar."""
        return bool(_APPROVE_RE.match(command or "")
                    or _CANCEL_RE.match(command or ""))

    def effective_decision(
        self, capability: str, source: str
    ) -> CapabilityDecision:
        """Return the currently effective decision without creating a request."""
        capability = (capability or "unknown").strip().lower()
        source = (source or "local").strip().lower()
        with self._lock:
            return self._decision_for(capability, source)

    def query_audit(
        self,
        *,
        capability: str | None = None,
        source: str | None = None,
        event: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return a newest-first page of redacted audit records.

        Audit storage contains only the request summary and bounded error text;
        action metadata and payloads are intentionally never persisted.  An
        unavailable audit database is treated as an empty result so the GUI
        remains usable when durable memory is disabled.
        """
        where, params = self._audit_filters(capability, source, event)
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        sql = (
            "SELECT id, timestamp, request_id, capability, source, summary, "
            "event, error FROM capability_audit"
            f"{where} ORDER BY id DESC LIMIT ? OFFSET ?"
        )
        params.extend((limit, offset))
        if not self._db_path or self._db_path == ":memory:":
            return []
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                return [dict(row) for row in conn.execute(sql, params).fetchall()]
        except Exception as exc:
            log.warning("[Capabilities] Audit query failed: %s", exc)
            return []

    def count_audit(
        self,
        *,
        capability: str | None = None,
        source: str | None = None,
        event: str | None = None,
    ) -> int:
        """Return the number of redacted audit records matching the filters."""
        where, params = self._audit_filters(capability, source, event)
        if not self._db_path or self._db_path == ":memory:":
            return 0
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM capability_audit{where}", params
                ).fetchone()
                return int(row[0]) if row else 0
        except Exception as exc:
            log.warning("[Capabilities] Audit count failed: %s", exc)
            return 0

    def _decision_for(self, capability: str, source: str) -> CapabilityDecision:
        if not self._config.get("enabled", True):
            return CapabilityDecision.ALLOW
        configured = self._config.get("policies") or {}
        override = configured.get(capability)
        default_policy = DEFAULT_POLICIES.get(capability, {})
        if isinstance(override, str):
            raw = override
        else:
            # Partial configuration must not erase safer built-in decisions for
            # other sources or capabilities.
            policy = dict(default_policy)
            if isinstance(override, Mapping):
                policy.update(override)
            raw = policy.get(source, policy.get("default"))
        if raw is None:
            raw = self._config.get("default_decision", "confirm")
        try:
            return CapabilityDecision(str(raw).lower())
        except ValueError:
            log.warning("[Capabilities] Invalid decision %r; using confirm", raw)
            return CapabilityDecision.CONFIRM

    @staticmethod
    def _audit_filters(
        capability: str | None, source: str | None, event: str | None
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("capability", capability), ("source", source), ("event", event)
        ):
            if value:
                clauses.append(f"{column} = ?")
                params.append(str(value).strip().lower())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params

    def _select_pending_locked(
        self, source: str, request_prefix: str | None
    ) -> _PendingAction | None:
        candidates = [
            item for item in self._pending.values()
            if item.request.source == source
            and (not request_prefix
                 or item.request.request_id.startswith(request_prefix.lower()))
        ]
        if len(candidates) != 1:
            return None
        return candidates[0]

    def _purge_expired_locked(self) -> None:
        now = time.time()
        expired = [
            request_id for request_id, item in self._pending.items()
            if item.request.expires_at <= now
        ]
        for request_id in expired:
            item = self._pending.pop(request_id)
            self._audit(item.request, "expired")

    def _execute(self, request: CapabilityRequest, executor: Callable[[], Any]) -> dict:
        try:
            result = executor()
            outcome_success = (
                result.get("success", True) if isinstance(result, dict) else True
            )
            self.record_outcome(request, success=bool(outcome_success))
        except Exception as exc:
            self.record_outcome(request, success=False, error=str(exc))
            log.exception("[Capabilities] Approved action failed: %s", request.summary)
            return {
                "response": f"Approved action failed: {exc}",
                "talent": "capability_broker",
                "success": False,
                "actions_taken": [],
            }
        if isinstance(result, dict):
            result.setdefault("talent", "capability_broker")
            result.setdefault("success", True)
            result.setdefault("actions_taken", [])
            return result
        return {
            "response": str(result or "Done."),
            "talent": "capability_broker",
            "success": True,
            "actions_taken": [],
        }

    def _init_audit_table(self) -> None:
        if not self._db_path or self._db_path == ":memory:":
            return
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS capability_audit (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        request_id TEXT NOT NULL,
                        capability TEXT NOT NULL,
                        source TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        event TEXT NOT NULL,
                        error TEXT NOT NULL DEFAULT ''
                    )
                    """
                )
        except Exception as exc:
            log.warning("[Capabilities] Audit table unavailable: %s", exc)

    def _audit(self, request: CapabilityRequest, event: str, error: str = "") -> None:
        log.info(
            "[Capabilities] %s %s from %s: %s",
            event, request.capability, request.source, request.summary,
        )
        if not self._db_path or self._db_path == ":memory:":
            return
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO capability_audit
                        (timestamp, request_id, capability, source, summary, event, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        time.time(), request.request_id, request.capability,
                        request.source, request.summary[:300], event, error[:300],
                    ),
                )
        except Exception as exc:
            log.warning("[Capabilities] Audit write failed: %s", exc)
