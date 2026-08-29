"""Capability declarations and coverage reporting for talents and MCP tools."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from core.capabilities import DEFAULT_POLICIES


READ_ONLY = "read_only"
BROKERED = "brokered"
UNDECLARED = "undeclared"
_ENFORCEMENT = frozenset({"host", "internal"})
_SANDBOX_KEYS = frozenset({
    "network", "subprocess", "llm", "filesystem_read", "filesystem_write",
})


# Built-ins use one central registry so the coverage report itself is the
# authoritative checklist. Third-party talents must declare the same structure
# as a ``capability_manifest`` class attribute.
BUILTIN_CAPABILITY_MANIFESTS: dict[str, dict[str, Any]] = {
    "clipboard_transform": {
        "access": BROKERED, "capabilities": ("clipboard_write",),
        "enforcement": "host",
    },
    "cowork_bridge": {
        "access": BROKERED, "capabilities": ("local_data_write",),
        "enforcement": "internal",
    },
    "desktop_control": {
        "access": BROKERED,
        "capabilities": ("desktop_control", "destructive_file_ops"),
        "enforcement": "internal",
    },
    "email": {
        "access": BROKERED,
        "capabilities": ("external_send", "external_account_write"),
        "enforcement": "internal",
    },
    "file_organizer": {
        "access": BROKERED, "capabilities": ("destructive_file_ops",),
        "enforcement": "internal",
    },
    "hermes_api": {
        "access": BROKERED, "capabilities": ("credential_write",),
        "enforcement": "host",
        "triggers": ("add", "create", "new", "rotate", "revoke", "remove", "delete"),
    },
    "hue_lights": {
        "access": BROKERED, "capabilities": ("device_control",),
        "enforcement": "host",
    },
    "itunes": {
        "access": BROKERED, "capabilities": ("device_control",),
        "enforcement": "host",
    },
    "job_search": {
        "access": BROKERED, "capabilities": ("local_data_write",),
        "enforcement": "host",
    },
    "job_tracker": {
        "access": BROKERED, "capabilities": ("local_data_write",),
        "enforcement": "host",
    },
    "lora_train": {
        "access": BROKERED, "capabilities": ("process_execution",),
        "enforcement": "host",
    },
    "notes": {
        "access": BROKERED, "capabilities": ("local_data_write",),
        "enforcement": "host",
        "triggers": (
            "save", "add", "write", "remember", "jot", "take a note",
            "make a note", "note:", "note to self", "scribble",
            "delete", "remove", "erase",
        ),
    },
    "reminder": {
        "access": BROKERED, "capabilities": ("local_data_write",),
        "enforcement": "host",
        "triggers": (
            "set", "remind", "timer", "alarm", "schedule",
            "cancel", "delete", "remove",
        ),
    },
    "rules": {
        "access": BROKERED, "capabilities": ("rule_write",),
        "enforcement": "internal",
    },
    "scheduler": {
        "access": BROKERED, "capabilities": ("local_data_write",),
        "enforcement": "host",
        "triggers": (
            "create", "schedule", "every", "run", "cancel", "delete", "remove",
        ),
    },
    "talent_builder": {
        "access": BROKERED, "capabilities": ("plugin_install",),
        "enforcement": "host",
        "triggers": (
            "install it", "install that", "save it", "save that",
            "looks good", "looks great", "perfect", "do it", "load it",
            "activate it", "yes", "yep", "yeah",
        ),
    },

    # These talents read data, report status, or orchestrate other talents. Any
    # downstream side effect is checked by the invoked talent's own manifest.
    "history": {"access": READ_ONLY},
    "news": {"access": READ_ONLY},
    "news_digest": {"access": READ_ONLY},
    "planner": {"access": READ_ONLY},
    "signal_remote": {"access": READ_ONLY},
    "task_assist": {"access": READ_ONLY},
    "threat_digest": {"access": READ_ONLY},
    "weather": {"access": READ_ONLY},
    "web_browser": {"access": READ_ONLY},
    "web_search": {"access": READ_ONLY},
}


@dataclass(frozen=True)
class CapabilityInventoryItem:
    owner: str
    owner_type: str
    access: str
    capabilities: tuple[str, ...]
    enforcement: str
    status: str
    detail: str = ""
    sandbox: str = "built_in"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _raw_manifest(talent) -> dict[str, Any] | None:
    raw = getattr(talent, "_source_manifest", None)
    if raw is None:
        raw = getattr(type(talent), "capability_manifest", None)
    module_name = str(getattr(type(talent), "__module__", ""))
    is_builtin_module = (
        module_name.startswith("talents.")
        and not module_name.startswith("talents.user.")
    )
    if raw is None and is_builtin_module:
        raw = BUILTIN_CAPABILITY_MANIFESTS.get(str(getattr(talent, "name", "")))
    return raw if isinstance(raw, dict) else None


def _sandbox_validation(raw: dict[str, Any], capabilities=()) -> tuple[list[str], str]:
    sandbox = raw.get("sandbox") or {}
    if not isinstance(sandbox, dict):
        return ["sandbox must be a literal dict"], "invalid"
    problems = []
    unknown = sorted(set(sandbox) - _SANDBOX_KEYS)
    if unknown:
        problems.append(f"unknown sandbox permissions: {', '.join(unknown)}")
    for key in ("network", "subprocess", "llm"):
        if key in sandbox and not isinstance(sandbox[key], bool):
            problems.append(f"sandbox.{key} must be true or false")
    for key in ("filesystem_read", "filesystem_write"):
        roots = sandbox.get(key, ())
        if not isinstance(roots, (list, tuple)):
            problems.append(f"sandbox.{key} must be a list of paths")
            continue
        if len(roots) > 16 or any(
            not isinstance(root, str) or not root.strip() or "\x00" in root
            for root in roots
        ):
            problems.append(f"sandbox.{key} contains invalid paths")
    caps = set(capabilities)
    if sandbox.get("subprocess") and "process_execution" not in caps:
        problems.append("sandbox subprocess access requires process_execution")
    write_caps = {
        "local_data_write", "destructive_file_ops", "plugin_install",
        "credential_write", "external_account_write",
    }
    if sandbox.get("filesystem_write") and not (caps & write_caps):
        problems.append("sandbox filesystem writes require a write capability")

    permissions = []
    if sandbox.get("network"):
        permissions.append("network")
    if sandbox.get("subprocess"):
        permissions.append("subprocess")
    if sandbox.get("filesystem_read"):
        permissions.append(f"read:{len(sandbox['filesystem_read'])}")
    if sandbox.get("filesystem_write"):
        permissions.append(f"write:{len(sandbox['filesystem_write'])}")
    if sandbox.get("llm", True):
        permissions.append("LLM proxy")
    return problems, ", ".join(permissions) if permissions else "default deny"


def inspect_talent(talent) -> CapabilityInventoryItem:
    """Validate and normalize a talent's capability declaration."""
    owner = str(getattr(talent, "name", "") or type(talent).__name__)
    class_manifest = getattr(type(talent), "capability_manifest", None)
    source_manifest = getattr(talent, "_source_manifest", None)
    third_party = isinstance(source_manifest, dict) or class_manifest is not None
    sandbox = "required" if third_party else "built_in"
    raw = _raw_manifest(talent)
    if raw is None:
        return CapabilityInventoryItem(
            owner, "talent", UNDECLARED, (), "none", UNDECLARED,
            "No capability_manifest declaration", sandbox,
        )

    access = str(raw.get("access", "")).strip().lower()
    if access == READ_ONLY:
        sandbox_problems, sandbox_summary = _sandbox_validation(raw)
        if sandbox_problems:
            return CapabilityInventoryItem(
                owner, "talent", READ_ONLY, (), "none", UNDECLARED,
                "; ".join(sandbox_problems), sandbox,
            )
        return CapabilityInventoryItem(
            owner, "talent", READ_ONLY, (), "none", READ_ONLY,
            (
                f"No privileged side effects declared; {sandbox_summary}"
                if third_party else "No privileged side effects declared"
            ),
            sandbox,
        )
    if access != BROKERED:
        return CapabilityInventoryItem(
            owner, "talent", UNDECLARED, (), "none", UNDECLARED,
            f"Invalid manifest access: {access or '(missing)'}", sandbox,
        )

    raw_caps = raw.get("capabilities") or ()
    if isinstance(raw_caps, str):
        raw_caps = (raw_caps,)
    capabilities = tuple(dict.fromkeys(
        str(value).strip().lower() for value in raw_caps if str(value).strip()
    ))
    enforcement = str(raw.get("enforcement", "")).strip().lower()
    unknown = tuple(cap for cap in capabilities if cap not in DEFAULT_POLICIES)
    untrusted_internal = third_party and enforcement == "internal"
    sandbox_problems, sandbox_summary = _sandbox_validation(raw, capabilities)
    if (not capabilities or enforcement not in _ENFORCEMENT
            or unknown or untrusted_internal or sandbox_problems):
        problems = []
        if not capabilities:
            problems.append("no capabilities listed")
        if enforcement not in _ENFORCEMENT:
            problems.append("enforcement must be host or internal")
        if unknown:
            problems.append(f"unknown capabilities: {', '.join(unknown)}")
        if untrusted_internal:
            problems.append("third-party brokered manifests must use host enforcement")
        problems.extend(sandbox_problems)
        return CapabilityInventoryItem(
            owner, "talent", BROKERED, capabilities, enforcement or "none",
            UNDECLARED, "; ".join(problems), sandbox,
        )
    return CapabilityInventoryItem(
        owner, "talent", BROKERED, capabilities, enforcement, "protected",
        (
            ("Host preflight" if enforcement == "host"
             else "Action-aware internal checks")
            + (f"; {sandbox_summary}" if third_party else "")
        ),
        sandbox,
    )


def validate_third_party_manifest(
    raw: dict[str, Any] | None, owner: str = "talent"
) -> CapabilityInventoryItem:
    """Validate an already-parsed third-party manifest with no code import."""
    class _Declaration:
        pass

    declaration = _Declaration()
    declaration.name = owner
    declaration._source_manifest = raw
    return inspect_talent(declaration)


def inspect_source_manifest(path: str | Path) -> CapabilityInventoryItem:
    """Validate a user talent's literal manifest without importing its code."""
    source_path = Path(path)
    owner = source_path.stem
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return CapabilityInventoryItem(
            owner, "talent", UNDECLARED, (), "none", UNDECLARED,
            f"Source cannot be validated: {exc}", "blocked",
        )

    talent_class = None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if any(
            (isinstance(base, ast.Name) and base.id == "BaseTalent")
            or (isinstance(base, ast.Attribute) and base.attr == "BaseTalent")
            for base in node.bases
        ):
            talent_class = node
            break
    if talent_class is None:
        return CapabilityInventoryItem(
            owner, "talent", UNDECLARED, (), "none", UNDECLARED,
            "No top-level BaseTalent subclass", "blocked",
        )

    raw = None
    for item in talent_class.body:
        if not isinstance(item, ast.Assign):
            continue
        for target in item.targets:
            if isinstance(target, ast.Name) and target.id == "name":
                try:
                    owner = str(ast.literal_eval(item.value))
                except (ValueError, TypeError):
                    pass
            if isinstance(target, ast.Name) and target.id == "capability_manifest":
                try:
                    raw = ast.literal_eval(item.value)
                except (ValueError, TypeError):
                    return CapabilityInventoryItem(
                        owner, "talent", UNDECLARED, (), "none", UNDECLARED,
                        "capability_manifest must be a literal dict", "blocked",
                    )

    # Use the same normalizer but suppress the built-in-name fallback: a user
    # file cannot inherit trust merely by naming itself after a built-in.
    if not isinstance(raw, dict):
        return CapabilityInventoryItem(
            owner, "talent", UNDECLARED, (), "none", UNDECLARED,
            "No capability_manifest declaration", "blocked",
        )
    result = validate_third_party_manifest(raw, owner)
    return replace(
        result,
        sandbox="required" if result.status != UNDECLARED else "blocked",
    )


def host_capability_for(talent, command: str) -> str | None:
    """Return the host-preflight capability for this invocation, if any."""
    item = inspect_talent(talent)
    if item.status != "protected" or item.enforcement != "host":
        return None
    raw = _raw_manifest(talent) or {}
    triggers = tuple(str(value).lower() for value in (raw.get("triggers") or ()))
    if triggers and not any(trigger in (command or "").lower() for trigger in triggers):
        return None
    return item.capabilities[0]


def build_inventory(talents: Iterable, mcp=None) -> list[CapabilityInventoryItem]:
    items = [inspect_talent(talent) for talent in talents]
    if mcp is not None and hasattr(mcp, "capability_inventory"):
        for record in mcp.capability_inventory():
            items.append(CapabilityInventoryItem(**record))
    return sorted(items, key=lambda item: (item.owner_type, item.owner.lower()))


def coverage_counts(items: Iterable[CapabilityInventoryItem]) -> dict[str, int]:
    counts = {"protected": 0, READ_ONLY: 0, UNDECLARED: 0}
    for item in items:
        status = (
            item.get("status", UNDECLARED)
            if isinstance(item, dict) else item.status
        )
        counts[status] = counts.get(status, 0) + 1
    return counts
