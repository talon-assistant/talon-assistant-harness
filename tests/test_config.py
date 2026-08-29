"""Tests for portable runtime configuration helpers."""

import json
from pathlib import Path

from core.config import (
    deep_merge,
    format_contact_line,
    get_setting,
    get_user_profile,
    load_runtime_settings,
    resolve_configured_path,
)


def test_deep_merge_empty_dicts():
    assert deep_merge({}, {}) == {}


def test_deep_merge_flat_override_replaces_base():
    base = {"a": 1, "b": 2}
    override = {"b": 99}
    result = deep_merge(base, override)
    assert result == {"a": 1, "b": 99}


def test_deep_merge_nested_recursive():
    base = {"a": {"x": 1, "y": 2}}
    override = {"a": {"y": 99}}
    result = deep_merge(base, override)
    assert result == {"a": {"x": 1, "y": 99}}


def test_deep_merge_base_keys_preserved():
    base = {"a": 1, "b": 2, "c": 3}
    override = {"b": 20}
    result = deep_merge(base, override)
    assert result["a"] == 1
    assert result["c"] == 3


def test_deep_merge_override_adds_new_keys():
    base = {"a": 1}
    override = {"b": 2, "c": 3}
    result = deep_merge(base, override)
    assert result == {"a": 1, "b": 2, "c": 3}


def test_deep_merge_mixed_scalar_and_dict():
    """When override replaces a dict with a scalar, the scalar wins."""
    base = {"a": {"nested": True}}
    override = {"a": "flat_value"}
    result = deep_merge(base, override)
    assert result["a"] == "flat_value"


def test_deep_merge_deep_nesting_three_levels():
    base = {"a": {"b": {"c": 1, "d": 2}}}
    override = {"a": {"b": {"c": 99}}}
    result = deep_merge(base, override)
    assert result == {"a": {"b": {"c": 99, "d": 2}}}


def test_deep_merge_none_values_in_override():
    base = {"a": 1, "b": 2}
    override = {"a": None}
    result = deep_merge(base, override)
    assert result["a"] is None
    assert result["b"] == 2


def test_deep_merge_no_mutation_of_inputs():
    base = {"a": {"x": 1}}
    override = {"a": {"y": 2}}
    base_copy = {"a": {"x": 1}}
    override_copy = {"a": {"y": 2}}

    deep_merge(base, override)

    assert base == base_copy, "base dict was mutated"
    assert override == override_copy, "override dict was mutated"


def test_deep_merge_empty_override_returns_copy_of_base():
    base = {"a": 1, "b": {"c": 2}}
    result = deep_merge(base, {})
    assert result == base
    assert result is not base  # Must be a new dict


def test_runtime_settings_merge_example_and_untracked_user_config(tmp_path):
    (tmp_path / "settings.example.json").write_text(json.dumps({
        "resume": {"template_path": "~/Documents/default.docx", "cap": 2},
        "user_profile": {"display_name": "", "email": ""},
    }), encoding="utf-8")
    (tmp_path / "settings.json").write_text(json.dumps({
        "resume": {"template_path": "%TEST_PROFILE%/custom.docx"},
        "user_profile": {"display_name": "Local User"},
    }), encoding="utf-8")

    settings = load_runtime_settings(tmp_path)

    assert settings["resume"]["cap"] == 2
    assert settings["resume"]["template_path"] == "%TEST_PROFILE%/custom.docx"
    assert get_setting("user_profile.display_name", settings=settings) == "Local User"


def test_profile_and_paths_are_resolved_without_public_identity_defaults(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("TEST_PROFILE", str(tmp_path))
    settings = {
        "user_profile": {
            "display_name": "Local User", "email": "local@example.com",
            "phone": "", "location": "Remote",
        },
        "resume": {"template_path": "%TEST_PROFILE%/resume.docx"},
    }

    profile = get_user_profile(settings=settings)
    path = resolve_configured_path(
        "resume.template_path", "unused", settings=settings
    )

    assert profile["display_name"] == "Local User"
    assert format_contact_line(profile) == "local@example.com | Remote"
    assert path == tmp_path / "resume.docx"
