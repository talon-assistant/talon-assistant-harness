"""Tests for talents.base module."""

import os
import pytest
from unittest.mock import MagicMock

from talents.base import BaseTalent, TalentResult
from core.llm_client import LLMError


class SampleTalent(BaseTalent):
    """Minimal concrete talent subclass for testing BaseTalent logic."""

    name = "sample"
    description = "A sample talent for testing"
    keywords = ["sample", "test thing", "demo"]
    examples = ["run a sample", "test thing please"]
    priority = 50

    def execute(self, command, context):
        return TalentResult(success=True, response="sample executed")


# ── keyword_match() ─────────────────────────────────────────────────────────

def test_keyword_match_single_word():
    t = SampleTalent()
    assert t.keyword_match("run the sample now") is True


def test_keyword_match_multi_word():
    t = SampleTalent()
    assert t.keyword_match("please test thing for me") is True


def test_keyword_match_no_match():
    t = SampleTalent()
    assert t.keyword_match("turn on the lights") is False


def test_keyword_match_case_insensitive():
    t = SampleTalent()
    assert t.keyword_match("RUN THE SAMPLE NOW") is True


def test_keyword_match_word_boundary_no_partial():
    """Single-word keyword 'sample' should not match inside 'resampled'."""
    t = SampleTalent()
    assert t.keyword_match("resampled data") is False


# ── can_handle() ─────────────────────────────────────────────────────────────

def test_can_handle_delegates_to_keyword_match():
    t = SampleTalent()
    assert t.can_handle("run the sample") is True
    assert t.can_handle("unrelated command") is False


# ── check_requirements() ────────────────────────────────────────────────────

def test_check_requirements_all_met():
    t = SampleTalent()
    t.required_config = []
    t.required_env = []
    problems = t.check_requirements({"a": "b"})
    assert problems == []


def test_check_requirements_missing_config():
    t = SampleTalent()
    t.required_config = ["hue.bridge_ip"]
    problems = t.check_requirements({"hue": {}})
    assert len(problems) == 1
    assert "bridge_ip" in problems[0]


def test_check_requirements_missing_env(monkeypatch):
    t = SampleTalent()
    t.required_env = ["TALON_TEST_MISSING_VAR"]
    monkeypatch.delenv("TALON_TEST_MISSING_VAR", raising=False)
    problems = t.check_requirements({})
    assert len(problems) == 1
    assert "TALON_TEST_MISSING_VAR" in problems[0]


# ── _extract_arg() ──────────────────────────────────────────────────────────

def test_extract_arg_successful():
    t = SampleTalent()
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "  London  "
    result = t._extract_arg(mock_llm, "weather in London", "location")
    assert result == "London"


def test_extract_arg_llm_returns_none_value():
    t = SampleTalent()
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "NONE"
    result = t._extract_arg(mock_llm, "weather", "location")
    assert result is None


def test_extract_arg_llm_error_returns_none():
    t = SampleTalent()
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = LLMError("timeout")
    result = t._extract_arg(mock_llm, "weather", "location")
    assert result is None


def test_extract_arg_with_options():
    t = SampleTalent()
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "red"
    result = t._extract_arg(
        mock_llm, "set color to red", "colour",
        options=["red", "green", "blue"],
    )
    assert result == "red"
    call_prompt = mock_llm.generate.call_args[0][0]
    assert "red" in call_prompt and "green" in call_prompt


# ── get_config_schema() ─────────────────────────────────────────────────────

def test_get_config_schema_returns_empty_dict():
    t = SampleTalent()
    assert t.get_config_schema() == {}


# ── subprocess_isolated ─────────────────────────────────────────────────────

def test_subprocess_isolated_default_is_false():
    t = SampleTalent()
    assert t.subprocess_isolated is False
