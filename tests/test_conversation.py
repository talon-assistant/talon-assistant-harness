"""Tests for core.conversation.ConversationEngine.

Focuses on pure-logic fast paths that can be tested without real LLM/memory.
"""

import os
import re
from datetime import datetime
from collections import deque
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from core.conversation import ConversationEngine


# ── Fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    """Create a ConversationEngine with a fully mocked assistant."""
    mock_assistant = MagicMock()
    mock_assistant.memory.log_command = MagicMock()
    mock_assistant.memory.get_last_session_reflection.return_value = ""
    mock_assistant.memory.get_relevant_corrections.return_value = []
    mock_assistant.memory.get_document_context.return_value = ""
    mock_assistant.llm = MagicMock()
    mock_assistant.llm.generate.return_value = "mock response"
    mock_assistant.vision = MagicMock()
    mock_assistant.security = MagicMock()
    mock_assistant.security.check_output.return_value = (False, None)
    mock_assistant.security.check_semantic.return_value = (False, None)
    mock_assistant.talents = []
    mock_assistant._RULE_INDICATORS = [
        "whenever", "when i say", "if i say", "every time i say",
    ]
    mock_assistant._detect_and_store_rule.return_value = None
    mock_assistant._detect_preference = MagicMock()

    eng = ConversationEngine(mock_assistant)
    return eng


# ── Time/date fast-path ─────────────────────────────────────────────────────

def test_time_fastpath_returns_time(engine):
    result = engine.handle("what time is it", MagicMock())
    assert re.search(r"\d{1,2}:\d{2}\s*(AM|PM)", result)


def test_date_fastpath_returns_date(engine):
    result = engine.handle("what day is it today", MagicMock())
    today = datetime.now()
    assert today.strftime("%Y") in result or today.strftime("%B") in result


# ── System facts fast-path ──────────────────────────────────────────────────

def test_system_facts_username(engine):
    result = engine.handle("what is my username", MagicMock())
    expected = os.environ.get("USERNAME") or os.environ.get("USER", "unknown")
    assert expected in result


# ── Rule definition detection ───────────────────────────────────────────────

def test_rule_definition_detected(engine):
    """When a rule indicator is present and detection succeeds, return ack."""
    engine._a._detect_and_store_rule.return_value = {
        "trigger": "goodnight",
        "action": "turn off the lights",
    }
    result = engine.handle("whenever I say goodnight turn off the lights", MagicMock())
    assert "goodnight" in result
    assert "turn off" in result


# ── Promise pattern extraction ──────────────────────────────────────────────

def test_detect_promise_search():
    mock_assistant = MagicMock()
    mock_assistant.memory.get_last_session_reflection.return_value = ""
    eng = ConversationEngine(mock_assistant)

    result = eng.detect_promise("I'll search the web for pizza recipes")
    assert result is not None
    assert "pizza recipes" in result


def test_detect_promise_no_match():
    mock_assistant = MagicMock()
    mock_assistant.memory.get_last_session_reflection.return_value = ""
    eng = ConversationEngine(mock_assistant)

    result = eng.detect_promise("The weather today is sunny.")
    assert result is None


# ── Buffer capacity ─────────────────────────────────────────────────────────

def test_buffer_capacity_maxlen_16(engine):
    """Adding 17 turns keeps only the last 16."""
    for i in range(17):
        engine.conversation_buffer.append({"role": "user", "text": f"msg {i}"})
    assert len(engine.conversation_buffer) == 16
    assert engine.conversation_buffer[0]["text"] == "msg 1"


# ── Vision phrase detection ─────────────────────────────────────────────────

def test_vision_phrase_detected(engine):
    """Commands containing vision phrases should trigger screenshot capture."""
    ctx = MagicMock()
    ctx.get.side_effect = lambda k, d=None: {
        "rag_explicit": False,
        "attachments": [],
        "_planner_substep": False,
        "memory_context": "",
    }.get(k, d)

    engine._a.vision.capture_screenshot.return_value = "base64data"
    engine._a.llm.generate.return_value = "I can see a terminal."
    engine._documents_exist = False

    engine.handle("what's on my screen", ctx)
    engine._a.vision.capture_screenshot.assert_called()


# ── Session summary injection format ────────────────────────────────────────

def test_session_summary_injection(engine):
    """When a session summary exists, the prompt should include it."""
    engine._session_summary = "Discussed lights and weather"
    engine.conversation_buffer.append({"role": "user", "text": "hello"})
    engine.conversation_buffer.append({"role": "talon", "text": "hi there"})

    ctx = MagicMock()
    ctx.get.side_effect = lambda k, d=None: {
        "rag_explicit": False,
        "attachments": [],
        "_planner_substep": False,
        "memory_context": "",
    }.get(k, d)
    engine._documents_exist = False

    engine.handle("what was I saying", ctx)

    # Check that the generate call included the session summary
    call_args = engine._a.llm.generate.call_args
    prompt = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
    assert "Session so far:" in prompt
