"""Shared fixtures for Talon Assistant test suite."""

import sqlite3
import pytest
from unittest.mock import MagicMock, PropertyMock

from talents.base import BaseTalent, TalentResult


# ── Mock LLM client ─────────────────────────────────────────────────────────

class MockLLMClient:
    """Configurable mock LLM client for testing.

    Set ``response`` to control what generate() returns.
    Set ``side_effect`` to make generate() raise an exception.
    """

    def __init__(self, response="mock response"):
        self.response = response
        self.side_effect = None
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if self.side_effect:
            raise self.side_effect
        return self.response

    def test_connection(self):
        return True


@pytest.fixture
def mock_llm():
    """Return a MockLLMClient with a default response."""
    return MockLLMClient()


# ── Minimal config dict ─────────────────────────────────────────────────────

@pytest.fixture
def mock_config():
    """Minimal settings dict matching the structure expected by core modules."""
    return {
        "llm": {
            "endpoint": "http://localhost:5001/api/v1/generate",
            "max_length": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "rep_pen": 1.1,
            "timeout": 30,
            "stop_sequences": ["<|im_end|>"],
            "prompt_template": {
                "user_prefix": "<|im_start|>user\n",
                "user_suffix": "<|im_end|>\n",
                "assistant_prefix": "<|im_start|>assistant\n",
                "vision_prefix": "[img-1]\n",
            },
            "api_format": "koboldcpp",
        },
        "memory": {
            "db_path": ":memory:",
            "chroma_path": "data/chroma_db",
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "reranker_model": "BAAI/bge-reranker-base",
        },
        "security": {
            "input_filter": {"enabled": True, "action": "block", "patterns": []},
            "output_scan": {"enabled": True, "action": "log"},
            "rate_limit": {"enabled": True, "action": "block", "requests_per_minute": 30},
            "confirmation_gates": {"enabled": True, "gates": []},
            "audit_log": {"enabled": False},
            "semantic_classifier": {"enabled": False},
        },
    }


# ── Temp SQLite database ────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite database file and return its path string."""
    db_file = tmp_path / "test_memory.db"
    return str(db_file)


# ── Sample talent for testing ────────────────────────────────────────────────

class SampleTalent(BaseTalent):
    """Minimal concrete talent subclass for testing BaseTalent logic."""

    name = "sample"
    description = "A sample talent for testing"
    keywords = ["sample", "test thing", "demo"]
    examples = ["run a sample", "test thing please"]
    priority = 50

    def execute(self, command, context):
        return TalentResult(success=True, response="sample executed")


@pytest.fixture
def sample_talent():
    """Return a fresh SampleTalent instance."""
    return SampleTalent()
