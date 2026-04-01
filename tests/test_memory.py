"""Tests for core.memory.MemorySystem.

Uses a real SQLite temp file but mocks ChromaDB and embedding models
so tests are fast and require no GPU/model downloads.
"""

import sqlite3
import time
import pytest
from unittest.mock import patch, MagicMock, PropertyMock


# ── Fixture: lightweight MemorySystem with mocked ChromaDB ──────────────────

@pytest.fixture
def memory_system(tmp_path):
    """Create a MemorySystem with real SQLite but mocked ChromaDB + embeddings."""
    db_file = str(tmp_path / "test.db")

    # Mock embedding functions to return dummy vectors
    dummy_embed = [[0.1] * 768]

    with patch("core.memory.chromadb") as mock_chroma_mod, \
         patch("core.memory._emb") as mock_emb, \
         patch("core.memory._reranker"), \
         patch("core.memory.DocumentRetriever"):

        mock_emb.embed_documents.return_value = dummy_embed
        mock_emb.embed_queries.return_value = dummy_embed

        # Mock ChromaDB client and collections
        mock_client = MagicMock()
        mock_chroma_mod.PersistentClient.return_value = mock_client

        mock_memory_col = MagicMock()
        mock_docs_col = MagicMock()
        mock_notes_col = MagicMock()
        mock_rules_col = MagicMock()
        mock_corrections_col = MagicMock()

        mock_rules_col.count.return_value = 0

        def get_or_create(name, **kwargs):
            return {
                "talon_memory": mock_memory_col,
                "talon_documents": mock_docs_col,
                "talon_notes": mock_notes_col,
                "talon_rules": mock_rules_col,
                "talon_corrections": mock_corrections_col,
            }[name]

        mock_client.get_or_create_collection.side_effect = get_or_create

        from core.memory import MemorySystem
        ms = MemorySystem(
            db_path=db_file,
            chroma_path=str(tmp_path / "chroma"),
            embedding_model="test-model",
            reranker_model="test-reranker",
        )

        # Attach mocks for test inspection
        ms._mock_memory_col = mock_memory_col
        ms._mock_notes_col = mock_notes_col
        ms._mock_rules_col = mock_rules_col
        ms._mock_corrections_col = mock_corrections_col
        ms._mock_emb = mock_emb

        yield ms


# ── init_database() ─────────────────────────────────────────────────────────

def test_init_creates_expected_tables(memory_system):
    conn = sqlite3.connect(memory_system.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()
    for expected in ("commands", "actions", "notes", "rules", "corrections",
                     "security_alerts", "goals"):
        assert expected in tables, f"Table '{expected}' missing"


# ── log_command() ────────────────────────────────────────────────────────────

def test_log_command_stores_and_retrieves(memory_system):
    cmd_id = memory_system.log_command("test command", success=True, response="ok")
    assert isinstance(cmd_id, int)

    conn = sqlite3.connect(memory_system.db_path)
    row = conn.execute(
        "SELECT command_text, response FROM commands WHERE id = ?", (cmd_id,)
    ).fetchone()
    conn.close()
    assert row[0] == "test command"
    assert row[1] == "ok"


# ── add_rule() ──────────────────────────────────────────────────────────────

def test_add_rule_stores_in_sqlite(memory_system):
    rule_id = memory_system.add_rule("goodnight", "turn off lights", "whenever I say goodnight")
    assert isinstance(rule_id, int)

    conn = sqlite3.connect(memory_system.db_path)
    row = conn.execute(
        "SELECT trigger_phrase, action_text FROM rules WHERE id = ?", (rule_id,)
    ).fetchone()
    conn.close()
    assert row[0] == "goodnight"
    assert row[1] == "turn off lights"


# ── match_rule() ─────────────────────────────────────────────────────────────

def test_match_rule_finds_matching(memory_system):
    """When ChromaDB returns a close match within threshold, match_rule returns it."""
    # Set up: add a rule first so _check_rules_exist returns True
    memory_system.add_rule("goodnight", "turn off lights")

    # Mock ChromaDB query result
    memory_system.rules_collection.query.return_value = {
        "documents": [["goodnight"]],
        "metadatas": [[{"rule_id": 1, "action_text": "turn off lights", "timestamp": "now"}]],
        "distances": [[0.1]],
    }

    result = memory_system.match_rule("goodnight")
    assert result is not None
    assert result["action_text"] == "turn off lights"


def test_match_rule_no_match_when_distance_exceeds_threshold(memory_system):
    memory_system.add_rule("goodnight", "turn off lights")

    memory_system.rules_collection.query.return_value = {
        "documents": [["goodnight"]],
        "metadatas": [[{"rule_id": 1, "action_text": "turn off lights", "timestamp": "now"}]],
        "distances": [[0.95]],  # Beyond max_distance 0.8
    }

    result = memory_system.match_rule("something unrelated")
    assert result is None


def test_match_rule_empty_collection_triggers_rebuild(memory_system):
    """When ChromaDB returns empty results but SQLite has rules, rebuild is attempted."""
    memory_system.add_rule("goodnight", "turn off lights")

    # First query returns empty, second (after rebuild) also empty
    memory_system.rules_collection.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    # _check_rules_exist will find the SQLite rule
    result = memory_system.match_rule("goodnight")
    # Should still be None since rebuild also returns empty in mock
    assert result is None


# ── delete_rule() ────────────────────────────────────────────────────────────

def test_delete_rule_removes_from_sqlite(memory_system):
    rule_id = memory_system.add_rule("test", "test action")
    assert memory_system.delete_rule(rule_id) is True

    conn = sqlite3.connect(memory_system.db_path)
    row = conn.execute("SELECT id FROM rules WHERE id = ?", (rule_id,)).fetchone()
    conn.close()
    assert row is None


def test_delete_rule_nonexistent_returns_false(memory_system):
    assert memory_system.delete_rule(99999) is False


# ── toggle_rule() ────────────────────────────────────────────────────────────

def test_toggle_rule_disables_and_enables(memory_system):
    rule_id = memory_system.add_rule("test", "action")

    assert memory_system.toggle_rule(rule_id, False) is True
    conn = sqlite3.connect(memory_system.db_path)
    row = conn.execute("SELECT enabled FROM rules WHERE id = ?", (rule_id,)).fetchone()
    conn.close()
    assert row[0] == 0

    assert memory_system.toggle_rule(rule_id, True) is True


def test_toggle_rule_nonexistent_returns_false(memory_system):
    assert memory_system.toggle_rule(99999, True) is False


# ── _check_rules_exist() ────────────────────────────────────────────────────

def test_check_rules_exist_true_when_rules(memory_system):
    memory_system.add_rule("test", "action")
    memory_system._rules_exist = None  # Reset cache
    assert memory_system._check_rules_exist() is True


def test_check_rules_exist_false_when_no_rules(memory_system):
    memory_system._rules_exist = None
    assert memory_system._check_rules_exist() is False


def test_check_rules_exist_cache_invalidation(memory_system):
    """Adding and deleting rules should invalidate the cache."""
    assert memory_system._check_rules_exist() is False
    rule_id = memory_system.add_rule("test", "action")
    # add_rule sets _rules_exist = None (cache invalidated)
    assert memory_system._rules_exist is None
    assert memory_system._check_rules_exist() is True

    memory_system.delete_rule(rule_id)
    assert memory_system._rules_exist is None  # Invalidated again


# ── add_note() ──────────────────────────────────────────────────────────────

def test_add_note_stores_and_returns_id(memory_system):
    note_id = memory_system.add_note("buy groceries", tags=["shopping"])
    assert isinstance(note_id, int)

    conn = sqlite3.connect(memory_system.db_path)
    row = conn.execute("SELECT content FROM notes WHERE id = ?", (note_id,)).fetchone()
    conn.close()
    assert row[0] == "buy groceries"


# ── search_notes() ──────────────────────────────────────────────────────────

def test_search_notes_returns_matching(memory_system):
    memory_system._mock_notes_col.query.return_value = {
        "documents": [["buy milk"]],
        "metadatas": [[{"note_id": 1, "timestamp": "2024-01-01", "tags": "[]"}]],
        "distances": [[0.2]],
    }
    results = memory_system.search_notes("groceries")
    assert len(results) == 1
    assert results[0]["content"] == "buy milk"


# ── store_correction() ──────────────────────────────────────────────────────

def test_store_correction_stores_pair(memory_system):
    memory_system.store_correction("turn on lamps", "turn on the living room lights")

    conn = sqlite3.connect(memory_system.db_path)
    row = conn.execute("SELECT prev_command, correction FROM corrections").fetchone()
    conn.close()
    assert row[0] == "turn on lamps"
    assert row[1] == "turn on the living room lights"


# ── get_relevant_corrections() ──────────────────────────────────────────────

def test_get_relevant_corrections_finds(memory_system):
    memory_system._mock_corrections_col.count.return_value = 1
    memory_system._mock_corrections_col.query.return_value = {
        "documents": [["turn on lamps"]],
        "metadatas": [[{"correction": "turn on living room lights"}]],
        "distances": [[0.3]],
    }
    results = memory_system.get_relevant_corrections("turn on the lamps")
    assert len(results) == 1
    assert results[0]["correction"] == "turn on living room lights"


# ── store_preference() ──────────────────────────────────────────────────────

def test_store_preference(memory_system):
    memory_system.store_preference("I prefer warm lighting", category="lighting")
    memory_system._mock_memory_col.add.assert_called()


# ── store_session_reflection() ──────────────────────────────────────────────

def test_store_session_reflection(memory_system):
    memory_system._mock_memory_col.get.return_value = {"ids": [], "metadatas": []}
    memory_system.store_session_reflection(
        summary="Discussed weather and lights",
        preferences=["warm tones"],
        failures=[],
    )
    memory_system._mock_memory_col.add.assert_called()


# ── WAL mode ────────────────────────────────────────────────────────────────

def test_wal_mode_enabled(memory_system):
    conn = sqlite3.connect(memory_system.db_path)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    assert mode == "wal"
