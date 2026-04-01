import os
import re
import sqlite3
import json
import time
from datetime import datetime

# Disable ChromaDB's PostHog telemetry — its background thread does SSL I/O
# that causes heap corruption / GIL crashes on Python 3.10 + PyQt6.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

import chromadb
from core import embeddings as _emb
from core import reranker as _reranker
from core.document_retriever import DocumentRetriever


class MemorySystem:
    """Handles structured memory (SQLite), semantic memory (ChromaDB), and document RAG"""

    def __init__(self, db_path="data/talon_memory.db", chroma_path="data/chroma_db",
                 embedding_model="BAAI/bge-base-en-v1.5",
                 reranker_model="BAAI/bge-reranker-base"):
        print("   [Memory] Initializing memory systems...")

        # SQLite for structured data
        self.db_path = db_path
        self.init_database()

        # ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        # Collection for conversation memory
        self.memory_collection = self.chroma_client.get_or_create_collection(
            name="talon_memory",
            metadata={"description": "Talon conversation and preference memory"}
        )

        # Collection for documents
        self.docs_collection = self.chroma_client.get_or_create_collection(
            name="talon_documents",
            metadata={"description": "User documents for RAG retrieval"}
        )

        # Collection for user notes (NotesTalent)
        self.notes_collection = self.chroma_client.get_or_create_collection(
            name="talon_notes",
            metadata={"description": "User notes for semantic search"}
        )

        # Collection for behavioral rules (trigger → action mappings)
        self.rules_collection = self.chroma_client.get_or_create_collection(
            name="talon_rules",
            metadata={"description": "Behavioral rules: trigger phrase semantic matching"}
        )

        # Sync check: if SQLite has rules but ChromaDB is empty (e.g. after
        # a crash that wiped ChromaDB storage), rebuild the index now.
        if self.rules_collection.count() == 0:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM rules WHERE enabled = 1")
            sql_count = cursor.fetchone()[0]
            conn.close()
            if sql_count > 0:
                self._rebuild_rules_collection()

        # Collection for correction learning (previous command → what user actually wanted)
        self.corrections_collection = self.chroma_client.get_or_create_collection(
            name="talon_corrections",
            metadata={"hnsw:space": "cosine",
                      "description": "Correction memory: maps bad commands to correct intent"}
        )

        # Embedding + reranker model names (models load lazily on first use)
        self._embed_model = embedding_model
        self._reranker_model = reranker_model

        # Pre-warm the embedding model so first-query latency is predictable
        print("   [Memory] Loading embedding model...")
        _emb.embed_documents(["warmup"], embedding_model)

        # Document retriever — RAG pipeline over docs_collection
        self._retriever = DocumentRetriever(
            self.docs_collection, self._embed_model, self._reranker_model)

        # Cache: None = unknown, True/False = cached result
        # Invalidated whenever a rule is added, deleted, or toggled.
        self._rules_exist: bool | None = None

        print("   [Memory] Memory systems ready!")

    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Commands table
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS commands
                       (
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           timestamp TEXT,
                           command_text TEXT,
                           success INTEGER,
                           response TEXT
                       )
                       """)

        # Actions table
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS actions
                       (
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           command_id INTEGER,
                           timestamp TEXT,
                           action_json TEXT,
                           result TEXT,
                           success INTEGER,
                           FOREIGN KEY (command_id) REFERENCES commands (id)
                       )
                       """)

        # Notes table (for NotesTalent)
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS notes
                       (
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           timestamp TEXT,
                           content TEXT,
                           tags TEXT,
                           chroma_id TEXT
                       )
                       """)

        # Rules table (behavioral rules: trigger → action)
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS rules
                       (
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           timestamp TEXT,
                           trigger_phrase TEXT,
                           action_text TEXT,
                           original_command TEXT,
                           enabled INTEGER DEFAULT 1,
                           chroma_id TEXT
                       )
                       """)

        # Corrections table (correction learning)
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS corrections
                       (
                           id           INTEGER PRIMARY KEY AUTOINCREMENT,
                           timestamp    TEXT,
                           prev_command TEXT,
                           correction   TEXT
                       )
                       """)

        # Security audit log
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS security_alerts
                       (
                           id           INTEGER PRIMARY KEY AUTOINCREMENT,
                           timestamp    REAL,
                           control      TEXT,
                           pattern_id   TEXT,
                           label        TEXT,
                           content      TEXT,
                           action_taken TEXT
                       )
                       """)

        # Self-initiated goals (personality / consciousness roadmap)
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS goals
                       (
                           id          INTEGER PRIMARY KEY AUTOINCREMENT,
                           created_at  TEXT,
                           updated_at  TEXT,
                           text        TEXT,
                           status      TEXT DEFAULT 'active',
                           progress    TEXT DEFAULT '',
                           source      TEXT DEFAULT 'reflection'
                       )
                       """)

        conn.commit()
        conn.close()

    def log_command(self, command_text, success=True, response=""):
        """Log a command to SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
                       INSERT INTO commands (timestamp, command_text, success, response)
                       VALUES (?, ?, ?, ?)
                       """, (datetime.now().isoformat(), command_text, 1 if success else 0, response))
        command_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return command_id

    def log_action(self, command_id, action_json, result, success=True):
        """Log an action to SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
                       INSERT INTO actions (command_id, timestamp, action_json, result, success)
                       VALUES (?, ?, ?, ?, ?)
                       """,
                       (command_id, datetime.now().isoformat(), json.dumps(action_json), result,
                        1 if success else 0))
        conn.commit()
        conn.close()

    def get_last_successful_action(self):
        """Get the most recent successful action"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
                       SELECT c.command_text, a.action_json, a.result
                       FROM actions a
                                JOIN commands c ON a.command_id = c.id
                       WHERE a.success = 1
                       ORDER BY a.timestamp DESC LIMIT 1
                       """)
        result = cursor.fetchone()
        conn.close()
        return result

    def store_preference(self, preference_text, category="general"):
        """Store a user preference in ChromaDB"""
        doc_id = f"pref_{int(time.time() * 1000)}"
        self.memory_collection.add(
            embeddings=_emb.embed_documents([preference_text], self._embed_model),
            documents=[preference_text],
            metadatas=[{"type": "preference", "category": category, "timestamp": datetime.now().isoformat()}],
            ids=[doc_id]
        )
        print(f"   [Memory] Stored preference: {preference_text}")

    def store_soft_hint(self, hint_text: str) -> None:
        """Store a soft behavioural hint derived from session reflection shortcuts.

        Soft hints are semantically retrieved at query time and injected as
        advisory context — they influence the LLM's reasoning without mandating
        a specific action (unlike hard rules in talon_rules).

        Capped at 30 hints; the oldest is pruned when the cap is reached.
        """
        hint_text = hint_text.strip()
        if not hint_text:
            return

        # Enforce 30-hint cap
        try:
            existing = self.memory_collection.get(
                where={"type": "soft_hint"},
                include=["metadatas"],
            )
            ids = existing.get("ids", [])
            metas = existing.get("metadatas", [])
            if len(ids) >= 30:
                paired = sorted(zip(ids, metas),
                                key=lambda x: x[1].get("timestamp", ""))
                self.memory_collection.delete(ids=[paired[0][0]])
        except Exception:
            pass

        doc_id = f"hint_{int(time.time() * 1000)}"
        self.memory_collection.add(
            embeddings=_emb.embed_documents([hint_text], self._embed_model),
            documents=[hint_text],
            metadatas=[{"type": "soft_hint",
                        "timestamp": datetime.now().isoformat()}],
            ids=[doc_id],
        )
        print(f"   [Memory] Stored soft hint: {hint_text[:80]}")

    def store_successful_pattern(self, command, actions, context=""):
        """Store a successful command pattern in ChromaDB"""
        doc_text = f"Command: {command}\nActions: {json.dumps(actions)}\nContext: {context}"
        doc_id = f"pattern_{int(time.time() * 1000)}"
        self.memory_collection.add(
            embeddings=_emb.embed_documents([doc_text], self._embed_model),
            documents=[doc_text],
            metadatas=[{"type": "pattern", "command": command, "timestamp": datetime.now().isoformat()}],
            ids=[doc_id]
        )

    # ── Correction learning ───────────────────────────────────────────

    def store_correction(self, prev_command: str, correction: str):
        """Store a user correction: what Talon tried vs what the user actually wanted.

        prev_command is embedded in ChromaDB so future similar commands can be
        matched semantically and the correction injected into the LLM prompt.
        """
        ts = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO corrections (timestamp, prev_command, correction) VALUES (?,?,?)",
            (ts, prev_command, correction)
        )
        conn.commit()
        conn.close()

        doc_id = f"correction_{int(time.time() * 1000)}"
        self.corrections_collection.add(
            embeddings=_emb.embed_documents([prev_command], self._embed_model),
            documents=[prev_command],   # embed the previous command for semantic retrieval
            metadatas=[{"correction": correction, "timestamp": ts}],
            ids=[doc_id]
        )
        print(f"   [Memory] Stored correction: '{prev_command}' → '{correction}'")

    def get_relevant_corrections(self, command: str, max_results: int = 2) -> list[dict]:
        """Return past corrections whose prev_command is semantically close to command.

        Returns a list of dicts: {"prev_command": str, "correction": str}
        Only returns results within cosine distance 0.55 (tight — avoids false positives).
        """
        try:
            count = self.corrections_collection.count()
            if count == 0:
                return []
            n = min(max_results, count)
            results = self.corrections_collection.query(
                query_embeddings=_emb.embed_queries([command], self._embed_model),
                n_results=n
            )
            out = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                if dist <= 0.55:
                    out.append({"prev_command": doc, "correction": meta["correction"]})
            return out
        except Exception as e:
            print(f"   [Memory] get_relevant_corrections error: {e}")
            return []

    def count_similar_corrections(self, command: str, threshold: float = 0.60) -> int:
        """Count stored corrections whose prev_command is semantically close to command.

        Uses a slightly looser threshold than get_relevant_corrections (0.60 vs 0.55)
        to catch near-duplicates for frequency counting purposes.
        """
        try:
            total = self.corrections_collection.count()
            if total == 0:
                return 0
            n = min(total, 20)
            results = self.corrections_collection.query(
                query_embeddings=_emb.embed_queries([command], self._embed_model),
                n_results=n,
                include=["distances"],
            )
            distances = results["distances"][0] if results["distances"] else []
            return sum(1 for d in distances if d <= threshold)
        except Exception as e:
            print(f"   [Memory] count_similar_corrections error: {e}")
            return 0

    # ── Session reflection ────────────────────────────────────────────

    def get_session_commands(self, since_timestamp: str, limit: int = 40) -> list[dict]:
        """Return commands logged at or after since_timestamp (ISO format).

        Returns:
            list[dict] with keys: command, success, response
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT command_text, success, response FROM commands "
            "WHERE timestamp >= ? ORDER BY id LIMIT ?",
            (since_timestamp, limit),
        ).fetchall()
        conn.close()
        return [{"command": r[0], "success": bool(r[1]), "response": r[2]} for r in rows]

    def search_commands(
        self,
        keyword: str | None = None,
        start_ts: str | None = None,
        end_ts: str | None = None,
        success_filter: bool | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search command history with optional keyword, date range, and success filters.

        Returns:
            list[dict] with keys: timestamp, command, success, response
        """
        conditions, params = [], []
        if start_ts:
            conditions.append("timestamp >= ?")
            params.append(start_ts)
        if end_ts:
            conditions.append("timestamp <= ?")
            params.append(end_ts)
        if keyword:
            conditions.append("(command_text LIKE ? OR response LIKE ?)")
            params += [f"%{keyword}%", f"%{keyword}%"]
        if success_filter is not None:
            conditions.append("success = ?")
            params.append(1 if success_filter else 0)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            f"SELECT timestamp, command_text, success, response FROM commands "
            f"{where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()
        conn.close()
        return [
            {"timestamp": r[0], "command": r[1], "success": bool(r[2]), "response": r[3]}
            for r in rows
        ]

    def store_session_reflection(
        self,
        summary: str,
        preferences: list[str],
        failures: list[str],
        shortcuts: list[str] | None = None,
    ) -> None:
        """Store a session reflection in talon_memory ChromaDB.

        Keeps only the last 5 reflections — the oldest is deleted before
        adding a new one whenever the count would exceed 5.
        """
        # Build document text
        parts = [f"Session Summary: {summary}"]
        if preferences:
            parts.append("Preferences: " + "; ".join(preferences))
        if failures:
            parts.append("Failures: " + "; ".join(failures))
        if shortcuts:
            parts.append("Shortcuts: " + "; ".join(shortcuts))
        doc_text = "\n".join(parts)

        # Enforce 5-reflection cap
        try:
            existing = self.memory_collection.get(
                where={"type": "session_reflection"},
                include=["metadatas"],
            )
            ids = existing.get("ids", [])
            metas = existing.get("metadatas", [])
            if len(ids) >= 5:
                # Sort ascending by timestamp — delete the oldest
                paired = sorted(zip(ids, metas),
                                key=lambda x: x[1].get("timestamp", ""))
                oldest_id = paired[0][0]
                self.memory_collection.delete(ids=[oldest_id])
                print(f"   [Memory] Pruned oldest session reflection ({oldest_id})")
        except Exception as e:
            print(f"   [Memory] Could not prune old reflections: {e}")

        doc_id = f"reflect_{int(time.time() * 1000)}"
        timestamp = datetime.now().isoformat()
        self.memory_collection.add(
            embeddings=_emb.embed_documents([doc_text], self._embed_model),
            documents=[doc_text],
            metadatas=[{"type": "session_reflection", "timestamp": timestamp}],
            ids=[doc_id],
        )
        print(f"   [Memory] Stored session reflection ({doc_id})")

    def get_last_session_reflection(self) -> str:
        """Return the most recent session reflection text, or '' if none exists."""
        try:
            results = self.memory_collection.get(
                where={"type": "session_reflection"},
                include=["documents", "metadatas"],
            )
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
            if not docs:
                return ""
            # Sort descending by timestamp — most recent first
            paired = sorted(
                zip(docs, metas),
                key=lambda x: x[1].get("timestamp", ""),
                reverse=True,
            )
            return paired[0][0]
        except Exception as e:
            print(f"   [Memory] Could not retrieve session reflection: {e}")
            return ""

    def store_free_thought(self, text: str, thought_num: int = 1,
                           *, valence: int | None = None) -> None:
        """Store a single free-thought reflection in talon_memory.

        Keeps a rolling cap of 60 free thoughts — oldest pruned when exceeded.
        They embed naturally alongside other memories and surface in ambient RAG
        when semantically relevant to a future user query.

        If *valence* is provided (1–10), it is stored in metadata so the
        reflection loop can prefer higher-rated thoughts when seeding future
        reflection context.
        """
        try:
            existing = self.memory_collection.get(
                where={"type": "free_thought"},
                include=["metadatas"],
            )
            ids   = existing.get("ids", [])
            metas = existing.get("metadatas", [])
            if len(ids) >= 60:
                # When pruning, prefer dropping low-valence thoughts first.
                paired = sorted(
                    zip(ids, metas),
                    key=lambda x: (
                        x[1].get("valence", 5),   # low valence first
                        x[1].get("timestamp", ""), # then oldest
                    ),
                )
                oldest_id = paired[0][0]
                self.memory_collection.delete(ids=[oldest_id])
        except Exception as e:
            print(f"   [Memory] Could not prune free thoughts: {e}")

        ts     = datetime.now().isoformat()
        doc_id = f"freethought_{int(time.time() * 1000)}_{thought_num}"
        meta   = {"type": "free_thought", "timestamp": ts,
                  "thought_num": thought_num}
        if valence is not None:
            meta["valence"] = valence
        self.memory_collection.add(
            embeddings=_emb.embed_documents([text], self._embed_model),
            documents=[text],
            metadatas=[meta],
            ids=[doc_id],
        )

    def get_free_thoughts(self) -> list[dict]:
        """Return all stored free thoughts, newest first.

        Each entry: {"id": str, "timestamp": str, "thought_num": int,
                     "text": str, "valence": int | None}
        """
        try:
            results = self.memory_collection.get(
                where={"type": "free_thought"},
                include=["documents", "metadatas"],
            )
            docs   = results.get("documents", [])
            metas  = results.get("metadatas", [])
            ids    = results.get("ids", [])
            paired = sorted(
                zip(ids, metas, docs),
                key=lambda x: x[1].get("timestamp", ""),
                reverse=True,
            )
            return [
                {"id": i, "timestamp": m.get("timestamp", ""),
                 "thought_num": m.get("thought_num", 1), "text": d,
                 "valence": m.get("valence")}
                for i, m, d in paired
            ]
        except Exception as e:
            print(f"   [Memory] Could not retrieve free thoughts: {e}")
            return []

    def delete_free_thought(self, doc_id: str) -> None:
        """Delete a single free thought by ChromaDB id."""
        try:
            self.memory_collection.delete(ids=[doc_id])
        except Exception as e:
            print(f"   [Memory] Could not delete free thought {doc_id}: {e}")

    def clear_free_thoughts(self) -> int:
        """Delete all free thoughts. Returns count deleted."""
        try:
            existing = self.memory_collection.get(
                where={"type": "free_thought"},
                include=["metadatas"],
            )
            ids = existing.get("ids", [])
            if ids:
                self.memory_collection.delete(ids=ids)
            return len(ids)
        except Exception as e:
            print(f"   [Memory] Could not clear free thoughts: {e}")
            return 0

    # ── Self-initiated goals ─────────────────────────────────────────────────

    def store_goal(self, text: str, source: str = "reflection") -> int:
        """Create a new active goal. Returns the row id."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO goals (created_at, updated_at, text, status, source) "
            "VALUES (?, ?, ?, 'active', ?)",
            (now, now, text, source),
        )
        gid = cursor.lastrowid
        conn.commit()
        conn.close()
        print(f"   [Goals] Created goal #{gid}: {text[:80]}")
        return gid

    def get_active_goals(self) -> list[dict]:
        """Return all active goals, newest first."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, created_at, updated_at, text, progress "
            "FROM goals WHERE status = 'active' ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {"id": r[0], "created_at": r[1], "updated_at": r[2],
             "text": r[3], "progress": r[4]}
            for r in rows
        ]

    def update_goal_progress(self, goal_id: int, progress: str) -> None:
        """Append a progress note to an existing goal."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute(
            "UPDATE goals SET progress = progress || ? || char(10), "
            "updated_at = ? WHERE id = ?",
            (f"[{now[:16]}] {progress}", now, goal_id),
        )
        conn.commit()
        conn.close()

    def complete_goal(self, goal_id: int, status: str = "completed") -> None:
        """Mark a goal as completed or abandoned."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute(
            "UPDATE goals SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, goal_id),
        )
        conn.commit()
        conn.close()
        print(f"   [Goals] Goal #{goal_id} → {status}")

    def get_all_goals(self) -> list[dict]:
        """Return all goals (active, completed, abandoned), newest first."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, created_at, updated_at, text, status, progress, source "
            "FROM goals ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {"id": r[0], "created_at": r[1], "updated_at": r[2],
             "text": r[3], "status": r[4], "progress": r[5], "source": r[6]}
            for r in rows
        ]

    def delete_goal(self, goal_id: int) -> None:
        """Permanently delete a goal by id."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM goals WHERE id = ?", (goal_id,))
        conn.commit()
        conn.close()
        print(f"   [Goals] Deleted goal #{goal_id}")

    # ── Anticipation — user behaviour patterns ────────────────────────────────

    def get_command_patterns(self, lookback_days: int = 14) -> list[dict]:
        """Analyse recent command history for temporal patterns.

        Returns a list of pattern dicts:
          {"hour": int, "day_name": str, "topic": str, "count": int}
        Grouped by (hour, day-of-week, first-two-words-of-command).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT timestamp, command_text FROM commands "
            "WHERE success = 1 AND timestamp >= datetime('now', ?)",
            (f"-{lookback_days} days",),
        )
        rows = cursor.fetchall()
        conn.close()

        from collections import Counter
        buckets: Counter = Counter()
        for ts_str, cmd in rows:
            try:
                dt = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                continue
            # Coarse topic: first two meaningful words
            words = cmd.lower().split()[:3]
            topic = " ".join(w for w in words if len(w) > 2)[:40]
            if not topic:
                continue
            buckets[(dt.hour, dt.strftime("%a"), topic)] += 1

        # Only surface patterns that repeated ≥ 2 times
        return [
            {"hour": h, "day_name": d, "topic": t, "count": c}
            for (h, d, t), c in buckets.most_common(20)
            if c >= 2
        ]

    def search_memory(self, query, n_results=3, memory_type=None,
                       max_distance=1.2):
        """Search conversation memory semantically.

        Args:
            max_distance: Maximum cosine distance (0 = identical, 2 = opposite).
                          Results further than this are considered irrelevant
                          and filtered out.  Default 1.2 keeps only reasonably
                          related matches.
        """
        where_filter = {"type": memory_type} if memory_type else None

        results = self.memory_collection.query(
            query_embeddings=_emb.embed_queries([query], self._embed_model),
            n_results=n_results,
            where=where_filter,
            include=["documents", "distances"]
        )

        if results['documents'] and len(results['documents'][0]) > 0:
            # Filter by distance threshold
            filtered = []
            for doc, dist in zip(results['documents'][0],
                                 results['distances'][0]):
                if dist <= max_distance:
                    filtered.append(doc)
            return filtered
        return []

    def search_documents(self, query, n_results=5):
        """Search user documents for relevant information"""
        try:
            results = self.docs_collection.query(
                query_texts=[query],
                n_results=n_results
            )

            if results['documents'] and len(results['documents'][0]) > 0:
                docs = []
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    docs.append({
                        'text': doc,
                        'filename': metadata.get('filename', 'unknown'),
                        'type': metadata.get('file_type', 'unknown')
                    })
                return docs
            return []
        except Exception as e:
            print(f"   [Memory] Document search error: {e}")
            return []

    def get_relevant_context(self, query, max_items=3, include_documents=False):
        """Get relevant context for a query from memory (preferences and patterns only).

        Document RAG is now handled by get_document_context() on the
        conversation path with distance-threshold gating.

        Uses a distance threshold so vague queries (like 'hi') don't pull in
        unrelated memories that cause the LLM to hallucinate past actions.
        """
        # Skip memory lookup for very short / generic queries
        if len(query.strip()) < 4:
            return ""

        # Search preferences (tight threshold — must be clearly relevant)
        prefs = self.search_memory(query, n_results=2,
                                   memory_type="preference", max_distance=1.0)

        # Search past patterns (tight threshold)
        patterns = self.search_memory(query, n_results=2,
                                      memory_type="pattern", max_distance=0.9)

        # Search soft hints (slightly looser — advisory nudges from reflection)
        hints = self.search_memory(query, n_results=2,
                                   memory_type="soft_hint", max_distance=1.1)

        context = ""

        if prefs:
            context += ("Background — user preferences (do NOT act on these, "
                        "just keep in mind):\n")
            context += "\n".join([f"- {p}" for p in prefs]) + "\n\n"

        if patterns:
            context += ("Background — similar past interactions (do NOT repeat "
                        "these actions unless explicitly asked):\n")
            context += "\n".join([f"- {p}" for p in patterns[:1]]) + "\n\n"

        if hints:
            context += ("Past experience — consider these approaches but use "
                        "your judgment; they are suggestions, not rules:\n")
            context += "\n".join([f"- {h}" for h in hints]) + "\n\n"

        return context

    # ── RAG helpers ───────────────────────────────────────────────

    def get_document_context(self, query: str, **kwargs) -> str:
        """Delegate to DocumentRetriever for RAG pipeline."""
        return self._retriever.get_document_context(query, **kwargs)

    # ── Notes CRUD (used by NotesTalent) ──────────────────────────────

    def add_note(self, content, tags=None):
        """Save a note to SQLite + ChromaDB for semantic search.

        Returns:
            int: the note's SQLite row ID.
        """
        tags = tags or []
        chroma_id = f"note_{int(time.time() * 1000)}"
        timestamp = datetime.now().isoformat()

        # SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO notes (timestamp, content, tags, chroma_id) VALUES (?, ?, ?, ?)",
            (timestamp, content, json.dumps(tags), chroma_id)
        )
        note_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # ChromaDB (semantic index)
        tag_str = ", ".join(tags) if tags else ""
        doc_text = f"{content}\nTags: {tag_str}" if tag_str else content
        self.notes_collection.add(
            embeddings=_emb.embed_documents([doc_text], self._embed_model),
            documents=[doc_text],
            metadatas=[{
                "note_id": note_id,
                "timestamp": timestamp,
                "tags": json.dumps(tags),
            }],
            ids=[chroma_id],
        )

        print(f"   [Memory] Saved note #{note_id}: {content[:60]}...")
        return note_id

    def search_notes(self, query, n_results=5):
        """Semantic search across notes.

        Returns:
            list[dict] with keys: id, content, tags, timestamp, distance
        """
        try:
            results = self.notes_collection.query(
                query_embeddings=_emb.embed_queries([query], self._embed_model),
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            notes = []
            if results["documents"] and results["documents"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    notes.append({
                        "id": meta.get("note_id", "?"),
                        "content": doc,
                        "tags": json.loads(meta.get("tags", "[]")),
                        "timestamp": meta.get("timestamp", ""),
                        "distance": dist,
                    })
            return notes
        except Exception as e:
            print(f"   [Memory] Note search error: {e}")
            return []

    def list_notes(self, limit=10):
        """Return the most recent notes from SQLite.

        Returns:
            list[dict] with keys: id, content, tags, timestamp
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, timestamp, content, tags FROM notes ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": r[0],
                "timestamp": r[1],
                "content": r[2],
                "tags": json.loads(r[3]) if r[3] else [],
            }
            for r in rows
        ]

    def delete_note(self, note_id):
        """Delete a note by SQLite ID (removes from both SQLite and ChromaDB).

        Returns:
            bool: True if deleted, False if not found.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get the chroma_id for this note
        cursor.execute("SELECT chroma_id FROM notes WHERE id = ?", (note_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False

        chroma_id = row[0]

        # Delete from SQLite
        cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        conn.commit()
        conn.close()

        # Delete from ChromaDB
        try:
            self.notes_collection.delete(ids=[chroma_id])
        except Exception as e:
            print(f"   [Memory] ChromaDB delete warning: {e}")

        print(f"   [Memory] Deleted note #{note_id}")
        return True

    # ── Rules CRUD (behavioral rules: trigger → action) ─────────────

    def _check_rules_exist(self) -> bool:
        """Return True if at least one enabled rule is stored in SQLite.

        Result is cached after the first query and invalidated by any
        mutation (add_rule, delete_rule, toggle_rule).  This avoids
        hitting ChromaDB on every command when no rules are defined.
        """
        if self._rules_exist is not None:
            return self._rules_exist

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM rules WHERE enabled = 1")
        count = cursor.fetchone()[0]
        conn.close()

        self._rules_exist = count > 0
        return self._rules_exist

    def _rebuild_rules_collection(self):
        """Rebuild the ChromaDB rules collection from SQLite ground truth.

        Called when the HNSW index is missing from disk (e.g. after chroma_db
        was wiped). SQLite is always the authoritative store for rules data.
        """
        print("   [Rules] Rebuilding ChromaDB index from SQLite...")

        # Drop and recreate the collection
        try:
            self.chroma_client.delete_collection(name="talon_rules")
        except Exception:
            pass

        self.rules_collection = self.chroma_client.get_or_create_collection(
            name="talon_rules",
            metadata={"description": "Behavioral rules: trigger phrase semantic matching"}
        )

        # Re-add all enabled rules from SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, trigger_phrase, action_text, timestamp, chroma_id "
            "FROM rules WHERE enabled = 1"
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            print("   [Rules] No enabled rules to restore")
            return

        for rule_id, trigger, action, timestamp, chroma_id in rows:
            try:
                self.rules_collection.add(
                    embeddings=_emb.embed_documents([trigger], self._embed_model),
                    documents=[trigger],
                    metadatas=[{
                        "rule_id": rule_id,
                        "action_text": action,
                        "timestamp": timestamp,
                    }],
                    ids=[chroma_id],
                )
            except Exception as e:
                print(f"   [Rules] Could not restore rule #{rule_id}: {e}")

        print(f"   [Rules] Restored {len(rows)} rule(s) from SQLite")

    def add_rule(self, trigger, action, original_command=""):
        """Store a behavioral rule in SQLite + ChromaDB.

        The ChromaDB document is the trigger phrase so incoming commands
        are semantically matched against triggers, not actions.

        Returns:
            int: the rule's SQLite row ID.
        """
        chroma_id = f"rule_{int(time.time() * 1000)}"
        timestamp = datetime.now().isoformat()

        # SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO rules (timestamp, trigger_phrase, action_text, "
            "original_command, enabled, chroma_id) VALUES (?, ?, ?, ?, 1, ?)",
            (timestamp, trigger, action, original_command, chroma_id)
        )
        rule_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # ChromaDB — document is the trigger phrase for semantic matching;
        # action_text stored in metadata for retrieval alongside the match.
        self.rules_collection.add(
            embeddings=_emb.embed_documents([trigger], self._embed_model),
            documents=[trigger],
            metadatas=[{
                "rule_id": rule_id,
                "action_text": action,
                "timestamp": timestamp,
            }],
            ids=[chroma_id],
        )

        self._rules_exist = None  # Invalidate cache
        print(f"   [Memory] Stored rule #{rule_id}: "
              f"'{trigger}' -> '{action}'")
        return rule_id

    def match_rule(self, command, max_distance=0.8):
        """Find a behavioral rule whose trigger semantically matches the command.

        Uses a distance threshold that adapts to command length:
          - Short commands (< 5 words): max_distance 0.8 (more lenient, since
            the user likely said just the trigger phrase)
          - Longer commands: max_distance 0.6 (tighter, to avoid false matches
            when the trigger is buried in a longer sentence)

        Returns:
            dict with keys: id, trigger_phrase, action_text, distance
            or None if no match within threshold.
        """
        # Fast-path: skip ChromaDB entirely when no enabled rules exist
        if not self._check_rules_exist():
            print("   [Rules] Skipping — no rules stored")
            return None

        # Adaptive threshold: short phrases are more likely to be triggers
        word_count = len(command.strip().split())
        threshold = max_distance if word_count < 5 else 0.6

        try:
            results = self.rules_collection.query(
                query_embeddings=_emb.embed_queries([command], self._embed_model),
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )

            if not results["documents"] or not results["documents"][0]:
                # ChromaDB empty but SQLite may have rules — try rebuild
                if self._check_rules_exist():
                    print("   [Rules] ChromaDB empty but SQLite has rules — rebuilding...")
                    self._rebuild_rules_collection()
                    results = self.rules_collection.query(
                        query_embeddings=_emb.embed_queries([command], self._embed_model),
                        n_results=1,
                        include=["documents", "metadatas", "distances"],
                    )
                    if not results["documents"] or not results["documents"][0]:
                        print("   [Rules] Still empty after rebuild")
                        return None
                else:
                    print("   [Rules] No rules in database to match against")
                    return None

            closest_trigger = results["documents"][0][0]
            distance = results["distances"][0][0]
            meta = results["metadatas"][0][0]

            print(f"   [Rules] Closest match: '{closest_trigger}' "
                  f"(distance={distance:.3f}, threshold={threshold:.1f})")

            if distance <= threshold:
                rule_id = int(meta.get("rule_id", 0))

                # Verify the rule is still enabled in SQLite
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT enabled FROM rules WHERE id = ?", (rule_id,))
                row = cursor.fetchone()
                conn.close()

                if row and row[0] == 1:
                    return {
                        "id": rule_id,
                        "trigger_phrase": closest_trigger,
                        "action_text": meta.get("action_text", ""),
                        "distance": distance,
                    }
                else:
                    print(f"   [Rules] Rule #{rule_id} matched but is disabled")
            else:
                print(f"   [Rules] No match — distance {distance:.3f} "
                      f"> threshold {threshold:.1f}")
        except Exception as e:
            err = str(e)
            if "Nothing found on disk" in err or "hnsw segment" in err.lower():
                # HNSW index missing — rebuild from SQLite and retry once
                self._rebuild_rules_collection()
                try:
                    results = self.rules_collection.query(
                        query_embeddings=_emb.embed_queries([command], self._embed_model),
                        n_results=1,
                        include=["documents", "metadatas", "distances"],
                    )
                    if results["documents"] and results["documents"][0]:
                        closest_trigger = results["documents"][0][0]
                        distance = results["distances"][0][0]
                        meta = results["metadatas"][0][0]
                        print(f"   [Rules] Closest match (after rebuild): "
                              f"'{closest_trigger}' (distance={distance:.3f})")
                        if distance <= threshold:
                            return {
                                "id": int(meta.get("rule_id", 0)),
                                "trigger_phrase": closest_trigger,
                                "action_text": meta.get("action_text", ""),
                                "distance": distance,
                            }
                except Exception as retry_e:
                    print(f"   [Rules] Retry after rebuild failed: {retry_e}")
            else:
                print(f"   [Memory] Rule match error: {e}")
        return None

    def list_rules(self, limit=20):
        """Return all rules from SQLite.

        Returns:
            list[dict] with keys: id, trigger_phrase, action_text, enabled, timestamp
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, timestamp, trigger_phrase, action_text, enabled "
            "FROM rules ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": r[0],
                "timestamp": r[1],
                "trigger_phrase": r[2],
                "action_text": r[3],
                "enabled": bool(r[4]),
            }
            for r in rows
        ]

    def delete_rule(self, rule_id):
        """Delete a rule by SQLite ID (removes from both SQLite and ChromaDB).

        Returns:
            bool: True if deleted, False if not found.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT chroma_id FROM rules WHERE id = ?", (rule_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False

        chroma_id = row[0]

        cursor.execute("DELETE FROM rules WHERE id = ?", (rule_id,))
        conn.commit()
        conn.close()

        try:
            self.rules_collection.delete(ids=[chroma_id])
        except Exception as e:
            print(f"   [Memory] ChromaDB rule delete warning: {e}")

        self._rules_exist = None  # Invalidate cache
        print(f"   [Memory] Deleted rule #{rule_id}")
        return True

    def toggle_rule(self, rule_id, enabled):
        """Enable or disable a rule in SQLite.

        Returns:
            bool: True if toggled, False if rule not found.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE rules SET enabled = ? WHERE id = ?",
            (1 if enabled else 0, rule_id)
        )
        changed = cursor.rowcount > 0
        conn.commit()
        conn.close()

        if changed:
            self._rules_exist = None  # Invalidate cache
            state = "enabled" if enabled else "disabled"
            print(f"   [Memory] Rule #{rule_id} {state}")
        return changed
