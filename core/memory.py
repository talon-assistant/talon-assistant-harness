import sqlite3
import json
import time
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer


class MemorySystem:
    """Handles structured memory (SQLite), semantic memory (ChromaDB), and document RAG"""

    def __init__(self, db_path="data/talon_memory.db", chroma_path="data/chroma_db",
                 embedding_model="all-MiniLM-L6-v2"):
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

        # Collection for correction learning (previous command → what user actually wanted)
        self.corrections_collection = self.chroma_client.get_or_create_collection(
            name="talon_corrections",
            metadata={"hnsw:space": "cosine",
                      "description": "Correction memory: maps bad commands to correct intent"}
        )

        # Sentence transformer for embeddings
        print("   [Memory] Loading embedding model...")
        self.embedder = SentenceTransformer(embedding_model)

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
            documents=[preference_text],
            metadatas=[{"type": "preference", "category": category, "timestamp": datetime.now().isoformat()}],
            ids=[doc_id]
        )
        print(f"   [Memory] Stored preference: {preference_text}")

    def store_successful_pattern(self, command, actions, context=""):
        """Store a successful command pattern in ChromaDB"""
        doc_text = f"Command: {command}\nActions: {json.dumps(actions)}\nContext: {context}"
        doc_id = f"pattern_{int(time.time() * 1000)}"
        self.memory_collection.add(
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
                query_texts=[command],
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
                query_texts=[command],
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
            query_texts=[query],
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

        context = ""

        if prefs:
            context += ("Background — user preferences (do NOT act on these, "
                        "just keep in mind):\n")
            context += "\n".join([f"- {p}" for p in prefs]) + "\n\n"

        if patterns:
            context += ("Background — similar past interactions (do NOT repeat "
                        "these actions unless explicitly asked):\n")
            context += "\n".join([f"- {p}" for p in patterns[:1]]) + "\n\n"

        return context

    def get_document_context(self, query: str, explicit: bool = False,
                             alt_queries: list | None = None) -> str:
        """Retrieve document chunks for RAG injection into the conversation path.

        Args:
            query:       Primary embedding query (expanded from user command).
            explicit:    If True, user explicitly asked for document search —
                         use loose distance cap (1.8) and return up to 8 chunks.
                         If False (ambient), only inject if distance <= 0.55 and
                         return at most 2 chunks.
            alt_queries: Optional list of alternate queries (synonyms / related
                         terms) whose results are unioned with the primary query
                         and deduplicated. Explicit mode only.

        Returns:
            Formatted string ready for injection, or "" if nothing qualifies.
        """
        if len(query.strip()) < 4:
            return ""

        n_results = 8 if explicit else 2
        max_distance = 1.8 if explicit else 0.55

        def _run_query(q: str) -> list[tuple[str, str, float, int | None]]:
            """Run one ChromaDB query and return (filename, text, dist, page_num) tuples."""
            try:
                results = self.docs_collection.query(
                    query_texts=[q],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"],
                )
                if not results["documents"] or not results["documents"][0]:
                    return []
                hits = []
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    if dist <= max_distance:
                        hits.append((meta.get("filename", "unknown file"), doc, dist,
                                     meta.get("page_number")))
                return hits
            except Exception:
                return []

        try:
            # Primary query
            all_chunks = _run_query(query)

            # Alt queries (explicit mode only) — union and deduplicate
            if explicit and alt_queries:
                seen: set[str] = {text[:100] for _, text, _, _pg in all_chunks}
                for aq in alt_queries:
                    if len(aq.strip()) < 4:
                        continue
                    for filename, text, dist, page_num in _run_query(aq):
                        key = text[:100]
                        if key not in seen:
                            seen.add(key)
                            all_chunks.append((filename, text, dist, page_num))

            # Text-match fallback (explicit mode only): use ChromaDB $contains
            # to pull chunks with exact keyword matches, bypassing the semantic
            # distance cutoff.  Sparse stat-block chunks embed poorly but are
            # textually exact — this guarantees they surface even when the
            # embedding distance would otherwise exclude them.
            if explicit:
                all_terms = [query] + (alt_queries or [])
                text_kws = sorted(
                    {w for t in all_terms for w in t.split() if len(w) > 3},
                    key=len, reverse=True,
                )[:8]
                seen_txt: set[str] = {t[:100] for _, t, _, _pg in all_chunks}
                for kw in text_kws:
                    # Try original case and title-case: "mana" → also "Mana"
                    # so that "Manabolt" is found even when query is lowercase.
                    for variant in {kw, kw.title()}:
                        try:
                            hits = self.docs_collection.get(
                                where_document={"$contains": variant},
                                limit=6,
                                include=["documents", "metadatas"],
                            )
                            for doc, meta in zip(
                                hits.get("documents", []),
                                hits.get("metadatas", []),
                            ):
                                key = doc[:100]
                                if key not in seen_txt:
                                    seen_txt.add(key)
                                    all_chunks.append(
                                        (meta.get("filename", "unknown file"), doc, 1.0,
                                         meta.get("page_number"))
                                    )
                        except Exception:
                            pass

            if not all_chunks:
                mode = "explicit" if explicit else "ambient"
                print(f"   [RAG] No chunks passed threshold "
                      f"(mode={mode}, threshold={max_distance:.2f})")
                return ""

            # Keyword re-ranking (explicit mode only): boost chunks that
            # literally contain search terms so structured stat blocks aren't
            # buried under prose that merely mentions the topic in passing.
            all_terms = [query] + (alt_queries or [])
            keywords = set(w.lower() for t in all_terms for w in t.split() if len(w) > 3)

            def _keyword_score(chunk_text: str) -> int:
                lower = chunk_text.lower()
                return sum(1 for kw in keywords if kw in lower)

            if explicit:
                # Re-sort: primary key = descending keyword hits,
                # secondary key = ascending semantic distance.
                all_chunks.sort(key=lambda x: (-_keyword_score(x[1]), x[2]))
            else:
                all_chunks.sort(key=lambda x: x[2])

            # Hard cap: $contains fallback can produce many hits (8 keywords ×
            # 2 case variants × 6 results = up to 96 extra chunks).  Capping
            # after ranking keeps the highest-signal chunks and prevents the
            # context window from being flooded with low-relevance noise.
            MAX_INJECT = 12 if explicit else 8
            all_chunks = all_chunks[:MAX_INJECT]

            print(f"   [RAG] Injecting {len(all_chunks)} unique chunk(s) "
                  f"(explicit={explicit}, best_dist={all_chunks[0][2]:.3f})")
            for _fn, _txt, _d, _pg in all_chunks:
                pg_label = f" p{_pg + 1}" if _pg is not None else ""
                print(f"      {_d:.3f} kw={_keyword_score(_txt)}  {_fn}{pg_label}  |  {_txt[:60].replace(chr(10),' ')!r}")

            if explicit:
                lines = [
                    "The following excerpts are from the user's own documents. "
                    "Prioritize this content — use it directly and cite the source filename. "
                    "Use ONLY what is explicitly stated in these excerpts. "
                    "For any specific stat, number, rule, or structured value — if it is not "
                    "present in the excerpts, say it was not found rather than substituting "
                    "from general knowledge. General knowledge may contradict the document."
                ]
            else:
                lines = [
                    "The following document excerpts may be relevant — "
                    "use them if helpful, ignore if not:"
                ]
            for filename, text, dist, page_num in all_chunks:
                truncated = text[:800] + "..." if len(text) > 800 else text
                source = f"{filename} (page {page_num + 1})" if page_num is not None else filename
                lines.append(f"- From {source}: {truncated}")

            return "\n".join(lines) + "\n"

        except Exception as e:
            print(f"   [RAG] Document context error: {e}")
            return ""

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
                query_texts=[query],
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
                query_texts=[command],
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )

            if not results["documents"] or not results["documents"][0]:
                print(f"   [Rules] No rules in database to match against")
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
                        query_texts=[command],
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
