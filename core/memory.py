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

    def get_document_context(self, query: str, explicit: bool = False) -> str:
        """Retrieve document chunks for RAG injection into the conversation path.

        Args:
            query:    The user's command text, used as the embedding query.
            explicit: If True, the user explicitly asked for document search —
                      use a loose distance cap (1.5) and return up to 5 chunks.
                      If False (ambient), only inject if distance <= 0.55 and
                      return at most 2 chunks.

        Returns:
            Formatted string ready for injection, or "" if nothing qualifies.
        """
        if len(query.strip()) < 4:
            return ""

        n_results = 5 if explicit else 2
        max_distance = 1.5 if explicit else 0.55

        try:
            results = self.docs_collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            if not results["documents"] or not results["documents"][0]:
                return ""

            chunks = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                if dist <= max_distance:
                    filename = meta.get("filename", "unknown file")
                    chunks.append((filename, doc, dist))

            if not chunks:
                mode = "explicit" if explicit else "ambient"
                print(f"   [RAG] No chunks passed threshold "
                      f"(mode={mode}, threshold={max_distance:.2f})")
                return ""

            print(f"   [RAG] Injecting {len(chunks)} chunk(s) "
                  f"(explicit={explicit}, best_dist={chunks[0][2]:.3f})")

            lines = [
                "The following document excerpts may be relevant — "
                "use them if helpful, ignore if not:"
            ]
            for filename, text, dist in chunks:
                truncated = text[:600] + "..." if len(text) > 600 else text
                lines.append(f"- From {filename}: {truncated}")

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
