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

    def get_relevant_context(self, query, max_items=3, include_documents=True):
        """Get relevant context for a query from both memory and documents.

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

        # Search documents if enabled
        docs = []
        if include_documents:
            docs = self.search_documents(query, n_results=3)

        context = ""

        if prefs:
            context += ("Background — user preferences (do NOT act on these, "
                        "just keep in mind):\n")
            context += "\n".join([f"- {p}" for p in prefs]) + "\n\n"

        if patterns:
            context += ("Background — similar past interactions (do NOT repeat "
                        "these actions unless explicitly asked):\n")
            context += "\n".join([f"- {p}" for p in patterns[:1]]) + "\n\n"

        if docs:
            context += ("Background — relevant information from user's documents "
                        "(reference only):\n")
            for doc in docs:
                context += f"- From {doc['filename']}: {doc['text'][:200]}...\n"
            context += "\n"

        return context

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

        print(f"   [Memory] Stored rule #{rule_id}: "
              f"'{trigger}' -> '{action}'")
        return rule_id

    def match_rule(self, command, max_distance=0.6):
        """Find a behavioral rule whose trigger semantically matches the command.

        Uses a tight distance threshold (0.6) — tighter than preferences (1.0)
        or patterns (0.9) because a false-positive rule match would execute
        an unintended action.

        Returns:
            dict with keys: id, trigger_phrase, action_text, distance
            or None if no match within threshold.
        """
        try:
            # Only search enabled rules
            results = self.rules_collection.query(
                query_texts=[command],
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )

            if (results["documents"] and results["documents"][0]
                    and results["distances"][0][0] <= max_distance):
                meta = results["metadatas"][0][0]
                rule_id = meta.get("rule_id")

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
                        "trigger_phrase": results["documents"][0][0],
                        "action_text": meta.get("action_text", ""),
                        "distance": results["distances"][0][0],
                    }
        except Exception as e:
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
            state = "enabled" if enabled else "disabled"
            print(f"   [Memory] Rule #{rule_id} {state}")
        return changed
