"""NotesTalent — save, search, list, and delete personal notes.

Notes are stored in both SQLite (for listing/deletion) and ChromaDB
(for semantic search). The LLM auto-generates tags from note content.

Examples:
    "save a note: meeting with Bob at 3pm"
    "note: pick up groceries after work"
    "find notes about the project"
    "search my notes for grocery list"
    "list my recent notes"
    "show my notes"
    "delete note about Bob"
"""

import re
import json
from talents.base import BaseTalent


class NotesTalent(BaseTalent):
    name = "notes"
    description = "Save, search, and manage personal notes and to-do items"
    keywords = [
        "note", "notes", "save a note", "write down", "remember this",
        "my notes", "find note", "search notes", "list notes",
        "delete note", "remove note",
        "todo", "to-do", "to do", "task", "task list", "todo list",
        "add a task", "my tasks",
    ]
    examples = [
        "save a note meeting with Bob at 3pm",
        "find notes about the project",
        "list my recent notes",
        "delete note about groceries",
        "add a task buy groceries",
        "what's on my todo list",
    ]
    priority = 45

    _NOTE_PHRASES = [
        "save a note", "save note", "note:", "add a note", "add note",
        "write down", "remember this", "jot down", "take a note",
        "make a note", "add a task", "add task", "new task", "new todo",
        "add to my list", "add to my todo",
    ]

    _SEARCH_PHRASES = [
        "find note", "find notes", "search note", "search notes",
        "search my notes", "find my notes", "notes about",
        "note about",
    ]

    _LIST_PHRASES = [
        "list notes", "list my notes", "show notes", "show my notes",
        "my notes", "recent notes", "all notes",
        "my tasks", "my todo", "my to-do", "todo list", "to-do list",
        "task list", "list tasks", "show tasks", "show my tasks",
        "what's on my list",
    ]

    _DELETE_PHRASES = [
        "delete note", "delete notes", "remove note", "remove notes",
        "erase note",
    ]

    _TAG_SYSTEM_PROMPT = (
        "You are a tagging assistant. Given a note, generate 2-5 short tags "
        "(single words or two-word phrases) that categorize the note. "
        "Return ONLY a JSON array of strings, nothing else.\n"
        'Example: ["meeting", "bob", "afternoon"]'
    )

    # ── Config schema ──────────────────────────────────────────────

    def get_config_schema(self) -> dict:
        return {
            "fields": [
                {"key": "max_search_results", "label": "Max Search Results",
                 "type": "int", "default": 5, "min": 1, "max": 20},
                {"key": "auto_tag", "label": "Auto-generate Tags",
                 "type": "bool", "default": True},
            ]
        }

    # ── Routing ────────────────────────────────────────────────────

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    # ── Execution ──────────────────────────────────────────────────

    def execute(self, command: str, context: dict) -> dict:
        cmd_lower = command.lower()

        # Determine action
        if any(phrase in cmd_lower for phrase in self._DELETE_PHRASES):
            return self._handle_delete(command, context)
        elif any(phrase in cmd_lower for phrase in self._NOTE_PHRASES):
            return self._handle_save(command, context)
        elif any(phrase in cmd_lower for phrase in self._SEARCH_PHRASES):
            return self._handle_search(command, context)
        elif any(phrase in cmd_lower for phrase in self._LIST_PHRASES):
            return self._handle_list(context)
        else:
            # Ambiguous — try search
            return self._handle_search(command, context)

    # ── Save a note ────────────────────────────────────────────────

    def _handle_save(self, command, context):
        """Extract note content from command and save it."""
        content = self._extract_note_content(command)
        if not content or len(content) < 3:
            return {
                "success": False,
                "response": "I couldn't figure out what to save. "
                            "Try: 'save a note: your note content here'",
                "actions_taken": [],
                "spoken": False,
            }

        # Auto-generate tags if enabled
        tags = []
        if self._config.get("auto_tag", True):
            tags = self._generate_tags(content, context)

        memory = context.get("memory")
        if not memory:
            return {
                "success": False,
                "response": "Memory system not available.",
                "actions_taken": [],
                "spoken": False,
            }

        note_id = memory.add_note(content, tags=tags)

        tag_str = ""
        if tags:
            tag_str = f"\nTags: {', '.join(tags)}"

        return {
            "success": True,
            "response": f"Note saved! (#{note_id}){tag_str}\n\n\"{content}\"",
            "actions_taken": [{"action": "note_save", "note_id": note_id}],
            "spoken": False,
        }

    # ── Search notes ───────────────────────────────────────────────

    def _handle_search(self, command, context):
        """Semantic search across saved notes."""
        query = self._extract_search_query(command)
        if not query or len(query) < 2:
            return {
                "success": False,
                "response": "What should I search for? Try: 'find notes about meetings'",
                "actions_taken": [],
                "spoken": False,
            }

        memory = context.get("memory")
        if not memory:
            return {
                "success": False,
                "response": "Memory system not available.",
                "actions_taken": [],
                "spoken": False,
            }

        max_results = self._config.get("max_search_results", 5)
        results = memory.search_notes(query, n_results=max_results)

        if not results:
            return {
                "success": True,
                "response": f"No notes found matching '{query}'.",
                "actions_taken": [{"action": "note_search", "query": query}],
                "spoken": False,
            }

        lines = [f"Found {len(results)} note(s) matching '{query}':\n"]
        for i, note in enumerate(results, 1):
            # Strip "Tags: ..." suffix from display
            display_content = note["content"]
            if "\nTags:" in display_content:
                display_content = display_content.split("\nTags:")[0]
            tags = note.get("tags", [])
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"{i}. (#{note['id']}) {display_content[:120]}{tag_str}")

        return {
            "success": True,
            "response": "\n".join(lines),
            "actions_taken": [{"action": "note_search", "query": query,
                               "count": len(results)}],
            "spoken": False,
        }

    # ── List recent notes ──────────────────────────────────────────

    def _handle_list(self, context):
        """List the most recent notes."""
        memory = context.get("memory")
        if not memory:
            return {
                "success": False,
                "response": "Memory system not available.",
                "actions_taken": [],
                "spoken": False,
            }

        max_results = self._config.get("max_search_results", 5)
        notes = memory.list_notes(limit=max_results)

        if not notes:
            return {
                "success": True,
                "response": "You don't have any saved notes yet.",
                "actions_taken": [{"action": "note_list"}],
                "spoken": False,
            }

        lines = [f"Your {len(notes)} most recent note(s):\n"]
        for note in notes:
            tags = note.get("tags", [])
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            ts = note["timestamp"][:16].replace("T", " ")  # compact timestamp
            lines.append(f"• (#{note['id']}) {note['content'][:100]}{tag_str}  — {ts}")

        return {
            "success": True,
            "response": "\n".join(lines),
            "actions_taken": [{"action": "note_list", "count": len(notes)}],
            "spoken": False,
        }

    # ── Delete a note ──────────────────────────────────────────────

    def _handle_delete(self, command, context):
        """Delete a note by ID or search query."""
        memory = context.get("memory")
        if not memory:
            return {
                "success": False,
                "response": "Memory system not available.",
                "actions_taken": [],
                "spoken": False,
            }

        # Check if user gave a note ID: "delete note #5" or "delete note 5"
        id_match = re.search(r'#?(\d+)', command)
        if id_match:
            note_id = int(id_match.group(1))
            deleted = memory.delete_note(note_id)
            if deleted:
                return {
                    "success": True,
                    "response": f"Note #{note_id} deleted.",
                    "actions_taken": [{"action": "note_delete", "note_id": note_id}],
                    "spoken": False,
                }
            else:
                return {
                    "success": False,
                    "response": f"No note found with ID #{note_id}.",
                    "actions_taken": [{"action": "note_delete_miss"}],
                    "spoken": False,
                }

        # No ID — search by content and delete first match
        query = self._extract_search_query(command)
        if query:
            results = memory.search_notes(query, n_results=1)
            if results:
                note_id = results[0]["id"]
                display = results[0]["content"][:80]
                if "\nTags:" in display:
                    display = display.split("\nTags:")[0]
                deleted = memory.delete_note(note_id)
                if deleted:
                    return {
                        "success": True,
                        "response": f"Deleted note #{note_id}: \"{display}\"",
                        "actions_taken": [{"action": "note_delete", "note_id": note_id}],
                        "spoken": False,
                    }

        return {
            "success": False,
            "response": "I couldn't find a matching note to delete. "
                        "Try: 'delete note #5' or 'delete note about meetings'",
            "actions_taken": [{"action": "note_delete_miss"}],
            "spoken": False,
        }

    # ── Content extraction helpers ─────────────────────────────────

    @staticmethod
    def _extract_note_content(command):
        """Pull the actual note content from the command string."""
        cmd = command

        # "note: <content>" or "note - <content>"
        for sep in [":", "-", "—"]:
            if sep in cmd:
                parts = cmd.split(sep, 1)
                # Only use the part after the separator if the part before
                # looks like a command prefix
                before = parts[0].lower().strip()
                if any(p in before for p in [
                    "note", "save", "write", "remember", "jot", "add"
                ]):
                    content = parts[1].strip()
                    if content:
                        return content

        # Strip command prefixes and use the rest
        lower = cmd.lower()
        for prefix in [
            "save a note about", "save a note", "save note about", "save note",
            "add a note about", "add a note", "add note about", "add note",
            "take a note about", "take a note", "take note",
            "write down that", "write down", "jot down",
            "remember this:", "remember this that", "remember this",
            "make a note about", "make a note that", "make a note",
            "note that", "note about",
            "add a task about", "add a task", "add task", "new task",
            "add to my list", "add to my todo",
        ]:
            if lower.startswith(prefix):
                content = cmd[len(prefix):].strip()
                if content:
                    return content

        # Last resort: return everything
        return cmd.strip()

    @staticmethod
    def _extract_search_query(command):
        """Pull the search query from the command string."""
        cmd = command.lower()
        for prefix in [
            "find notes about", "find notes for", "find note about",
            "search notes for", "search notes about", "search my notes for",
            "search my notes about", "find my notes about",
            "notes about", "note about",
            "find notes", "search notes", "find note", "search note",
            "delete note about", "delete notes about",
            "remove note about", "remove notes about",
            "erase note about",
        ]:
            if prefix in cmd:
                query = cmd.split(prefix, 1)[1].strip()
                query = query.rstrip("?.!")
                if query:
                    return query

        # Fallback: strip all command words
        for noise in [
            "find", "search", "my", "notes", "note", "about",
            "for", "the", "delete", "remove", "erase",
        ]:
            cmd = cmd.replace(noise, "")
        return cmd.strip()

    # ── Tag generation ─────────────────────────────────────────────

    def _generate_tags(self, content, context):
        """Use the LLM to auto-generate tags for a note."""
        llm = context.get("llm")
        if not llm:
            return []

        try:
            response = llm.generate(
                f"Generate tags for this note:\n\n{content}",
                system_prompt=self._TAG_SYSTEM_PROMPT,
                temperature=0.2,
            )

            # Extract JSON array
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                tags = json.loads(match.group())
                # Ensure all tags are strings, lowercase, reasonable length
                return [
                    str(t).lower().strip()[:30]
                    for t in tags
                    if isinstance(t, str) and len(t.strip()) > 0
                ][:5]
        except (json.JSONDecodeError, Exception) as e:
            print(f"   [Notes] Tag generation error: {e}")

        return []
