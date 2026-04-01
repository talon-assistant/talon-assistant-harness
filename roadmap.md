# Talon Assistant — Roadmap

## Recently Completed

### Architecture & Code Quality
- Structured logging system — 46 files converted from print() to Python logging module, rotating file handler at data/logs/talon.log
- ConversationEngine extracted from assistant.py (633 lines -> core/conversation.py)
- DocumentRetriever extracted from memory.py (417 lines -> core/document_retriever.py)
- TalentContext + TalentResult dataclasses with backward-compat dict access
- LLMError exceptions replace error-string returns across 36 callers
- Security utilities (wrap_external, injection patterns) moved to core/security.py
- deep_merge moved to core/config.py (breaks circular import)
- 97-test pytest suite covering config, security, LLM client, talents, memory, conversation, routing
- requirements.lock for reproducible builds, version-pinned requirements.txt
- GPL-3.0 LICENSE file
- Instance locking (data/.talon.lock) prevents concurrent runs / DB corruption

### Talent System
- Talent builder complete rewrite — full-file generation, code review before install, iterative refinement loop
- Subprocess isolation for talents with C-extension libraries (subprocess_isolated = True)
- required_packages attribute for declaring pip dependencies
- pynput replaced with Win32 RegisterHotKey for global hotkeys

### Bug Fixes
- Desktop control calculator fix (enter not equals key, increased keystroke delay)
- Email reply no longer adopts sender's identity/signature
- Signal attachment flag fixed (--attachment not -a)
- ChromaDB rules collection sync on startup (rebuild from SQLite when empty)
- Heap corruption eliminated (all style().unpolish/polish calls removed)
- ChromaDB PostHog telemetry disabled (background thread caused GIL crashes)
- Stock talent lazy imports + subprocess isolation to prevent heap corruption

### Earlier
- Vision-enhanced PDF ingestion (`--vision` flag, one chunk per page via Qwen3-VL)
- Multi-query RAG retrieval with alt_queries expansion
- Text-match `$contains` fallback for sparse stat-block chunks
- Keyword re-ranking, 8-chunk hard cap, page number citations in responses
- Training-knowledge contamination prevention in explicit RAG mode
- `_extract_arg()` helper on BaseTalent — standardised single-value LLM extraction
- Signal remote control (receive & send via signal-cli)
- File/photo attachment support in GUI
- Vision follow-up ("read me the answer", "what does it say") — any app
- Planner bug fixes (double-open, context contamination from stale buffer)
- Session summarisation — compact rolling summary replaces full raw buffer dump

## Queued

### Document Ingest — Expanded Format Support
Currently `ingest_documents.py` handles PDF only. Extend to cover the full
office document ecosystem so users can RAG against any file they have.

| Format | Library | Notes |
|---|---|---|
| PDF | `fitz` (PyMuPDF) | Already done; vision flag supported |
| Word (.docx) | `python-docx` | Paragraphs, tables, headings as chunks |
| Excel (.xlsx / .xls) | `openpyxl` / `xlrd` | Sheet-per-chunk or row-range chunks; preserve column headers |
| PowerPoint (.pptx) | `python-pptx` | One chunk per slide; include slide title + body + speaker notes |
| Plain text (.txt, .md) | built-in | Split on double-newline or fixed token count |
| CSV | `csv` stdlib | Row-range chunks with header row prepended to each |

**Design notes:**
- Add a `--format` flag to `ingest_documents.py` (or auto-detect by extension)
- All formats feed into the same `talon_documents` ChromaDB collection with
  consistent metadata: `source`, `page_number` (or `slide_number` / `sheet`),
  `chunk_index`
- Excel and CSV: preserve column headers in each chunk so the model has
  schema context — e.g. `"Name | Score | Date\nAlice | 95 | 2024-03-01"`
- PowerPoint: vision flag could render slide thumbnails via Pillow +
  `pptx.util` for diagram-heavy decks (same approach as PDF vision mode)
- Word: respect heading hierarchy — use `<h1>`, `<h2>` markers in chunk text
  so citations can reference section names rather than just page numbers
- Dependencies to add to requirements: `python-docx`, `openpyxl`, `python-pptx`
  (all pure Python, no system deps)

### GUI Attachment — Non-Image File Support
Currently the 📎 button only accepts image files (jpg, png, etc.) which are
sent to the vision model. Extend it so users can attach any document file for
the LLM to review inline — distinct from RAG ingest (no indexing, no ChromaDB,
just "look at this file right now as part of our conversation").

**How it works:**
- User attaches a PDF, DOCX, XLSX, PPTX, TXT, or CSV via the existing picker
- `_handle_conversation` detects non-image attachments alongside or instead of images
- A `DocumentExtractor` utility extracts plain text from the file
- Extracted text is injected into the prompt wrapped in `_wrap_external()` just
  like image attachments are today — the LLM reads and reasons about it directly
- Image attachments continue to work as before (vision model path)
- Mixed attachments (e.g. image + spreadsheet) are handled in the same call

**Text extraction per format:**

| Format | Method |
|---|---|
| PDF | `fitz` (already a dep) — `page.get_text()` per page, join with page markers |
| Word (.docx) | `python-docx` — paragraphs + table cells in reading order |
| Excel (.xlsx) | `openpyxl` — each sheet as a text table; column headers on every row block |
| PowerPoint (.pptx) | `python-pptx` — slide title + body text + speaker notes per slide |
| TXT / MD | `open(...).read()` — no parsing needed |
| CSV | `csv.reader` — header row prepended to every N-row block |

**Token budget considerations:**
- Extracted text for a large document could be huge. Cap injected text at
  ~6000 chars (≈ 1500 tokens) with a truncation notice, or chunk and only
  inject the first N pages/sheets unless the user asks for more
- Add a print log: `[Attachment] Extracted 2 340 chars from report.xlsx`

**GUI changes needed:**
- `text_input.py`: widen the file picker filter from `"Images (*.png *.jpg ...)"`
  to also include `"Documents (*.pdf *.docx *.xlsx *.pptx *.txt *.md *.csv)"`
- Show a different icon or label for non-image attachments in the attachment
  preview area so the user knows it was read as text, not rendered as an image

**Offer to ingest:**
After reviewing an attached document, Talon could offer:
*"Want me to add this to your document library so you can query it later?"*
This bridges the one-shot review path to the permanent RAG ingest pipeline.

**Dependencies:** same as Document Ingest section above — `python-docx`,
`openpyxl`, `python-pptx` (all already planned)

### Other Queued Items
- RAG: corpus-frequency stop words for `$contains` — skip generic terms like
  "damage" or "shadowrun" that match too broadly across the corpus
- **Configurable `--mdextract` domain schema** — The metadata extraction prompt is
  hardwired for RPG content (creatures, spells, stats, locations). Add a `--domain`
  flag (e.g. `--domain security`, `--domain networking`, `--domain programming`) that
  swaps in an appropriate entity schema, or auto-detect from filename/content.
  Until then `--mdextract` is not useful for technical books — for security books
  you'd want CVEs, tool names, techniques, protocols; for programming books you'd
  want function names, classes, parameters, APIs.

---

## Option 5 — Agentic Tool Calling (Major Architectural Shift)
> **On hold** — waiting for Qwen3.5-VL availability (expected via Unsloth shortly).
> Model upgrade and this refactor will land together to avoid touching all talent
> files twice.

### What it is
Replace the current keyword/embedding talent router with Qwen3's native
function-calling. Instead of Talon guessing which talent to invoke, the
model decides — and includes pre-extracted parameters — in its first
response token stream.

### How it would work
1. Each talent exposes a JSON schema describing its name, purpose, and
   parameters (e.g. `room`, `brightness`, `action` for Hue lights).
2. The LLM client switches from `generate(flat_string)` to
   `chat(messages[], tools[])` using Qwen3's chat template via
   `/v1/chat/completions`.
3. On each call the model either:
   - Answers directly (no tool needed), **or**
   - Emits a `<tool_call>` block with name + args.
4. If a tool call is emitted, Talon executes it and feeds the result back
   as a `<tool_response>` turn, then calls the model again for the
   final natural-language reply.

### What this replaces / eliminates
| Current component | Fate under tool calling |
|---|---|
| `_find_talent(command)` | **Removed** — model routes itself |
| `BaseTalent.keywords / triggers / examples` | Become the tool `description` + param hints |
| `BaseTalent._extract_arg()` | **Removed** — params arrive pre-parsed in the tool call |
| `talents/planner.py` | **Largely redundant** — model chains tool calls natively |
| RAG in `_handle_conversation` | Becomes a `search_documents(query)` tool |
| `_build_capabilities_summary()` | **Removed** — model knows tools from their schemas |
| `_classify_query_intent()` | **Removed** — no pre-emptive retrieval needed |
| Multi-query RAG expansion (extra LLM call) | Moves inside the tool; model writes its own query |
| `_CONVERSATION_SYSTEM_PROMPT` capability list | **Removed** — redundant with tool schemas |

### What stays / changes shape
| Current component | Change |
|---|---|
| `BaseTalent.execute()` | Stays; becomes the handler invoked after tool-call parse |
| `MemorySystem` / ChromaDB | Stays; `search_documents` wraps `get_document_context` |
| `VisionSystem` | Stays; screen capture becomes a `capture_screen()` tool |
| `LLMClient` | Needs `chat(messages, tools=None)` method added |
| Plugin talent loader | Stays; talents register schemas at load time |
| Conversation buffer + summarisation | Stays; becomes proper `messages[]` history |
| Injection-defence wrapping | Stays; tool responses are wrapped as external data |

### Multi-step without a Planner
The model handles chaining natively. "Open calculator, type 2+2, read the
answer" becomes three sequential tool calls in the same response stream
rather than a separate planning LLM call followed by three re-entrant
`process_command()` invocations. This eliminates the sub-step context
contamination bugs we have been working around with `_planner_substep` flags.

### RAG as a tool (not just RAG)
```
search_documents(query: str, max_results: int = 3) -> str
```
The model calls this only when it recognises a knowledge gap. For "tell me
a joke" it never calls it. For "what's the AC modifier for my paladin?" it
does, and it writes its own semantically precise query — better retrieval
than our current intent classifier heuristics.

The same pattern applies to every other external call the model currently
can't decide on for itself:
- `search_web(query)` — already a talent, becomes a first-class tool
- `get_weather(location)` — model decides when location context is needed
- `read_clipboard()` — model decides to look at clipboard vs. use screenshot
- `capture_screen()` — model requests vision when it genuinely needs it

### Per-talent implications

**Hue Lights** — `_extract_arg()` calls for room, action, colour go away.
Schema params handle extraction. The model already knows "living room" →
`room="living room"`. No extra LLM call needed for arg extraction.

**Desktop Control** — The model emits the full action list directly as tool
params rather than generating a JSON blob inside a prompted string. The
schema enforces valid action names, preventing hallucinated action types.

**Weather** — Becomes a clean `get_weather(location: str)` call. No routing
ambiguity with "what's the weather in London vs. what's my forecast".

**Notes / Reminders** — Create/list/delete become distinct tool calls with
typed params. Currently all routed through freeform command strings.

**Signal** — Send becomes `send_signal(recipient, message)`. Much cleaner
than the current "extract recipient and message from natural language" dance.

**Web Search** — Inline with the conversation flow. Model calls
`search_web(query)` mid-response if needed, rather than Talon pre-routing
the entire command to the search talent.

**Email** — Separate `list_emails()`, `read_email(id)`, `compose_email(...)`
tools replace the single catch-all email talent route.

### Conversation history as proper messages[]
The current flat-string buffer injection becomes actual `messages[]` turns:
```python
[
  {"role": "system",    "content": "You are Talon..."},
  {"role": "user",      "content": "what's the weather?"},
  {"role": "assistant", "content": "It's 12°C in London."},
  {"role": "user",      "content": "what about tomorrow?"},  # current turn
]
```
The session summariser (Option 3) still applies: older turns get replaced by
a summary assistant message to keep the history compact.

### Qwen3-specific notes
- Qwen3 has built-in tool-calling support in its training (unlike Qwen2.5
  which needed prompting workarounds).
- Thinking mode (`<think>` tags) can be disabled per-request with
  `enable_thinking: false` for latency-sensitive turns.
- KoboldCpp's `/v1/chat/completions` applies the Qwen3 chat template
  automatically — no manual template string formatting needed.
- Tool schemas follow the OpenAI `tools[]` format which KoboldCpp passes
  through to the model's context via its template.
- KV cache reuse (now enabled by removing `--multiuser`) means the system
  prompt + tool schemas prefix is cached across calls — the per-call cost
  is only the new tokens, not re-processing the full schema list every time.

### Estimated effort
- `LLMClient.chat()` method: ~1 day
- `BaseTalent.to_tool_schema()` + auto-generation from existing metadata: ~half day
- Core routing replacement (`_find_talent` → single chat call): ~1-2 days
- Per-talent schema tuning and testing: ~1 day per talent (~8 talents)
- RAG tool migration + removal of `_classify_query_intent`: ~half day
- Planner retirement: ~half day

**Total: ~2 weeks including per-talent work.**

High payoff: eliminates the entire routing heuristic layer, removes ~400
lines of intent/extraction boilerplate, makes multi-step more reliable,
and the prompt sandwich problem largely solves itself — tool schemas are
compact, conversation history becomes structured, and retrieval only happens
when genuinely needed.
