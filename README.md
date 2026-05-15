# Talon Assistant

A local-first desktop AI assistant for Windows with voice control, smart home integration, a talent plugin system, and a self-improvement pipeline. Talon is not OpenClaw, it is not autonomous.

You should install [CUDA 12 SDK](https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Windows) prior to installing this!

THIS IS AN ALPHA RELEASE!
Requires Python 3.10+. Tested on 3.10, 3.11, and 3.12. Python 3.13 may work; 3.14 is not yet supported (ChromaDB compatibility).

Talon runs entirely on your machine. It connects to a local LLM server for inference, uses Whisper for speech recognition, and provides a modular talent system for extending functionality. No cloud accounts required.

## Features

- **Voice control** with wake-word detection and text-to-speech responses
- **Local LLM inference** via KoboldCpp, llama.cpp, or any OpenAI-compatible API
- **Built-in llama.cpp server** with automatic download and GPU acceleration
- **Talent plugin system** with auto-discovery and per-talent configuration
- **Talent Marketplace** for browsing and installing community talents
- **Deep search agent** — agentic RAG over your document library with TOC/index-aware navigation, page flipping, section reading, and structural memory beyond simple vector similarity (see [Deep Search Agent](#deep-search-agent))
- **Document RAG** with vision-enhanced PDF ingestion, EPUB chapter structure, OCR cleanup, and per-book TOC indexing
- **Job search automation** — scrape LinkedIn, Dice, Built In, and SimplyHired; fit-score postings against your bullet library; auto-generate tailored resumes and cover letters; track applications with status, follow-ups, and LinkedIn recruiter recon (see [Job Search Automation](#job-search-automation))
- **Correction learning** — say "no I meant X" and Talon re-executes correctly, stores the correction, and uses it as a hint on similar future commands
- **Command history search** — "what did I ask yesterday?", "show failed commands from last week"
- **Conversation memory** — insights from older conversation turns are distilled to long-term memory before the buffer evicts them
- **Proactive rule proposals** — after correcting the same mistake 3 times, Talon suggests adding a permanent rule
- **Behavioral rules** with semantic triggers ("when I say X, do Y") backed by SQLite + ChromaDB
- **Training pair harvesting** — corrections and web search results are silently saved as Alpaca-format training pairs for future LoRA fine-tuning
- **LoRA self-improvement** — fine-tune the local LLM on your accumulated usage data (optional, disabled by default)
- **Security classifier** for prompt-injection defense on foreign inputs (job descriptions, RAG chunks, emails, web results)
- **Talent builder** — natural-language talent authoring ("create a talent that does X" produces reviewable Python in `talents/user/`)
- **Scheduler** for recurring tasks and time-based triggers
- **Signal remote** — accept commands from authorized phone numbers via signal-cli
- **Philips Hue** smart light control (colors, brightness, scenes)
- **iTunes / Apple Music** playback control
- **Web search** and news aggregation via DuckDuckGo (no API key needed)
- **Web browser** automation for following URLs cited in previous responses
- **News digest** with configurable feeds (RSS + scrape fallback)
- **Weather** from Open-Meteo, OpenWeatherMap, or WeatherAPI
- **Email** checking, displaying, sending, and deleting (IMAP/SMTP with OS keyring credentials)
- **Reminders and timers** with desktop notifications
- **Notes** with semantic search (SQLite + ChromaDB)
- **File organizer** for sorting and renaming files based on natural-language rules
- **Clipboard transform** — paste, transform via the LLM, copy back
- **Task assist** — collaborative task helper that captures screen context, clipboard, and intent for multi-step work
- **Cowork bridge** — integrate with the Cowork session host for collaborative agent runs
- **Desktop automation** (launch apps, type text, browser navigation)
- **Vision / screenshot analysis** via multimodal LLM
- **Reflection** — optional inner-life features (periodic free thought, valence rating, self-set goals, contradiction detection) — all disabled by default
- **Dual themes** (Catppuccin Mocha dark / Catppuccin Latte light)
- **System tray** with global hotkey (Ctrl+Shift+J)
- **Conversation** save, load, and export (text, markdown, JSON)

## Quick Start

### Prerequisites

- Python 3.10 or newer
- Windows 10/11 (primary target; Linux and macOS are not supported yet, though some features may work — credential storage and TTS definitely will not)
- An LLM server: [KoboldCpp](https://github.com/LostRuins/koboldcpp), [llama.cpp](https://github.com/ggerganov/llama.cpp), or any OpenAI-compatible endpoint
  - Or use the **built-in llama.cpp server** (downloaded automatically via File > LLM Server)

### Installation

```
git clone https://github.com/talon-assistant/talon-assistant.git
cd talon-assistant
pip install -r requirements.txt
python setup.py
python main.py
```

For reproducible installs, use `pip install -r requirements.lock` which pins exact versions.

### What setup.py does

- Creates required directories: `data/`, `documents/`, `talents/user/`, `config/`
- Copies `*.example.json` templates to actual config files on first run
- Deep-merges new configuration keys on upgrades without overwriting your values
- Safe to run repeatedly

## Running Modes

| Mode | Command | Description |
|------|---------|-------------|
| GUI (default) | `python main.py` | Full PyQt6 interface with voice, chat, and talents |
| Voice | `python main.py voice` | Terminal-based voice interface with wake-word detection |
| Text | `python main.py text` | Terminal-based text input (REPL) |
| Both | `python main.py both` | Simultaneous voice and text in the terminal |

## Configuration

All settings live in `config/settings.json`. Run `python setup.py` to create it from the included template. The available sections are:

| Section | Purpose |
|---------|---------|
| `llm` | LLM endpoint URL, API format, generation parameters, prompt template |
| `llm_server` | Built-in llama.cpp server: mode, model path, GPU layers, context size |
| `audio` | Microphone sample rate, energy thresholds, noise word filtering |
| `voice` | TTS voice name, wake words |
| `whisper` | Whisper model size, device preference, compute type |
| `memory` | SQLite, ChromaDB, and book-index paths; embedding model; reranker model |
| `documents` | Document ingestion directory for RAG |
| `desktop` | PyAutoGUI timing, failsafe toggle, app launch delays |
| `appearance` | Theme (dark/light), base font size |
| `system_tray` | Minimize-to-tray behavior, notifications, global hotkey |
| `task_assist` | Hotkey and screenshot resolution cap for the task-assist talent |
| `training` | Training pair harvesting toggle; LoRA settings live in Talent Config → lora_train |
| `web_browser` | Per-domain RSS overrides and disable list |
| `personality` | Optional inner-life features (reflection, valence, goals, coherence, anticipation, lora self-refinement) — all default off |
| `scheduler` | Time-based recurring tasks (cron-like, with `enabled` toggles per entry) |
| `security` | Input filter patterns, output scan checks, rate limit, confirmation gates, audit log |

## Built-in Talents

Talents are plugins that handle specific types of commands. When you send a message, Talon checks each talent in priority order and routes to the first one that matches.

| Talent | Priority | Description |
|--------|----------|-------------|
| planner | 85 | Multi-step routine executor |
| plan_executor | 82 | Executes pre-saved multi-step plans by name |
| task_assist | 80 | Collaborative screen+clipboard context helper for multi-step work |
| news | 80 | Latest headlines via DuckDuckGo News |
| news_digest | 78 | Aggregated digest from configurable RSS / scrape sources |
| weather | 75 | Current weather and forecast (Open-Meteo, OpenWeatherMap, WeatherAPI) |
| signal_remote | 72 | Accept commands from authorized phone numbers via signal-cli |
| hue_lights | 70 | Philips Hue smart light control via local bridge |
| itunes | 68 | iTunes / Apple Music playback control |
| reminder | 65 | Timers, reminders, and alarms with desktop notifications |
| scheduler_talent | 62 | Schedule recurring tasks and time-based triggers |
| rules | 61 | Manage behavioral rules ("when I say X, do Y") with semantic matching |
| job_search | 60 | Scrape job postings (LinkedIn, Dice, Built In, SimplyHired) with fit scoring |
| job_tracker | 60 | Application database, status tracking, resume/cover-letter generation, LinkedIn recon |
| lora_train | 60 | Fine-tune the local LLM on collected training pairs *(disabled by default)* |
| web_search | 60 | Web search with LLM-synthesized answers |
| web_browser | 58 | Open URLs cited in previous responses; basic scrape-and-summarize |
| email | 55 | Check, read, and send email (IMAP/SMTP) |
| talent_builder | 50 | Natural-language talent authoring — generates reviewable Python into `talents/user/` |
| file_organizer | 48 | Sort and rename files using natural-language rules |
| notes | 45 | Save, search, and manage personal notes (SQLite + ChromaDB) |
| clipboard_transform | 44 | Take clipboard contents, transform via the LLM, copy result back |
| history | 43 | Search past commands and responses from the command log |
| cowork_bridge | 42 | Integration with the Cowork session host for collaborative agent runs |
| desktop_control | 40 | Launch apps, type text, control mouse and keyboard |
| conversation | — | General LLM conversation (fallback when no talent matches) |

Higher priority means the talent is checked first. Disabled talents are skipped entirely. The conversation fallback also handles `deep search` / `deep research` triggers that route into the agentic RAG loop (see [Deep Search Agent](#deep-search-agent)).

## Correction Learning

When Talon misunderstands, you can correct it in plain English:

- "no I meant summarize it"
- "that's wrong, open Chrome instead"
- "actually I wanted the bedroom lights"

Talon detects the correction, extracts your intent, re-executes the right command, and stores the correction in memory. On similar future commands, the correction is injected as a hint into the LLM prompt. After the same mistake occurs 3 times, Talon appends a suggestion: *"Want to add a rule to prevent this?"*

## Deep Search Agent

Prefix any query with `deep search`, `deep research`, or `research deeply` to invoke the agentic RAG loop instead of the standard one-shot retrieval. The agent has tools to navigate documents the way a human does — by structure, position, and content — not just by vector similarity.

### Tools the agent can call

| Tool | Purpose |
|------|---------|
| `search` | Global semantic + keyword search across all ingested documents |
| `filter_source` | Search restricted to a single book |
| `lookup_in_toc` | Direct TOC entry lookup — book's curated map for "what page is X on" |
| `read_page` | Fetch the full text of one page or EPUB chapter |
| `read_nearby` | Fetch ±N adjacent pages from an anchor — page flipping |
| `read_section` | Read the entire TOC section a page belongs to |
| `next_section` | Jump to the start of the section after the current one |
| `list_sources` | Inventory of available books, with `has_toc` flag for navigable ones |

### Automatic behaviors

- **Index auto-follow:** when `read_page` returns a back-of-book index chunk, the system detects the "Term, page" density pattern, looks up the user's query term in the index, and silently appends the resolved page content. The agent doesn't have to reason "this is an index, the answer is on page 133."
- **Auto-extend on near-miss:** when a query keyword sits in the last 400 characters of a chunk (likely cut off at a page boundary), the system automatically fetches the next page and appends it.
- **Cross-encoder bypass:** when a candidate chunk contains all the query keywords, the reranker is skipped. The cross-encoder often demotes multi-topic pages (a spell catalog where one entry is the answer) below related-but-less-relevant chunks — strong keyword evidence overrides.
- **Search-engine-style scoring:** IDF weighting (rare terms outrank common ones), heading detection (keyword in a section title outranks the same keyword in body text), AND-gated stat-block detection (keyword followed by clustered uppercase column headers like `RANGE TYPE DURATION DV DAMAGE`).
- **Compound-word variant expansion:** queries like "mana bolt" also search for the concatenated form "manabolt" so chunks that render the term as one word in the source are still retrieved.
- **Keyword-context preview:** chunk previews shown to the agent center on the longest matching keyword, not the chunk's leading 200 characters, so the agent can identify the right page even when the answer sits deep in the chunk.

### TOC and EPUB structure indexing

- PDF TOCs are parsed at ingestion time (`core/document_index.py`) using regex pattern detection with dot-leader recognition and per-book page-offset detection (printed page numbers rarely match PDF page indices).
- EPUB structure comes from `ebooklib`'s nav doc — chapters are atomic units indexed by spine position.
- TOC entries are stored in SQLite (`data/talon_book_index.db`) for fast keyword and substring lookups, independent of ChromaDB.
- A backfill command (`build_toc_index.py`) walks already-ingested books in ChromaDB and reconstructs the TOC index without re-vision-ingesting hundreds of pages.

### Standalone debugger tools

- `python rquery.py "your query"` — runs the standard RAG pipeline standalone with full visibility into every stage (semantic candidates, `$contains` hits, reranker scores, kw boost, quality scoring).
- `python deep_query.py "your query"` — runs the agentic deep search loop standalone, dumps every iteration's tool call, args, and result summary.
- `python parse_book_toc.py "filename.pdf" --verbose` — runs only the TOC parser against one book to see what entries got extracted.
- `python build_toc_index.py --epub-dir <path>` — backfill TOC for a directory of EPUB source files, matched to ingested filenames by `DC:title` and `DC:creator` metadata.

## Job Search Automation

A pair of talents (`job_search` and `job_tracker`) plus a dedicated `gui/dialogs/job_inbox_dialog.py` UI handle the full pipeline from posting discovery to application submission.

### Sources

- **LinkedIn** (persistent Chrome profile, requires sign-in)
- **Dice**
- **Built In**
- **SimplyHired** (throwaway profile, no login)

Each scraper uses a desktop user-agent and standard Selenium WebDriver settings to avoid headless detection flags. Indeed and Glassdoor are dropped because Cloudflare blocks all headless approaches against them.

### Flow

1. **Scrape and score:** `job_search` runs every configured search URL, deduplicates by job URL, and calls the LLM to compute a fit score (0–100) against the bullet library.
2. **Inbox UI:** the **Job Inbox** dialog shows all postings with fit score, source, status, and per-row actions. Filter by status, source, minimum fit, or full-text search across company / position / location.
3. **Pipelines per row:** one-click buttons run the "tailored resume" / "cover letter" / "full prep" pipelines, each generating output into a per-application folder `jobappmaterials/{company}_{position}_{date}/`.
4. **Recon:** for LinkedIn postings, a "Recon" button uses the persistent Chrome profile to find the company's recruiter and first-degree connections, stored as inbox metadata.
5. **Status tracking:** dropdown per row for `new`, `interested`, `applied`, `interview`, `offer`, `rejected`, `archived`. Moving to `applied` triggers the LinkedIn "I'm Interested" auto-click for jobs sourced from LinkedIn.
6. **Auto-archive expiry:** archived applications are auto-purged after a configurable retention window (default 30 days) on Talon startup, with a CSV audit log at `data/job_archive_purge.csv` for recovery.

### Bullet library format

The bullet libraryis a structured markdown file organized by role (chronological), with numbered bullets that the resume/cover-letter generators can pull selectively based on JD match. A "Corrections Ledger" section at the top establishes global facts (titles, dates, metrics) so the LLM doesn't drift across documents.

## Training Pair Harvesting

Talon silently accumulates supervised training data from real use:

- **Corrections** — every corrected command becomes an (original → correct response) pair
- **Web searches** — synthesized web search answers are saved as (question → answer) pairs

Pairs are written to `data/training_pairs.jsonl` in Alpaca format. Harvesting is enabled by default and can be toggled in `config/settings.json` under `training.harvest_pairs`.

## LoRA Fine-tuning

Once enough training pairs have accumulated, you can fine-tune the local LLM on your own usage patterns:

1. Enable the `lora_train` talent in Settings → Talent Config → lora_train
2. Set `base_model_path` to your HuggingFace model directory (the safetensors version, not the GGUF)
3. Say "start LoRA training" — Talon pauses the inference server, trains in the background, and notifies you when done
4. Add `--lora data/lora_adapters/adapter.gguf` to your server extra args to load the adapter

Requires: `pip install unsloth trl` (see platform-specific instructions in `requirements.txt`)

## Marketplace

Browse and install community talents from the built-in marketplace:

- Open with **File > Talent Marketplace** or **Ctrl+M**
- Search by name or filter by category
- One-click install downloads the talent to `talents/user/`
- Installed talents are auto-discovered on next launch

Available categories: Productivity, Developer, Finance, Utilities.

## Creating Custom Talents

Create a Python file that subclasses `BaseTalent`:

```python
from talents.base import BaseTalent
from core.llm_client import LLMError

class MyTalent(BaseTalent):
    name = "my_talent"
    description = "Does something useful"
    examples = ["do the useful thing", "run my talent"]
    priority = 50

    # Run in a subprocess if your talent loads C-extension libraries
    # (numpy, pandas, yfinance, etc.) that can corrupt the host process.
    subprocess_isolated = False

    # Pip packages beyond base requirements (checked at load time).
    required_packages = []

    def execute(self, command, context):
        llm = context["llm"]

        # Extract a single value from the command using the LLM:
        color = self._extract_arg(llm, command, "color name",
                                  options=["red", "green", "blue"])

        try:
            response = llm.generate(command)
        except LLMError:
            return {"success": False, "response": "LLM unavailable."}

        return {
            "success": True,
            "response": response,
            "actions_taken": ["did_something"],
        }
```

You can also ask Talon to build a talent for you: say "create a talent that does X" and the talent builder will generate the code, ask you to review it, and install it into `talents/user/`.

### Required interface

| Attribute / Method | Description |
|--------------------|-------------|
| `name` | Unique string identifier |
| `description` | Human-readable description (used by the LLM router) |
| `examples` | Natural-language example commands (primary routing signal) |
| `priority` | Integer; higher = checked first (default 50) |
| `execute(command, context)` | Perform the action and return a result dict |

### Context dict

The `context` dictionary passed to `execute()` contains:

| Key | Type | Description |
|-----|------|-------------|
| `llm` | LLMClient | Send prompts to the LLM |
| `memory` | MemorySystem | Store and retrieve notes, search history |
| `vision` | VisionSystem | Capture and analyze screenshots |
| `voice` | VoiceSystem | Text-to-speech output |
| `config` | dict | Full settings.json contents |
| `memory_context` | str | Pre-fetched relevant memory for the current command |
| `notify` | callable | Send desktop notifications: `notify(title, message)` |
| `server_manager` | LLMServerManager \| None | Stop/start the built-in inference server (set only in builtin mode) |
| `assistant` | TalonAssistant | The main assistant instance |

### Optional: configurable settings

Expose GUI-editable settings by implementing `get_config_schema()`:

```python
def get_config_schema(self):
    return {
        "fields": [
            {"key": "max_results", "label": "Max Results", "type": "int",
             "default": 5, "min": 1, "max": 50},
            {"key": "api_key", "label": "API Key", "type": "password",
             "default": ""},
        ]
    }
```

Supported field types: `string`, `password`, `int`, `float`, `bool`, `choice`, `list`.

Password fields are stored in the OS keyring and never written to disk.

### Installation

Place the `.py` file in `talents/user/` or use **File > Import Talent** from the GUI.

## LLM Backend Options

Talon supports three API formats for connecting to an LLM server:

| Format | Endpoint | Compatible With |
|--------|----------|-----------------|
| `koboldcpp` | `/api/v1/generate` | KoboldCpp (default) |
| `llamacpp` | `/completion` | llama.cpp server, llama-server |
| `openai` | `/v1/chat/completions` | OpenAI API, LM Studio, Ollama, text-generation-webui |

Configure the format in `config/settings.json` under `llm.api_format`, or use the **File > LLM Server** dialog.

### Built-in server

Set `llm_server.mode` to `"builtin"` and provide a path to a GGUF model file. Talon will:

1. Download `llama-server` from the latest llama.cpp GitHub release (with CUDA support)
2. Launch the server as a managed subprocess
3. Automatically configure the LLM endpoint and API format
4. Poll `/health` until the server is ready

Use **File > LLM Server** to download the binary, select a model, and start/stop the server at runtime.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+S | Save conversation |
| Ctrl+O | Load conversation |
| Ctrl+Shift+E | Export conversation |
| Ctrl+M | Talent Marketplace |
| Ctrl+, | Settings |
| Ctrl+F1 | Help Topics |
| Ctrl+Q | Exit |
| Ctrl+L | Toggle Activity Log |
| Ctrl+B | Toggle Sidebar |
| Ctrl+= | Increase font size |
| Ctrl+- | Decrease font size |
| Ctrl+0 | Reset font size |
| Escape | Stop TTS playback |
| Up / Down | Navigate command history (in text input) |
| Ctrl+Shift+J | Global hotkey: show/hide window (from system tray) |
| Ctrl+Alt+J | Task Assist: capture screen context + clipboard for collaborative help |

## Architecture

```
main.py                    Entry point (loads models before QApplication)
core/
  assistant.py             TalonAssistant: command routing, talent orchestration,
                           correction detection, buffer eviction consolidation
  conversation.py          Conversation engine (extracted from assistant.py)
  llm_client.py            Multi-backend LLM API client (KoboldCpp / llama.cpp / OpenAI)
  llm_server.py            Built-in llama.cpp server manager
  voice.py                 Whisper STT, edge-tts TTS, wake-word detection
  memory.py                SQLite + ChromaDB memory, corrections, semantic search
  document_retriever.py    RAG retrieval pipeline with variant expansion, IDF +
                           heading + stat-block scoring, cross-encoder bypass on
                           strong keyword signal, page-flipping helpers
  document_extractor.py    Text extraction from PDF / EPUB / DOCX / etc.
  document_index.py        TOC parser, page-offset detection, EPUB nav-doc
                           extraction, back-of-book index detection
  toc_store.py             SQLite store for per-book TOC entries and metadata
  deep_search_agent.py     Agentic RAG: LLM drives retrieval via tool calls
                           (search, lookup_in_toc, read_page, read_nearby,
                           read_section, etc.) with index auto-follow and
                           auto-extend on near-miss
  reranker.py              Cross-encoder reranker with keyword boost
  embeddings.py            Embedding model wrapper (BAAI/bge-base-en-v1.5 default)
  resume_builder.py        Tailored resume generation from bullet library + JD
  resume_docx.py           DOCX writer for generated resumes
  scheduler.py             Recurring task scheduler
  reflection_loop.py       Optional inner-life features (free thought, valence,
                           goals, contradiction detection) — all off by default
  training_harvester.py    Silently appends training pairs to data/training_pairs.jsonl
  lora_trainer.py          In-process LoRA training (Unsloth)
  vision.py                Screenshot capture and multimodal analysis
  chat_store.py            Conversation save/load/export
  marketplace.py           Talent marketplace client (GitHub catalog)
  credential_store.py      OS keyring credential management
  config.py                Configuration utilities (deep merge)
  logging_config.py        Centralized logging with rotating file output
  security.py              Prompt injection defense, input/output filtering, alerts
  security_classifier.py   ML-based classifier for foreign-input prompt injection
  skill_router.py          On-demand talent loader for query-driven activation
  input_normalizer.py      Punctuation / whitespace / Unicode normalization
ingest_documents.py        Document ingestion for RAG (--vision flag for multimodal,
                           builds TOC index alongside chunks)
build_toc_index.py         Standalone TOC backfill for already-ingested books
parse_book_toc.py          TOC parser debugger (--list, --all, --verbose)
rquery.py                  RAG retrieval pipeline debugger
deep_query.py              Agentic deep-search debugger
gui/
  main_window.py           Main window layout and menus
  assistant_bridge.py      Signal bridge between core and GUI
  workers.py               QThread workers for blocking operations
  theme_manager.py         Catppuccin theme switching and font scaling
  system_tray.py           System tray icon and global hotkey
  widgets/                 ChatView, TextInput, VoicePanel, TalentSidebar, StatusBar
  dialogs/                 Settings, LLMSetup, Marketplace, TalentConfig, Help,
                           About, JobInboxDialog (job search inbox UI)
  styles/                  dark_theme.qss (Mocha), light_theme.qss (Latte)
talents/
  base.py                  BaseTalent abstract base class
  planner.py               Multi-step routine executor (priority 85)
  plan_executor.py         Executes pre-saved named plans
  task_assist.py           Screen+clipboard context helper for multi-step work
  weather.py               Weather talent (Open-Meteo, OpenWeatherMap, WeatherAPI)
  news.py                  News headlines (DuckDuckGo News)
  news_digest.py           Aggregated RSS / scrape digest
  signal_remote.py         signal-cli remote command bridge
  hue_lights.py            Philips Hue smart light control
  itunes.py                iTunes / Apple Music playback control
  reminder.py              Timers and reminders
  scheduler_talent.py      Schedule recurring tasks and time-based triggers
  rules.py                 Behavioral rule CRUD with semantic trigger matching
  job_search.py            Scrape job postings, fit-score against bullet library
  job_tracker.py           Application DB, status tracking, resume/cover-letter
                           generation, LinkedIn recruiter recon, auto-archive expiry
  lora_train.py            LoRA fine-tuning trigger talent (disabled by default)
  web_search.py            Web search with LLM synthesis + training pair harvest
  web_browser.py           Open URLs cited in previous responses
  email_talent.py          IMAP/SMTP email
  talent_builder.py        Natural-language talent authoring
  file_organizer.py        Sort/rename files via natural-language rules
  notes.py                 Notes with semantic search
  clipboard_transform.py   Clipboard → LLM transform → clipboard
  history.py               Command history search (date ranges, keyword, success)
  cowork_bridge.py         Cowork session-host integration
  desktop_control.py       Desktop automation (PyAutoGUI)
  user/                    User-installed talents (auto-discovered)
tests/                     97-test pytest suite covering core modules
data/
  talon_memory.db          SQLite: commands, corrections, rules, preferences
  talon_book_index.db      SQLite: per-book TOC entries and metadata
  job_tracker.db           SQLite: job applications, follow-ups, recon results
  job_archive_purge.csv    Audit log of auto-purged archived applications
  training_pairs.jsonl     Accumulated training pairs (Alpaca format)
  chroma_db/               ChromaDB: memory, documents, notes, rules, corrections
  lora_adapters/           LoRA adapter output (after training)
  jobappmaterials/         Per-application generated resumes and cover letters
  logs/                    Rotating log files (talon.log)
config/
  settings.example.json    Configuration template
  hue_config.example.json
  talents.example.json
  news_digest.json         News digest feed configuration
  scheduled_tasks.json     Scheduler task definitions
```

## Testing

Run the test suite:

```
python -m pytest tests/ -v
```

97 tests cover config, security, LLM client, talent base, memory, conversation, and routing.

## Logging

Talon uses centralized logging via `core/logging_config.py`.

| Output | Level | Details |
|--------|-------|---------|
| Console | INFO+ | Condensed output for interactive use |
| File | DEBUG+ | Full detail written to `data/logs/talon.log` |

The file handler uses a rotating strategy: 5 MB max per file, 3 backups retained.

## Disclaimer

This project is provided as-is, without warranty of any kind. Use at your own risk.

Talon can control your desktop (keyboard, mouse, screenshots), manage smart home
devices, and accept remote commands via Signal. Misconfiguration — particularly of
the Signal remote feature — could allow unintended access to your machine. You are
solely responsible for securing your setup, including keeping `authorized_numbers`
accurate and your signal-cli config directory private.

AI-generated responses may be inaccurate, incomplete, or inappropriate. Do not rely
on them for anything safety-critical. Document RAG responses are only as accurate as
the source material and the model's interpretation of it.

This project is not affiliated with or endorsed by any third party whose software or
services it integrates with.

## License

Licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE).
