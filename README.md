# Talon Assistant

A local-first desktop AI assistant for Windows with voice control, smart home integration, a talent plugin system, and a self-improvement pipeline. Talon is not OpenClaw, it is not autonomous.

You should install [CUDA 12 SDK](https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Windows) prior to installing this!

THIS IS AN ALPHA RELEASE!
Definitely works in Python 3.10.6, probably works in 3.11 and 3.12 branches, after that it gets dicey with chromadb and ctranslate2.

Talon runs entirely on your machine. It connects to a local LLM server for inference, uses Whisper for speech recognition, and provides a modular talent system for extending functionality. No cloud accounts required.

## Features

- **Voice control** with wake-word detection and text-to-speech responses
- **Local LLM inference** via KoboldCpp, llama.cpp, or any OpenAI-compatible API
- **Built-in llama.cpp server** with automatic download and GPU acceleration
- **Talent plugin system** with auto-discovery and per-talent configuration
- **Talent Marketplace** for browsing and installing community talents
- **Correction learning** — say "no I meant X" and Talon re-executes correctly, stores the correction, and uses it as a hint on similar future commands
- **Command history search** — "what did I ask yesterday?", "show failed commands from last week"
- **Conversation memory** — insights from older conversation turns are distilled to long-term memory before the buffer evicts them
- **Proactive rule proposals** — after correcting the same mistake 3 times, Talon suggests adding a permanent rule
- **Training pair harvesting** — corrections and web search results are silently saved as Alpaca-format training pairs for future LoRA fine-tuning
- **LoRA self-improvement** — fine-tune the local LLM on your accumulated usage data (optional, disabled by default)
- **Philips Hue** smart light control (colors, brightness, scenes)
- **Web search** and news aggregation via DuckDuckGo (no API key needed)
- **Weather** from Open-Meteo, OpenWeatherMap, or WeatherAPI
- **Email** checking, displaying, and eventually sending/deleting (IMAP/SMTP with OS keyring credentials)
- **Reminders and timers** with desktop notifications
- **Notes** with semantic search (SQLite + ChromaDB)
- **Desktop automation** (launch apps, type text, browser navigation)
- **Vision / screenshot analysis** via multimodal LLM
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
| `memory` | SQLite and ChromaDB paths, embedding model |
| `documents` | Document ingestion directory for RAG |
| `desktop` | PyAutoGUI timing, failsafe toggle, app launch delays |
| `appearance` | Theme (dark/light), base font size |
| `system_tray` | Minimize-to-tray behavior, notifications, global hotkey |
| `training` | Training pair harvesting toggle; LoRA settings live in Talent Config → lora_train |

## Built-in Talents

Talents are plugins that handle specific types of commands. When you send a message, Talon checks each talent in priority order and routes to the first one that matches.

| Talent | Priority | Description |
|--------|----------|-------------|
| planner | 85 | Multi-step routine executor |
| news | 80 | Latest headlines via DuckDuckGo News |
| weather | 75 | Current weather and forecast (Open-Meteo, OpenWeatherMap, WeatherAPI) |
| hue_lights | 70 | Philips Hue smart light control via local bridge |
| reminder | 65 | Timers, reminders, and alarms with desktop notifications |
| lora_train | 60 | Fine-tune the local LLM on collected training pairs *(disabled by default)* |
| web_search | 60 | Web search with LLM-synthesized answers |
| email | 55 | Check, read, and send email (IMAP/SMTP) |
| notes | 45 | Save, search, and manage personal notes (SQLite + ChromaDB) |
| history | 43 | Search past commands and responses from the command log |
| desktop_control | 40 | Launch apps, type text, control mouse and keyboard |
| conversation | — | General LLM conversation (fallback when no talent matches) |

Higher priority means the talent is checked first. Disabled talents are skipped entirely.

## Correction Learning

When Talon misunderstands, you can correct it in plain English:

- "no I meant summarize it"
- "that's wrong, open Chrome instead"
- "actually I wanted the bedroom lights"

Talon detects the correction, extracts your intent, re-executes the right command, and stores the correction in memory. On similar future commands, the correction is injected as a hint into the LLM prompt. After the same mistake occurs 3 times, Talon appends a suggestion: *"Want to add a rule to prevent this?"*

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

class MyTalent(BaseTalent):
    name = "my_talent"
    description = "Does something useful"
    keywords = ["my", "custom", "trigger"]
    priority = 50

    def can_handle(self, command):
        return self.keyword_match(command)

    def execute(self, command, context):
        llm = context["llm"]
        response = llm.generate(command)
        return {
            "success": True,
            "response": response,
            "actions_taken": [],
        }
```

### Required interface

| Attribute / Method | Description |
|--------------------|-------------|
| `name` | Unique string identifier |
| `description` | Human-readable description |
| `keywords` | List of trigger words for routing |
| `priority` | Integer; higher = checked first (default 50) |
| `can_handle(command)` | Return `True` if this talent should handle the command |
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

## Architecture

```
main.py                    Entry point (loads models before QApplication)
core/
  assistant.py             TalonAssistant: command routing, talent orchestration,
                           correction detection, buffer eviction consolidation
  llm_client.py            Multi-backend LLM API client
  llm_server.py            Built-in llama.cpp server manager
  voice.py                 Whisper STT, edge-tts TTS, wake-word detection
  memory.py                SQLite + ChromaDB memory, corrections, semantic search
  training_harvester.py    Silently appends training pairs to data/training_pairs.jsonl
  vision.py                Screenshot capture and multimodal analysis
  chat_store.py            Conversation save/load/export
  marketplace.py           Talent marketplace client (GitHub catalog)
  credential_store.py      OS keyring credential management
scripts/
  train_lora.py            Standalone LoRA fine-tuning script (Unsloth + trl)
  ingest_documents.py      Document ingestion for RAG (--vision flag for multimodal)
gui/
  main_window.py           Main window layout and menus
  assistant_bridge.py      Signal bridge between core and GUI
  workers.py               QThread workers for blocking operations
  theme_manager.py         Catppuccin theme switching and font scaling
  system_tray.py           System tray icon and global hotkey
  widgets/                 ChatView, TextInput, VoicePanel, TalentSidebar, StatusBar
  dialogs/                 Settings, LLMSetup, Marketplace, TalentConfig, Help, About
  styles/                  dark_theme.qss (Mocha), light_theme.qss (Latte)
talents/
  base.py                  BaseTalent abstract base class
  planner.py               Multi-step routine executor (priority 85)
  weather.py               Weather talent (Open-Meteo, OpenWeatherMap, WeatherAPI)
  news.py                  News headlines (DuckDuckGo News)
  web_search.py            Web search with LLM synthesis + training pair harvest
  hue_lights.py            Philips Hue smart light control
  reminder.py              Timers and reminders
  email_talent.py          IMAP/SMTP email
  notes.py                 Notes with semantic search
  history.py               Command history search (date ranges, keyword, success filter)
  lora_train.py            LoRA fine-tuning trigger talent (disabled by default)
  desktop_control.py       Desktop automation (PyAutoGUI)
  user/                    User-installed talents (auto-discovered)
data/
  talon_memory.db          SQLite: commands, corrections, rules, preferences
  training_pairs.jsonl     Accumulated training pairs (Alpaca format)
  chroma_db/               ChromaDB: memory, documents, notes, rules, corrections
  lora_adapters/           LoRA adapter output (after training)
config/
  settings.example.json    Configuration template
  hue_config.example.json
  talents.example.json
```

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
