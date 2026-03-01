from collections import OrderedDict
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QSplitter,
                             QListWidget, QListWidgetItem, QTextBrowser,
                             QLineEdit, QLabel, QPushButton, QWidget)
from PyQt6.QtCore import Qt


# ── Help content ──────────────────────────────────────────────────────────
# Each topic maps to an HTML string rendered in a QTextBrowser.
# Body text inherits color from QSS; only headings use inline accents.

HELP_TOPICS = OrderedDict()

HELP_TOPICS["Getting Started"] = """
<h2>Getting Started</h2>
<p>Talon Assistant is a local-first desktop AI assistant. It connects to a
local LLM server for inference and provides voice control, smart home
integration, an extensible talent plugin system, and a self-improvement
pipeline that learns from your corrections over time.</p>

<h3>First Run</h3>
<ol>
<li>Run <code>python setup.py</code> to create config files and directories.</li>
<li>Start your LLM server (KoboldCpp, llama.cpp, or any OpenAI-compatible endpoint).</li>
<li>Run <code>python main.py</code> to launch the GUI.</li>
</ol>

<h3>Basic Usage</h3>
<ul>
<li><b>Type a command</b> in the text input at the bottom and press Enter or click Send.</li>
<li><b>Toggle voice mode</b> with the microphone button in the voice panel.</li>
<li><b>Browse talents</b> in the sidebar on the left (toggle with Ctrl+B).</li>
<li><b>Check the status bar</b> at the bottom for LLM connection, voice state, and activity.</li>
<li><b>Correct mistakes</b> by saying "no I meant X" &mdash; Talon re-executes and learns.</li>
</ul>

<h3>System Tray</h3>
<p>When you close the window, Talon minimizes to the system tray. Use the
global hotkey <b>Ctrl+Shift+J</b> to show or hide the window from anywhere.
Double-click the tray icon to restore the window.</p>
"""

HELP_TOPICS["Voice Commands"] = """
<h2>Voice Commands</h2>
<p>Talon uses Whisper for speech-to-text and Microsoft Edge TTS for spoken responses.</p>

<h3>Enabling Voice</h3>
<p>Click the <b>microphone button</b> in the voice panel below the text input.
The status will change from "Voice: Off" to "Listening...".</p>

<h3>Wake Words</h3>
<p>When voice is enabled, Talon listens for a wake word before recording your command.
The default wake words are:</p>
<ul>
<li>"okay talon"</li>
<li>"ok talon"</li>
<li>"hey talon"</li>
<li>"talon"</li>
</ul>
<p>You can customize wake words in Settings (Ctrl+,) under the Voice tab.</p>

<h3>Voice Flow</h3>
<ol>
<li><b>Listening</b> &mdash; Talon monitors audio for a wake word.</li>
<li><b>Recording</b> &mdash; After the wake word, Talon records your command
(default 5 seconds or until silence).</li>
<li><b>Transcribing</b> &mdash; Whisper converts speech to text.</li>
<li><b>Processing</b> &mdash; The transcribed command is routed to talents.</li>
</ol>

<h3>Text-to-Speech</h3>
<p>Toggle TTS with the speaker button in the voice panel. When enabled, Talon
speaks responses aloud using the configured voice (default: en-US-AriaNeural).</p>
<p>Press <b>Escape</b> at any time to interrupt speech playback.</p>
"""

HELP_TOPICS["Talents"] = """
<h2>Talents</h2>
<p>Talents are plugins that handle specific types of commands. When you send a
message, the LLM router classifies your intent and routes to the best-matching
talent based on its description and examples.</p>

<h3>Managing Talents</h3>
<ul>
<li><b>Enable/disable</b> talents using the checkboxes in the sidebar.</li>
<li><b>Configure</b> a talent by clicking the gear icon next to its name.</li>
<li>Disabled talents are skipped during command routing.</li>
</ul>

<h3>Built-in Talents</h3>
<table cellpadding="4" cellspacing="0" border="0" width="100%">
<tr><td><b>Talent</b></td><td><b>Priority</b></td><td><b>Description</b></td></tr>
<tr><td>planner</td><td>85</td><td>Multi-step routine executor</td></tr>
<tr><td>news</td><td>80</td><td>Latest headlines via DuckDuckGo News</td></tr>
<tr><td>weather</td><td>75</td><td>Current weather and forecast</td></tr>
<tr><td>hue_lights</td><td>70</td><td>Philips Hue smart light control</td></tr>
<tr><td>reminder</td><td>65</td><td>Timers, reminders, and alarms</td></tr>
<tr><td>lora_train</td><td>60</td><td>Fine-tune the LLM on usage data <i>(disabled by default)</i></td></tr>
<tr><td>web_search</td><td>60</td><td>Web search with LLM-synthesized answers</td></tr>
<tr><td>email</td><td>55</td><td>Check, read, and send email</td></tr>
<tr><td>notes</td><td>45</td><td>Save, search, and manage notes</td></tr>
<tr><td>history</td><td>43</td><td>Search past commands by date, keyword, or outcome</td></tr>
<tr><td>desktop_control</td><td>40</td><td>Launch apps, type text, automation</td></tr>
<tr><td>conversation</td><td>&mdash;</td><td>General LLM conversation (fallback)</td></tr>
</table>

<h3>Example Commands</h3>
<ul>
<li>"What's the weather like?" &rarr; weather talent</li>
<li>"Show me the latest news" &rarr; news talent</li>
<li>"Turn the lights blue" &rarr; hue_lights talent</li>
<li>"Remind me to call the dentist in 30 minutes" &rarr; reminder talent</li>
<li>"Search for Python decorators" &rarr; web_search talent</li>
<li>"Check my email" &rarr; email talent</li>
<li>"Save a note: buy groceries tomorrow" &rarr; notes talent</li>
<li>"What did I ask yesterday?" &rarr; history talent</li>
<li>"Open notepad" &rarr; desktop_control talent</li>
</ul>
"""

HELP_TOPICS["Correction Learning"] = """
<h2>Correction Learning</h2>
<p>When Talon gets something wrong, just say so in plain English. Talon will
re-execute the right command immediately and remember the correction for next time.</p>

<h3>Correction Phrases</h3>
<p>Talon recognises phrases like:</p>
<ul>
<li>"no I meant summarize it"</li>
<li>"that's wrong, open Chrome instead"</li>
<li>"actually I wanted the bedroom lights"</li>
<li>"try again but make it shorter"</li>
<li>"I said email, not notes"</li>
</ul>

<h3>What Happens</h3>
<ol>
<li>Talon detects the correction phrase and extracts your intended command.</li>
<li>The corrected command is executed immediately.</li>
<li>The correction (original command &rarr; correct response) is stored in memory.</li>
<li>On future commands that resemble the original mistake, the correction is
injected as a hint into the LLM prompt.</li>
</ol>

<h3>Proactive Rule Proposals</h3>
<p>If Talon makes the same type of mistake 3 times, it appends a suggestion to
the correction response:</p>
<blockquote><i>&#128161; I've made this mistake 3 times. Want to add a rule?
Try: "whenever I ask [X], always [Y]".</i></blockquote>
<p>Rules are stored permanently in ChromaDB and persist across restarts.</p>

<h3>Training Pair Harvesting</h3>
<p>Every correction is also saved as a supervised training pair in
<code>data/training_pairs.jsonl</code> (Alpaca format). Web search results
are harvested the same way. These pairs accumulate silently and can later be
used to fine-tune the local LLM via the <b>lora_train</b> talent.</p>
<p>Toggle harvesting in <code>config/settings.json</code> under
<code>training.harvest_pairs</code> (default: true).</p>
"""

HELP_TOPICS["History Search"] = """
<h2>History Search</h2>
<p>Every command you send &mdash; and Talon's response &mdash; is logged to the
SQLite database. The <b>history</b> talent lets you query that log in natural language.</p>

<h3>Example Commands</h3>
<ul>
<li>"What did I ask you yesterday?"</li>
<li>"Did I ever ask about Python async?"</li>
<li>"Show me my recent commands"</li>
<li>"Show failed commands from today"</li>
<li>"What did I ask last Monday?"</li>
<li>"Show me everything about lights this week"</li>
</ul>

<h3>Filters Available</h3>
<table cellpadding="4" cellspacing="0" border="0" width="100%">
<tr><td><b>Filter</b></td><td><b>How to use it</b></td></tr>
<tr><td>Date range</td><td>today, yesterday, this week, last week, this month, last Monday&hellip;</td></tr>
<tr><td>Keyword</td><td>Any topic word — searches both the command text and Talon's response</td></tr>
<tr><td>Success filter</td><td>"failed commands" / "successful commands"</td></tr>
<tr><td>Limit</td><td>Default 10 results; say "show me all" for up to 30</td></tr>
</table>

<h3>Output Format</h3>
<p>Results are listed newest-first with a timestamp, a ✓/✗ status icon, and the command text:</p>
<pre>Found 3 command(s) matching 'lights':
  ✓ 2026-03-01 14:23 — turn the living room lights blue
  ✓ 2026-03-01 09:11 — dim the lights to 40%
  ✗ 2026-02-28 21:05 — set lights to warm white</pre>
"""

HELP_TOPICS["LoRA Fine-tuning"] = """
<h2>LoRA Fine-tuning</h2>
<p>After accumulating enough training pairs, you can fine-tune the local LLM
on your own usage patterns. The <b>lora_train</b> talent handles this entirely
in the background — it pauses the inference server, trains, and restarts it
automatically.</p>

<h3>Setup</h3>
<ol>
<li>Install the optional dependencies:
<pre>pip install unsloth trl</pre>
(Windows + CUDA 12 needs a platform-specific unsloth wheel &mdash; see <code>requirements.txt</code>)
</li>
<li>Enable the talent: <b>Settings &rarr; Talent Config &rarr; lora_train</b></li>
<li>Set <b>HuggingFace Model Path</b> to the directory containing your model's
<code>.safetensors</code> weights (not the GGUF &mdash; the full HuggingFace format).</li>
<li>Adjust hyperparameters if desired (rank, alpha, epochs, batch size).</li>
</ol>

<h3>Running a Training Session</h3>
<p>Once enabled and configured, say one of:</p>
<ul>
<li>"start LoRA training"</li>
<li>"fine tune the model"</li>
<li>"train from history"</li>
<li>"retrain from corrections"</li>
</ul>
<p>Talon will:</p>
<ol>
<li>Check prerequisites (minimum pairs, base model path, unsloth installed).</li>
<li>Stop the inference server.</li>
<li>Launch <code>scripts/train_lora.py</code> in the background.</li>
<li>Return immediately with an ETA estimate.</li>
<li>Notify you when training completes and restart the server.</li>
</ol>

<h3>Loading the Adapter</h3>
<p>After training, the adapter is saved to <code>data/lora_adapters/</code>.
To load it in KoboldCpp, add to <b>Settings &rarr; LLM Server &rarr; Extra Args</b>:</p>
<pre>--lora data/lora_adapters/adapter.gguf</pre>
<p>The adapter is non-destructive — your base GGUF stays unchanged. Swap or
remove the adapter at any time by editing Extra Args.</p>

<h3>Talent Config Options</h3>
<table cellpadding="4" cellspacing="0" border="0" width="100%">
<tr><td><b>Setting</b></td><td><b>Description</b></td><td><b>Default</b></td></tr>
<tr><td>Enabled</td><td>Whether this talent is active</td><td>Off</td></tr>
<tr><td>HuggingFace Model Path</td><td>Path to safetensors model directory</td><td>(required)</td></tr>
<tr><td>Adapter Output Directory</td><td>Where the trained adapter is saved</td><td>data/lora_adapters</td></tr>
<tr><td>Min Pairs to Train</td><td>Minimum collected pairs before training is allowed</td><td>100</td></tr>
<tr><td>LoRA Rank</td><td>Adapter rank (higher = more capacity, more VRAM)</td><td>16</td></tr>
<tr><td>LoRA Alpha</td><td>Adapter alpha (usually equal to rank)</td><td>16</td></tr>
<tr><td>Training Epochs</td><td>Number of full passes through the training data</td><td>3</td></tr>
<tr><td>Batch Size</td><td>Examples per gradient step (reduce if OOM)</td><td>2</td></tr>
<tr><td>Max Sequence Length</td><td>Context length during training</td><td>2048</td></tr>
</table>
"""

HELP_TOPICS["Marketplace"] = """
<h2>Talent Marketplace</h2>
<p>The marketplace lets you browse and install community-built talents
without leaving the app.</p>

<h3>Opening the Marketplace</h3>
<ul>
<li><b>File &gt; Talent Marketplace</b> or press <b>Ctrl+M</b></li>
<li>Click the <b>Marketplace</b> button at the bottom of the talent sidebar</li>
</ul>

<h3>Browsing</h3>
<ul>
<li>Search by name using the search bar at the top.</li>
<li>Filter by category: Productivity, Developer, Finance, Utilities.</li>
<li>Each entry shows the talent name, description, author, and version.</li>
</ul>

<h3>Installing</h3>
<p>Click <b>Install</b> on any talent. The plugin file is downloaded to
<code>talents/user/</code> and immediately loaded into the running app.
Dependencies (if any) must be installed manually with pip.</p>

<h3>Removing</h3>
<p>Click <b>Remove</b> on an installed marketplace talent to delete the file
and unload it from the running app.</p>

<h3>Available Talents</h3>
<ul>
<li><b>Productivity:</b> Todo List, Pomodoro Timer, File Organizer, Clipboard History</li>
<li><b>Developer:</b> GitHub, Docker, Regex Tester, JSON Formatter, Code Snippets</li>
<li><b>Finance:</b> Stock Prices, Crypto Prices</li>
<li><b>Utilities:</b> Unit Converter</li>
</ul>
"""

HELP_TOPICS["LLM Setup"] = """
<h2>LLM Setup</h2>
<p>Talon needs a running LLM server for inference. Three API formats are supported:</p>

<table cellpadding="4" cellspacing="0" border="0" width="100%">
<tr><td><b>Format</b></td><td><b>Endpoint</b></td><td><b>Compatible With</b></td></tr>
<tr><td>koboldcpp</td><td>/api/v1/generate</td><td>KoboldCpp (default)</td></tr>
<tr><td>llamacpp</td><td>/completion</td><td>llama.cpp server, llama-server</td></tr>
<tr><td>openai</td><td>/v1/chat/completions</td><td>OpenAI API, LM Studio, Ollama</td></tr>
</table>

<h3>External Server (Default)</h3>
<p>Install and run your LLM server separately, then configure the endpoint URL
and API format in <b>File &gt; LLM Server</b> on the External tab.</p>

<h3>Built-in Server</h3>
<p>Talon can download and manage a llama.cpp server automatically:</p>
<ol>
<li>Open <b>File &gt; LLM Server</b>.</li>
<li>On the <b>Built-in Server</b> tab, click <b>Download llama.cpp</b>.</li>
<li>Select a GGUF model file with the Browse button.</li>
<li>Adjust GPU layers (-1 for all), context size, and threads.</li>
<li>Click <b>Start Server</b>.</li>
</ol>
<p>The server starts as a subprocess and Talon automatically configures the
endpoint and API format. The status bar shows "Built-in (Running)" when ready.</p>

<h3>Recommended Models</h3>
<p>Any GGUF-format model works. For general conversation with vision support,
try models in the 7B to 13B parameter range. Larger models need more VRAM.</p>
"""

HELP_TOPICS["Keyboard Shortcuts"] = """
<h2>Keyboard Shortcuts</h2>

<table cellpadding="6" cellspacing="0" border="0" width="100%">
<tr><td><b>Shortcut</b></td><td><b>Action</b></td></tr>
<tr><td>Ctrl+S</td><td>Save conversation</td></tr>
<tr><td>Ctrl+O</td><td>Load conversation</td></tr>
<tr><td>Ctrl+Shift+E</td><td>Export conversation</td></tr>
<tr><td>Ctrl+M</td><td>Talent Marketplace</td></tr>
<tr><td>Ctrl+,</td><td>Settings</td></tr>
<tr><td>Ctrl+F1</td><td>Help Topics</td></tr>
<tr><td>Ctrl+Q</td><td>Exit</td></tr>
<tr><td>Ctrl+L</td><td>Toggle Activity Log</td></tr>
<tr><td>Ctrl+B</td><td>Toggle Sidebar</td></tr>
<tr><td>Ctrl+=</td><td>Increase font size</td></tr>
<tr><td>Ctrl+-</td><td>Decrease font size</td></tr>
<tr><td>Ctrl+0</td><td>Reset font size</td></tr>
<tr><td>Escape</td><td>Stop TTS playback</td></tr>
<tr><td>Up / Down</td><td>Navigate command history (in text input)</td></tr>
<tr><td>Ctrl+Shift+J</td><td>Global hotkey: show/hide window</td></tr>
</table>
"""

HELP_TOPICS["Configuration"] = """
<h2>Configuration</h2>
<p>All settings are stored in <code>config/settings.json</code>. The file is
created from <code>config/settings.example.json</code> on first run.</p>

<h3>Editing Settings</h3>
<ul>
<li>Use <b>File &gt; Settings</b> (Ctrl+,) for a GUI editor with validation.</li>
<li>Or edit <code>config/settings.json</code> directly (restart required for some changes).</li>
</ul>

<h3>Settings Sections</h3>
<table cellpadding="4" cellspacing="0" border="0" width="100%">
<tr><td><b>Section</b></td><td><b>Purpose</b></td></tr>
<tr><td>llm</td><td>LLM endpoint, API format, generation parameters, prompt template</td></tr>
<tr><td>llm_server</td><td>Built-in server: mode, model path, GPU layers, context size, port</td></tr>
<tr><td>audio</td><td>Microphone sample rate, energy thresholds, noise word filtering</td></tr>
<tr><td>voice</td><td>TTS voice name, wake words</td></tr>
<tr><td>whisper</td><td>Whisper model size, device preference (cuda/cpu), compute type</td></tr>
<tr><td>memory</td><td>SQLite DB path, ChromaDB path, embedding model</td></tr>
<tr><td>documents</td><td>Directory for document ingestion (RAG)</td></tr>
<tr><td>desktop</td><td>PyAutoGUI timing, failsafe toggle, app launch delay</td></tr>
<tr><td>appearance</td><td>Theme (dark/light), base font size</td></tr>
<tr><td>system_tray</td><td>Minimize to tray, notifications, global hotkey</td></tr>
<tr><td>training</td><td>Training pair harvesting toggle; LoRA settings live in Talent Config</td></tr>
</table>

<h3>Training Settings</h3>
<p>The <code>training</code> section has one key you might want to change:</p>
<ul>
<li><code>harvest_pairs</code> (true/false) &mdash; whether corrections and web searches
are automatically saved as training data. Default: true.</li>
</ul>
<p>All LoRA hyperparameters (rank, alpha, epochs, etc.) are configured per-talent
in <b>Settings &rarr; Talent Config &rarr; lora_train</b>.</p>

<h3>Upgrading</h3>
<p>When new settings are added in an update, run <code>python setup.py</code>
again. It deep-merges new keys from the example template without overwriting
your existing values.</p>

<h3>Resetting to Defaults</h3>
<p>Delete <code>config/settings.json</code> and run <code>python setup.py</code>
to regenerate it from the template.</p>
"""

HELP_TOPICS["Creating Talents"] = """
<h2>Creating Talents</h2>
<p>You can extend Talon by creating custom talent plugins. The LLM router uses
your talent's <code>description</code> and <code>examples</code> to decide when
to route commands to it &mdash; no keyword matching needed.</p>

<h3>Basic Structure</h3>
<pre>
from talents.base import BaseTalent

class MyTalent(BaseTalent):
    name = "my_talent"
    description = "Does something useful"
    examples = [
        "do the custom thing",
        "run my custom action",
    ]

    def execute(self, command, context):
        llm = context["llm"]
        response = llm.generate(command)
        return {
            "success": True,
            "response": response,
            "actions_taken": [{"action": "my_action"}],
        }
</pre>

<h3>Required Attributes</h3>
<table cellpadding="4" cellspacing="0" border="0" width="100%">
<tr><td><b>Attribute</b></td><td><b>Type</b></td><td><b>Description</b></td></tr>
<tr><td>name</td><td>str</td><td>Unique identifier</td></tr>
<tr><td>description</td><td>str</td><td>What this talent does (shown to the LLM router)</td></tr>
<tr><td>examples</td><td>list</td><td>Natural-language example commands (primary routing)</td></tr>
</table>

<h3>Optional Attributes</h3>
<table cellpadding="4" cellspacing="0" border="0" width="100%">
<tr><td><b>Attribute</b></td><td><b>Type</b></td><td><b>Description</b></td></tr>
<tr><td>keywords</td><td>list</td><td>Fallback trigger words (used only when LLM is unavailable)</td></tr>
<tr><td>priority</td><td>int</td><td>Sidebar display order (higher = listed first, default 50)</td></tr>
</table>

<h3>Required Methods</h3>
<table cellpadding="4" cellspacing="0" border="0" width="100%">
<tr><td><b>Method</b></td><td><b>Description</b></td></tr>
<tr><td>execute(command, context)</td><td>Perform the action and return a result dict</td></tr>
</table>

<h3>Optional Methods</h3>
<table cellpadding="4" cellspacing="0" border="0" width="100%">
<tr><td><b>Method</b></td><td><b>Description</b></td></tr>
<tr><td>can_handle(command)</td><td>Keyword fallback for degraded mode (default uses keyword_match)</td></tr>
<tr><td>routing_available</td><td>Property: return False to hide from the LLM router</td></tr>
</table>

<h3>Context Dict Keys</h3>
<table cellpadding="4" cellspacing="0" border="0" width="100%">
<tr><td><b>Key</b></td><td><b>Type</b></td><td><b>Description</b></td></tr>
<tr><td>llm</td><td>LLMClient</td><td>Send prompts to the LLM</td></tr>
<tr><td>memory</td><td>MemorySystem</td><td>Store and search notes/history</td></tr>
<tr><td>vision</td><td>VisionSystem</td><td>Capture and analyze screenshots</td></tr>
<tr><td>voice</td><td>VoiceSystem</td><td>Text-to-speech output</td></tr>
<tr><td>config</td><td>dict</td><td>Full settings.json contents</td></tr>
<tr><td>memory_context</td><td>str</td><td>Pre-fetched relevant context</td></tr>
<tr><td>notify</td><td>callable</td><td>Send desktop notifications</td></tr>
<tr><td>server_manager</td><td>LLMServerManager | None</td><td>Stop/start the built-in server (builtin mode only)</td></tr>
<tr><td>assistant</td><td>TalonAssistant</td><td>The main assistant instance</td></tr>
</table>

<h3>Optional: GUI-Configurable Settings</h3>
<p>Implement <code>get_config_schema()</code> to expose settings in the GUI:</p>
<pre>
def get_config_schema(self):
    return {
        "fields": [
            {"key": "max_results", "label": "Max Results",
             "type": "int", "default": 5, "min": 1, "max": 50},
            {"key": "api_key", "label": "API Key",
             "type": "password", "default": ""},
        ]
    }
</pre>
<p>Field types: <code>string</code>, <code>password</code>, <code>int</code>,
<code>float</code>, <code>bool</code>, <code>choice</code>, <code>list</code>.</p>
<p>Password fields are stored securely in the OS keyring.</p>

<h3>Installation</h3>
<ul>
<li>Place the <code>.py</code> file in <code>talents/user/</code></li>
<li>Or use <b>File &gt; Import Talent</b> from the GUI</li>
<li>Talon auto-discovers talents on startup</li>
</ul>
"""

HELP_TOPICS["Troubleshooting"] = """
<h2>Troubleshooting</h2>

<h3>LLM Connection Failed</h3>
<ul>
<li>Verify your LLM server is running and accessible at the configured endpoint.</li>
<li>Check the API format matches your server (koboldcpp, llamacpp, or openai).</li>
<li>Open <b>File &gt; LLM Server</b> and click <b>Test Connection</b> on the External tab.</li>
<li>The status bar shows "LLM: Disconnected" when the server is unreachable.</li>
</ul>

<h3>Voice Not Working</h3>
<ul>
<li>Ensure a microphone is connected and not muted.</li>
<li>Check that the voice toggle (mic button) is enabled.</li>
<li>Increase the energy threshold in Settings &gt; Audio if the mic picks up too
much background noise.</li>
<li>On first run, the Whisper model downloads automatically (may take a minute).</li>
</ul>

<h3>Hue Lights Not Responding</h3>
<ul>
<li>The Hue bridge must be on the same local network.</li>
<li>Run Talon once and press the physical button on the Hue bridge when prompted
to authorize the connection.</li>
<li>Verify the bridge IP in <code>config/hue_config.json</code>.</li>
</ul>

<h3>Slow Startup</h3>
<p>Talon loads the Whisper speech model, sentence-transformer embeddings, and
connects to services on startup. This typically takes 10-30 seconds on first
launch, faster on subsequent runs once models are cached.</p>
<p>Technical note: Models are loaded <i>before</i> the Qt application starts.
This is required because CTranslate2 (used by Whisper) can crash if loaded
after Qt initializes its platform plugins.</p>

<h3>TTS Not Speaking</h3>
<ul>
<li>Check that TTS is toggled on (speaker button in voice panel).</li>
<li>edge-tts requires an internet connection for Azure Neural Voices.</li>
<li>Verify the TTS voice name in Settings &gt; Voice is valid.</li>
</ul>

<h3>LoRA Training Won't Start</h3>
<ul>
<li>Enable the talent first: <b>Settings &rarr; Talent Config &rarr; lora_train &rarr; Enable LoRA Training</b>.</li>
<li>Set a valid <b>HuggingFace Model Path</b> (the safetensors directory, not a .gguf file).</li>
<li>Install unsloth and trl: see platform-specific instructions in <code>requirements.txt</code>.</li>
<li>At least <b>100 training pairs</b> (default minimum) must be collected before training is allowed.
Check <code>data/training_pairs.jsonl</code> line count.</li>
</ul>

<h3>Correction Learning Not Working</h3>
<ul>
<li>Start your correction with a recognised phrase: "no I meant", "that's wrong",
"actually I wanted", etc.</li>
<li>Make sure the previous command response is still in the conversation buffer
(within the last ~16 turns).</li>
<li>Check the console for <code>[Correction]</code> log lines to confirm detection.</li>
</ul>

<h3>Resetting Configuration</h3>
<ol>
<li>Delete <code>config/settings.json</code> (or the specific config file).</li>
<li>Run <code>python setup.py</code> to regenerate from the template.</li>
<li>Your user data (conversations, notes, memory, training pairs) in <code>data/</code> is not affected.</li>
</ol>
"""


# ── Help dialog ───────────────────────────────────────────────────────────

class HelpDialog(QDialog):
    """Help Topics dialog with a searchable table of contents and HTML content."""

    def __init__(self, parent=None, initial_topic=None):
        super().__init__(parent)
        self.setWindowTitle("Talon Help")
        self.setMinimumSize(800, 550)
        self.resize(920, 620)

        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Help Topics")
        header.setObjectName("help_header")
        layout.addWidget(header)

        # Splitter: left nav | right content
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: search + topic list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter topics...")
        self.search_input.setObjectName("help_search")
        self.search_input.textChanged.connect(self._filter_topics)
        left_layout.addWidget(self.search_input)

        self.topic_list = QListWidget()
        self.topic_list.setObjectName("help_topic_list")
        self.topic_list.currentItemChanged.connect(self._on_topic_changed)
        left_layout.addWidget(self.topic_list)

        splitter.addWidget(left_panel)

        # Right panel: content browser
        self.content_browser = QTextBrowser()
        self.content_browser.setObjectName("help_content_pane")
        self.content_browser.setReadOnly(True)
        self.content_browser.setOpenExternalLinks(True)
        splitter.addWidget(self.content_browser)

        splitter.setSizes([220, 680])
        layout.addWidget(splitter)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        # Populate topic list
        for topic_name in HELP_TOPICS:
            self.topic_list.addItem(QListWidgetItem(topic_name))

        # Select initial topic
        if initial_topic:
            self._select_topic(initial_topic)
        elif self.topic_list.count() > 0:
            self.topic_list.setCurrentRow(0)

    def _filter_topics(self, text):
        """Filter topic list by name and content."""
        search = text.lower().strip()
        for i in range(self.topic_list.count()):
            item = self.topic_list.item(i)
            topic_name = item.text()
            if not search:
                item.setHidden(False)
                continue
            content = HELP_TOPICS.get(topic_name, "")
            visible = (search in topic_name.lower()
                       or search in content.lower())
            item.setHidden(not visible)

    def _on_topic_changed(self, current, previous):
        """Display the selected topic's HTML content."""
        if current is None:
            self.content_browser.clear()
            return
        topic_name = current.text()
        html = HELP_TOPICS.get(topic_name, "")
        self.content_browser.setHtml(html)

    def _select_topic(self, topic_name):
        """Navigate to a specific topic by name."""
        for i in range(self.topic_list.count()):
            item = self.topic_list.item(i)
            if item.text() == topic_name:
                self.topic_list.setCurrentRow(i)
                return
