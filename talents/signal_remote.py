"""signal_remote.py — Remote control Talon via Signal (signal-cli).

Disabled by default. Enable via Settings → Talent Config → signal_remote,
then configure your phone number, signal-cli path, and authorized numbers.

Prerequisites (do once, manually):
  1. Install JRE 21+:  winget install Microsoft.OpenJDK.21
  2. Download signal-cli from https://github.com/AsamK/signal-cli/releases
  3. Register:  signal-cli -a +1YOURNUM --config data/signal-cli-config register
  4. Verify:    signal-cli -a +1YOURNUM --config data/signal-cli-config verify CODE

Flow:
  - Background thread polls `signal-cli receive` every N seconds
  - Messages from authorized numbers that start with the configured prefix
    (default "talon: ") are forwarded to assistant.process_command()
  - The response is sent back as a Signal reply to the sender
"""

import json
import os
import subprocess
import threading
import time
from datetime import datetime

from talents.base import BaseTalent


class SignalRemoteTalent(BaseTalent):
    name = "signal_remote"
    description = "Receive commands from and send responses to authorized Signal contacts"
    keywords = [
        "signal status", "check signal", "signal remote",
        "signal listener", "signal messages",
    ]
    examples = [
        "what's the signal remote status",
        "check for signal messages now",
        "is the signal listener running",
        "show signal stats",
    ]
    priority = 48   # between notes (45) and email (55)

    def get_config_schema(self) -> dict:
        return {
            "fields": [
                {"key": "enabled",
                 "label": "Enable Signal Remote",
                 "type": "bool",
                 "default": False},
                {"key": "signal_cli_path",
                 "label": "signal-cli Path",
                 "type": "string",
                 "default": "signal-cli"},
                {"key": "config_dir",
                 "label": "signal-cli Config Dir",
                 "type": "string",
                 "default": "data/signal-cli-config"},
                {"key": "account_number",
                 "label": "Talon's Signal Number (+E.164)",
                 "type": "password",
                 "default": ""},
                {"key": "authorized_numbers",
                 "label": "Authorized Numbers (one per line)",
                 "type": "list",
                 "default": []},
                {"key": "command_prefix",
                 "label": "Command Prefix",
                 "type": "string",
                 "default": "talon: "},
                {"key": "poll_interval",
                 "label": "Poll Interval (seconds)",
                 "type": "int",
                 "default": 10,
                 "min": 5,
                 "max": 300},
                {"key": "max_response_chars",
                 "label": "Max Response Length (chars)",
                 "type": "int",
                 "default": 1000,
                 "min": 100,
                 "max": 4000},
            ]
        }

    # ── Lifecycle ──────────────────────────────────────────────────

    def __init__(self):
        super().__init__()
        self._assistant = None
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stats: dict = {
            "messages_received": 0,
            "commands_processed": 0,
            "last_seen": None,       # ISO timestamp of most recent message
            "last_sender": None,
        }

    def set_assistant(self, assistant) -> None:
        """Called by TalonAssistant.__init__() after talent discovery.

        Stores the assistant reference so the background thread can call
        process_command() and trigger notifications.
        """
        self._assistant = assistant
        if self.talent_config.get("enabled", False):
            self._start_polling()

    def update_config(self, config: dict) -> None:
        """Called by GUI when the user saves talent config changes."""
        super().update_config(config)
        if config.get("enabled", False):
            self._restart_polling()
        else:
            self._stop_polling()

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    # ── Thread management ──────────────────────────────────────────

    def _start_polling(self) -> None:
        if not self._validate_config():
            return
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="signal-remote-poll",
        )
        self._poll_thread.start()
        print("   [Signal] Polling started.")

    def _stop_polling(self) -> None:
        self._stop_event.set()
        # Daemon thread — no join needed; it will exit at the next wait() tick

    def _restart_polling(self) -> None:
        self._stop_polling()
        time.sleep(0.25)
        self._start_polling()

    # ── Validation ─────────────────────────────────────────────────

    def _validate_config(self) -> bool:
        """Check all prerequisites. Returns False and logs if anything is missing."""
        cfg = self.talent_config

        if self._assistant is None:
            print("   [Signal] Cannot start: assistant not set yet.")
            return False

        if not cfg.get("account_number", "").strip():
            print("   [Signal] Cannot start: account_number not configured.")
            print("   [Signal]   Set it in Settings → Talent Config → signal_remote.")
            return False

        authorized = self._get_authorized_numbers()
        if not authorized:
            print("   [Signal] Cannot start: no authorized_numbers configured.")
            print("   [Signal]   Add at least one phone number to the authorized list.")
            return False

        cli = cfg.get("signal_cli_path", "signal-cli")
        try:
            result = subprocess.run(
                [cli, "--version"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                raise OSError("non-zero exit")
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            print(f"   [Signal] Cannot start: signal-cli not found at {cli!r}.")
            print("   [Signal]   Install JRE 21+ and download signal-cli from:")
            print("   [Signal]   https://github.com/AsamK/signal-cli/releases")
            return False

        return True

    def _get_authorized_numbers(self) -> list[str]:
        """Return a clean list of authorized phone numbers from config."""
        raw = self.talent_config.get("authorized_numbers", [])
        if isinstance(raw, list):
            return [n.strip() for n in raw if str(n).strip()]
        if isinstance(raw, str):
            return [n.strip() for n in raw.splitlines() if n.strip()]
        return []

    # ── Polling loop ───────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Background daemon thread: poll Signal periodically."""
        interval = max(5, int(self.talent_config.get("poll_interval", 10)))
        while not self._stop_event.wait(interval):
            try:
                self._check_messages()
            except Exception as e:
                print(f"   [Signal] Poll error: {e}")

    def _check_messages(self) -> None:
        """Run one `signal-cli receive` pass and process any incoming messages."""
        cfg = self.talent_config
        cli = cfg.get("signal_cli_path", "signal-cli")
        config_dir = cfg.get("config_dir", "data/signal-cli-config")
        account = cfg.get("account_number", "")

        # --output=json MUST be a global flag, before -a and the subcommand
        result = subprocess.run(
            [
                cli,
                "--output=json",
                "--config", config_dir,
                "-a", account,
                "receive",
                "--timeout", "3",   # short — we control cadence via poll_interval
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )

        if result.returncode != 0:
            stderr_snippet = result.stderr.strip()[:200]
            if stderr_snippet:
                print(f"   [Signal] receive failed: {stderr_snippet}")
            return

        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                envelope = json.loads(line)
                self._handle_envelope(envelope)
            except json.JSONDecodeError:
                continue

    # ── Message handling ───────────────────────────────────────────

    def _handle_envelope(self, envelope: dict) -> None:
        """Inspect one received envelope and process it if it's a valid command."""
        cfg = self.talent_config

        try:
            inner = envelope.get("envelope", {})
            sender = inner.get("source", "")

            # dataMessage: messages received FROM another account TO ours
            data_msg = inner.get("dataMessage") or {}
            text = (data_msg.get("message") or "").strip()

            # syncMessage.sentMessage: Note to Self — messages the user sends
            # FROM their phone are delivered to linked devices as a sync, not
            # a dataMessage.  We also extract the destination so we can reply
            # to the right place (the user's own number).
            sync_dest = None
            if not text:
                sync_msg = inner.get("syncMessage") or {}
                sent_msg = sync_msg.get("sentMessage") or {}
                text = (sent_msg.get("message") or "").strip()
                if text:
                    sync_dest = sent_msg.get("destination") or sender
        except (AttributeError, TypeError):
            return

        # 1. Skip non-text envelopes (receipts, typing indicators, calls, etc.)
        if not text:
            return

        # 2. Authorization check
        authorized = self._get_authorized_numbers()
        if sender not in authorized:
            print(f"   [Signal] Ignored message from unauthorized sender: {sender}")
            return

        # 3. Prefix check (case-insensitive)
        prefix = cfg.get("command_prefix", "talon: ").lower()
        if not text.lower().startswith(prefix):
            return   # Normal Signal chat with this number — ignore silently

        command = text[len(prefix):].strip()
        if not command:
            return

        # 4. Update stats
        now_iso = datetime.now().isoformat()
        with self._lock:
            self._stats["messages_received"] += 1
            self._stats["last_seen"] = now_iso
            self._stats["last_sender"] = sender

        print(f"   [Signal] Command from {sender}: {command!r}")

        # 5. Process command
        #    _executing_rule=True — prevents conversation buffer pollution from
        #    a background thread; speak_response=False — user isn't at the PC
        if self._assistant is None:
            print("   [Signal] Assistant not available, cannot process command.")
            return
        try:
            result = self._assistant.process_command(
                command,
                speak_response=False,
                _executing_rule=True,
            )
            response = (result.get("response") or "").strip()
        except Exception as e:
            response = f"Error processing command: {e}"
            print(f"   [Signal] process_command error: {e}")

        # 6. Truncate if needed
        max_chars = int(cfg.get("max_response_chars", 1000))
        if len(response) > max_chars:
            response = response[:max_chars - 3] + "..."

        # 6b. Collect file attachments from actions_taken (e.g. screenshots)
        attachments = []
        for action_result in (result.get("actions_taken") or []):
            ar = action_result.get("result", "")
            if isinstance(ar, str) and ar.startswith("Screenshot: "):
                path = ar[len("Screenshot: "):].strip()
                if path and os.path.exists(path):
                    attachments.append(path)

        with self._lock:
            self._stats["commands_processed"] += 1

        # 7. Send reply — for Note to Self (syncMessage), reply to sync_dest
        #    (the user's own number) rather than sender (also their number, same thing,
        #    but sync_dest is more explicit and correct)
        reply_to = sync_dest or sender
        self._send_reply(reply_to, response or "(no response)", attachments=attachments)

    def _send_reply(self, recipient: str, message: str,
                    attachments: list | None = None) -> None:
        """Send a Signal message (with optional file attachments) back to the sender.

        When recipient is the account's own number (Note-to-Self), signal-cli's
        'send' command strips the self-recipient and returns "No recipients given".
        Use 'send --note-to-self' flag instead (no recipient argument needed).
        """
        cfg = self.talent_config
        cli = cfg.get("signal_cli_path", "signal-cli")
        config_dir = cfg.get("config_dir", "data/signal-cli-config")
        account = cfg.get("account_number", "")

        # Detect Note-to-Self: recipient is our own number or empty.
        # signal-cli strips self from 'send' recipients ("No recipients given");
        # use the --note-to-self flag instead (no recipient argument needed).
        is_self = not recipient or recipient == account
        if is_self:
            cmd = [cli, "--config", config_dir, "-a", account,
                   "send", "--note-to-self", "-m", message]
        else:
            cmd = [cli, "--config", config_dir, "-a", account,
                   "send", "-m", message, recipient]

        for path in (attachments or []):
            cmd += ["--attachment", path]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"   [Signal] Send failed: {result.stderr.strip()[:200]}")
            else:
                att_note = f" (+{len(attachments)} attachment(s))" if attachments else ""
                dest = "Note-to-Self" if is_self else recipient
                print(f"   [Signal] Reply sent to {dest}{att_note}.")
        except Exception as e:
            print(f"   [Signal] Send error: {e}")

    # ── Execute (user-typed commands about Signal) ─────────────────

    def execute(self, command: str, context: dict) -> dict:
        cmd = command.lower()

        # "check signal messages now" → immediate poll
        if any(w in cmd for w in ("check", "now", "poll", "fetch")):
            running = bool(self._poll_thread and self._poll_thread.is_alive())
            if running:
                try:
                    self._check_messages()
                    response = "Checked Signal for new messages."
                except Exception as e:
                    response = f"Signal check failed: {e}"
            else:
                response = (
                    "Signal listener is not running. "
                    "Enable it in Settings → Talent Config → signal_remote."
                )
            return {"success": True, "response": response, "actions_taken": []}

        # Default: status report
        running = bool(self._poll_thread and self._poll_thread.is_alive())
        with self._lock:
            stats = dict(self._stats)

        cfg = self.talent_config
        prefix = cfg.get("command_prefix", "talon: ")
        interval = cfg.get("poll_interval", 10)
        authorized = self._get_authorized_numbers()
        account = cfg.get("account_number", "")
        masked = (account[:4] + "***" + account[-3:]) if len(account) > 7 else account

        lines = [
            f"Signal Remote: {'🟢 Running' if running else '🔴 Stopped'}",
            f"Account: {masked or '(not configured)'}",
            f"Poll interval: every {interval}s",
            f"Authorized numbers: {len(authorized)}",
            f"Command prefix: '{prefix}'",
            f"Messages received: {stats['messages_received']}",
            f"Commands processed: {stats['commands_processed']}",
        ]
        if stats["last_seen"]:
            ts = stats["last_seen"][:19].replace("T", " ")
            lines.append(f"Last message: {ts} from {stats['last_sender']}")

        return {
            "success": True,
            "response": "\n".join(lines),
            "actions_taken": [],
        }
