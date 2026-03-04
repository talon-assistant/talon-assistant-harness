"""signal_remote.py — Remote control Talon via Signal (signal-cli daemon mode).

Disabled by default. Enable via Settings → Talent Config → signal_remote,
then configure your phone number, signal-cli path, and authorized numbers.

Prerequisites (do once, manually):
  1. Install JRE 21+:  winget install Microsoft.OpenJDK.21
  2. Download signal-cli from https://github.com/AsamK/signal-cli/releases
  3. Register:  signal-cli -a +1YOURNUM --config data/signal-cli-config register
  4. Verify:    signal-cli -a +1YOURNUM --config data/signal-cli-config verify CODE

Flow:
  - Starts signal-cli in daemon mode (persistent JVM, TCP JSON-RPC on daemon_port)
  - Background thread calls daemon's `receive` endpoint every poll_interval seconds
  - Messages from authorized numbers starting with command_prefix are forwarded to
    assistant.process_command(); replies are sent back via daemon's `send` endpoint
"""

import json
import os
import socket
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
                {"key": "daemon_port",
                 "label": "Daemon TCP Port",
                 "type": "int",
                 "default": 7583,
                 "min": 1024,
                 "max": 65535},
                {"key": "poll_interval",
                 "label": "Poll Interval (seconds)",
                 "type": "int",
                 "default": 5,
                 "min": 2,
                 "max": 60},
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
        self._daemon_proc: subprocess.Popen | None = None
        self._rpc_id = 0
        self._lock = threading.Lock()
        self._stats: dict = {
            "messages_received": 0,
            "commands_processed": 0,
            "last_seen": None,
            "last_sender": None,
        }

    def set_assistant(self, assistant) -> None:
        """Called by TalonAssistant.__init__() after talent discovery."""
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

    # ── Thread & daemon management ─────────────────────────────────

    def _start_polling(self) -> None:
        if not self._validate_config():
            return
        if not self._start_daemon():
            return
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="signal-remote-poll",
        )
        self._poll_thread.start()
        print("   [Signal] Poll thread started.")

    def _stop_polling(self) -> None:
        self._stop_event.set()
        self._stop_daemon()

    def _restart_polling(self) -> None:
        self._stop_polling()
        time.sleep(0.5)
        self._start_polling()

    def _start_daemon(self) -> bool:
        """Launch signal-cli in daemon mode. Returns True when the TCP port is ready."""
        cfg = self.talent_config
        cli = cfg.get("signal_cli_path", "signal-cli")
        config_dir = cfg.get("config_dir", "data/signal-cli-config")
        account = cfg.get("account_number", "")
        port = int(cfg.get("daemon_port", 7583))

        cmd = [
            cli,
            "--config", config_dir,
            "-a", account,
            "daemon",
            "--tcp", f"localhost:{port}",
        ]

        print(f"   [Signal] Starting daemon on port {port}...")
        try:
            self._daemon_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as e:
            print(f"   [Signal] Failed to launch daemon: {e}")
            return False

        # Wait up to 10s for the TCP port to become ready
        for _ in range(20):
            time.sleep(0.5)
            if self._daemon_proc.poll() is not None:
                # Process already exited
                try:
                    stderr = self._daemon_proc.stderr.read().decode(errors="replace")[:500]
                except Exception:
                    stderr = "(unreadable)"
                print(f"   [Signal] Daemon exited early: {stderr}")
                return False
            try:
                with socket.create_connection(("localhost", port), timeout=1):
                    pass
                print(f"   [Signal] Daemon ready on port {port}.")
                return True
            except (ConnectionRefusedError, OSError):
                pass

        print("   [Signal] Daemon did not become ready within 10s.")
        self._stop_daemon()
        return False

    def _stop_daemon(self) -> None:
        if self._daemon_proc is not None:
            try:
                self._daemon_proc.terminate()
                self._daemon_proc.wait(timeout=5)
            except Exception:
                try:
                    self._daemon_proc.kill()
                except Exception:
                    pass
            self._daemon_proc = None
            print("   [Signal] Daemon stopped.")

    # ── JSON-RPC client ────────────────────────────────────────────

    def _rpc_call(self, method: str, params: dict | None = None,
                  timeout: float = 10.0) -> object:
        """Send a JSON-RPC 2.0 request to the daemon and return the result.

        Opens a fresh TCP connection per call (cheap on localhost).
        Raises RuntimeError on RPC-level errors.
        """
        with self._lock:
            self._rpc_id += 1
            rpc_id = self._rpc_id

        port = int(self.talent_config.get("daemon_port", 7583))
        request: dict = {"jsonrpc": "2.0", "method": method, "id": rpc_id}
        if params:
            request["params"] = params

        payload = (json.dumps(request) + "\n").encode()

        with socket.create_connection(("localhost", port), timeout=timeout) as sock:
            sock.sendall(payload)

            # Read response (newline-delimited)
            buf = b""
            sock.settimeout(timeout)
            while b"\n" not in buf:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk

        response = json.loads(buf.decode().strip())

        if "error" in response:
            raise RuntimeError(f"RPC error: {response['error']}")

        return response.get("result")

    # ── Polling loop ───────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Background daemon thread: poll Signal periodically via JSON-RPC."""
        interval = max(2, int(self.talent_config.get("poll_interval", 5)))
        while not self._stop_event.wait(interval):
            # Health check — restart daemon if it died
            if self._daemon_proc is None or self._daemon_proc.poll() is not None:
                print("   [Signal] Daemon died — restarting...")
                if not self._start_daemon():
                    print("   [Signal] Daemon restart failed; backing off 30s.")
                    self._stop_event.wait(30)
                    continue

            try:
                self._check_messages()
            except Exception as e:
                print(f"   [Signal] Poll error: {e}")

    def _check_messages(self) -> None:
        """Call daemon `receive` and dispatch any incoming envelopes."""
        try:
            envelopes = self._rpc_call("receive", {"timeout": 3}) or []
        except Exception as e:
            print(f"   [Signal] receive RPC failed: {e}")
            return

        if not isinstance(envelopes, list):
            return

        for envelope in envelopes:
            try:
                self._handle_envelope(envelope)
            except Exception as e:
                print(f"   [Signal] Envelope error: {e}")

    # ── Message handling ───────────────────────────────────────────

    def _handle_envelope(self, envelope: dict) -> None:
        """Inspect one received envelope and process it if it's a valid command."""
        cfg = self.talent_config

        try:
            inner = envelope.get("envelope", {})
            sender = inner.get("source", "")

            # dataMessage: regular message from another account
            data_msg = inner.get("dataMessage") or {}
            text = (data_msg.get("message") or "").strip()

            # syncMessage.sentMessage: Note-to-Self — the user messaged their own
            # number from their phone; delivered to linked devices as a sync event.
            sync_dest = None
            if not text:
                sync_msg = inner.get("syncMessage") or {}
                sent_msg = sync_msg.get("sentMessage") or {}
                text = (sent_msg.get("message") or "").strip()
                if text:
                    sync_dest = sent_msg.get("destination") or sender
        except (AttributeError, TypeError):
            return

        # 1. Skip non-text envelopes (receipts, typing indicators, calls…)
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
            return   # normal Signal chat — ignore silently

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
        #    _executing_rule=True — prevents conversation buffer pollution
        #    speak_response=False — user isn't at the PC
        if self._assistant is None:
            print("   [Signal] Assistant not available, cannot process command.")
            return

        result = {}
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

        # 7. Collect file attachments from actions_taken (e.g. screenshots)
        attachments = []
        for action_result in (result.get("actions_taken") or []):
            ar = action_result.get("result", "")
            if isinstance(ar, str) and ar.startswith("Screenshot: "):
                path = ar[len("Screenshot: "):].strip()
                if path and os.path.exists(path):
                    attachments.append(path)

        with self._lock:
            self._stats["commands_processed"] += 1

        # 8. Send reply — for Note-to-Self, reply to sync_dest
        reply_to = sync_dest or sender
        self._send_reply(reply_to, response or "(no response)", attachments=attachments)

    def _send_reply(self, recipient: str, message: str,
                    attachments: list | None = None) -> None:
        """Send a Signal message back to the sender via daemon JSON-RPC."""
        account = self.talent_config.get("account_number", "")
        is_self = not recipient or recipient == account

        if is_self:
            params: dict = {"noteToSelf": True, "message": message}
        else:
            params = {"recipient": [recipient], "message": message}

        if attachments:
            params["attachment"] = attachments

        try:
            self._rpc_call("send", params, timeout=30)
            att_note = f" (+{len(attachments)} attachment(s))" if attachments else ""
            dest = "Note-to-Self" if is_self else recipient
            print(f"   [Signal] Reply sent to {dest}{att_note}.")
        except Exception as e:
            print(f"   [Signal] Send error: {e}")

    # ── Validation ─────────────────────────────────────────────────

    def _validate_config(self) -> bool:
        """Check prerequisites. Returns False and logs if anything is missing."""
        cfg = self.talent_config

        if self._assistant is None:
            print("   [Signal] Cannot start: assistant not set yet.")
            return False

        if not cfg.get("account_number", "").strip():
            print("   [Signal] Cannot start: account_number not configured.")
            return False

        if not self._get_authorized_numbers():
            print("   [Signal] Cannot start: no authorized_numbers configured.")
            return False

        cli = cfg.get("signal_cli_path", "signal-cli")
        try:
            r = subprocess.run(
                [cli, "--version"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
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

    # ── Execute (status / manual poll) ────────────────────────────

    def execute(self, command: str, context: dict) -> dict:
        cmd = command.lower()

        # "check signal messages now" → immediate poll
        if any(w in cmd for w in ("check", "now", "poll", "fetch")):
            daemon_alive = (self._daemon_proc is not None
                            and self._daemon_proc.poll() is None)
            if daemon_alive:
                try:
                    self._check_messages()
                    response = "Checked Signal for new messages."
                except Exception as e:
                    response = f"Signal check failed: {e}"
            else:
                response = (
                    "Signal daemon is not running. "
                    "Enable it in Settings → Talent Config → signal_remote."
                )
            return {"success": True, "response": response, "actions_taken": []}

        # Default: status report
        daemon_alive = (self._daemon_proc is not None
                        and self._daemon_proc.poll() is None)
        thread_alive = bool(self._poll_thread and self._poll_thread.is_alive())
        with self._lock:
            stats = dict(self._stats)

        cfg = self.talent_config
        prefix = cfg.get("command_prefix", "talon: ")
        interval = cfg.get("poll_interval", 5)
        port = cfg.get("daemon_port", 7583)
        authorized = self._get_authorized_numbers()
        account = cfg.get("account_number", "")
        masked = (account[:4] + "***" + account[-3:]) if len(account) > 7 else account

        status = "🟢 Running" if (daemon_alive and thread_alive) else "🔴 Stopped"
        lines = [
            f"Signal Remote: {status}",
            f"Account: {masked or '(not configured)'}",
            f"Daemon port: {port} ({'up' if daemon_alive else 'down'})",
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
