"""signal_remote.py — Remote control Talon via Signal (signal-cli direct mode).

Disabled by default. Enable via Settings → Talent Config → signal_remote,
then configure your phone number, signal-cli path, and authorized numbers.

Prerequisites (do once, manually):
  1. Install JRE 25+:  https://www.oracle.com/java/technologies/downloads/
  2. Download signal-cli from https://github.com/AsamK/signal-cli/releases
  3. Register:  signal-cli -a +1YOURNUM --config data/signal-cli-config register
  4. Verify:    signal-cli -a +1YOURNUM --config data/signal-cli-config verify CODE

Flow:
  - One persistent `signal-cli jsonRpc` process. Incoming messages arrive as
    JSON-RPC "receive" notifications on its stdout — no polling, no per-message
    JVM. The JVM (and its ~22 MB native-lib extraction) starts once.
  - Only Note-to-Self messages (syncMessage.sentMessage) with command_prefix are
    forwarded to assistant.process_command(); replies are "send" requests
    written to the process's stdin.
"""

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime

from talents.base import BaseTalent

import logging
log = logging.getLogger(__name__)


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
                {"key": "command_prefix",
                 "label": "Command Prefix",
                 "type": "string",
                 "default": "talon: "},
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
        # Dedicated JVM temp dir for signal-cli. libsignal extracts its ~22 MB
        # native lib (signal_jni_amd64.dll) to java.io.tmpdir on every single
        # invocation and never cleans up. We pin java.io.tmpdir here (see
        # _run_signal_cli) and purge it so the leak can never fill the drive.
        self._jni_tmp = os.path.join(tempfile.gettempdir(), "talon-signal-jni")
        self._stop_event = threading.Event()
        self._proc: subprocess.Popen | None = None   # persistent signal-cli jsonRpc
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._supervisor_thread: threading.Thread | None = None
        self._write_lock = threading.Lock()          # serialize stdin writes
        self._req_id = 0
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
        # Poll only when the talent is enabled (top-level toggle) AND the
        # listener is switched on in config. Gating on self.enabled matters:
        # set_assistant() runs for every talent regardless of the top-level
        # toggle, so without this check, turning the talent "off" left the
        # poll thread running and leaking a signal-cli JVM every interval.
        if self.enabled and self.talent_config.get("enabled", False):
            self._start_listener()

    def update_config(self, config: dict) -> None:
        """Called by GUI when the user saves talent config changes."""
        super().update_config(config)
        if self.enabled and config.get("enabled", False):
            self._restart_listener()
        else:
            self._stop_listener()

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    def _build_cmd(self, cli: str, args: list) -> list:
        """Build the command list to invoke signal-cli.

        For .bat wrappers we invoke java directly (bypassing cmd.exe) so that
        multiline message strings are not truncated at embedded newlines, which
        cmd.exe treats as command separators even inside quoted arguments.
        """
        if not cli.lower().endswith(".bat"):
            return [cli] + args

        import glob as _glob

        # Find java executable via JAVA_HOME (user or system env).
        java_home = os.environ.get("JAVA_HOME", "")
        java_exe = (os.path.join(java_home, "bin", "java.exe")
                    if java_home else "java.exe")

        # Derive app_home: .../bin/signal-cli.bat  →  .../
        bat_dir = os.path.dirname(os.path.abspath(cli))
        app_home = os.path.dirname(bat_dir)
        lib_dir = os.path.join(app_home, "lib")

        all_jars = _glob.glob(os.path.join(lib_dir, "*.jar"))
        if not all_jars:
            # Fallback if we can't find the jars (shouldn't happen).
            return ["cmd", "/c", cli] + args

        classpath = ";".join(all_jars)

        return [
            java_exe,
            "--enable-native-access=ALL-UNNAMED",
            # Pin native-lib extraction to our managed dir so _run_signal_cli
            # can purge it (belt-and-suspenders with the TEMP/TMP env below).
            f"-Djava.io.tmpdir={self._jni_tmp}",
            "-Xms32m",   # start small — signal-cli is a short-lived CLI tool
            "-Xmx128m",  # cap heap; default is ~1 GB which exhausts the page file
            "-classpath", classpath,
            "org.asamk.signal.Main",
        ] + args

    # ── signal-cli invocation (leak-contained) ─────────────────────

    def _purge_jni_tmp(self) -> None:
        """Delete libsignal* native-lib extraction dirs from the managed tmp.

        A DLL still loaded by a concurrent/killed JVM stays locked on Windows;
        those dirs are skipped (ignore_errors) and cleared on the next pass.
        """
        try:
            for name in os.listdir(self._jni_tmp):
                if name.startswith("libsignal"):
                    shutil.rmtree(os.path.join(self._jni_tmp, name),
                                  ignore_errors=True)
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def _run_signal_cli(self, args: list, timeout: int):
        """Run signal-cli with its JVM temp pinned to a managed dir, purging
        the leaked native-lib extraction before and after.

        Returns the CompletedProcess. Exceptions (FileNotFoundError,
        TimeoutExpired, OSError) propagate so callers handle them as before.
        """
        cli = self.talent_config.get("signal_cli_path", "signal-cli")
        os.makedirs(self._jni_tmp, exist_ok=True)
        self._purge_jni_tmp()  # sweep any leftover a prior locked run left
        env = dict(os.environ)
        # Java derives java.io.tmpdir from TMP/TEMP on Windows; redirect both
        # so even the plain-signal-cli (non-.bat) path extracts into our dir.
        env["TEMP"] = env["TMP"] = env["TMPDIR"] = self._jni_tmp
        try:
            return subprocess.run(
                self._build_cmd(cli, args),
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=timeout, env=env,
            )
        finally:
            self._purge_jni_tmp()

    # ── Thread management ──────────────────────────────────────────

    def _kill_orphan_signal_processes(self) -> None:
        """Kill any Java processes running signal-cli left from previous sessions."""
        try:
            result = subprocess.run(
                ["wmic", "process", "where",
                 "name='java.exe' and commandline like '%signal-cli%'",
                 "get", "processid", "/value"],
                capture_output=True, text=True, timeout=10,
            )
            pids = [
                line.split("=")[1].strip()
                for line in result.stdout.splitlines()
                if line.strip().startswith("ProcessId=")
            ]
            for pid in pids:
                subprocess.run(["taskkill", "/F", "/T", "/PID", pid],
                                capture_output=True)
                log.info(f"[Signal] Killed orphaned signal-cli JVM (PID {pid}).")
            if pids:
                time.sleep(1.0)
        except Exception:
            pass

    def _start_listener(self) -> None:
        """Spawn the persistent signal-cli jsonRpc process and its threads."""
        if self._proc and self._proc.poll() is None:
            return  # already running
        if not self._validate_config():
            return
        self._kill_orphan_signal_processes()
        self._stop_event.clear()
        if not self._spawn_process():
            return
        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="signal-rpc-reader")
        self._reader_thread.start()
        self._supervisor_thread = threading.Thread(
            target=self._supervise, daemon=True, name="signal-rpc-supervisor")
        self._supervisor_thread.start()
        log.info("[Signal] JSON-RPC listener started (persistent process).")

    def _stop_listener(self) -> None:
        """Signal shutdown and terminate the jsonRpc process."""
        self._stop_event.set()
        proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass
        self._purge_jni_tmp()
        log.info("[Signal] JSON-RPC listener stopped.")

    def _restart_listener(self) -> None:
        self._stop_listener()
        time.sleep(0.25)
        self._start_listener()

    # ── Persistent JSON-RPC transport ──────────────────────────────

    def _spawn_process(self) -> bool:
        """Launch one long-lived `signal-cli jsonRpc` process.

        Replaces the old poll-a-fresh-JVM-every-interval design: the JVM (and
        its ~22 MB native-lib extraction) starts once, incoming messages arrive
        as push notifications on stdout, and replies are written to stdin.
        """
        cfg = self.talent_config
        cli = cfg.get("signal_cli_path", "signal-cli")
        config_dir = cfg.get("config_dir", "data/signal-cli-config")
        account = cfg.get("account_number", "")

        os.makedirs(self._jni_tmp, exist_ok=True)
        self._purge_jni_tmp()
        env = dict(os.environ)
        env["TEMP"] = env["TMP"] = env["TMPDIR"] = self._jni_tmp

        args = ["--config", config_dir, "-a", account,
                "jsonRpc", "--receive-mode=on-start", "--ignore-stories"]
        try:
            self._proc = subprocess.Popen(
                self._build_cmd(cli, args),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                errors="replace", bufsize=1, env=env,
            )
        except (FileNotFoundError, OSError) as e:
            log.error(f"[Signal] Failed to launch jsonRpc process: {e}")
            self._proc = None
            return False

        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(self._proc,),
            daemon=True, name="signal-rpc-stderr")
        self._stderr_thread.start()
        log.info(f"[Signal] signal-cli jsonRpc running (pid {self._proc.pid}).")
        return True

    def _read_loop(self) -> None:
        """Read newline-delimited JSON-RPC from stdout and dispatch each line."""
        proc = self._proc
        if not proc or not proc.stdout:
            return
        while not self._stop_event.is_set():
            try:
                line = proc.stdout.readline()
            except Exception as e:
                log.error(f"[Signal] stdout read error: {e}")
                break
            if line == "":          # EOF: process exited
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                self._dispatch_rpc(msg)
            except Exception as e:
                log.error(f"[Signal] dispatch error: {e}")

    def _dispatch_rpc(self, msg: dict) -> None:
        """Route one JSON-RPC message: 'receive' notifications and errors."""
        if msg.get("method") == "receive":
            params = msg.get("params") or {}
            # on-start mode: params holds 'envelope'; manual mode nests it under
            # 'result'. Unwrap so _handle_envelope always sees {'envelope': ...}.
            if "envelope" not in params and isinstance(params.get("result"), dict):
                params = params["result"]
            self._handle_envelope(params)
        elif msg.get("error") is not None:
            log.error(f"[Signal] RPC error (id={msg.get('id')}): {msg['error']}")

    def _rpc_send(self, method: str, params: dict) -> bool:
        """Write one JSON-RPC request to the process's stdin."""
        proc = self._proc
        if not proc or proc.poll() is not None or not proc.stdin:
            log.error("[Signal] Cannot send: jsonRpc process not running.")
            return False
        with self._write_lock:
            self._req_id += 1
            req = {"jsonrpc": "2.0", "method": method,
                   "params": params, "id": self._req_id}
            try:
                proc.stdin.write(json.dumps(req) + "\n")
                proc.stdin.flush()
                return True
            except (BrokenPipeError, OSError) as e:
                log.error(f"[Signal] Failed writing to jsonRpc stdin: {e}")
                return False

    def _drain_stderr(self, proc) -> None:
        """Forward signal-cli's stderr logging into our logger."""
        stream = getattr(proc, "stderr", None)
        if stream is None:
            return
        try:
            for line in stream:
                line = line.rstrip()
                if not line:
                    continue
                if "ERROR" in line or "Exception" in line:
                    log.error(f"[Signal][cli] {line}")
                elif "WARN" in line:
                    log.warning(f"[Signal][cli] {line}")
                else:
                    log.debug(f"[Signal][cli] {line}")
        except Exception:
            pass

    def _supervise(self) -> None:
        """Restart the jsonRpc process if it dies unexpectedly, with backoff."""
        backoff = 2
        fast_failures = 0
        while not self._stop_event.is_set():
            proc = self._proc
            if proc is None:
                break
            started = time.monotonic()
            try:
                rc = proc.wait()
            except Exception:
                break
            if self._stop_event.is_set():
                break
            ran_for = time.monotonic() - started
            fast_failures = fast_failures + 1 if ran_for < 15 else 0
            if fast_failures >= 5:
                log.error("[Signal] jsonRpc keeps exiting immediately; giving "
                          "up. Check account registration and signal-cli path.")
                break
            log.error(f"[Signal] jsonRpc exited (code {rc}) after {ran_for:.0f}s; "
                      f"restarting in {backoff}s.")
            if self._stop_event.wait(backoff):
                break
            if not (self.enabled and self.talent_config.get("enabled", False)):
                break
            if self._spawn_process():
                self._reader_thread = threading.Thread(
                    target=self._read_loop, daemon=True, name="signal-rpc-reader")
                self._reader_thread.start()
                backoff = 2
            else:
                fast_failures += 1
                backoff = min(backoff * 2, 60)
        log.info("[Signal] supervisor exited.")

    # ── Message handling ───────────────────────────────────────────

    def _handle_envelope(self, envelope: dict) -> None:
        """Inspect one received envelope and process it if it's a valid command."""
        cfg = self.talent_config

        try:
            inner = envelope.get("envelope", {})
            sender = inner.get("source", "")

            # Only process Note-to-Self (syncMessage.sentMessage).
            # These arrive when the user sends a message to their own number
            # from their phone; linked devices receive it as a sync event.
            sync_msg = inner.get("syncMessage") or {}
            sent_msg = sync_msg.get("sentMessage") or {}
            text = (sent_msg.get("message") or "").strip()
            sync_dest = sent_msg.get("destination") or sender
            # Extract incoming attachments (images, files)
            incoming_attachments = sent_msg.get("attachments") or []
        except (AttributeError, TypeError):
            return

        # Skip empty envelopes
        if not text and not incoming_attachments:
            return

        # Authorization boundary: Note-to-Self only.
        # We deliberately do not keep an allowlist of authorized contacts. An
        # allowlist would *widen* access by letting other numbers drive the
        # assistant. Instead we require the message to be a sync of one the
        # account sent to itself, which the destination==sender check below
        # establishes: a syncMessage.sentMessage only reaches a linked device
        # when the account's own primary device sent it, and we further demand
        # its destination be that same account. So the sole party who can issue
        # commands is whoever holds the linked Signal account — the owner.
        if not sync_dest or sync_dest != sender:
            return

        # Prefix check (case-insensitive).
        prefix = cfg.get("command_prefix", "talon: ").lower()
        has_prefix = text.lower().startswith(prefix)

        # Images in Note-to-Self are allowed without prefix
        has_images = any(
            (a.get("contentType") or "").startswith("image/")
            for a in incoming_attachments
        )
        if not has_prefix and not has_images:
            return

        command = text[len(prefix):].strip() if text.lower().startswith(prefix) else text.strip()
        if not command and not has_images:
            return

        # Update stats
        now_iso = datetime.now().isoformat()
        with self._lock:
            self._stats["messages_received"] += 1
            self._stats["last_seen"] = now_iso
            self._stats["last_sender"] = sender

        log.info(f"[Signal] Command from {sender}: {command!r}")

        # Semantic injection check on the raw Signal command before processing
        from core.security import get_security_filter as _gsf
        _sf = _gsf()
        if _sf:
            _blocked, _alert = _sf.check_semantic_input(command, "signal_in")
            if _blocked:
                log.info(f"[Signal] Command blocked by semantic classifier: {command!r}")
                return

        if self._assistant is None:
            log.error("[Signal] Assistant not available, cannot process command.")
            return

        # Resolve incoming image attachments to local file paths
        image_paths = []
        config_dir = cfg.get("config_dir", "data/signal-cli-config")
        for att in incoming_attachments:
            # signal-cli stores attachments with an "id" field; the file
            # lives at <config_dir>/attachments/<id>
            att_id = att.get("id")
            att_type = att.get("contentType", "")
            if att_id and att_type.startswith("image/"):
                att_path = os.path.join(config_dir, "attachments", att_id)
                if os.path.exists(att_path):
                    image_paths.append(att_path)
                    log.info(f"[Signal] Received image attachment: {att_id}")
                else:
                    log.warning(f"[Signal] Attachment file missing: {att_path}")

        result = {}
        try:
            result = self._assistant.process_command(
                command or "describe this image",
                speak_response=False,
                _executing_rule=True,
                command_source="signal",
                attachments=image_paths or None,
            )
            response = (result.get("response") or "").strip()
        except Exception as e:
            response = f"Error processing command: {e}"
            log.error(f"[Signal] process_command error: {e}")

        # Truncate if needed
        max_chars = int(cfg.get("max_response_chars", 1000))
        if len(response) > max_chars:
            response = response[:max_chars - 3] + "..."

        # Collect file attachments (e.g. screenshots)
        attachments = []
        for action_result in (result.get("actions_taken") or []):
            ar = action_result.get("result", "")
            if isinstance(ar, str) and ar.startswith("Screenshot: "):
                path = ar[len("Screenshot: "):].strip()
                if path and os.path.exists(path):
                    attachments.append(path)

        with self._lock:
            self._stats["commands_processed"] += 1

        reply_to = sync_dest or sender
        self._send_reply(reply_to, response or "(no response)", attachments=attachments)

    def _send_reply(self, recipient: str, message: str,
                    attachments: list | None = None) -> None:
        """Send a reply as a 'send' request over the persistent jsonRpc stdin."""
        account = self.talent_config.get("account_number", "")
        is_self = not recipient or recipient == account

        params: dict = {"message": message}
        if is_self:
            params["noteToSelf"] = True
        else:
            params["recipient"] = [recipient]
        if attachments:
            params["attachment"] = list(attachments)

        if self._rpc_send("send", params):
            att_note = (f" (+{len(attachments)} attachment(s))"
                        if attachments else "")
            dest = "Note-to-Self" if is_self else recipient
            log.info(f"[Signal] Reply queued to {dest}{att_note}.")

    # ── Validation ─────────────────────────────────────────────────

    def _validate_config(self) -> bool:
        cfg = self.talent_config

        if self._assistant is None:
            log.error("[Signal] Cannot start: assistant not set yet.")
            return False

        if not cfg.get("account_number", "").strip():
            log.error("[Signal] Cannot start: account_number not configured.")
            return False

        cli = cfg.get("signal_cli_path", "signal-cli")
        try:
            r = self._run_signal_cli(["--version"], timeout=15)
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "").strip()[:300]
                log.error(f"[Signal] Cannot start: signal-cli --version failed "
                      f"(exit {r.returncode}) at {cli!r}.")
                if err:
                    log.info(f"[Signal] Output: {err}")
                return False
        except FileNotFoundError:
            log.error(f"[Signal] Cannot start: signal-cli not found at {cli!r}.")
            return False
        except subprocess.TimeoutExpired:
            log.error(f"[Signal] Cannot start: signal-cli --version timed out at {cli!r}.")
            return False
        except OSError as e:
            log.error(f"[Signal] Cannot start: OS error running {cli!r}: {e}")
            return False

        return True

    # ── Execute (status / manual poll) ────────────────────────────

    def execute(self, command: str, context: dict) -> dict:
        cmd = command.lower()
        alive = bool(self._proc and self._proc.poll() is None)

        # Messages arrive automatically over the persistent JSON-RPC stream, so
        # there is no manual poll — just report listener state.
        if any(w in cmd for w in ("check", "now", "poll", "fetch")):
            if alive:
                response = ("Signal listener is running. Messages arrive "
                            "automatically over the JSON-RPC stream.")
            else:
                response = ("Signal listener is not running. "
                            "Enable it in Settings → Talent Config → signal_remote.")
            return {"success": True, "response": response, "actions_taken": []}

        # Default: status report
        with self._lock:
            stats = dict(self._stats)

        cfg = self.talent_config
        prefix = cfg.get("command_prefix", "talon: ")
        account = cfg.get("account_number", "")
        masked = (account[:4] + "***" + account[-3:]) if len(account) > 7 else account

        status = "🟢 Running" if alive else "🔴 Stopped"
        lines = [
            f"Signal Remote: {status}",
            f"Account: {masked or '(not configured)'}",
            "Mode: JSON-RPC (push, persistent process)",
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
