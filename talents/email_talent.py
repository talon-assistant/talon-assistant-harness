"""EmailTalent — check, read, send, reply, delete, and manage email via IMAP/SMTP.

Uses stdlib imaplib + smtplib (no third-party mail libraries).
Credentials are managed centrally by the CredentialStore (OS keyring).

Examples:
    "check my email"
    "read latest email from John"
    "read email 2"
    "read me the email about the hockey game"
    "how many unread emails do I have"
    "send email to john@example.com about the meeting"
    "reply to the last email from Alice"
    "delete email 3"
    "move email from Bob to Archive"
    "list my email folders"
"""

import os
import re
import html as _html_mod
import json
import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from datetime import datetime
from talents.base import BaseTalent
from core.llm_client import LLMError
from core.security import wrap_external, INJECTION_DEFENSE_CLAUSE

import logging
log = logging.getLogger(__name__)


class EmailTalent(BaseTalent):
    name = "email"
    description = "Check, read, send, reply, delete, and manage email via IMAP/SMTP"
    keywords = [
        "email", "mail", "inbox", "unread", "send email",
        "compose", "check email", "read email", "new email",
        "send a message", "send message to",
        "reply", "reply to", "respond to",
        "delete email", "remove email",
        "move email", "archive email",
        "list folders", "show folders",
    ]
    examples = [
        "check my email",
        "how many unread emails do I have",
        "read latest email from John",
        "read email 2",
        "read me the email about the hockey game",
        "send email to alice about the meeting",
        "reply to the last email from Bob",
        "delete email 3",
        "list my email folders",
        "move email to archive",
    ]
    priority = 55

    # Native tool-calling: typed parameters so the model fills structured
    # fields in their own slots instead of cramming everything into one
    # free-form string and dropping the recipient address (the failure that
    # motivated this migration).
    tool_parameters = {
        "action": {
            "type": "string",
            "enum": ["send", "check", "read", "reply", "delete", "move"],
            "description": ("What to do: 'send' a new email, 'check' the inbox, "
                            "'read' a message, 'reply' to one, 'delete' or "
                            "'move' one."),
        },
        "to": {"type": "string",
               "description": "Recipient email address. Required when action is 'send'."},
        "subject": {"type": "string", "description": "Subject line (for 'send')."},
        "body": {"type": "string",
                 "description": "Body text of the email to send (for 'send')."},
        "attachment_path": {"type": "string",
                            "description": "Absolute path of a local file to attach (for 'send')."},
        "query": {"type": "string",
                  "description": ("Which message to act on for read/reply/delete/move — "
                                  "e.g. 'from Bob', 'about the invoice', or an index like '2'.")},
        "folder": {"type": "string", "description": "Target folder name (for 'move')."},
    }
    tool_required = ["action"]

    _EMAIL_PHRASES = [
        "email", "mail", "inbox", "unread",
        "send email", "compose email", "send a message",
        "check email", "check my email", "read email",
        "new email", "latest email", "read my mail",
        "reply", "delete email", "list folders",
    ]

    _SYSTEM_PROMPT_READ = (
        "You are an email summarizer. "
        "Using ONLY the email data provided, give a concise summary. "
        "Include sender name, subject, and a 1-2 sentence summary of the body. "
        "Do NOT add information not in the data."
        + INJECTION_DEFENSE_CLAUSE
    )

    _SYSTEM_PROMPT_COMPOSE = (
        "You are an email composition assistant. "
        "Given the user's request, generate ONLY a JSON object with these keys:\n"
        '  "to": email address (string)\n'
        '  "subject": email subject line (string)\n'
        '  "body": email body text (string)\n'
        "\n"
        "Write the body in a professional, friendly tone unless the user specifies otherwise. "
        "Keep it concise. Return ONLY the JSON object, no markdown, no explanation."
    )

    _SYSTEM_PROMPT_REPLY = (
        "You are an email reply assistant. "
        "Given the original email and the user's instruction, write ONLY the reply body text. "
        "Do not repeat or quote the original email. Keep it concise and professional. "
        "Return ONLY the reply body text — no subject line, no greeting headers. "
        "IMPORTANT: You are writing a reply ON BEHALF of the recipient (the user), "
        "NOT as the original sender. Never use the sender's name, title, or signature. "
        "Do not add any sign-off name or signature — the user will add their own."
        + INJECTION_DEFENSE_CLAUSE
    )

    def __init__(self):
        super().__init__()
        self._imap = None
        self._connected = False
        # Cache of last fetched email list for index/subject-based addressing.
        # Keyed by IMAP UID (stable across reconnects) rather than sequence
        # number, which shifts when messages are expunged between commands.
        # Each entry: {uid, from, subject, date, message_id}
        self._email_cache: list[dict] = []

    # ── Config schema ──────────────────────────────────────────────

    def get_config_schema(self) -> dict:
        return {
            "fields": [
                {"key": "imap_server", "label": "IMAP Server",
                 "type": "string", "default": ""},
                {"key": "imap_port", "label": "IMAP Port",
                 "type": "int", "default": 993, "min": 1, "max": 65535},
                {"key": "smtp_server", "label": "SMTP Server",
                 "type": "string", "default": ""},
                {"key": "smtp_port", "label": "SMTP Port",
                 "type": "int", "default": 587, "min": 1, "max": 65535},
                {"key": "username", "label": "Email Address",
                 "type": "string", "default": ""},
                {"key": "password", "label": "Password / App Password",
                 "type": "password", "default": ""},
                {"key": "max_fetch", "label": "Max Emails to Fetch",
                 "type": "int", "default": 5, "min": 1, "max": 50},
            ]
        }

    def update_config(self, config: dict) -> None:
        """Store config.  Credential storage is handled centrally by the
        CredentialStore via the bridge — no keyring logic needed here."""
        self._config = config
        # Force reconnect next time
        self._disconnect()

    # ── Routing ────────────────────────────────────────────────────

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    # ── Execution ──────────────────────────────────────────────────

    def execute(self, command: str, context: dict) -> dict:
        cmd_lower = command.lower()

        # Check if email is configured
        if not self._config.get("imap_server") or not self._config.get("username"):
            return {
                "success": False,
                "response": "Email is not configured yet. Please set up your IMAP/SMTP "
                            "settings in the email talent configuration (click the gear icon).",
                "actions_taken": [{"action": "email_not_configured"}],
                "spoken": False,
            }

        # Native tool-calling path: structured args (action + typed fields)
        # take priority over keyword-sniffing the command string.
        args = context.get("tool_args") or {}
        action = (args.get("action") or "").lower().strip()
        if action:
            return self._route_typed(action, args, command, context)

        # Route to the right sub-command (most specific first)
        if any(w in cmd_lower for w in ["delete", "trash", "remove"]) and \
                any(w in cmd_lower for w in ["email", "mail", "message"]):
            return self._handle_delete(command, context)
        elif any(w in cmd_lower for w in ["reply", "respond"]):
            return self._handle_reply(command, context)
        elif any(w in cmd_lower for w in ["folder", "folders", "move to", "archive",
                                           "move email", "move message"]):
            return self._handle_folders(command, context)
        elif any(w in cmd_lower for w in ["send", "compose", "write"]) or \
                self._extract_attach_paths(command):
            # Also route to send when the command contains a real file path —
            # e.g. when {last_result} expansion injects a digest path alongside
            # words like "from" that would otherwise match the read branch.
            return self._handle_send(command, context)
        elif any(w in cmd_lower for w in ["read", "latest", "last", "recent",
                                           "from", "about", "regarding"]):
            return self._handle_read(command, context)
        else:
            # Default: check inbox (unread count + summaries)
            return self._handle_check(command, context)

    def _route_typed(self, action, args, command, context):
        """Dispatch a native tool-call (typed args) to the right handler.

        'send' passes the structured fields straight through so the recipient
        is never re-extracted (and lost). read/reply/delete/move hand the
        'query' string to the existing resolver-based handlers.
        """
        if action == "send":
            return self._handle_send(command, context, typed=args)
        q = (args.get("query") or "").strip()
        ref_command = q or command
        if action == "reply":
            return self._handle_reply(ref_command, context)
        if action == "delete":
            return self._handle_delete(ref_command, context)
        if action == "move":
            folder = (args.get("folder") or "").strip()
            move_command = f"move {q} to {folder}".strip() if folder else ref_command
            return self._handle_move(move_command, context)
        if action == "read":
            return self._handle_read(ref_command, context)
        # check / unknown → inbox check
        return self._handle_check(command, context)

    # ── Check inbox ────────────────────────────────────────────────

    def _handle_check(self, command, context):
        """Show unread count and brief summaries of recent emails."""
        try:
            imap = self._connect_imap()
            imap.select("INBOX", readonly=True)

            # Get unread count
            status, data = imap.uid('search', None, "UNSEEN")
            unread_ids = data[0].split() if data[0] else []
            unread_count = len(unread_ids)

            max_fetch = self._config.get("max_fetch", 5)

            # Fetch the most recent unread (or all recent if none unread)
            if unread_ids:
                fetch_ids = unread_ids[-max_fetch:]
            else:
                status, data = imap.uid('search', None, "ALL")
                all_ids = data[0].split() if data[0] else []
                fetch_ids = all_ids[-max_fetch:] if all_ids else []

            if not fetch_ids:
                self._disconnect()
                return {
                    "success": True,
                    "response": "Your inbox is empty.",
                    "actions_taken": [{"action": "email_check", "unread": 0}],
                    "spoken": False,
                }

            summaries = self._fetch_summaries(imap, fetch_ids)
            self._disconnect()

            # Populate index cache for subsequent commands
            self._email_cache = [
                {
                    "uid":        mid,
                    "from":       s["from"],
                    "subject":    s["subject"],
                    "date":       s["date"],
                    "message_id": s.get("message_id", ""),
                }
                for mid, s in zip(fetch_ids, summaries)
            ]

            # Build response
            header = f"You have {unread_count} unread email{'s' if unread_count != 1 else ''}.\n"
            if summaries:
                header += f"\nHere are the {'unread' if unread_ids else 'most recent'} messages:\n\n"
                for i, s in enumerate(summaries, 1):
                    header += f"{i}. From: {s['from']}\n   Subject: {s['subject']}\n   Date: {s['date']}\n\n"

            return {
                "success": True,
                "response": header.strip(),
                "actions_taken": [{"action": "email_check", "unread": unread_count}],
                "spoken": False,
            }

        except Exception as e:
            self._disconnect()
            return {
                "success": False,
                "response": f"Error checking email: {str(e)}",
                "actions_taken": [{"action": "email_check_error"}],
                "spoken": False,
            }

    # ── Read specific email ────────────────────────────────────────

    def _handle_read(self, command, context):
        """Read and summarize a specific email.

        Resolves by: numbered index > sender > subject keywords > most recent.
        """
        try:
            imap = self._connect_imap()
            imap.select("INBOX", readonly=True)

            # Try resolving from cache first (index / sender / subject keywords)
            cache_entry = self._resolve_email_ref(command)
            if cache_entry:
                fetch_ids = [cache_entry["uid"]]
            else:
                # Cold start — fall back to sender filter or most recent
                sender_filter = self._extract_sender(command)
                if sender_filter:
                    status, data = imap.uid('search', None, f'FROM "{sender_filter}"')
                else:
                    status, data = imap.uid('search', None, "ALL")
                msg_ids = data[0].split() if data[0] else []
                if not msg_ids:
                    self._disconnect()
                    return {
                        "success": True,
                        "response": "No emails found.",
                        "actions_taken": [{"action": "email_read"}],
                        "spoken": False,
                    }
                fetch_ids = [msg_ids[-1]]

            details = self._fetch_full_emails(imap, fetch_ids)
            self._disconnect()

            if not details:
                return {
                    "success": False,
                    "response": "Couldn't read the email content.",
                    "actions_taken": [{"action": "email_read_error"}],
                    "spoken": False,
                }

            email_data = details[0]
            llm = context.get("llm")
            if llm:
                email_block = (
                    f"From: {email_data['from']}\n"
                    f"Subject: {email_data['subject']}\n"
                    f"Date: {email_data['date']}\n"
                    f"Body:\n{email_data['body'][:2000]}"
                )
                # Semantic injection check before passing email body to LLM
                from core.security import get_security_filter as _gsf
                _sf = _gsf()
                if _sf:
                    _blocked, _alert = _sf.check_semantic_input(email_block, "email")
                    if _blocked:
                        return {"response": "[Email content blocked by security filter]",
                                "actions_taken": [], "success": False}
                user_msg = (
                    f"{wrap_external(email_block, 'email content')}\n\n"
                    f"User asked: {command}\n\n"
                    f"Summarize this email concisely."
                )
                try:
                    response = llm.generate(
                        user_msg,
                        system_prompt=self._SYSTEM_PROMPT_READ,
                        temperature=0.3,
                    )
                except LLMError as e:
                    return {"success": False, "response": f"LLM unavailable: {e}", "actions_taken": [], "spoken": False}
            else:
                response = (
                    f"From: {email_data['from']}\n"
                    f"Subject: {email_data['subject']}\n"
                    f"Date: {email_data['date']}\n\n"
                    f"{email_data['body'][:500]}"
                )

            return {
                "success": True,
                "response": response,
                "actions_taken": [{"action": "email_read", "subject": email_data["subject"]}],
                "spoken": False,
            }

        except Exception as e:
            self._disconnect()
            return {
                "success": False,
                "response": f"Error reading email: {str(e)}",
                "actions_taken": [{"action": "email_read_error"}],
                "spoken": False,
            }

    # ── Send email (draft + confirm flow) ─────────────────────────

    # Regex to pull Windows/Unix absolute file paths out of a command string.
    _PATH_RE = re.compile(
        r'(?:[A-Za-z]:\\(?:[^\s,\'"]+\\)*[^\s,\'"]+|/(?:[^\s,\'"]+/)*[^\s,\'"]+)'
    )

    def _extract_attach_paths(self, command):
        """Return list of existing local file paths found in command text."""
        paths = []
        for p in self._PATH_RE.findall(command):
            p = p.rstrip(".,;:!?")   # strip sentence-ending punctuation
            if os.path.isfile(p):
                paths.append(p)
        return paths

    @staticmethod
    def _remote_draft_confirmation(broker, auth, *, to, subject, body,
                                   attach_paths=None, reply=False):
        """Render an informed approval prompt for a non-desktop channel.

        Put the approval token first so Signal's response-length limit cannot
        truncate the only way to authorize the action. Draft bodies are
        previewed in-channel and bounded to keep the response practical.
        """
        preview_limit = 600
        body_preview = body or "(empty body)"
        if len(body_preview) > preview_limit:
            body_preview = body_preview[:preview_limit] + "... [truncated]"
        attachments = list(attach_paths or [])
        attach_note = (
            "\nAttachments: "
            + ", ".join(os.path.basename(path) for path in attachments)
            if attachments else ""
        )
        label = "reply" if reply else "email"
        return (
            f"{broker.confirmation_message(auth)}\n\n"
            f"Draft {label}:\n"
            f"To: {to}\n"
            f"Subject: {subject}{attach_note}\n\n"
            f"{body_preview}"
        )

    def _handle_send(self, command, context, typed=None):
        """Compose/send an email (or stage it for review).

        typed: structured tool-call args ({"to","subject","body",
        "attachment_path"}). When a recipient is supplied there it is used
        verbatim, so the native router can't lose the address; only a missing
        subject/body is filled by the LLM composer. Without typed args this
        falls back to the original LLM-extracts-everything path.
        """
        llm = context.get("llm")
        typed = typed or {}

        # Extract any file paths present in the command (e.g. from {last_result}
        # substitution) so they can be attached rather than pasted into the body.
        attach_paths = self._extract_attach_paths(command)
        typed_path = (typed.get("attachment_path") or "").strip()
        if typed_path and os.path.isfile(typed_path) and typed_path not in attach_paths:
            attach_paths.append(typed_path)

        to      = (typed.get("to") or "").strip()
        subject = (typed.get("subject") or "").strip()
        body    = (typed.get("body") or "").strip()

        if to:
            # Structured recipient from the tool call — never re-extract it.
            # Only compose what's missing, and keep `to` fixed.
            if (not subject or not body) and llm:
                try:
                    composed = self._parse_json_draft(llm.generate(
                        f"Compose an email based on this request:\n\n{command}",
                        system_prompt=self._SYSTEM_PROMPT_COMPOSE,
                        temperature=0.3)) or {}
                except LLMError:
                    composed = {}
                subject = subject or composed.get("subject", "")
                body = body or composed.get("body", "")
            subject = subject or "No Subject"
            if not body:
                body = "Please see the attached file." if attach_paths else ""
        else:
            # Free-form path: the LLM extracts to/subject/body from the command.
            if not llm:
                return {
                    "success": False,
                    "response": "LLM not available for composing email.",
                    "actions_taken": [],
                    "spoken": False,
                }
            try:
                response = llm.generate(
                    f"Compose an email based on this request:\n\n{command}",
                    system_prompt=self._SYSTEM_PROMPT_COMPOSE,
                    temperature=0.3,
                )
            except LLMError as e:
                return {"success": False, "response": f"LLM unavailable: {e}", "actions_taken": [], "spoken": False}

            composed = self._parse_json_draft(response)
            if not composed or "to" not in composed:
                return {
                    "success": False,
                    "response": "I couldn't figure out who to send the email to. "
                                "Try: 'send email to user@example.com about ...'",
                    "actions_taken": [{"action": "email_compose_fail"}],
                    "spoken": False,
                }

            to      = composed["to"]
            subject = composed.get("subject", "No Subject")
            body    = composed.get("body", "")

        command_source = context.get("command_source", "local")
        broker = context.get("capabilities")

        def _perform_send():
            self._send_smtp(to, subject, body, attach_paths=attach_paths)
            attach_note = (f" with {len(attach_paths)} attachment(s)"
                           if attach_paths else "")
            return {
                "success": True,
                "response": f"Email sent to {to}{attach_note}.",
                "actions_taken": [{"action": "email_sent", "to": to}],
                "spoken": False,
            }

        auth = None
        if broker:
            auth = broker.request(
                "external_send",
                source=command_source,
                summary=f"Send email to {to} with subject {subject!r}",
                metadata={"channel": "email", "recipient": to},
                executor=_perform_send,
            )
            if not auth.allowed and not auth.confirmation_required:
                return {
                    "success": False,
                    "response": broker.denial_message(auth),
                    "actions_taken": [],
                    "spoken": False,
                }
            if auth.allowed:
                try:
                    result = _perform_send()
                    broker.record_outcome(auth.request, success=True)
                    return result
                except Exception as exc:
                    broker.record_outcome(auth.request, success=False,
                                          error=str(exc))
                    return {
                        "success": False,
                        "response": f"Failed to send email: {exc}",
                        "actions_taken": [],
                        "spoken": False,
                    }
            if command_source != "local":
                return {
                    "success": False,
                    "response": self._remote_draft_confirmation(
                        broker, auth, to=to, subject=subject, body=body,
                        attach_paths=attach_paths),
                    "actions_taken": [{"action": "email_send_pending",
                                       "to": to}],
                    "spoken": False,
                }

        if not broker:
            from core.security import get_security_filter as _gsf
            _sf = _gsf()
            gate_enabled = (not _sf) or _sf.gate_required("external_send")
            if not gate_enabled:
                try:
                    return _perform_send()
                except Exception as exc:
                    return {
                        "success": False,
                        "response": f"Failed to send email: {exc}",
                        "actions_taken": [],
                        "spoken": False,
                    }

        # Local GUI commands use the compose dialog as the confirmation UI.
        # The request id follows the draft so the worker can verify approval
        # immediately before SMTP is invoked.
        attach_note = (f"\nAttachments: {', '.join(os.path.basename(p) for p in attach_paths)}"
                       if attach_paths else "")
        pending_email = {"to": to, "subject": subject, "body": body,
                         "attach_paths": attach_paths}
        if auth:
            pending_email["capability_request_id"] = auth.request.request_id
        return {
            "success": True,
            "response": (
                f"Here is my draft email:\n\n"
                f"To: {to}\n"
                f"Subject: {subject}{attach_note}\n\n"
                f"{body}\n\n"
                f"Review the compose window, then click Send or Cancel."
            ),
            "actions_taken": [{"action": "email_draft_created", "to": to}],
            "spoken": False,
            "pending_email": pending_email,
        }

    # ── Reply to email ─────────────────────────────────────────────

    def _handle_reply(self, command, context):
        """Fetch the referenced email, draft a reply body, return for compose dialog."""
        try:
            imap = self._connect_imap()
            imap.select("INBOX", readonly=True)

            cache_entry = self._resolve_email_ref(command)
            if cache_entry:
                fetch_ids = [cache_entry["uid"]]
            else:
                sender_filter = self._extract_sender(command)
                if sender_filter:
                    status, data = imap.uid('search', None, f'FROM "{sender_filter}"')
                else:
                    status, data = imap.uid('search', None, "ALL")
                msg_ids = data[0].split() if data[0] else []
                if not msg_ids:
                    self._disconnect()
                    return {
                        "success": False,
                        "response": "No email found to reply to.",
                        "actions_taken": [],
                        "spoken": False,
                    }
                fetch_ids = [msg_ids[-1]]

            details = self._fetch_full_emails(imap, fetch_ids)
            self._disconnect()

            if not details:
                return {
                    "success": False,
                    "response": "Could not read the email to reply to.",
                    "actions_taken": [],
                    "spoken": False,
                }

            original = details[0]
            reply_to_addr = self._extract_reply_addr(original["from"])
            reply_subject = original["subject"]
            if not reply_subject.lower().startswith("re:"):
                reply_subject = f"Re: {reply_subject}"
            original_msg_id = original.get("message_id", "")

            # LLM drafts the reply body
            reply_body = ""
            llm = context.get("llm")
            if llm:
                original_block = (
                    f"From: {original['from']}\n"
                    f"Subject: {original['subject']}\n"
                    f"Body:\n{original['body'][:1500]}"
                )
                # Semantic injection check before passing email body to LLM
                from core.security import get_security_filter as _gsf
                _sf = _gsf()
                if _sf:
                    _blocked, _alert = _sf.check_semantic_input(original_block, "email")
                    if _blocked:
                        return {"response": "[Email content blocked by security filter]",
                                "actions_taken": [], "success": False}
                prompt = (
                    f"{wrap_external(original_block, 'original email to reply to')}\n\n"
                    f"User instruction: {command}\n\n"
                    f"Write the reply body."
                )
                try:
                    reply_body = llm.generate(
                        prompt,
                        system_prompt=self._SYSTEM_PROMPT_REPLY,
                        temperature=0.4,
                    )
                except LLMError as e:
                    return {"success": False, "response": f"LLM unavailable: {e}", "actions_taken": [], "spoken": False}

            command_source = context.get("command_source", "local")
            broker = context.get("capabilities")

            def _perform_reply():
                self._send_smtp_reply(reply_to_addr, reply_subject,
                                      reply_body, original_msg_id)
                return {
                    "success": True,
                    "response": f"Reply sent to {reply_to_addr}.",
                    "actions_taken": [{"action": "email_reply_sent",
                                       "to": reply_to_addr}],
                    "spoken": False,
                }

            auth = None
            if broker:
                auth = broker.request(
                    "external_send",
                    source=command_source,
                    summary=(f"Reply to {reply_to_addr} with subject "
                             f"{reply_subject!r}"),
                    metadata={"channel": "email", "recipient": reply_to_addr,
                              "reply": True},
                    executor=_perform_reply,
                )
                if not auth.allowed and not auth.confirmation_required:
                    return {
                        "success": False,
                        "response": broker.denial_message(auth),
                        "actions_taken": [],
                        "spoken": False,
                    }
                if auth.allowed:
                    try:
                        result = _perform_reply()
                        broker.record_outcome(auth.request, success=True)
                        return result
                    except Exception as exc:
                        broker.record_outcome(auth.request, success=False,
                                              error=str(exc))
                        return {
                            "success": False,
                            "response": f"Failed to send reply: {exc}",
                            "actions_taken": [],
                            "spoken": False,
                        }
                if command_source != "local":
                    return {
                        "success": False,
                        "response": self._remote_draft_confirmation(
                            broker, auth, to=reply_to_addr,
                            subject=reply_subject, body=reply_body,
                            reply=True),
                        "actions_taken": [{"action": "email_reply_pending",
                                           "to": reply_to_addr}],
                        "spoken": False,
                    }

            # With no broker, retain the legacy security switch. Disabling the
            # old gate explicitly still permits immediate sends for embedders
            # that construct EmailTalent outside TalonAssistant.
            if not broker:
                from core.security import get_security_filter as _gsf
                _sf = _gsf()
                gate_enabled = (not _sf) or _sf.gate_required("external_send")
                if not gate_enabled:
                    try:
                        return _perform_reply()
                    except Exception as exc:
                        return {
                            "success": False,
                            "response": f"Failed to send reply: {exc}",
                            "actions_taken": [],
                            "spoken": False,
                        }

            # Gate enabled, local command — present draft for user review.
            pending_email = {
                "to":           reply_to_addr,
                "subject":      reply_subject,
                "body":         reply_body,
                "reply_to_uid": original_msg_id,
            }
            if auth:
                pending_email["capability_request_id"] = auth.request.request_id

            return {
                "success": True,
                "response": (
                    f"Here is my draft reply:\n\n"
                    f"To: {reply_to_addr}\n"
                    f"Subject: {reply_subject}\n\n"
                    f"{reply_body}\n\n"
                    f"Review the compose window, then click Send or Cancel."
                ),
                "actions_taken": [{"action": "email_reply_draft", "to": reply_to_addr}],
                "spoken": False,
                "pending_email": pending_email,
            }

        except Exception as e:
            self._disconnect()
            return {
                "success": False,
                "response": f"Error preparing reply: {str(e)}",
                "actions_taken": [],
                "spoken": False,
            }

    # ── Delete email ───────────────────────────────────────────────

    def _handle_delete(self, command, context):
        return self._authorize_account_change(
            command,
            context,
            summary=f"Delete email matching: {command[:160]}",
            executor=lambda: self._delete_now(command),
        )

    def _delete_now(self, command):
        """Delete the referenced email (by index, sender, or subject)."""
        try:
            imap = self._connect_imap()
            imap.select("INBOX")   # read-write

            cache_entry = self._resolve_email_ref(command)
            if cache_entry:
                target_uid = cache_entry["uid"]
                subject_hint = cache_entry.get("subject", "")
            else:
                # Fall back to live search by sender
                sender_filter = self._extract_sender(command)
                if sender_filter:
                    status, data = imap.uid('search', None, f'FROM "{sender_filter}"')
                else:
                    status, data = imap.uid('search', None, "ALL")
                msg_ids = data[0].split() if data[0] else []
                if not msg_ids:
                    self._disconnect()
                    return {
                        "success": False,
                        "response": "No matching email found to delete.",
                        "actions_taken": [],
                        "spoken": False,
                    }
                target_uid = msg_ids[-1]
                subject_hint = ""

            imap.uid('store', target_uid, '+FLAGS', '\\Deleted')
            imap.expunge()
            self._disconnect()

            # Remove from cache
            self._email_cache = [
                e for e in self._email_cache if e["uid"] != target_uid
            ]

            desc = f" ({subject_hint})" if subject_hint else ""
            return {
                "success": True,
                "response": f"Email deleted{desc}.",
                "actions_taken": [{"action": "email_delete"}],
                "spoken": False,
            }

        except Exception as e:
            self._disconnect()
            return {
                "success": False,
                "response": f"Error deleting email: {str(e)}",
                "actions_taken": [],
                "spoken": False,
            }

    # ── Folder management ──────────────────────────────────────────

    def _handle_folders(self, command, context):
        """List folders or move an email to a folder."""
        cmd_lower = command.lower()
        if any(w in cmd_lower for w in ["move", "archive"]):
            return self._handle_move(command, context)
        # Default: list folders
        try:
            imap = self._connect_imap()
            status, folder_list = imap.list()
            self._disconnect()

            folders = []
            for item in folder_list:
                if isinstance(item, bytes):
                    # IMAP LIST: (\Flags) "delim" "Name"
                    m = re.search(r'"([^"]+)"\s*$|(\S+)\s*$', item.decode())
                    if m:
                        name = (m.group(1) or m.group(2)).strip('"')
                        folders.append(name)

            if not folders:
                return {"success": True, "response": "No folders found.",
                        "actions_taken": [], "spoken": False}

            folder_str = "\n".join(f"• {f}" for f in folders[:25])
            return {
                "success": True,
                "response": f"Your email folders:\n\n{folder_str}",
                "actions_taken": [{"action": "email_list_folders"}],
                "spoken": False,
            }

        except Exception as e:
            self._disconnect()
            return {
                "success": False,
                "response": f"Error listing folders: {str(e)}",
                "actions_taken": [],
                "spoken": False,
            }

    def _handle_move(self, command, context):
        return self._authorize_account_change(
            command,
            context,
            summary=f"Move or archive email matching: {command[:160]}",
            executor=lambda: self._move_now(command),
        )

    def _move_now(self, command):
        """Move the referenced email to a target folder."""
        m = re.search(
            r'(?:move|archive)\s+(?:email\s+)?(?:\d+\s+)?(?:to\s+)?([a-zA-Z0-9_\-/ ]+)',
            command, re.IGNORECASE
        )
        folder = m.group(1).strip() if m else "Archive"
        # Strip trailing noise
        folder = re.sub(r'\s+(email|message|it|this).*$', '', folder,
                        flags=re.IGNORECASE).strip()

        try:
            imap = self._connect_imap()
            imap.select("INBOX")

            cache_entry = self._resolve_email_ref(command)
            if cache_entry:
                target_uid = cache_entry["uid"]
            else:
                sender_filter = self._extract_sender(command)
                if sender_filter:
                    status, data = imap.uid('search', None, f'FROM "{sender_filter}"')
                else:
                    status, data = imap.uid('search', None, "ALL")
                msg_ids = data[0].split() if data[0] else []
                if not msg_ids:
                    self._disconnect()
                    return {
                        "success": False,
                        "response": "No email found to move.",
                        "actions_taken": [],
                        "spoken": False,
                    }
                target_uid = msg_ids[-1]

            imap.uid('copy', target_uid, folder)
            imap.uid('store', target_uid, '+FLAGS', '\\Deleted')
            imap.expunge()
            self._disconnect()

            # Remove from cache
            self._email_cache = [
                e for e in self._email_cache if e["uid"] != target_uid
            ]

            return {
                "success": True,
                "response": f"Email moved to {folder}.",
                "actions_taken": [{"action": "email_move", "folder": folder}],
                "spoken": False,
            }

        except Exception as e:
            self._disconnect()
            return {
                "success": False,
                "response": f"Error moving email: {str(e)}",
                "actions_taken": [],
                "spoken": False,
            }

    def _authorize_account_change(
        self, command, context, *, summary, executor
    ):
        """Broker destructive IMAP mutations before opening a write session."""
        broker = context.get("capabilities")
        if broker is None:
            return {
                "success": False,
                "response": "Email account changes require the capability broker.",
                "actions_taken": [],
                "spoken": False,
            }
        authorization = broker.request(
            "external_account_write",
            source=context.get("command_source", "local"),
            summary=summary,
            metadata={"channel": "email", "command": command[:160]},
            executor=executor,
        )
        if authorization.confirmation_required:
            return {
                "success": False,
                "response": broker.confirmation_message(authorization),
                "actions_taken": [],
                "spoken": False,
            }
        if not authorization.allowed:
            return {
                "success": False,
                "response": broker.denial_message(authorization),
                "actions_taken": [],
                "spoken": False,
            }
        result = executor()
        success = result.get("success", True)
        broker.record_outcome(
            authorization.request,
            success=bool(success),
            error="" if success else result.get("response", ""),
        )
        return result

    # ── Email reference resolver ───────────────────────────────────

    def _resolve_email_ref(self, command: str) -> dict | None:
        """Return a cache entry matching the email referenced in the command.

        Resolution priority:
          1. Numeric index ("email 2", "#3")
          2. Sender name  ("email from Bob")
          3. Subject keywords ("email about the hockey game")
          4. Most recent in cache (silent fallback)
        """
        if not self._email_cache:
            return None

        cmd_lower = command.lower()

        # 1. Numeric index (1-based, user-facing)
        m = re.search(r'\b(\d+)\b', command)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(self._email_cache):
                return self._email_cache[idx]

        # 2. Sender name
        sender = self._extract_sender(command)
        if sender:
            for entry in reversed(self._email_cache):
                if sender.lower() in entry["from"].lower():
                    return entry

        # 3. Subject keyword matching — strip action/filler words, score by hits
        topic = re.sub(
            r'\b(read|delete|reply|send|move|email|message|about|regarding|'
            r'the|latest|recent|last|me|my|to|from|re|it|this)\b',
            '', cmd_lower
        ).strip()
        if topic:
            keywords = [w for w in topic.split() if len(w) > 2]
            if keywords:
                best_entry, best_score = None, 0
                for entry in reversed(self._email_cache):
                    subj_lower = entry["subject"].lower()
                    score = sum(1 for kw in keywords if kw in subj_lower)
                    if score > best_score:
                        best_score = score
                        best_entry = entry
                if best_entry and best_score > 0:
                    return best_entry

        # 4. Fallback: most recent
        return self._email_cache[-1]

    # ── JSON draft parser ──────────────────────────────────────────

    @staticmethod
    def _parse_json_draft(llm_response: str) -> dict | None:
        """Extract and parse a JSON email draft from an LLM response."""
        try:
            # Strip markdown fences if present
            clean = re.sub(r'```(?:json)?\s*', '', llm_response).strip('`').strip()
            json_match = re.search(r'\{.*\}', clean, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return None

    # ── IMAP helpers ───────────────────────────────────────────────

    def _get_password(self):
        """Retrieve password from the in-memory config."""
        return self._config.get("password", "")

    def _connect_imap(self):
        """Connect to IMAP server and login."""
        server = self._config["imap_server"]
        port = self._config.get("imap_port", 993)
        username = self._config["username"]
        password = self._get_password()

        if not password:
            raise ValueError("No email password configured. Use the gear icon to set it.")

        log.info(f"[Email] Connecting to IMAP {server}:{port}...")
        imap = imaplib.IMAP4_SSL(server, port)

        try:
            imap.login(username, password)
        except imaplib.IMAP4.error as e:
            err = str(e).lower()
            if "unmatch" in err or "quote" in err or "parse" in err:
                imap = imaplib.IMAP4_SSL(server, port)
                auth_string = f"\x00{username}\x00{password}"
                imap.authenticate("PLAIN",
                                  lambda _: auth_string.encode("utf-8"))
            else:
                raise

        self._imap = imap
        self._connected = True
        log.info(f"[Email] IMAP connected!")
        return imap

    def _disconnect(self):
        """Close IMAP connection."""
        if self._imap:
            try:
                self._imap.close()
            except Exception:
                pass
            try:
                self._imap.logout()
            except Exception:
                pass
            self._imap = None
            self._connected = False

    def _fetch_summaries(self, imap, msg_ids):
        """Fetch brief headers (from, subject, date, message_id) for given message IDs."""
        summaries = []
        for mid in msg_ids:
            try:
                status, data = imap.uid('fetch', mid, "(BODY.PEEK[HEADER])")
                if status != "OK":
                    continue
                if not data or not data[0] or not isinstance(data[0], tuple):
                    continue
                raw_header = data[0][1]
                if not isinstance(raw_header, bytes):
                    continue
                msg = email.message_from_bytes(raw_header)
                summaries.append({
                    "from":       self._decode_header(msg.get("From", "Unknown")),
                    "subject":    self._decode_header(msg.get("Subject", "(no subject)")),
                    "date":       msg.get("Date", "Unknown"),
                    "message_id": msg.get("Message-ID", ""),
                })
            except Exception as e:
                log.error(f"[Email] Error fetching summary: {e}")
                continue
        return summaries

    def _fetch_full_emails(self, imap, msg_ids):
        """Fetch full email content for given message IDs."""
        emails = []
        for mid in msg_ids:
            try:
                status, data = imap.uid('fetch', mid, "(BODY.PEEK[])")
                if status != "OK":
                    continue
                if not data or not data[0] or not isinstance(data[0], tuple):
                    continue
                raw = data[0][1]
                if not isinstance(raw, bytes):
                    continue
                msg = email.message_from_bytes(raw)

                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        ct = part.get_content_type()
                        if ct == "text/plain":
                            payload = part.get_payload(decode=True)
                            if isinstance(payload, bytes):
                                body = payload.decode("utf-8", errors="replace")
                            elif isinstance(payload, str):
                                body = payload
                            if body:
                                break
                    if not body:
                        for part in msg.walk():
                            ct = part.get_content_type()
                            if ct == "text/html":
                                payload = part.get_payload(decode=True)
                                if isinstance(payload, bytes):
                                    raw_html = payload.decode("utf-8", errors="replace")
                                elif isinstance(payload, str):
                                    raw_html = payload
                                else:
                                    continue
                                body = self._strip_html(raw_html)
                                if body:
                                    break
                else:
                    ct = msg.get_content_type()
                    payload = msg.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        text = payload.decode("utf-8", errors="replace")
                    elif isinstance(payload, str):
                        text = payload
                    else:
                        text = ""
                    body = self._strip_html(text) if ct == "text/html" else text

                emails.append({
                    "from":       self._decode_header(msg.get("From", "Unknown")),
                    "subject":    self._decode_header(msg.get("Subject", "(no subject)")),
                    "date":       msg.get("Date", "Unknown"),
                    "body":       body,
                    "message_id": msg.get("Message-ID", ""),
                })
            except Exception as e:
                log.error(f"[Email] Error fetching email: {e}")
                continue
        return emails

    @staticmethod
    def _strip_html(html_text):
        """Convert HTML email body to readable plain text."""
        if not html_text:
            return ""
        text = re.sub(r'<style[^>]*>.*?</style>', '', html_text,
                      flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<script[^>]*>.*?</script>', '', text,
                      flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<br\s*/?>',  '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>',       '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</div>',     '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</tr>',      '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</li>',      '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        text = _html_mod.unescape(text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _decode_header(value):
        """Decode an email header that may be encoded (e.g., =?UTF-8?Q?...)."""
        if not value:
            return ""
        try:
            parts = decode_header(str(value))
        except Exception:
            return str(value)
        decoded = []
        for part, charset in parts:
            if isinstance(part, bytes):
                try:
                    decoded.append(part.decode(charset or "utf-8", errors="replace"))
                except (LookupError, AttributeError):
                    decoded.append(part.decode("utf-8", errors="replace"))
            elif isinstance(part, str):
                decoded.append(part)
            else:
                decoded.append(str(part))
        return " ".join(decoded)

    def _extract_sender(self, command):
        """Extract a sender name or email from the command."""
        cmd = command.lower()
        match = re.search(r'from\s+([a-z0-9@._\- ]+)', cmd)
        if match:
            sender = match.group(1).strip()
            for noise in ["about", "regarding", "the", "today", "this"]:
                sender = re.sub(rf'\s+{noise}.*$', '', sender)
            return sender.strip()
        return ""

    @staticmethod
    def _extract_reply_addr(from_header: str) -> str:
        """Extract the bare email address from a From header like 'Name <addr@example.com>'."""
        match = re.search(r'<([^>]+)>', from_header)
        if match:
            return match.group(1)
        return from_header.strip()

    # ── SMTP helpers ───────────────────────────────────────────────

    def _send_smtp(self, to_addr, subject, body, attach_paths=None):
        """Send an email via SMTP (STARTTLS).

        attach_paths: optional list of local file paths to attach.
        """
        import mimetypes
        from email.mime.base import MIMEBase
        from email import encoders as _enc

        server = self._config.get("smtp_server", "")
        port = self._config.get("smtp_port", 587)
        username = self._config["username"]
        password = self._get_password()

        if not server:
            raise ValueError("SMTP server not configured.")
        if not password:
            raise ValueError("No email password configured.")

        msg = MIMEMultipart()
        msg["From"] = username
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        for path in (attach_paths or []):
            if not os.path.isfile(path):
                log.warning(f"[Email] Attachment not found, skipping: {path}")
                continue
            ctype, encoding = mimetypes.guess_type(path)
            if ctype is None or encoding is not None:
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)
            with open(path, "rb") as f:
                part = MIMEBase(maintype, subtype)
                part.set_payload(f.read())
            _enc.encode_base64(part)
            part.add_header("Content-Disposition", "attachment",
                            filename=os.path.basename(path))
            msg.attach(part)
            log.info(f"[Email] Attached: {os.path.basename(path)}")

        log.info(f"[Email] Sending via SMTP {server}:{port} to {to_addr}...")
        with smtplib.SMTP(server, port) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(username, password)
            smtp.send_message(msg)

        log.info(f"[Email] Sent!")

    def _send_smtp_reply(self, to_addr, subject, body, reply_uid):
        """Send a reply email with proper In-Reply-To / References threading headers."""
        server = self._config.get("smtp_server", "")
        port = self._config.get("smtp_port", 587)
        username = self._config["username"]
        password = self._get_password()

        if not server:
            raise ValueError("SMTP server not configured.")
        if not password:
            raise ValueError("No email password configured.")

        msg = MIMEMultipart()
        msg["From"] = username
        msg["To"] = to_addr
        msg["Subject"] = subject
        if reply_uid:
            msg["In-Reply-To"] = reply_uid
            msg["References"]  = reply_uid
        msg.attach(MIMEText(body, "plain"))

        log.info(f"[Email] Sending reply via SMTP {server}:{port} to {to_addr}...")
        with smtplib.SMTP(server, port) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(username, password)
            smtp.send_message(msg)

        log.info(f"[Email] Reply sent!")
