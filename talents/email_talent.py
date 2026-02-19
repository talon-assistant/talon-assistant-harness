"""EmailTalent — check, read, and send email via IMAP/SMTP.

Uses stdlib imaplib + smtplib (no third-party mail libraries).
Credentials are managed centrally by the CredentialStore (OS keyring).

Examples:
    "check my email"
    "read latest email from John"
    "how many unread emails do I have"
    "send email to john@example.com about the meeting"
"""

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


class EmailTalent(BaseTalent):
    name = "email"
    description = "Check, read, and send email via IMAP/SMTP"
    keywords = [
        "email", "mail", "inbox", "unread", "send email",
        "compose", "check email", "read email", "new email",
        "send a message", "send message to",
    ]
    examples = [
        "check my email",
        "how many unread emails do I have",
        "read latest email from John",
        "send email to alice about the meeting",
    ]
    priority = 55

    _EMAIL_PHRASES = [
        "email", "mail", "inbox", "unread",
        "send email", "compose email", "send a message",
        "check email", "check my email", "read email",
        "new email", "latest email", "read my mail",
    ]

    _SYSTEM_PROMPT_READ = (
        "You are an email summarizer. "
        "Using ONLY the email data provided, give a concise summary. "
        "Include sender name, subject, and a 1-2 sentence summary of the body. "
        "Do NOT add information not in the data."
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

    def __init__(self):
        super().__init__()
        self._imap = None
        self._connected = False

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

        # Route to the right sub-command
        if any(w in cmd_lower for w in ["send", "compose", "write"]):
            return self._handle_send(command, context)
        elif any(w in cmd_lower for w in ["read", "latest", "last", "recent", "from"]):
            return self._handle_read(command, context)
        else:
            # Default: check inbox (unread count + summaries)
            return self._handle_check(command, context)

    # ── Check inbox ────────────────────────────────────────────────

    def _handle_check(self, command, context):
        """Show unread count and brief summaries of recent emails."""
        try:
            imap = self._connect_imap()
            imap.select("INBOX", readonly=True)

            # Get unread count
            status, data = imap.search(None, "UNSEEN")
            unread_ids = data[0].split() if data[0] else []
            unread_count = len(unread_ids)

            max_fetch = self._config.get("max_fetch", 5)

            # Fetch the most recent unread (or all recent if few unread)
            if unread_ids:
                fetch_ids = unread_ids[-max_fetch:]
            else:
                # No unread — show most recent
                status, data = imap.search(None, "ALL")
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
        """Read and summarize a specific email (e.g., "read latest email from John")."""
        try:
            imap = self._connect_imap()
            imap.select("INBOX", readonly=True)

            # Try to extract a sender filter
            sender_filter = self._extract_sender(command)
            max_fetch = self._config.get("max_fetch", 5)

            if sender_filter:
                # Search by sender
                status, data = imap.search(None, f'FROM "{sender_filter}"')
            else:
                status, data = imap.search(None, "ALL")

            msg_ids = data[0].split() if data[0] else []

            if not msg_ids:
                self._disconnect()
                who = f" from '{sender_filter}'" if sender_filter else ""
                return {
                    "success": True,
                    "response": f"No emails found{who}.",
                    "actions_taken": [{"action": "email_read", "filter": sender_filter}],
                    "spoken": False,
                }

            # Fetch the latest matching email(s)
            fetch_ids = msg_ids[-min(max_fetch, len(msg_ids)):]
            details = self._fetch_full_emails(imap, fetch_ids[-1:])  # just the latest
            self._disconnect()

            if not details:
                return {
                    "success": False,
                    "response": "Couldn't read the email content.",
                    "actions_taken": [{"action": "email_read_error"}],
                    "spoken": False,
                }

            # Use LLM to summarize
            email_data = details[0]
            llm = context.get("llm")
            if llm:
                user_msg = (
                    f"=== EMAIL ===\n"
                    f"From: {email_data['from']}\n"
                    f"Subject: {email_data['subject']}\n"
                    f"Date: {email_data['date']}\n"
                    f"Body:\n{email_data['body'][:2000]}\n"
                    f"=== END EMAIL ===\n\n"
                    f"User asked: {command}\n\n"
                    f"Summarize this email concisely."
                )
                response = llm.generate(
                    user_msg,
                    system_prompt=self._SYSTEM_PROMPT_READ,
                    temperature=0.3,
                )
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

    # ── Send email ─────────────────────────────────────────────────

    def _handle_send(self, command, context):
        """Compose and send an email using LLM for drafting."""
        llm = context.get("llm")
        if not llm:
            return {
                "success": False,
                "response": "LLM not available for composing email.",
                "actions_taken": [],
                "spoken": False,
            }

        # Use LLM to extract to/subject/body
        response = llm.generate(
            f"Compose an email based on this request:\n\n{command}",
            system_prompt=self._SYSTEM_PROMPT_COMPOSE,
            temperature=0.3,
        )

        # Parse JSON
        composed = None
        try:
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                composed = json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        if not composed or "to" not in composed:
            return {
                "success": False,
                "response": "I couldn't figure out who to send the email to. "
                            "Try: 'send email to user@example.com about ...'",
                "actions_taken": [{"action": "email_compose_fail"}],
                "spoken": False,
            }

        # Actually send
        try:
            self._send_smtp(
                to_addr=composed["to"],
                subject=composed.get("subject", "No Subject"),
                body=composed.get("body", ""),
            )

            return {
                "success": True,
                "response": f"Email sent to {composed['to']}!\n"
                            f"Subject: {composed.get('subject', 'No Subject')}",
                "actions_taken": [{"action": "email_send", "to": composed["to"],
                                   "subject": composed.get("subject", "")}],
                "spoken": False,
            }

        except Exception as e:
            return {
                "success": False,
                "response": f"Failed to send email: {str(e)}",
                "actions_taken": [{"action": "email_send_error"}],
                "spoken": False,
            }

    # ── IMAP helpers ───────────────────────────────────────────────

    def _get_password(self):
        """Retrieve password from the in-memory config.

        The real password is injected from the OS keyring at startup by
        TalonAssistant._inject_secrets(), so it's always available here
        as long as credentials have been configured.
        """
        return self._config.get("password", "")

    def _connect_imap(self):
        """Connect to IMAP server and login."""
        server = self._config["imap_server"]
        port = self._config.get("imap_port", 993)
        username = self._config["username"]
        password = self._get_password()

        if not password:
            raise ValueError("No email password configured. Use the gear icon to set it.")

        print(f"   [Email] Connecting to IMAP {server}:{port}...")
        imap = imaplib.IMAP4_SSL(server, port)

        try:
            imap.login(username, password)
        except imaplib.IMAP4.error as e:
            err = str(e).lower()
            if "unmatch" in err or "quote" in err or "parse" in err:
                # Password has special chars that break IMAP quoting;
                # open a fresh connection and use SASL PLAIN instead.
                imap = imaplib.IMAP4_SSL(server, port)
                auth_string = f"\x00{username}\x00{password}"
                imap.authenticate("PLAIN",
                                  lambda _: auth_string.encode("utf-8"))
            else:
                raise

        self._imap = imap
        self._connected = True
        print(f"   [Email] IMAP connected!")
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
        """Fetch brief headers (from, subject, date) for given message IDs."""
        summaries = []
        for mid in msg_ids:
            try:
                # BODY.PEEK[HEADER] avoids marking as read and works on
                # servers (like iCloud) where RFC822.HEADER returns empty.
                status, data = imap.fetch(mid, "(BODY.PEEK[HEADER])")
                if status != "OK":
                    continue
                if not data or not data[0] or not isinstance(data[0], tuple):
                    continue
                raw_header = data[0][1]
                if not isinstance(raw_header, bytes):
                    continue
                msg = email.message_from_bytes(raw_header)
                summaries.append({
                    "from": self._decode_header(msg.get("From", "Unknown")),
                    "subject": self._decode_header(msg.get("Subject", "(no subject)")),
                    "date": msg.get("Date", "Unknown"),
                })
            except Exception as e:
                print(f"   [Email] Error fetching summary: {e}")
                continue
        return summaries

    def _fetch_full_emails(self, imap, msg_ids):
        """Fetch full email content for given message IDs."""
        emails = []
        for mid in msg_ids:
            try:
                # BODY.PEEK[] avoids marking as read and works on servers
                # (like iCloud) where RFC822 returns empty responses.
                status, data = imap.fetch(mid, "(BODY.PEEK[])")
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
                    # Pass 1: prefer text/plain
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
                    # Pass 2: fall back to text/html if no plain text
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
                    if ct == "text/html":
                        body = self._strip_html(text)
                    else:
                        body = text

                emails.append({
                    "from": self._decode_header(msg.get("From", "Unknown")),
                    "subject": self._decode_header(msg.get("Subject", "(no subject)")),
                    "date": msg.get("Date", "Unknown"),
                    "body": body,
                })
            except Exception as e:
                print(f"   [Email] Error fetching email: {e}")
                continue
        return emails

    @staticmethod
    def _strip_html(html_text):
        """Convert HTML email body to readable plain text."""
        if not html_text:
            return ""
        # Remove <style> and <script> blocks entirely
        text = re.sub(r'<style[^>]*>.*?</style>', '', html_text,
                       flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<script[^>]*>.*?</script>', '', text,
                       flags=re.DOTALL | re.IGNORECASE)
        # Replace <br>, <p>, <div>, <tr>, <li> with newlines
        text = re.sub(r'<br\s*/?>',  '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>',       '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</div>',     '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</tr>',      '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</li>',      '\n', text, flags=re.IGNORECASE)
        # Strip all remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode HTML entities (&amp; &lt; &#39; etc.)
        text = _html_mod.unescape(text)
        # Collapse excessive whitespace
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
        # "from <name>" pattern
        match = re.search(r'from\s+([a-z0-9@._\- ]+)', cmd)
        if match:
            sender = match.group(1).strip()
            # Remove trailing noise words
            for noise in ["about", "regarding", "the", "today", "this"]:
                sender = re.sub(rf'\s+{noise}.*$', '', sender)
            return sender.strip()
        return ""

    # ── SMTP helpers ───────────────────────────────────────────────

    def _send_smtp(self, to_addr, subject, body):
        """Send an email via SMTP (STARTTLS)."""
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

        print(f"   [Email] Sending via SMTP {server}:{port} to {to_addr}...")
        with smtplib.SMTP(server, port) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(username, password)
            smtp.send_message(msg)

        print(f"   [Email] Sent!")
