"""Centralized credential storage using the OS keyring.

Wraps the `keyring` library to provide a single interface for all talents
to store and retrieve secrets (API keys, passwords, tokens).  Falls back
to plaintext config with a console warning when keyring is unavailable.

Key format:  service = "talon_assistant"
             username = "{talent_name}.{field_key}"

For large values (e.g. Playwright auth state), use ``store_blob`` /
``get_blob`` which compress, base64-encode, and split across multiple
keyring entries to stay within the Windows Credential Manager size limit.
"""

import base64
import zlib

try:
    import keyring
    _HAS_KEYRING = True
except ImportError:
    _HAS_KEYRING = False

_SERVICE = "talon_assistant"

# Windows Credential Manager has a ~2560-byte limit per entry.
# Use 2000 to leave headroom for key name overhead.
_CHUNK_MAX = 2000


class CredentialStore:
    """OS-level secret storage for Talon talent credentials."""

    # Legacy service names from before centralisation (email_talent used this)
    _LEGACY_SERVICES = {"talon_email"}

    def __init__(self):
        if not _HAS_KEYRING:
            print("   [CredentialStore] WARNING: keyring not installed. "
                  "Secrets will remain in plaintext config.")

    # ── Public API ────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True when the keyring backend is usable."""
        return _HAS_KEYRING

    def store_secret(self, talent_name: str, field_key: str, value: str) -> bool:
        """Store a secret in the OS keyring.

        Returns True on success, False if keyring is unavailable or errored.
        """
        if not _HAS_KEYRING or not value:
            return False
        try:
            keyring.set_password(_SERVICE, f"{talent_name}.{field_key}", value)
            return True
        except Exception as e:
            print(f"   [CredentialStore] Failed to store "
                  f"{talent_name}.{field_key}: {e}")
            return False

    def get_secret(self, talent_name: str, field_key: str) -> str:
        """Retrieve a secret from the OS keyring.

        Returns the secret string, or "" if not found / unavailable.
        """
        if not _HAS_KEYRING:
            return ""
        try:
            value = keyring.get_password(_SERVICE, f"{talent_name}.{field_key}")
            return value or ""
        except Exception as e:
            print(f"   [CredentialStore] Failed to read "
                  f"{talent_name}.{field_key}: {e}")
            return ""

    def delete_secret(self, talent_name: str, field_key: str) -> bool:
        """Remove a secret from the OS keyring."""
        if not _HAS_KEYRING:
            return False
        try:
            keyring.delete_password(_SERVICE, f"{talent_name}.{field_key}")
            return True
        except keyring.errors.PasswordDeleteError:
            return False  # already absent
        except Exception as e:
            print(f"   [CredentialStore] Failed to delete "
                  f"{talent_name}.{field_key}: {e}")
            return False

    def has_secret(self, talent_name: str, field_key: str) -> bool:
        """Check whether a secret exists without returning its value."""
        return bool(self.get_secret(talent_name, field_key))

    # ── Large-value storage (blob) ─────────────────────────────────

    def store_blob(self, talent_name: str, field_key: str, value: str) -> bool:
        """Compress, encode, and store a large string across multiple entries.

        Works around the Windows Credential Manager ~2560-byte limit by
        splitting into numbered chunks.  Returns True on full success.
        """
        if not _HAS_KEYRING or not value:
            return False

        # Compress → base64 so it's keyring-safe ASCII
        compressed = zlib.compress(value.encode("utf-8"), level=9)
        encoded = base64.b64encode(compressed).decode("ascii")

        # Split into chunks
        chunks = [encoded[i:i + _CHUNK_MAX]
                  for i in range(0, len(encoded), _CHUNK_MAX)]

        # Clear any old chunks first
        self._delete_blob_chunks(talent_name, field_key)

        # Store chunk count in a metadata entry
        base_key = f"{talent_name}.{field_key}"
        try:
            keyring.set_password(_SERVICE, f"{base_key}._chunks",
                                 str(len(chunks)))
            for i, chunk in enumerate(chunks):
                keyring.set_password(_SERVICE, f"{base_key}.{i}", chunk)
            return True
        except Exception as e:
            print(f"   [CredentialStore] Failed to store blob "
                  f"{base_key}: {e}")
            # Clean up partial writes
            self._delete_blob_chunks(talent_name, field_key)
            return False

    def get_blob(self, talent_name: str, field_key: str) -> str:
        """Retrieve and decompress a blob stored by ``store_blob``.

        Returns the original string, or "" if not found.
        """
        if not _HAS_KEYRING:
            return ""

        base_key = f"{talent_name}.{field_key}"
        try:
            count_str = keyring.get_password(_SERVICE, f"{base_key}._chunks")
            if not count_str:
                return ""
            count = int(count_str)

            parts = []
            for i in range(count):
                chunk = keyring.get_password(_SERVICE, f"{base_key}.{i}")
                if chunk is None:
                    print(f"   [CredentialStore] Blob chunk {i} missing "
                          f"for {base_key}")
                    return ""
                parts.append(chunk)

            encoded = "".join(parts)
            compressed = base64.b64decode(encoded)
            return zlib.decompress(compressed).decode("utf-8")
        except Exception as e:
            print(f"   [CredentialStore] Failed to read blob "
                  f"{base_key}: {e}")
            return ""

    def delete_blob(self, talent_name: str, field_key: str) -> bool:
        """Remove all chunks of a stored blob."""
        return self._delete_blob_chunks(talent_name, field_key)

    def _delete_blob_chunks(self, talent_name: str, field_key: str) -> bool:
        """Internal: remove metadata + all numbered chunks."""
        if not _HAS_KEYRING:
            return False
        base_key = f"{talent_name}.{field_key}"
        try:
            count_str = keyring.get_password(_SERVICE, f"{base_key}._chunks")
            if count_str:
                count = int(count_str)
                for i in range(count):
                    try:
                        keyring.delete_password(_SERVICE, f"{base_key}.{i}")
                    except Exception:
                        pass
                try:
                    keyring.delete_password(_SERVICE, f"{base_key}._chunks")
                except Exception:
                    pass
            return True
        except Exception:
            return False

    # ── Legacy migration ──────────────────────────────────────────

    def migrate_legacy_email(self, username: str) -> None:
        """Migrate email password from old 'talon_email' service to the
        new centralised 'talon_assistant' service.

        Called once during startup.  Safe to call repeatedly (no-op if
        already migrated or no legacy entry exists).
        """
        if not _HAS_KEYRING or not username:
            return
        try:
            old_pw = keyring.get_password("talon_email", username)
            if old_pw:
                self.store_secret("email", "password", old_pw)
                try:
                    keyring.delete_password("talon_email", username)
                except Exception:
                    pass  # non-critical
                print("   [CredentialStore] Migrated email password "
                      "from legacy keyring entry")
        except Exception:
            pass  # no legacy entry — nothing to do
