"""One-time script to capture Playwright auth state into the OS keyring.

Run once:
    python scripts/save_linkedin_auth.py [key]

A Chromium window will open to linkedin.com/login. Log in manually,
then press Enter in this terminal. Your session cookies and localStorage
will be stored in the Windows Credential Manager (via keyring) under
the service 'talon_assistant', key 'cowork_bridge.auth_state.<key>'.

The optional [key] argument defaults to 'default'. Use different keys
for different sites:
    python scripts/save_linkedin_auth.py linkedin
    python scripts/save_linkedin_auth.py workday

No plaintext files are written.
"""
import json
import sys

from playwright.sync_api import sync_playwright

# Allow running from repo root
sys.path.insert(0, ".")
from core.credential_store import CredentialStore

KEY = sys.argv[1] if len(sys.argv) > 1 else "default"
LOGIN_URL = "https://www.linkedin.com/login"

store = CredentialStore()
if not store.available:
    print("ERROR: keyring is not available. Install it: pip install keyring")
    sys.exit(1)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto(LOGIN_URL)
    input("Log in manually, then press Enter here to save session...")

    # Capture auth state as dict (not file)
    state = context.storage_state()
    browser.close()

# Serialize and store in OS keyring (compressed + chunked for large auth state)
state_json = json.dumps(state)
field_key = f"auth_state.{KEY}"

if store.store_blob("cowork_bridge", field_key, state_json):
    print(f"Auth state saved to OS keyring: talon_assistant / "
          f"cowork_bridge.{field_key}")
    print(f"Original size: {len(state_json):,} chars (compressed + chunked)")
else:
    print("ERROR: Failed to store auth state in keyring.")
    sys.exit(1)
