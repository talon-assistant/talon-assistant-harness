"""One-time script to capture LinkedIn auth state for Playwright.

Run once:
    python scripts/save_linkedin_auth.py

A Chromium window will open to linkedin.com/login. Log in manually,
then press Enter in this terminal. Your session cookies and localStorage
will be saved to config/linkedin_auth.json for headless reuse.
"""
from playwright.sync_api import sync_playwright

AUTH_PATH = "config/linkedin_auth.json"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://www.linkedin.com/login")
    input("Log in manually, then press Enter here to save session...")
    context.storage_state(path=AUTH_PATH)
    browser.close()
    print(f"Auth state saved to {AUTH_PATH}")
