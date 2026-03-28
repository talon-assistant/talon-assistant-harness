"""Cowork Bridge Talent — file-based task relay between Cowork and Talon.

Cowork (Claude desktop) writes task JSON files to:
    ~/OneDrive/Documents/cowork_bridge/tasks/{task_id}.json

This talent polls that folder on a scheduler interval, executes each task,
and writes results to:
    ~/OneDrive/Documents/cowork_bridge/results/{task_id}.json

Processed task files are moved to:
    ~/OneDrive/Documents/cowork_bridge/logs/{task_id}.json
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from talents.base import BaseTalent

# ── paths ─────────────────────────────────────────────────────────────────────

_BRIDGE_ROOT = Path.home() / "OneDrive" / "Documents" / "cowork_bridge"
_TASKS_DIR   = _BRIDGE_ROOT / "tasks"
_RESULTS_DIR = _BRIDGE_ROOT / "results"
_LOGS_DIR    = _BRIDGE_ROOT / "logs"


def _ensure_dirs() -> None:
    for d in (_TASKS_DIR, _RESULTS_DIR, _LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_result(task_id: str, status: str, data: dict | None,
                  error: str | None) -> None:
    result = {
        "task_id":   task_id,
        "status":    status,
        "completed": datetime.now().isoformat(),
        "data":      data or {},
        "error":     error,
    }
    out_path = _RESULTS_DIR / f"{task_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"   [CoworkBridge] Result written: {out_path.name}")


def _archive_task(task_path: Path) -> None:
    dest = _LOGS_DIR / task_path.name
    shutil.move(str(task_path), str(dest))


# ── fetcher ───────────────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

MAX_CHARS = 32_000


def _fetch_url(url: str) -> str | None:
    """Fetch URL content using trafilatura -> BS4 fallback."""
    try:
        import requests
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:
        print(f"   [CoworkBridge] HTTP error for {url}: {exc}")
        return None

    try:
        import trafilatura
        text = trafilatura.extract(html, include_comments=False,
                                   include_tables=True)
        if text and len(text.split()) > 30:
            return text[:MAX_CHARS]
    except Exception:
        pass

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text[:MAX_CHARS]
    except Exception as exc:
        print(f"   [CoworkBridge] BS4 error: {exc}")

    return None


# ── task handlers ─────────────────────────────────────────────────────────────

def _handle_ping(task_id: str, payload: dict) -> None:
    _write_result(task_id, "success",
                  {"pong": True, "message": "Talon bridge is live"}, None)


def _handle_scrape_url(task_id: str, payload: dict) -> None:
    url = payload.get("url", "").strip()
    if not url:
        _write_result(task_id, "error", None, "No URL provided in payload")
        return

    print(f"   [CoworkBridge] Scraping: {url}")
    content = _fetch_url(url)
    if content:
        _write_result(task_id, "success", {"content": content, "url": url}, None)
    else:
        _write_result(
            task_id, "error", None,
            f"Could not retrieve content from {url} — "
            "site may block automated access or require authentication"
        )


def _handle_browser_fetch(task_id: str, payload: dict) -> None:
    """Fetch a page using Playwright with optional saved auth state."""
    try:
        from playwright.sync_api import sync_playwright
        from playwright.sync_api import TimeoutError as PWTimeout
    except ImportError:
        _write_result(task_id, "error", None,
                      "playwright is not installed — run: "
                      "pip install playwright && playwright install chromium")
        return

    url      = payload.get("url", "").strip()
    wait_for = payload.get("wait_for")
    extract  = payload.get("extract", "text")
    auth_file = payload.get("auth_state", "config/linkedin_auth.json")

    if not url:
        _write_result(task_id, "error", None, "No URL provided")
        return

    print(f"   [CoworkBridge] browser_fetch: {url}")

    try:
        with sync_playwright() as p:
            context_kwargs = {}
            if os.path.exists(auth_file):
                context_kwargs["storage_state"] = auth_file
                print(f"   [CoworkBridge] Using auth state: {auth_file}")

            browser = p.chromium.launch(headless=True)
            context = browser.new_context(**context_kwargs)
            page    = context.new_page()

            page.goto(url, wait_until="domcontentloaded", timeout=30_000)

            if wait_for:
                try:
                    page.wait_for_selector(wait_for, timeout=15_000)
                except PWTimeout:
                    print(f"   [CoworkBridge] Selector '{wait_for}' not found, "
                          "continuing with available content")

            if extract == "html":
                content = page.content()
            elif extract == "links":
                links = page.eval_on_selector_all(
                    "a[href]",
                    "els => els.map(e => ({text: e.innerText.trim(), href: e.href}))"
                )
                content = json.dumps(links, indent=2)
            else:
                content = page.inner_text("body")

            browser.close()

            if content:
                _write_result(task_id, "success",
                              {"content": content[:MAX_CHARS], "url": url}, None)
            else:
                _write_result(task_id, "error", None,
                              "Page loaded but no content extracted")

    except PWTimeout:
        _write_result(task_id, "error", None, f"Timeout loading {url}")
    except Exception as exc:
        _write_result(task_id, "error", None, str(exc))


_HANDLERS = {
    "ping":          _handle_ping,
    "scrape_url":    _handle_scrape_url,
    "browser_fetch": _handle_browser_fetch,
}


# ── talent ────────────────────────────────────────────────────────────────────

class CoworkBridgeTalent(BaseTalent):
    name        = "cowork_bridge"
    description = (
        "Internal bridge talent. Polls the cowork_bridge/tasks folder for "
        "tasks written by the Cowork desktop agent and writes results back. "
        "Triggered by the scheduler — not by user commands."
    )
    examples  = ["cowork bridge check"]
    keywords  = ["cowork bridge"]
    priority  = 10

    @property
    def routing_available(self) -> bool:
        # Scheduler-only — never appear in LLM routing roster
        return False

    def initialize(self, config: dict) -> None:
        _ensure_dirs()
        print(f"   [CoworkBridge] Bridge initialized. Watching: {_TASKS_DIR}")

    def execute(self, command: str, context: dict) -> dict:
        _ensure_dirs()

        task_files = sorted(_TASKS_DIR.glob("*.json"))
        if not task_files:
            return {
                "success":       True,
                "response":      "",
                "actions_taken": [],
                "spoken":        False,
            }

        processed = 0
        errors    = 0

        for task_path in task_files:
            try:
                with open(task_path, "r", encoding="utf-8") as f:
                    task = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"   [CoworkBridge] Could not read {task_path.name}: {exc}")
                continue

            task_id   = task.get("task_id", task_path.stem)
            task_type = task.get("type", "unknown")
            payload   = task.get("payload", {})

            print(f"   [CoworkBridge] Processing task {task_id} (type={task_type})")

            # Skip if result already exists (duplicate delivery guard)
            result_path = _RESULTS_DIR / f"{task_id}.json"
            if result_path.exists():
                _archive_task(task_path)
                continue

            handler = _HANDLERS.get(task_type)
            if handler:
                try:
                    handler(task_id, payload)
                    processed += 1
                except Exception as exc:
                    print(f"   [CoworkBridge] Handler error for {task_id}: {exc}")
                    _write_result(task_id, "error", None, str(exc))
                    errors += 1
            else:
                _write_result(task_id, "error", None,
                              f"Unknown task type: {task_type!r}")
                errors += 1

            _archive_task(task_path)

        summary = f"Cowork bridge: {processed} task(s) processed"
        if errors:
            summary += f", {errors} error(s)"

        return {
            "success":       True,
            "response":      summary if processed or errors else "",
            "actions_taken": [{"action": "cowork_bridge_poll",
                               "processed": processed}],
            "spoken":        False,
        }
