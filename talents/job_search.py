"""JobSearchTalent -- automated job search across LinkedIn, Dice, and Built In.

Scrapes job listings from configured search URLs, identifies new postings
not yet in the job tracker, adds them automatically, and runs automated
resume-fit analysis via Claude CLI in the background.

Supported sites:
    - LinkedIn (requires one-time login via dedicated Chrome profile)
    - Dice.com (no auth required)
    - Built In / builtin.com (no auth required)

Designed to run on a schedule via the Talon scheduler — e.g.,
"schedule a job search every 2 hours"

Examples:
    "search for jobs"
    "run a job hunt"
    "check for new job listings"
    "add a search URL https://dice.com/jobs?q=..."
    "show my search URLs"
    "remove the second search URL"
    "job search login"
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from talents.base import BaseTalent

import logging
log = logging.getLogger(__name__)


def _data_dir() -> str:
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(d, exist_ok=True)
    return d


def _detect_site(url: str) -> str:
    """Identify which job board a URL belongs to."""
    host = urlparse(url).netloc.lower()
    if "linkedin.com" in host:
        return "linkedin"
    if "dice.com" in host:
        return "dice"
    if "builtin.com" in host:
        return "builtin"
    return "unknown"


def _clean_search_url(url: str) -> str:
    """Strip ephemeral/session params from a search URL."""
    site = _detect_site(url)
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)

    if site == "linkedin":
        for key in ("currentJobId", "origin"):
            params.pop(key, None)

    # Flatten single-value lists for urlencode
    flat = {k: v[0] if len(v) == 1 else v for k, v in params.items()}
    return urlunparse(parsed._replace(query=urlencode(flat, doseq=True)))


class JobSearchTalent(BaseTalent):
    """Search LinkedIn, Dice, and Built In for job listings."""

    name = "job_search"
    description = (
        "Search for job listings on LinkedIn, Dice, and Built In; "
        "track new postings and hand matches to Cowork for fit analysis"
    )
    keywords = [
        "job search", "job searches", "job hunt", "job hunting",
        "hunt for jobs", "search for jobs", "find jobs", "find job",
        "check linkedin", "check dice", "new job listings",
        "search url", "search urls", "job listings",
        "todays job", "today's job", "do a job",
    ]
    examples = [
        "search for jobs",
        "run a job hunt",
        "do todays job searches",
        "check for new job listings",
        "find me some jobs",
        "show top jobs",
        "show best matches",
        "score my jobs",
        "run fit analysis",
        "add a search URL https://dice.com/jobs?q=...",
        "show my search URLs",
        "remove the first search URL",
        "job search login",
    ]
    priority = 62
    required_packages = ["selenium"]

    def __init__(self) -> None:
        super().__init__()
        self._profile_dir = os.path.join(_data_dir(), "job_search_chrome_profile")
        self._config_file = os.path.join(_data_dir(), "job_search_config.json")
        self._search_config: dict[str, Any] = {"urls": [], "auto_cowork": True}
        self._load_search_config()
        log.info(f"[JobSearch] Config: {self._config_file} "
                 f"(exists={os.path.exists(self._config_file)}, "
                 f"urls={len(self._search_config.get('urls', []))})")

    # ── Persistent config ────────────────────────────────────────────────────

    def _load_search_config(self) -> None:
        if os.path.exists(self._config_file):
            try:
                with open(self._config_file, encoding="utf-8") as f:
                    self._search_config.update(json.load(f))
            except (json.JSONDecodeError, OSError) as e:
                log.warning(f"[JobSearch] Could not load config: {e}")

    def _save_search_config(self) -> None:
        os.makedirs(os.path.dirname(self._config_file), exist_ok=True)
        with open(self._config_file, "w", encoding="utf-8") as f:
            json.dump(self._search_config, f, indent=2)

    # ── GUI config schema ────────────────────────────────────────────────────

    def get_config_schema(self) -> dict:
        return {
            "fields": [
                {
                    "key": "auto_cowork",
                    "label": "Auto-run fit analysis on new listings via Claude CLI",
                    "type": "bool",
                    "default": True,
                },
            ]
        }

    # ── Main dispatch ────────────────────────────────────────────────────────

    def execute(self, command: str, context: dict) -> dict:
        cmd = command.lower()

        # URL management
        if "add" in cmd and ("url" in cmd or "search url" in cmd):
            return self._handle_add_url(command)
        if ("show" in cmd or "list" in cmd) and ("url" in cmd or "search" in cmd):
            return self._handle_list_urls()
        if ("remove" in cmd or "delete" in cmd) and ("url" in cmd or "search" in cmd):
            return self._handle_remove_url(command)

        # Login helper (LinkedIn only)
        if "login" in cmd:
            return self._handle_login()

        # "show top jobs" / "best matches" — handle directly
        if re.search(r'\b(top jobs|best match|top candidates|best jobs|'
                     r'top match|strongest|highest fit|best fit|ranked)\b',
                     cmd):
            return self._handle_top_candidates()

        # "score my jobs" / "run fit analysis" — score unscored existing jobs
        if re.search(r'\b(score|fit analysis|analyze|evaluate)\b', cmd):
            return self._handle_score_existing()

        # "cover letter" — delegate to job_tracker via decline
        if "cover letter" in cmd:
            return {"success": False, "response": "",
                    "actions_taken": [], "spoken": False}

        # Default: run the search
        return self._handle_search(context)

    # ── URL management ───────────────────────────────────────────────────────

    def _handle_add_url(self, command: str) -> dict:
        url_match = re.search(r'https?://[^\s]+', command)
        if not url_match:
            return _fail(
                "Include the full URL. Example: "
                "add search URL https://dice.com/jobs?q=security"
            )
        url = _clean_search_url(url_match.group())
        site = _detect_site(url)
        if site == "unknown":
            return _fail(
                "Unsupported site. Supported: LinkedIn, Dice, Built In."
            )
        self._search_config["urls"].append(url)
        self._save_search_config()
        n = len(self._search_config["urls"])
        return _ok(
            f"Added {site.title()} search URL ({n} total).",
            actions=[{"action": "add_search_url", "url": url, "site": site}],
        )

    def _handle_list_urls(self) -> dict:
        urls = self._search_config.get("urls", [])
        if not urls:
            return _ok("No search URLs configured. Say 'add a search URL' followed by the link.")
        lines = []
        for i, u in enumerate(urls, 1):
            site = _detect_site(u).title()
            display = u[:90] + "..." if len(u) > 90 else u
            lines.append(f"{i}. [{site}] {display}")
        return _ok("Search URLs:\n" + "\n".join(lines))

    def _handle_remove_url(self, command: str) -> dict:
        urls = self._search_config.get("urls", [])
        if not urls:
            return _fail("No search URLs to remove.")
        idx_match = re.search(r'(\d+)', command)
        if idx_match:
            idx = int(idx_match.group()) - 1
            if 0 <= idx < len(urls):
                urls.pop(idx)
                self._save_search_config()
                return _ok(f"Removed URL #{idx + 1}. {len(urls)} remaining.")
        return _fail(f"Specify which URL to remove (1\u2013{len(urls)}).")

    # ── Login (LinkedIn) ─────────────────────────────────────────────────────

    def _handle_login(self) -> dict:
        """Open Chrome with the dedicated profile so user can log into LinkedIn."""
        try:
            driver = self._create_driver(headless=False)
            driver.get("https://www.linkedin.com/login")
            return _ok(
                "Chrome opened to LinkedIn login. Sign in, then close the "
                "browser when done. Your session will be saved for future "
                "searches.",
                actions=[{"action": "login_prompt"}],
            )
        except Exception as e:
            return _fail(f"Could not open Chrome: {e}")

    # ── Selenium driver ──────────────────────────────────────────────────────

    def _create_driver(self, headless: bool = True):
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service

        options = Options()
        options.add_argument(f"--user-data-dir={self._profile_dir}")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        if headless:
            options.add_argument("--headless=new")

        # Prefer webdriver-manager if installed; fall back to system chromedriver
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())
        except ImportError:
            service = Service()

        driver = webdriver.Chrome(service=service, options=options)
        # Mask webdriver fingerprint
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', "
                       "{get: () => undefined})"},
        )
        return driver

    # ── Score existing unscored jobs ────────────────────────────────────────

    def _handle_score_existing(self) -> dict:
        """Run fit analysis on tracker jobs that haven't been scored yet."""
        from talents.job_tracker import _DB, _data_dir as tracker_data_dir

        db_path = os.path.join(tracker_data_dir(), "job_tracker.db")
        if not os.path.exists(db_path):
            return _fail("No job tracker database found. Run a search first.")

        db = _DB(db_path)
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM applications WHERE archived = 0 "
                "AND status = 'new' AND fit_score = 0 "
                "ORDER BY id DESC"
            ).fetchall()
        unscored = [dict(r) for r in rows]

        if not unscored:
            return _ok("All tracked jobs already have fit scores.")

        # Convert to the format _run_fit_analysis expects
        jobs = []
        for app in unscored:
            jobs.append({
                "id": app["id"],
                "company": app["company"],
                "position": app["position"],
                "location": app.get("location", ""),
                "job_url": app.get("job_url", ""),
                "source": app.get("source", ""),
            })

        count = self._run_fit_analysis(jobs)
        return _ok(
            f"Scoring {count} unscored job(s) in the background. "
            f"Check back in a few minutes with 'show top jobs'."
        )

    # ── Search orchestrator ──────────────────────────────────────────────────

    # ── Top candidates (queries tracker DB directly) ───────────────────────

    def _handle_top_candidates(self) -> dict:
        """Show new jobs ranked by fit score from the tracker DB."""
        from talents.job_tracker import _DB, _data_dir as tracker_data_dir

        db_path = os.path.join(tracker_data_dir(), "job_tracker.db")
        if not os.path.exists(db_path):
            return _fail("No job tracker database found. Run a search first.")

        db = _DB(db_path)
        apps = db.list_top_candidates(limit=15)

        if not apps:
            return _ok(
                "No scored candidates yet. Fit scores are added "
                "automatically after a job search. Try 'search for jobs' "
                "first, then check back in a few minutes."
            )

        lines = [f"**Top candidates** ({len(apps)}):\n"]
        for app in apps:
            loc = f" ({app['location']})" if app.get("location") else ""
            rec = ""
            notes = app.get("notes", "")
            if "Recommendation:" in notes:
                rec_part = notes.split("Recommendation:")[-1].strip()
                rec = f" [{rec_part}]"
            lines.append(
                f"  #{app['id']} [{app['fit_score']}%] "
                f"**{app['company']}** -- {app['position']}{loc}{rec}"
            )

        lines.append(
            "\nSay 'write a cover letter for #ID' or "
            "'write a cover letter for [company]' to generate one."
        )

        return _ok("\n".join(lines))

    # ── Search orchestrator ──────────────────────────────────────────────────

    def _handle_search(self, context: dict) -> dict:
        urls = self._search_config.get("urls", [])
        if not urls:
            return _fail(
                "No search URLs configured. Say 'add a search URL' "
                "followed by your search link."
            )

        all_jobs: list[dict] = []
        errors: list[str] = []
        sites_searched: set[str] = set()

        for url in urls:
            site = _detect_site(url)
            sites_searched.add(site)
            try:
                scraper = {
                    "linkedin": self._scrape_linkedin,
                    "dice":     self._scrape_dice,
                    "builtin":  self._scrape_builtin,
                }.get(site)
                if scraper:
                    jobs = scraper(url)
                    all_jobs.extend(jobs)
                else:
                    log.warning(f"[JobSearch] Unknown site for URL: {url[:60]}")
            except Exception as e:
                log.error(f"[JobSearch] Scrape failed for {url[:60]}: {e}")
                errors.append(str(e))

        if not all_jobs and errors:
            err_text = " ".join(errors).lower()
            if "login" in err_text or "authwall" in err_text:
                return _fail(
                    "LinkedIn requires login. Say 'job search login' to "
                    "open the browser and sign in."
                )
            return _fail(f"Search failed: {errors[0]}")

        sites_label = ", ".join(s.title() for s in sorted(sites_searched))
        if not all_jobs:
            return _ok(
                f"Searched {sites_label} ({len(urls)} URL(s)) \u2014 "
                "no listings found.",
                actions=[{"action": "search", "found": 0, "new": 0}],
            )

        # Dedup against existing tracker DB
        new_jobs = self._dedup_jobs(all_jobs)

        if not new_jobs:
            return _ok(
                f"Searched {sites_label}, found {len(all_jobs)} "
                f"listing(s) \u2014 all already tracked.",
                actions=[{"action": "search", "found": len(all_jobs), "new": 0}],
            )

        # Add new jobs to tracker DB
        added = self._add_to_tracker(new_jobs)

        # Automated fit analysis via Claude CLI
        fit_count = 0
        if self._search_config.get("auto_cowork") and added:
            fit_count = self._run_fit_analysis(added)

        # Build response
        lines = [f"Found {len(all_jobs)} listings across {sites_label}, "
                 f"{len(added)} new:"]
        for job in added[:10]:
            src = job.get("source", "")
            loc = f" ({job['location']})" if job.get("location") else ""
            lines.append(
                f"  \u2022 [{src}] {job.get('company', '?')} \u2014 "
                f"{job['position']}{loc}"
            )
        if len(added) > 10:
            lines.append(f"  \u2026 and {len(added) - 10} more")
        if fit_count:
            lines.append(
                f"\nFit analysis running in background for {fit_count} listing(s)."
            )

        notify = context.get("notify")
        if notify:
            notify("Job Search", f"{len(added)} new listing(s) found")

        return _ok(
            "\n".join(lines),
            actions=[{
                "action": "search",
                "found": len(all_jobs),
                "new": len(added),
                "fit_analysis": fit_count,
                "sites": list(sites_searched),
            }],
        )

    # ═══════════════════════════════════════════════════════════════════════════
    #  Site-specific scrapers — based on live DOM inspection (April 2026)
    # ═══════════════════════════════════════════════════════════════════════════

    # ── LinkedIn ─────────────────────────────────────────────────────────────
    #
    # Verified structure: div[data-job-id] cards inside
    # .scaffold-layout__list-container.  Each card contains:
    #   - a[href*="/jobs/view/"] for title + URL
    #   - .job-card-container__company-name for company
    #   - .job-card-container__metadata-item for location
    #
    # NOTE: LinkedIn "semantic search" URLs (/jobs/search-results/) render a
    # single-job detail view, NOT a list.  Standard search (/jobs/search/)
    # with f_TPR (time), f_WT (remote), etc. gives a proper list.

    def _scrape_linkedin(self, url: str) -> list[dict]:
        """Scrape job listings from a LinkedIn jobs search page."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        driver = self._create_driver(headless=True)
        jobs: list[dict] = []

        try:
            driver.get(url)
            time.sleep(5)

            if "/login" in driver.current_url or "authwall" in driver.current_url:
                raise RuntimeError(
                    "LinkedIn requires login \u2014 say 'job search login'"
                )

            # Wait for job cards or job links
            try:
                WebDriverWait(driver, 12).until(
                    EC.presence_of_element_located((
                        By.CSS_SELECTOR,
                        'div[data-job-id], a[href*="/jobs/view/"]',
                    ))
                )
            except Exception:
                pass

            time.sleep(2)

            # Strategy 1: card-based (standard search page)
            cards = driver.find_elements(By.CSS_SELECTOR, "div[data-job-id]")
            if cards:
                seen: set[str] = set()
                for card in cards:
                    try:
                        job = self._parse_linkedin_card(card)
                        if job and job.get("position"):
                            uid = job.get("job_url", job["position"])
                            if uid not in seen:
                                seen.add(uid)
                                jobs.append(job)
                    except Exception as e:
                        log.debug(f"[JobSearch] LinkedIn card parse: {e}")

            # Strategy 2: link-based fallback (semantic search or alt layout)
            if not jobs:
                seen_urls: set[str] = set()
                for link in driver.find_elements(
                    By.CSS_SELECTOR, 'a[href*="/jobs/view/"]'
                ):
                    text = link.text.strip()
                    href = (link.get_attribute("href") or "").split("?")[0]
                    if not text or len(text) < 4 or href in seen_urls:
                        continue
                    seen_urls.add(href)
                    jobs.append({
                        "source": "LinkedIn",
                        "date_found": date.today().isoformat(),
                        "position": text,
                        "job_url": href,
                    })

            log.info(f"[JobSearch] LinkedIn: {len(jobs)} listings")
        finally:
            driver.quit()

        return jobs

    @staticmethod
    def _parse_linkedin_card(card) -> dict | None:
        """Parse a LinkedIn div[data-job-id] card element."""
        from selenium.webdriver.common.by import By

        job: dict[str, str] = {
            "source": "LinkedIn",
            "date_found": date.today().isoformat(),
        }

        job_id = card.get_attribute("data-job-id")
        if job_id:
            job["job_url"] = f"https://www.linkedin.com/jobs/view/{job_id}"

        # Title from the job link
        for link in card.find_elements(
            By.CSS_SELECTOR, 'a[href*="/jobs/view/"]'
        ):
            text = link.text.strip()
            # LinkedIn sometimes duplicates text (e.g. "Title\nTitle with verification")
            # Take first line only
            if text:
                text = text.split("\n")[0].strip()
            if text and 3 < len(text) < 200:
                job["position"] = text
                if not job.get("job_url"):
                    job["job_url"] = (link.get_attribute("href") or "").split("?")[0]
                break

        if not job.get("position"):
            return None

        # Company
        for sel in (".job-card-container__company-name",
                    ".artdeco-entity-lockup__subtitle",
                    ".job-card-container__primary-description"):
            for elem in card.find_elements(By.CSS_SELECTOR, sel):
                text = elem.text.strip()
                if text and len(text) > 1:
                    job["company"] = text
                    break
            if job.get("company"):
                break

        # Location
        for sel in (".job-card-container__metadata-item",
                    ".artdeco-entity-lockup__caption"):
            for elem in card.find_elements(By.CSS_SELECTOR, sel):
                text = elem.text.strip()
                if text and ("remote" in text.lower() or "," in text
                             or len(text) > 5):
                    job["location"] = text
                    break
            if job.get("location"):
                break

        return job

    # ── Dice ─────────────────────────────────────────────────────────────────
    #
    # Verified structure (April 2026): Dice is a JS SPA. Job cards are <div>
    # elements with Tailwind classes (rounded-lg border bg-surface-primary).
    # Each card contains a[href*="/job-detail/"] links. The card's full text
    # splits into lines like:
    #   [Company, "Easy Apply"/"Apply Now", Title, Location, dot, "Today"]
    #
    # The most reliable extraction: find all unique /job-detail/ URLs, then
    # walk up to the card-level parent and parse its text lines.

    def _scrape_dice(self, url: str) -> list[dict]:
        """Scrape job listings from Dice.com search results."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        driver = self._create_driver(headless=True)
        jobs: list[dict] = []

        try:
            driver.get(url)
            time.sleep(5)

            # Wait for job links to render
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((
                        By.CSS_SELECTOR,
                        'a[href*="/job-detail/"]',
                    ))
                )
            except Exception:
                pass

            time.sleep(3)

            # Collect unique job URLs and their card parents
            seen_urls: set[str] = set()
            links = driver.find_elements(
                By.CSS_SELECTOR, 'a[href*="/job-detail/"]'
            )

            for link in links:
                href = (link.get_attribute("href") or "").split("?")[0]
                if not href or href in seen_urls:
                    continue
                seen_urls.add(href)

                # Walk up to find the card container (div with multiple text lines)
                card_el = link
                for _ in range(8):
                    card_el = card_el.find_element(By.XPATH, "..")
                    card_text = card_el.text or ""
                    lines = [ln.strip() for ln in card_text.split("\n")
                             if ln.strip()]
                    # A proper card has at least 3 text lines:
                    # company, apply button, title, location...
                    if len(lines) >= 3:
                        break
                else:
                    continue

                job = self._parse_dice_card_text(lines, href)
                if job:
                    jobs.append(job)

            log.info(f"[JobSearch] Dice: {len(jobs)} listings")
        finally:
            driver.quit()

        return jobs

    @staticmethod
    def _parse_dice_card_text(lines: list[str], job_url: str) -> dict | None:
        """Parse a Dice job card from its text lines.

        Typical line order:
            [Company, "Easy Apply"/"Apply Now", Title, Location, dot, "Today"/"Yesterday"]
        But order can vary, so we use heuristics.
        """
        _SKIP = {"easy apply", "apply now", "today", "yesterday",
                 "\u00b7", "posted", "new"}

        # Filter noise lines
        meaningful = []
        for ln in lines:
            if ln.lower() in _SKIP or len(ln) <= 1:
                continue
            if ln.lower().startswith("posted "):
                continue
            meaningful.append(ln)

        if len(meaningful) < 2:
            return None

        # Heuristic: company is typically first, title is the longest
        # meaningful line, location contains comma or "Remote"
        company = meaningful[0]
        title = ""
        location = ""

        for ln in meaningful[1:]:
            if not title and len(ln) > 5:
                title = ln
            elif not location and ("," in ln or "remote" in ln.lower()):
                location = ln

        # If title looks like a location, swap
        if title and ("," in title and not location):
            # Check if next meaningful line is longer (more likely a title)
            remaining = [ln for ln in meaningful[1:] if ln != title]
            for r in remaining:
                if len(r) > len(title):
                    location = title
                    title = r
                    break

        if not title:
            return None

        return {
            "source": "Dice",
            "date_found": date.today().isoformat(),
            "position": title,
            "company": company,
            "location": location,
            "job_url": job_url,
        }

    # ── Built In ─────────────────────────────────────────────────────────────
    #
    # Verified structure (April 2026): Job links use a[href*="/job/"] pattern
    # with paths like /job/title-slug/12345.  The page is server-rendered so
    # links are available without waiting for JS.  Simple link extraction is
    # the most reliable approach.

    def _scrape_builtin(self, url: str) -> list[dict]:
        """Scrape job listings from builtin.com search results."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        driver = self._create_driver(headless=True)
        jobs: list[dict] = []

        try:
            driver.get(url)
            time.sleep(4)

            # Wait for any job links
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((
                        By.CSS_SELECTOR, 'a[href*="/job/"]',
                    ))
                )
            except Exception:
                pass

            time.sleep(2)

            # Extract job links
            seen_urls: set[str] = set()
            links = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/job/"]')

            for link in links:
                text = link.text.strip()
                href = (link.get_attribute("href") or "").split("?")[0]
                if not text or len(text) < 4 or len(text) > 200:
                    continue
                if href in seen_urls:
                    continue
                if not href.startswith("http"):
                    href = "https://builtin.com" + href
                seen_urls.add(href)
                jobs.append({
                    "source": "Built In",
                    "date_found": date.today().isoformat(),
                    "position": text,
                    "job_url": href,
                })

            # Scroll and check for more / load-more button
            if jobs:
                try:
                    driver.execute_script(
                        "window.scrollTo(0, document.body.scrollHeight)"
                    )
                    time.sleep(3)
                    more_links = driver.find_elements(
                        By.CSS_SELECTOR, 'a[href*="/job/"]'
                    )
                    for link in more_links:
                        text = link.text.strip()
                        href = (link.get_attribute("href") or "").split("?")[0]
                        if (text and 4 <= len(text) <= 200
                                and href not in seen_urls):
                            if not href.startswith("http"):
                                href = "https://builtin.com" + href
                            seen_urls.add(href)
                            jobs.append({
                                "source": "Built In",
                                "date_found": date.today().isoformat(),
                                "position": text,
                                "job_url": href,
                            })
                except Exception:
                    pass

            log.info(f"[JobSearch] Built In: {len(jobs)} listings")
        finally:
            driver.quit()

        return jobs

    # ── Dedup against tracker DB ─────────────────────────────────────────────

    @staticmethod
    def _dedup_jobs(jobs: list[dict]) -> list[dict]:
        """Filter out jobs already present in the job tracker database."""
        from talents.job_tracker import _DB, _data_dir as tracker_data_dir
        from talents.job_tracker import _normalize_company

        db_path = os.path.join(tracker_data_dir(), "job_tracker.db")
        if not os.path.exists(db_path):
            return list(jobs)

        db = _DB(db_path)
        existing = db.list_all()

        existing_keys: set[tuple[str, str]] = set()
        existing_urls: set[str] = set()
        for app in existing:
            existing_keys.add((
                _normalize_company(app["company"]),
                app["position"].lower().strip(),
            ))
            if app.get("job_url"):
                existing_urls.add(app["job_url"].split("?")[0])

        new: list[dict] = []
        for job in jobs:
            url_base = (job.get("job_url") or "").split("?")[0]
            if url_base and url_base in existing_urls:
                continue
            key = (
                _normalize_company(job.get("company", "")),
                job.get("position", "").lower().strip(),
            )
            if key in existing_keys:
                continue
            new.append(job)
            existing_keys.add(key)
            if url_base:
                existing_urls.add(url_base)

        return new

    # ── Add to tracker DB ────────────────────────────────────────────────────

    @staticmethod
    def _add_to_tracker(jobs: list[dict]) -> list[dict]:
        """Insert new jobs into the job_tracker database."""
        from talents.job_tracker import _DB, _data_dir as tracker_data_dir

        db_path = os.path.join(tracker_data_dir(), "job_tracker.db")
        db = _DB(db_path)

        added: list[dict] = []
        for job in jobs:
            try:
                app_id = db.add_application(
                    company=job.get("company", "Unknown"),
                    position=job.get("position", ""),
                    location=job.get("location", ""),
                    source=job.get("source", ""),
                    status="new",
                    date_found=job.get("date_found", date.today().isoformat()),
                    job_url=job.get("job_url", ""),
                    salary_range=job.get("salary_range", ""),
                )
                job["id"] = app_id
                added.append(job)
            except Exception as e:
                log.error(f"[JobSearch] Failed to add {job.get('company')}: {e}")

        if added:
            log.info(f"[JobSearch] Added {len(added)} new applications to tracker")
        return added

    # ── Automated fit analysis via Claude CLI ──────────────────────────────

    def _run_fit_analysis(self, jobs: list[dict]) -> int:
        """Run fit analysis in a background thread using claude -p.

        Calls Claude CLI to score each job against the user's resume,
        then writes fit_score, notes, and recommendation back to the
        tracker database.  Runs in a daemon thread so it doesn't block
        the user response.
        """
        if not jobs:
            return 0

        count = len(jobs)
        thread = threading.Thread(
            target=self._fit_analysis_worker,
            args=(list(jobs),),
            daemon=True,
        )
        thread.start()
        log.info(f"[JobSearch] Fit analysis started in background ({count} jobs)")
        return count

    def _fit_analysis_worker(self, jobs: list[dict]) -> None:
        """Background worker: call claude -p for fit analysis, store results.

        Processes in batches of 5 to avoid output limits and URL fetch
        timeouts from overwhelming a single call.
        """
        batch_size = 5
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i + batch_size]
            try:
                self._score_batch(batch)
            except Exception as e:
                log.error(f"[JobSearch] Batch {i // batch_size + 1} failed: {e}")

    def _score_batch(self, jobs: list[dict]) -> None:
        """Score a single batch of jobs via claude -p."""
        resume_path = Path.home() / "OneDrive" / "Documents" / "resume_master.md"

        # Read resume content directly so Claude doesn't need file access
        try:
            resume_text = resume_path.read_text(encoding="utf-8")
        except Exception as e:
            log.error(f"[JobSearch] Cannot read resume: {e}")
            return

        job_lines = []
        for j in jobs:
            parts = [f"tracker_id={j.get('id')}"]
            parts.append(f"company={j.get('company', 'Unknown')}")
            parts.append(f"position={j.get('position', '')}")
            if j.get("location"):
                parts.append(f"location={j['location']}")
            job_lines.append(" | ".join(parts))

        prompt = (
            "TASK: Score each job listing for fit against this resume.\n\n"
            f"RESUME:\n{resume_text}\n\n"
            "JOB LISTINGS:\n"
            + "\n".join(job_lines) + "\n\n"
            "For each job, score fit 0-100 based on title, company, and "
            "location match to the resume. 80+ = strong match on seniority "
            "and domain. 50-79 = partial match. Below 50 = poor fit.\n\n"
            "Return ONLY a JSON array, nothing else. Each element:\n"
            '{"tracker_id": <int>, "fit_score": <int>, '
            '"analysis": "<2-3 sentences>", '
            '"recommendation": "<apply|skip|maybe>"}'
        )

        try:
            claude_bin = shutil.which("claude")
            if not claude_bin:
                log.error("[JobSearch] claude CLI not found in PATH")
                return

            result = subprocess.run(
                [claude_bin, "-p", "--output-format", "text"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(Path.home()),
            )

            if result.returncode != 0:
                log.error(
                    f"[JobSearch] claude -p failed (rc={result.returncode}): "
                    f"{result.stderr[:200]}"
                )
                return

            raw = result.stdout.strip()
            if not raw:
                log.error(
                    f"[JobSearch] claude -p returned empty output. "
                    f"stderr: {result.stderr[:300]}"
                )
                return

            # Strip markdown fences if Claude wraps them
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())

            # Extract JSON array from anywhere in the response
            # (Claude sometimes adds preamble text before the JSON)
            arr_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if not arr_match:
                log.error(
                    f"[JobSearch] No JSON array found in response. "
                    f"First 500 chars: {raw[:500]}"
                )
                return

            scores = json.loads(arr_match.group())
            if not isinstance(scores, list):
                log.error("[JobSearch] Fit analysis returned non-list JSON")
                return

            self._store_fit_results(scores)

        except subprocess.TimeoutExpired:
            log.error("[JobSearch] claude -p timed out (300s)")
        except json.JSONDecodeError as e:
            log.error(
                f"[JobSearch] Fit analysis JSON parse error: {e}. "
                f"First 500 chars: {raw[:500]}"
            )
        except Exception as e:
            log.error(f"[JobSearch] Fit analysis failed: {e}")

    @staticmethod
    def _store_fit_results(scores: list[dict]) -> None:
        """Write fit analysis results back to the job tracker database."""
        from talents.job_tracker import _DB, _data_dir as tracker_data_dir

        db_path = os.path.join(tracker_data_dir(), "job_tracker.db")
        db = _DB(db_path)

        updated = 0
        for entry in scores:
            app_id = entry.get("tracker_id")
            if not app_id:
                continue
            try:
                fields = {}
                if "fit_score" in entry:
                    fields["fit_score"] = int(entry["fit_score"])
                note_parts = []
                if entry.get("analysis"):
                    note_parts.append(entry["analysis"])
                if entry.get("recommendation"):
                    note_parts.append(
                        f"Recommendation: {entry['recommendation']}"
                    )
                if note_parts:
                    fields["notes"] = " | ".join(note_parts)
                if fields:
                    db.update_application(app_id, **fields)
                    updated += 1
            except Exception as e:
                log.error(
                    f"[JobSearch] Failed to store fit for #{app_id}: {e}"
                )

        log.info(f"[JobSearch] Fit analysis complete: {updated}/{len(scores)} updated")


# ── Response helpers ─────────────────────────────────────────────────────────

def _ok(response: str, *, actions: list | None = None) -> dict:
    return {
        "success": True,
        "response": response,
        "actions_taken": actions or [],
        "spoken": False,
    }


def _fail(response: str) -> dict:
    return {
        "success": False,
        "response": response,
        "actions_taken": [],
        "spoken": False,
    }
