"""JobSearchTalent -- automated job search across LinkedIn, Dice, and Built In.

Scrapes job listings from configured search URLs, identifies new postings
not yet in the job tracker, adds them automatically, and optionally sends
a batch to Cowork for resume-fit analysis.

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
import time
from datetime import date, datetime
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
        "job search", "job hunt", "hunt for jobs", "search for jobs",
        "find jobs", "check linkedin", "check dice", "new job listings",
        "search url", "search urls",
    ]
    examples = [
        "search for jobs",
        "run a job hunt",
        "check for new job listings",
        "add a search URL https://dice.com/jobs?q=...",
        "show my search URLs",
        "remove the first search URL",
        "job search login",
    ]
    priority = 58
    required_packages = ["selenium"]

    def __init__(self) -> None:
        super().__init__()
        self._profile_dir = os.path.join(_data_dir(), "job_search_chrome_profile")
        self._config_file = os.path.join(_data_dir(), "job_search_config.json")
        self._search_config: dict[str, Any] = {"urls": [], "auto_cowork": True}
        self._load_search_config()

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
                    "label": "Auto-send new listings to Cowork for fit analysis",
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

        # Cowork handoff
        cowork_sent = 0
        if self._search_config.get("auto_cowork") and added:
            cowork_sent = self._send_to_cowork(added)

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
        if cowork_sent:
            lines.append(f"\nSent {cowork_sent} to Cowork for fit analysis.")

        notify = context.get("notify")
        if notify:
            notify("Job Search", f"{len(added)} new listing(s) found")

        return _ok(
            "\n".join(lines),
            actions=[{
                "action": "search",
                "found": len(all_jobs),
                "new": len(added),
                "cowork": cowork_sent,
                "sites": list(sites_searched),
            }],
        )

    # ═══════════════════════════════════════════════════════════════════════════
    #  Site-specific scrapers
    # ═══════════════════════════════════════════════════════════════════════════

    # ── LinkedIn ─────────────────────────────────────────────────────────────

    def _scrape_linkedin(self, url: str) -> list[dict]:
        """Scrape job listings from a LinkedIn search results page."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        driver = self._create_driver(headless=True)
        jobs: list[dict] = []

        try:
            driver.get(url)
            time.sleep(3)

            if "/login" in driver.current_url or "authwall" in driver.current_url:
                raise RuntimeError(
                    "LinkedIn requires login \u2014 say 'job search login'"
                )

            try:
                WebDriverWait(driver, 12).until(
                    EC.presence_of_element_located((
                        By.CSS_SELECTOR,
                        "[data-job-id], .job-search-card, "
                        ".jobs-search-results-list, "
                        ".scaffold-layout__list-container",
                    ))
                )
            except Exception:
                pass

            time.sleep(2)
            jobs = self._collect_cards(
                driver, self._parse_linkedin_card, "LinkedIn"
            )
            log.info(f"[JobSearch] LinkedIn: {len(jobs)} listings")
        finally:
            driver.quit()

        return jobs

    @staticmethod
    def _parse_linkedin_card(card) -> dict | None:
        from selenium.webdriver.common.by import By

        job: dict[str, str] = {
            "source": "LinkedIn",
            "date_found": date.today().isoformat(),
        }

        job_id = card.get_attribute("data-job-id")
        if job_id:
            job["job_url"] = f"https://www.linkedin.com/jobs/view/{job_id}"

        # Title
        for sel in (".job-card-list__title", ".base-search-card__title",
                     "a.job-card-container__link",
                     ".artdeco-entity-lockup__title",
                     "a[data-control-name='job_card_title']"):
            for elem in card.find_elements(By.CSS_SELECTOR, sel):
                text = elem.text.strip()
                if 3 < len(text) < 200:
                    job["position"] = text
                    if not job.get("job_url"):
                        href = elem.get_attribute("href") or ""
                        if "linkedin.com" in href:
                            job["job_url"] = href.split("?")[0]
                    break
            if job.get("position"):
                break

        if not job.get("position"):
            for a in card.find_elements(By.TAG_NAME, "a"):
                text = a.text.strip()
                if 3 < len(text) < 200:
                    job["position"] = text
                    href = a.get_attribute("href") or ""
                    if "linkedin.com" in href and not job.get("job_url"):
                        job["job_url"] = href.split("?")[0]
                    break

        # Company
        for sel in (".job-card-container__company-name",
                     ".base-search-card__subtitle",
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
                     ".job-search-card__location",
                     ".artdeco-entity-lockup__caption"):
            for elem in card.find_elements(By.CSS_SELECTOR, sel):
                text = elem.text.strip()
                if text and ("remote" in text.lower() or "," in text
                             or len(text) > 5):
                    job["location"] = text
                    break
            if job.get("location"):
                break

        # Salary
        for elem in card.find_elements(
            By.CSS_SELECTOR,
            ".job-card-container__metadata-item--salary, .salary-text"
        ):
            text = elem.text.strip()
            if "$" in text or "k" in text.lower():
                job["salary_range"] = text
                break

        return job if job.get("position") else None

    # ── Dice ─────────────────────────────────────────────────────────────────

    def _scrape_dice(self, url: str) -> list[dict]:
        """Scrape job listings from Dice.com search results."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        driver = self._create_driver(headless=True)
        jobs: list[dict] = []

        try:
            driver.get(url)
            time.sleep(3)

            # Wait for Dice's JS-rendered job cards
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((
                        By.CSS_SELECTOR,
                        "a.card-title-link, "
                        "[data-cy='card-title-link'], "
                        "dhi-search-card, "
                        ".search-card",
                    ))
                )
            except Exception:
                pass

            time.sleep(2)

            # Dice renders search results as <dhi-search-card> custom elements
            # or as cards with a.card-title-link anchors
            cards = (
                driver.find_elements(By.CSS_SELECTOR, "dhi-search-card")
                or driver.find_elements(By.CSS_SELECTOR, ".search-card")
            )

            if cards:
                jobs = self._parse_dice_cards(cards)
            else:
                # Fallback: extract from title links directly
                jobs = self._parse_dice_links(driver)

            # Scroll for more results
            if len(jobs) < 20:
                try:
                    driver.execute_script(
                        "window.scrollTo(0, document.body.scrollHeight)"
                    )
                    time.sleep(3)
                    cards2 = (
                        driver.find_elements(By.CSS_SELECTOR, "dhi-search-card")
                        or driver.find_elements(By.CSS_SELECTOR, ".search-card")
                    )
                    if cards2:
                        more = self._parse_dice_cards(cards2)
                    else:
                        more = self._parse_dice_links(driver)
                    # Merge only new ones
                    seen = {j.get("job_url", j["position"]) for j in jobs}
                    for j in more:
                        uid = j.get("job_url", j["position"])
                        if uid not in seen:
                            seen.add(uid)
                            jobs.append(j)
                except Exception:
                    pass

            log.info(f"[JobSearch] Dice: {len(jobs)} listings")
        finally:
            driver.quit()

        return jobs

    @staticmethod
    def _parse_dice_cards(cards) -> list[dict]:
        """Parse Dice <dhi-search-card> or .search-card elements."""
        from selenium.webdriver.common.by import By
        jobs: list[dict] = []
        seen: set[str] = set()

        for card in cards:
            job: dict[str, str] = {
                "source": "Dice",
                "date_found": date.today().isoformat(),
            }

            # Title + URL — prefer data-cy selectors (stable)
            for sel in ("a[data-cy='card-title-link']",
                        "a.card-title-link", "h5 a", "a"):
                elems = card.find_elements(By.CSS_SELECTOR, sel)
                for elem in elems:
                    text = elem.text.strip()
                    if 3 < len(text) < 200:
                        job["position"] = text
                        href = elem.get_attribute("href") or ""
                        if href and "dice.com" in href:
                            job["job_url"] = href.split("?")[0]
                        break
                if job.get("position"):
                    break

            if not job.get("position"):
                continue

            # Company
            for sel in ("a[data-cy='search-result-company-name']",
                        "[data-cy='companyNameLink']",
                        ".card-company a",
                        ".card-company span",
                        "h6 a"):
                elems = card.find_elements(By.CSS_SELECTOR, sel)
                for elem in elems:
                    text = elem.text.strip()
                    if text and len(text) > 1:
                        job["company"] = text
                        break
                if job.get("company"):
                    break

            # Location
            for sel in ("[data-cy='search-result-location']",
                        "span.search-result-location",
                        ".card-location"):
                elems = card.find_elements(By.CSS_SELECTOR, sel)
                for elem in elems:
                    text = elem.text.strip()
                    if text:
                        job["location"] = text
                        break
                if job.get("location"):
                    break

            # Salary — Dice sometimes shows this on cards
            for sel in ("[data-cy='compensationText']",
                        ".card-salary"):
                elems = card.find_elements(By.CSS_SELECTOR, sel)
                for elem in elems:
                    text = elem.text.strip()
                    if "$" in text or "k" in text.lower():
                        job["salary_range"] = text
                        break
                if job.get("salary_range"):
                    break

            uid = job.get("job_url", job["position"])
            if uid not in seen:
                seen.add(uid)
                jobs.append(job)

        return jobs

    @staticmethod
    def _parse_dice_links(driver) -> list[dict]:
        """Fallback: extract jobs from a.card-title-link elements directly."""
        from selenium.webdriver.common.by import By
        jobs: list[dict] = []
        seen: set[str] = set()

        links = driver.find_elements(By.CSS_SELECTOR, "a.card-title-link")
        for link in links:
            text = link.text.strip()
            href = (link.get_attribute("href") or "").split("?")[0]
            if not text or len(text) < 3 or href in seen:
                continue
            seen.add(href)
            jobs.append({
                "source": "Dice",
                "date_found": date.today().isoformat(),
                "position": text,
                "company": "",  # Not available from link alone
                "job_url": href,
            })

        return jobs

    # ── Built In ─────────────────────────────────────────────────────────────

    def _scrape_builtin(self, url: str) -> list[dict]:
        """Scrape job listings from builtin.com search results."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        driver = self._create_driver(headless=True)
        jobs: list[dict] = []

        try:
            driver.get(url)
            time.sleep(3)

            # Wait for job cards to render
            try:
                WebDriverWait(driver, 12).until(
                    EC.presence_of_element_located((
                        By.CSS_SELECTOR,
                        "[data-id='job-card'], "
                        ".job-card, "
                        "[class*='JobCard'], "
                        "a[href*='/job/'], "
                        "[class*='job-list'] a",
                    ))
                )
            except Exception:
                pass

            time.sleep(2)

            # Built In renders job cards in various structures
            # Try specific card containers first, then fall back to link extraction
            cards = (
                driver.find_elements(By.CSS_SELECTOR, "[data-id='job-card']")
                or driver.find_elements(By.CSS_SELECTOR, ".job-card")
                or driver.find_elements(By.CSS_SELECTOR, "[class*='JobCard']")
            )

            if cards:
                jobs = self._parse_builtin_cards(cards)
            else:
                # Fallback: extract from job links on the page
                jobs = self._parse_builtin_links(driver)

            # Scroll for more
            if len(jobs) < 20:
                try:
                    driver.execute_script(
                        "window.scrollTo(0, document.body.scrollHeight)"
                    )
                    time.sleep(3)
                    # Try "Load more" or "Show more" button
                    for sel in ("button[class*='load-more']",
                                "button[class*='LoadMore']",
                                "a[class*='load-more']"):
                        btns = driver.find_elements(By.CSS_SELECTOR, sel)
                        if btns:
                            try:
                                btns[0].click()
                                time.sleep(3)
                            except Exception:
                                pass
                            break

                    cards2 = (
                        driver.find_elements(By.CSS_SELECTOR, "[data-id='job-card']")
                        or driver.find_elements(By.CSS_SELECTOR, ".job-card")
                        or driver.find_elements(By.CSS_SELECTOR, "[class*='JobCard']")
                    )
                    if cards2:
                        more = self._parse_builtin_cards(cards2)
                    else:
                        more = self._parse_builtin_links(driver)

                    seen = {j.get("job_url", j["position"]) for j in jobs}
                    for j in more:
                        uid = j.get("job_url", j["position"])
                        if uid not in seen:
                            seen.add(uid)
                            jobs.append(j)
                except Exception:
                    pass

            log.info(f"[JobSearch] Built In: {len(jobs)} listings")
        finally:
            driver.quit()

        return jobs

    @staticmethod
    def _parse_builtin_cards(cards) -> list[dict]:
        """Parse Built In job card elements."""
        from selenium.webdriver.common.by import By
        jobs: list[dict] = []
        seen: set[str] = set()

        for card in cards:
            job: dict[str, str] = {
                "source": "Built In",
                "date_found": date.today().isoformat(),
            }

            # Title + URL
            for sel in ("h2 a", "h3 a", "[class*='title'] a",
                        "[class*='Title'] a", "a[href*='/job/']"):
                elems = card.find_elements(By.CSS_SELECTOR, sel)
                for elem in elems:
                    text = elem.text.strip()
                    if 3 < len(text) < 200:
                        job["position"] = text
                        href = elem.get_attribute("href") or ""
                        if href:
                            if not href.startswith("http"):
                                href = "https://builtin.com" + href
                            job["job_url"] = href.split("?")[0]
                        break
                if job.get("position"):
                    break

            if not job.get("position"):
                continue

            # Company
            for sel in ("[class*='company'] a", "[class*='Company'] a",
                        "[class*='company-name']", "[class*='CompanyName']",
                        "span[class*='company']"):
                elems = card.find_elements(By.CSS_SELECTOR, sel)
                for elem in elems:
                    text = elem.text.strip()
                    if text and len(text) > 1:
                        job["company"] = text
                        break
                if job.get("company"):
                    break

            # Location
            for sel in ("[class*='location']", "[class*='Location']"):
                elems = card.find_elements(By.CSS_SELECTOR, sel)
                for elem in elems:
                    text = elem.text.strip()
                    if text:
                        job["location"] = text
                        break
                if job.get("location"):
                    break

            # Salary
            for sel in ("[class*='salary']", "[class*='Salary']",
                        "[class*='compensation']"):
                elems = card.find_elements(By.CSS_SELECTOR, sel)
                for elem in elems:
                    text = elem.text.strip()
                    if "$" in text or "k" in text.lower():
                        job["salary_range"] = text
                        break
                if job.get("salary_range"):
                    break

            uid = job.get("job_url", job["position"])
            if uid not in seen:
                seen.add(uid)
                jobs.append(job)

        return jobs

    @staticmethod
    def _parse_builtin_links(driver) -> list[dict]:
        """Fallback: extract jobs from a[href*='/job/'] links."""
        from selenium.webdriver.common.by import By
        jobs: list[dict] = []
        seen: set[str] = set()

        links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/job/']")
        for link in links:
            text = link.text.strip()
            href = (link.get_attribute("href") or "").split("?")[0]
            if not text or len(text) < 3 or href in seen:
                continue
            # Skip nav/footer links
            if len(text) > 200 or "/" not in href:
                continue
            seen.add(href)
            if not href.startswith("http"):
                href = "https://builtin.com" + href
            jobs.append({
                "source": "Built In",
                "date_found": date.today().isoformat(),
                "position": text,
                "company": "",
                "job_url": href,
            })

        return jobs

    # ═══════════════════════════════════════════════════════════════════════════
    #  Shared helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _collect_cards(self, driver, parser, site_name: str) -> list[dict]:
        """Generic card collector with scroll-to-load-more."""
        from selenium.webdriver.common.by import By

        all_selectors = {
            "LinkedIn": (
                "[data-job-id]",
                ".job-search-card",
                ".jobs-search-results__list-item",
                "li.ember-view.occludable-update",
            ),
        }
        selectors = all_selectors.get(site_name, ())

        cards = []
        for sel in selectors:
            cards = driver.find_elements(By.CSS_SELECTOR, sel)
            if cards:
                break

        jobs: list[dict] = []
        seen: set[str] = set()
        for card in cards:
            try:
                job = parser(card)
                if job and job.get("position"):
                    uid = job.get("job_url", job["position"])
                    if uid not in seen:
                        seen.add(uid)
                        jobs.append(job)
            except Exception as e:
                log.debug(f"[JobSearch] {site_name} card parse failed: {e}")

        # Scroll for more
        if len(jobs) < 25:
            try:
                driver.execute_script(
                    "window.scrollTo(0, document.body.scrollHeight)"
                )
                time.sleep(3)
                for sel in selectors:
                    more_cards = driver.find_elements(By.CSS_SELECTOR, sel)
                    if more_cards:
                        for card in more_cards:
                            try:
                                job = parser(card)
                                if job and job.get("position"):
                                    uid = job.get("job_url", job["position"])
                                    if uid not in seen:
                                        seen.add(uid)
                                        jobs.append(job)
                            except Exception:
                                pass
                        break
            except Exception:
                pass

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

    # ── Cowork handoff ───────────────────────────────────────────────────────

    @staticmethod
    def _send_to_cowork(jobs: list[dict]) -> int:
        """Batch-send newly found jobs to Cowork for fit analysis."""
        import uuid
        from pathlib import Path

        bridge_tasks = Path.home() / "OneDrive" / "Documents" / "cowork_bridge" / "tasks"
        bridge_tasks.mkdir(parents=True, exist_ok=True)

        task_id = f"job_fit_{uuid.uuid4().hex[:8]}"
        task = {
            "task_id": task_id,
            "task_type": "job_fit_batch",
            "created": datetime.now().isoformat(),
            "payload": {
                "jobs": [
                    {
                        "tracker_id": j.get("id"),
                        "company": j.get("company"),
                        "position": j.get("position"),
                        "location": j.get("location", ""),
                        "job_url": j.get("job_url", ""),
                        "source": j.get("source", ""),
                    }
                    for j in jobs
                ],
                "instructions": (
                    "For each job listing, visit the job URL to read the full "
                    "description, then evaluate fit against the user's resume. "
                    "Return a JSON array where each element has: tracker_id, "
                    "fit_score (0-100), analysis (2-3 sentences), and "
                    "recommendation (apply / skip / maybe)."
                ),
            },
        }

        task_path = bridge_tasks / f"{task_id}.json"
        try:
            with open(task_path, "w", encoding="utf-8") as f:
                json.dump(task, f, indent=2)
            log.info(
                f"[JobSearch] Cowork fit-analysis task: "
                f"{task_path.name} ({len(jobs)} jobs)"
            )
            return len(jobs)
        except Exception as e:
            log.error(f"[JobSearch] Failed to write Cowork task: {e}")
            return 0


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
