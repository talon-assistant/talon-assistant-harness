"""Web browser talent — fetch and summarise web pages.

Supports:
  - RSS feeds for known news sites (clean, reliable headlines)
  - trafilatura for general article/page extraction
  - requests + BeautifulSoup fallback

The planner can also use this as a sub-step: fetch page → answer question.
"""
from __future__ import annotations

import re
import os
from talents.base import BaseTalent
from core.llm_client import LLMError

import logging
log = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

MAX_CHARS = 24_000   # ~6 000 tokens — leaves headroom for prompt + response

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Built-in fallback domain→RSS map.  Overridden/extended at runtime from
# settings.json ("news_digest.feeds" and "web_browser.rss_feeds").
_RSS_FEEDS_DEFAULT: dict[str, str] = {
    # CNN's rss.cnn.com feeds are abandoned/stale — omitted intentionally
    "bbc.com":              "https://feeds.bbci.co.uk/news/rss.xml",
    "bbc.co.uk":            "https://feeds.bbci.co.uk/news/rss.xml",
    "reuters.com":          "https://feeds.reuters.com/reuters/topNews",
    "apnews.com":           "https://rsshub.app/apnews/topics/apf-topnews",
    "npr.org":              "https://feeds.npr.org/1001/rss.xml",
    "theguardian.com":      "https://www.theguardian.com/world/rss",
    "nytimes.com":          "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "foxnews.com":          "https://moxie.foxnews.com/google-publisher/latest.xml",
    "washingtonpost.com":   "https://feeds.washingtonpost.com/rss/world",
    "techcrunch.com":       "https://techcrunch.com/feed/",
    "arstechnica.com":      "https://feeds.arstechnica.com/arstechnica/index",
    "wired.com":            "https://www.wired.com/feed/rss",
    "theverge.com":         "https://www.theverge.com/rss/index.xml",
    "ycombinator.com":      "https://news.ycombinator.com/rss",
    "hackernews":           "https://news.ycombinator.com/rss",
    "engadget.com":         "https://www.engadget.com/rss.xml",
    "zdnet.com":            "https://www.zdnet.com/news/rss.xml",
    "krebsonsecurity.com":  "https://krebsonsecurity.com/feed/",
    "schneier.com":         "https://www.schneier.com/feed/atom/",
    "thehackernews.com":    "https://feeds.feedburner.com/TheHackersNews",
    "bleepingcomputer.com": "https://www.bleepingcomputer.com/feed/",
    "isc.sans.edu":         "https://isc.sans.edu/rssfeed.xml",
}

# How old (in hours) a feed's newest entry can be before we skip it and
# fall back to a direct page fetch.  Prevents abandoned feeds from surfacing
# year-old content as if it were current news.
_RSS_MAX_AGE_HOURS = 48


def _normalise_url(raw: str) -> str:
    """Ensure url has a scheme; strip trailing punctuation."""
    raw = raw.strip().rstrip(".,;:)'\"")
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    return raw


def _domain_from_url(url: str) -> str:
    """Extract bare domain (e.g. 'cnn.com') from a full URL."""
    m = re.search(r'(?:https?://)?(?:www\.)?([^/?\s]+)', url)
    return m.group(1).lower() if m else ""


# ── talent ───────────────────────────────────────────────────────────────────

class WebBrowserTalent(BaseTalent):
    name        = "web_browser"
    description = (
        "Fetch and read web pages, URLs, and websites. "
        "Summarise news headlines, read articles, check what a site says. "
        "Use when the user mentions a URL, domain, or asks to browse/visit/read a website."
    )
    examples = [
        "summarize the headlines at cnn.com",
        "what's the top news on bbc.com today",
        "read the article at https://techcrunch.com/2024/01/example",
        "go to reuters.com and give me the top stories",
        "fetch https://example.com and tell me what it says",
        "what does theverge.com say about AI today",
        "check hacker news",
        "browse to arstechnica.com",
        "what are the latest stories on npr.org",
        "connect to dallasobserver.com and find things to do this weekend",
        "connect to the dallas observer website",
        "can you connect to that site and read it",
    ]
    keywords = [
        "headlines at", "stories at", "browse to", "fetch http",
        "read http", "visit http", "go to http", "summarize http",
        "what does .com", "hacker news",
    ]
    priority = 47   # above most talents, below planner (85)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self, config: dict) -> None:
        """Build runtime RSS map from defaults + config feeds."""
        # Start from built-in defaults
        self._rss_feeds: dict[str, str] = dict(_RSS_FEEDS_DEFAULT)

        # Auto-register feeds listed under news_digest (single source of truth)
        for feed in config.get("news_digest", {}).get("feeds", []):
            url = feed.get("url", "")
            if url:
                domain = _domain_from_url(url)
                if domain:
                    self._rss_feeds[domain] = url

        # Allow explicit web_browser.rss_feeds overrides in settings.json
        for domain, url in config.get("web_browser", {}).get("rss_feeds", {}).items():
            if url:
                self._rss_feeds[domain] = url
            else:
                # null/empty entry = explicitly disable this domain's RSS
                self._rss_feeds.pop(domain, None)

    # ── URL / domain extraction ───────────────────────────────────────────────

    def _extract_url(self, command: str, llm, context: dict | None = None) -> str | None:
        """Return a fully-qualified URL from the command, or None."""
        # 1. Explicit http(s) URL
        m = re.search(r'https?://\S+', command)
        if m:
            return _normalise_url(m.group(0))

        # 2. Bare domain pattern (e.g. "cnn.com", "www.bbc.co.uk")
        m = re.search(
            r'\b(?:www\.)?'
            r'([a-z0-9-]+\.(?:com|org|net|io|gov|co\.uk|news|ai|dev))'
            r'\b', command, re.IGNORECASE
        )
        if m:
            return _normalise_url(m.group(0))

        # 3. Known shorthand names
        cmd_lower = command.lower()
        for alias, feed in (
            ("hacker news", "https://news.ycombinator.com"),
            ("ycombinator", "https://news.ycombinator.com"),
            ("reddit", "https://www.reddit.com"),
        ):
            if alias in cmd_lower:
                return alias.replace(" ", "")   # return alias key for RSS lookup

        # 4. Scan conversation buffer for URLs from recent assistant responses.
        # When the user says "summarize the culturemap list", the previous
        # web_search response may have included a culturemap.com URL.
        assistant = (context or {}).get("assistant")
        if assistant and hasattr(assistant, "conversation_buffer"):
            url_re = re.compile(r'https?://[^\s)\]>,"\']+')
            for entry in reversed(list(assistant.conversation_buffer)):
                if entry.get("role") != "talon":
                    continue
                urls = url_re.findall(entry.get("text", ""))
                if not urls:
                    continue
                # If only one URL, use it. If multiple, ask the LLM to pick.
                if len(urls) == 1:
                    log.info(f"[WebBrowser] URL from buffer: {urls[0]}")
                    return _normalise_url(urls[0])
                pick = self._extract_arg(
                    llm,
                    f"Command: {command}\nURLs found: {', '.join(urls)}",
                    "the single URL most relevant to the command",
                    max_length=120,
                )
                if pick and pick.upper() != "NONE" and pick.startswith("http"):
                    log.info(f"[WebBrowser] URL from buffer (LLM pick): {pick}")
                    return _normalise_url(pick)
                # Fall through to LLM extraction

        # 5. LLM extraction — last resort
        raw = self._extract_arg(
            llm, command, "website URL or domain name (e.g. cnn.com)", max_length=50
        )
        if raw and raw.upper() != "NONE":
            return _normalise_url(raw)

        return None

    # ── fetchers ─────────────────────────────────────────────────────────────

    def _fetch_rss(self, domain: str, url: str) -> str | None:
        """Try RSS feed for known news sites; return formatted headlines or None."""
        feed_url = self._rss_feeds.get(domain) or self._rss_feeds.get(url)
        if not feed_url:
            return None
        try:
            import feedparser
            import time as _time
            feed = feedparser.parse(feed_url)
            entries = feed.entries
            if not entries:
                return None

            # Staleness check — skip feeds whose newest entry is too old
            newest_ts = None
            for e in entries[:5]:
                t = e.get("published_parsed") or e.get("updated_parsed")
                if t:
                    ts = _time.mktime(t)
                    if newest_ts is None or ts > newest_ts:
                        newest_ts = ts
            if newest_ts is not None:
                age_h = (_time.time() - newest_ts) / 3600
                if age_h > _RSS_MAX_AGE_HOURS:
                    log.warning(f"[WebBrowser] RSS stale ({age_h:.0f}h old) — skipping, "
                          f"falling back to direct fetch")
                    return None

            feed_title = feed.feed.get("title", domain)
            lines = [f"Headlines from {feed_title}:\n"]
            for entry in entries[:20]:
                title   = entry.get("title",   "").strip()
                summary = entry.get("summary", "").strip()
                # Strip HTML tags from summary
                summary = re.sub(r'<[^>]+>', '', summary).strip()
                if title:
                    lines.append(f"• {title}")
                    if summary and len(summary) < 300:
                        lines.append(f"  {summary}")
            return "\n".join(lines)
        except ImportError:
            log.warning("[WebBrowser] feedparser not installed — skipping RSS")
            return None
        except Exception as e:
            log.error(f"[WebBrowser] RSS fetch error: {e}")
            return None

    def _fetch_page(self, url: str) -> str | None:
        """Fetch arbitrary URL; try trafilatura first, fall back to BS4."""
        try:
            import requests
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            log.error(f"[WebBrowser] HTTP fetch error: {e}")
            return None

        # trafilatura — best for article extraction
        try:
            import trafilatura
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )
            if text and len(text.split()) > 40:
                return text
        except ImportError:
            pass
        except Exception as e:
            log.error(f"[WebBrowser] trafilatura error: {e}")

        # BeautifulSoup fallback
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer",
                              "header", "aside", "form"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            text = re.sub(r'\n{3,}', '\n\n', text)
            if len(text.split()) > 20:
                return text
        except Exception as e:
            log.error(f"[WebBrowser] BS4 error: {e}")

        return None

    # ── execute ───────────────────────────────────────────────────────────────

    def execute(self, command: str, context: dict) -> dict:
        llm            = context["llm"]
        speak_response = context.get("speak_response", False)
        voice          = context.get("voice")

        # ── Step 1: resolve URL ───────────────────────────────────────────────
        url = self._extract_url(command, llm, context)
        if not url:
            return {"response": "", "talent": self.name, "success": False}

        domain = _domain_from_url(url)
        log.info(f"[WebBrowser] Target: {url}  (domain={domain})")

        # ── Step 2: fetch ─────────────────────────────────────────────────────
        content      = None
        source_label = "web page"
        wants_news   = any(w in command.lower()
                           for w in ("headline", "news", "top stor", "latest"))

        # Try RSS first when asking for headlines / news
        if wants_news or domain in self._rss_feeds or url in self._rss_feeds:
            content = self._fetch_rss(domain, url)
            if content:
                source_label = "RSS feed"

        # Direct page fetch
        if not content:
            if url.startswith("http"):
                content = self._fetch_page(url)

        if not content:
            return {
                "success": False,
                "response": (
                    f"I wasn't able to retrieve content from {url}. "
                    "The site may block automated access or require JavaScript rendering."
                ),
                "actions_taken": [{"action": "web_fetch", "url": url, "success": False}],
            }

        # Truncate
        if len(content) > MAX_CHARS:
            content = content[:MAX_CHARS] + f"\n\n[Content truncated at {MAX_CHARS} chars]"
        log.info(f"[WebBrowser] Got {len(content):,} chars via {source_label}")

        # ── Step 3: summarise / answer ────────────────────────────────────────
        # If command is "open [browser] and go to X", reframe as a content request
        # so the LLM doesn't respond with "I cannot open a browser"
        display_command = command.strip()
        if re.search(
            r'\b(?:open|launch|start)\b.*\b(?:browser|chrome|firefox|edge|safari)\b',
            display_command, re.IGNORECASE
        ):
            display_command = f"Show me what is on {url} — summarise the main content or headlines."

        prompt = (
            f"The following content was already fetched from {url} ({source_label}).\n"
            f"Present this content helpfully to the user. "
            f"Do NOT say you cannot open browsers or navigate — the content is already here.\n"
            f"Use ONLY the fetched content below. Do not add information from training data.\n\n"
            f"--- FETCHED CONTENT ---\n{content}\n--- END CONTENT ---\n\n"
            f"User request: {display_command}"
        )
        try:
            response = llm.generate(prompt, max_length=600)
        except LLMError as e:
            return {"success": False, "response": f"LLM unavailable: {e}", "actions_taken": [], "spoken": False}

        if speak_response and voice:
            voice.speak(response)
            return {
                "success": True,
                "response":      response,
                "actions_taken": [{"action": "web_fetch",
                                   "url": url, "source": source_label}],
                "spoken":        True,
            }

        return {
            "success":       True,
            "response":      response,
            "actions_taken": [{"action": "web_fetch",
                               "url": url, "source": source_label}],
        }
