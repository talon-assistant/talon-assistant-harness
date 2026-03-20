import logging
import requests
from talents.base import BaseTalent
from ddgs import DDGS
from core.assistant import _wrap_external, _INJECTION_DEFENSE_CLAUSE

log = logging.getLogger(__name__)


class WebSearchTalent(BaseTalent):
    name = "web_search"
    description = (
        "Search the web for information, events, news, or facts using a search query. "
        "Use when the user wants to find something online without specifying a URL or website. "
        "Do NOT use for browsing a specific URL or domain."
    )
    keywords = ["search", "google", "look up", "find out",
                "what is", "who is", "how to", "when did", "where is",
                "define", "explain"]
    examples = [
        "search the web for events in Dallas this weekend",
        "search for best pizza places nearby",
        "find interesting things to do in Austin",
        "who is the CEO of Tesla",
        "how does photosynthesis work",
        "look up Python list comprehension",
        "what's happening in New York this week",
        "find out the latest news about SpaceX",
    ]
    priority = 60

    # System prompt that forces the model to be a search-result synthesizer
    _SEARCH_SYSTEM_PROMPT = (
        "You are a search-result answering assistant. "
        "You MUST answer using ONLY the search results provided in the user message. "
        "NEVER use your own knowledge or training data. "
        "If the search results contain the answer, state it directly with specific numbers and facts. "
        "If the search results do NOT contain enough information, say "
        "\"I couldn't find specific information about that in the search results.\" "
        "NEVER make up data. NEVER guess. NEVER say 'check website X'."
        + _INJECTION_DEFENSE_CLAUSE
    )

    # Conversational openers to strip before prefix detection
    _CONVERSATIONAL_OPENERS = [
        "can you ", "could you ", "would you ", "will you ",
        "please ", "are you able to ", "i need you to ",
        "i want you to ", "i'd like you to ",
    ]

    # Lighter touch on query cleaning — only strip command prefixes, keep the question
    _COMMAND_PREFIXES = [
        "search the web for", "search the internet for",
        "search for", "search on", "search",
        "google", "look up", "find out",
        "tell me about", "tell me",
    ]

    # Commands that refer to a *previous* query rather than stating one directly.
    # e.g. "can you web search for it" / "look that up" / "search for what I asked"
    _REFERENTIAL_TRIGGERS = [
        "search for it", "search it", "look it up", "look that up",
        "search that", "web search for it", "web search that",
        "search for what i asked", "search for the above",
        "search my question", "search for my question",
        "find it", "find that",
    ]

    # Commands that are complaints/corrections about a prior answer — not searches.
    # These should not be used as literal search queries.
    _META_COMPLAINT_TRIGGERS = [
        "this is not the answer", "not the answer i",
        "not what i was looking for", "this should have gone to",
        "this should have", "you should have", "that's not what",
        "that was not", "this was wrong", "wrong answer",
        "missed the point", "not correct", "that is incorrect",
        "not what i wanted", "that's wrong",
    ]

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    # ── Query resolution ───────────────────────────────────────────

    def _get_prior_query(self, context: dict) -> str | None:
        """Walk the conversation buffer backwards to find the last real user query.

        Skips referential commands ('search for it') and meta-complaints so we
        don't recursively resolve to another non-query turn.
        """
        assistant = context.get("assistant")
        if not assistant or not hasattr(assistant, "conversation_buffer"):
            return None
        for entry in reversed(list(assistant.conversation_buffer)):
            if entry["role"] != "user":
                continue
            text = entry["text"].lower()
            if any(t in text for t in self._REFERENTIAL_TRIGGERS):
                continue
            if any(t in text for t in self._META_COMPLAINT_TRIGGERS):
                continue
            return entry["text"]
        return None

    def _resolve_query(self, command: str, context: dict) -> str | None:
        """Return the search query to use, or None to decline the command.

        None → talent declines; assistant falls through to conversation path
              (correct for pure meta-complaints with no recoverable prior query).
        """
        cmd_lower = command.lower()

        # Case 1: Meta-complaint about a prior response (e.g. "that was wrong,
        # you should have searched for…").  Try to recover the original topic;
        # if we can't, decline so the conversation handler can explain.
        if any(t in cmd_lower for t in self._META_COMPLAINT_TRIGGERS):
            prior = self._get_prior_query(context)
            if prior:
                print(f"   [WebSearch] Meta-complaint detected; "
                      f"recovering prior query: {prior!r}")
                return prior
            print("   [WebSearch] Meta-complaint with no recoverable prior query "
                  "— declining to conversation.")
            return None

        # Case 2: Referential command ("search for it", "look that up", etc.).
        # Strip conversational openers first ("can you", "could you", etc.),
        # then strip command prefixes ("search the web for", "look up", etc.).
        stripped = cmd_lower
        for opener in self._CONVERSATIONAL_OPENERS:
            if stripped.startswith(opener):
                stripped = stripped[len(opener):].strip()
                break
        for prefix in self._COMMAND_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):].strip()
                break

        is_referential = (
            len(stripped) < 5
            or any(t in cmd_lower for t in self._REFERENTIAL_TRIGGERS)
        )
        if is_referential:
            prior = self._get_prior_query(context)
            if prior:
                print(f"   [WebSearch] Referential query; "
                      f"using prior query: {prior!r}")
                return prior
            # No prior query found — use the stripped text as best effort
            return command

        # Case 3: Normal query — use stripped form (or full command if too short)
        return stripped if len(stripped) >= 5 else command

    # ── Talent entry point ─────────────────────────────────────────

    def execute(self, command: str, context: dict) -> dict:
        search_query = self._resolve_query(command, context)

        # Decline: meta-complaint with no recoverable prior query
        if search_query is None:
            return {"success": False, "response": "", "actions_taken": []}

        # Perform search (use configured max_results)
        max_results = self._config.get("max_results", 5)
        provider = self._config.get("provider", "DuckDuckGo")
        web_results = self._search(search_query, max_results=max_results,
                                   provider=provider)

        # Log what we got back for debugging
        print(f"   -> Search query sent: '{search_query}' via {provider}")
        print(f"   -> Results preview: {web_results[:300]}...")

        # Semantic injection check before passing web content to LLM
        from core.security import get_security_filter as _gsf
        _sf = _gsf()
        if _sf:
            _blocked, _alert = _sf.check_semantic_input(web_results, "web")
            if _blocked:
                return {"response": "[Search results blocked by security filter]",
                        "actions_taken": [], "success": False}

        # Build the user message — wrap results in structural injection-defence markers
        user_message = (
            f"{_wrap_external(web_results, 'web search results')}\n\n"
            f"Question: {command}\n\n"
            f"Answer the question using ONLY the search results above. "
            f"Include specific facts, numbers, and data found in the results."
        )

        # Ask LLM to synthesize — use system prompt + low temperature for factual grounding
        llm = context["llm"]
        response = llm.generate(
            user_message,
            system_prompt=self._SEARCH_SYSTEM_PROMPT,
            temperature=0.3,  # low temp = stick to the facts in the prompt
        )

        print(f"   -> LLM response: {response[:200]}...")

        # Harvest as training pair — web search results are high-signal facts
        # the base model may not have known, worth learning from.
        if context.get("config", {}).get("training", {}).get("harvest_pairs", True):
            try:
                from core.training_harvester import append_training_pair
                append_training_pair(command, response, source="web_search")
            except Exception as e:
                print(f"   [Harvest] Failed: {e}")

        return {
            "success": True,
            "response": response,
            "actions_taken": [{"action": "web_search", "query": search_query,
                               "provider": provider}],
            "spoken": False
        }

    def get_config_schema(self) -> dict:
        return {
            "fields": [
                {"key": "provider", "label": "Search Provider",
                 "type": "choice", "default": "DuckDuckGo",
                 "choices": ["DuckDuckGo", "Google", "Bing", "SearXNG"]},
                {"key": "api_key", "label": "API Key",
                 "type": "password", "default": ""},
                {"key": "api_endpoint", "label": "API Endpoint (SearXNG)",
                 "type": "string", "default": ""},
                {"key": "max_results", "label": "Max Search Results",
                 "type": "int", "default": 5, "min": 1, "max": 20},
            ]
        }

    # ── Provider dispatch ──────────────────────────────────────────

    def _search(self, query, max_results=5, provider="DuckDuckGo"):
        """Search the web using the selected provider."""
        providers = {
            "DuckDuckGo": self._search_duckduckgo,
            "Google": self._search_google,
            "Bing": self._search_bing,
            "SearXNG": self._search_searxng,
        }
        search_fn = providers.get(provider, self._search_duckduckgo)
        return search_fn(query, max_results)

    # ── DuckDuckGo (default, no key) ──────────────────────────────

    def _search_duckduckgo(self, query, max_results):
        """Search via DuckDuckGo (free, no API key)."""
        try:
            print(f"   -> Searching DuckDuckGo: '{query}'")
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if not results:
                print("   -> WARNING: DuckDuckGo returned zero results!")
                return "No search results found."

            print(f"   -> Got {len(results)} results from DuckDuckGo")
            return self._format_results(results,
                                        title_key="title", body_key="body",
                                        url_key="href")
        except Exception as e:
            print(f"   -> ERROR in DuckDuckGo search: {e}")
            return f"Error searching web: {str(e)}"

    # ── Google Custom Search ──────────────────────────────────────

    def _search_google(self, query, max_results):
        """Search via Google Custom Search API (requires API key + CX).

        API key format in config: "<api_key>:<cx_id>"
        e.g. "AIzaSyABC123:017576662512468239146:abcdef12345"
        """
        api_key = self._config.get("api_key", "")
        if not api_key or ":" not in api_key:
            return ("Google Search requires an API key and Search Engine ID.\n"
                    "Set the API Key field to:  YOUR_API_KEY:YOUR_CX_ID\n"
                    "Get keys at https://programmablesearchengine.google.com/")

        key, cx = api_key.split(":", 1)

        try:
            print(f"   -> Searching Google CSE: '{query}'")
            url = "https://www.googleapis.com/customsearch/v1"
            params = {"key": key, "cx": cx, "q": query, "num": min(max_results, 10)}
            resp = requests.get(url, params=params, timeout=15)

            if resp.status_code == 403:
                return "Google API key invalid or quota exceeded."
            resp.raise_for_status()

            data = resp.json()
            items = data.get("items", [])

            if not items:
                return "No search results found."

            print(f"   -> Got {len(items)} results from Google")
            formatted = ""
            for i, item in enumerate(items, 1):
                formatted += f"Result {i}:\n"
                formatted += f"  Title: {item.get('title', '')}\n"
                formatted += f"  Content: {item.get('snippet', '')}\n"
                formatted += f"  URL: {item.get('link', '')}\n\n"
            return formatted

        except Exception as e:
            print(f"   -> ERROR in Google search: {e}")
            return f"Error searching Google: {str(e)}"

    # ── Bing Web Search ───────────────────────────────────────────

    def _search_bing(self, query, max_results):
        """Search via Bing Web Search API v7 (requires API key).

        Get a key at https://www.microsoft.com/en-us/bing/apis/bing-web-search-api
        """
        api_key = self._config.get("api_key", "")
        if not api_key:
            return ("Bing Search requires an API key.\n"
                    "Get one at https://www.microsoft.com/en-us/bing/apis/bing-web-search-api")

        try:
            print(f"   -> Searching Bing: '{query}'")
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": api_key}
            params = {"q": query, "count": max_results, "mkt": "en-US"}
            resp = requests.get(url, headers=headers, params=params, timeout=15)

            if resp.status_code == 401:
                return "Bing API key is invalid."
            resp.raise_for_status()

            data = resp.json()
            pages = data.get("webPages", {}).get("value", [])

            if not pages:
                return "No search results found."

            print(f"   -> Got {len(pages)} results from Bing")
            formatted = ""
            for i, page in enumerate(pages, 1):
                formatted += f"Result {i}:\n"
                formatted += f"  Title: {page.get('name', '')}\n"
                formatted += f"  Content: {page.get('snippet', '')}\n"
                formatted += f"  URL: {page.get('url', '')}\n\n"
            return formatted

        except Exception as e:
            print(f"   -> ERROR in Bing search: {e}")
            return f"Error searching Bing: {str(e)}"

    # ── SearXNG (self-hosted) ─────────────────────────────────────

    def _search_searxng(self, query, max_results):
        """Search via a SearXNG instance (self-hosted, no API key required usually).

        Set the API Endpoint field to your SearXNG base URL,
        e.g. "https://search.example.com"
        """
        endpoint = self._config.get("api_endpoint", "").rstrip("/")
        if not endpoint:
            return ("SearXNG requires an API endpoint.\n"
                    "Set the API Endpoint field to your SearXNG URL,\n"
                    "e.g. https://search.example.com")

        api_key = self._config.get("api_key", "")

        try:
            print(f"   -> Searching SearXNG at {endpoint}: '{query}'")
            url = f"{endpoint}/search"
            params = {"q": query, "format": "json", "number_of_results": max_results}
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()

            data = resp.json()
            results = data.get("results", [])

            if not results:
                return "No search results found."

            print(f"   -> Got {len(results)} results from SearXNG")
            formatted = ""
            for i, r in enumerate(results[:max_results], 1):
                formatted += f"Result {i}:\n"
                formatted += f"  Title: {r.get('title', '')}\n"
                formatted += f"  Content: {r.get('content', '')}\n"
                formatted += f"  URL: {r.get('url', '')}\n\n"
            return formatted

        except Exception as e:
            print(f"   -> ERROR in SearXNG search: {e}")
            return f"Error searching SearXNG: {str(e)}"

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _format_results(results, title_key="title", body_key="body",
                        url_key="href"):
        """Format a list of result dicts into a readable string."""
        formatted = ""
        for i, result in enumerate(results, 1):
            formatted += f"Result {i}:\n"
            formatted += f"  Title: {result.get(title_key, '')}\n"
            formatted += f"  Content: {result.get(body_key, '')}\n"
            formatted += f"  URL: {result.get(url_key, '')}\n\n"
        return formatted
