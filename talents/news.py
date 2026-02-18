import logging
import requests
from talents.base import BaseTalent
from ddgs import DDGS

log = logging.getLogger(__name__)


class NewsTalent(BaseTalent):
    name = "news"
    description = "Get latest news headlines from configurable providers"
    keywords = ["news", "headlines", "latest", "stories", "top stories",
                "breaking", "current events", "happening"]
    examples = [
        "what's in the news today",
        "give me the latest headlines",
        "any breaking news about technology",
        "what are the top stories right now",
    ]
    priority = 80

    # Words/phrases that indicate a news-like request even without the word "news"
    NEWS_PHRASES = [
        "news", "headlines", "stories", "top stories", "breaking",
        "current events", "happening", "latest on",
    ]

    # Known news sites — if the user mentions one, this is a news request
    NEWS_SITES = [
        "cnn", "bbc", "fox", "reuters", "nytimes", "nyt",
        "washington post", "ap news", "msnbc", "cnbc", "npr",
        "guardian", "aljazeera",
    ]

    # System prompt for news synthesis
    _NEWS_SYSTEM_PROMPT = (
        "You are a news summarizer. "
        "You MUST summarize using ONLY the news articles provided in the user message. "
        "NEVER use your own knowledge or training data. "
        "Include specific details: names, dates, events, numbers from the articles. "
        "If the articles don't cover the requested topic, say "
        "\"I couldn't find recent news about that topic.\" "
        "NEVER make up news. NEVER guess. NEVER say 'check website X'."
    )

    def can_handle(self, command: str) -> bool:
        cmd = command.lower()
        if any(phrase in cmd for phrase in self.NEWS_PHRASES):
            return True
        if any(site in cmd for site in self.NEWS_SITES):
            return True
        return False

    def execute(self, command: str, context: dict) -> dict:
        # Extract topic — strip out news-related noise words
        topic = command.lower()
        for word in self.NEWS_PHRASES + self.NEWS_SITES + [
            "on", "from", "about", "the", "what are", "what's",
            "tell me", "give me", ".com", ".org", ".net",
        ]:
            topic = topic.replace(word, "")
        topic = topic.strip()
        if not topic or len(topic) < 3:
            topic = "general"

        # Fetch news (use configured max_results and provider)
        max_results = self._config.get("max_results", 5)
        provider = self._config.get("provider", "DuckDuckGo")
        news_results = self._get_news(topic, max_results=max_results,
                                      provider=provider)

        print(f"   -> News topic: '{topic}' via {provider}")
        print(f"   -> Results preview: {news_results[:300]}...")

        # Build user message with clear delimiters
        user_message = (
            f"=== NEWS ARTICLES START ===\n"
            f"{news_results}\n"
            f"=== NEWS ARTICLES END ===\n\n"
            f"Request: {command}\n\n"
            f"Summarize the most important headlines using ONLY the articles above. "
            f"Include specific details — names, events, dates — from the articles."
        )

        # Ask LLM to synthesize with system prompt + low temperature
        llm = context["llm"]
        response = llm.generate(
            user_message,
            system_prompt=self._NEWS_SYSTEM_PROMPT,
            temperature=0.3,
        )

        print(f"   -> LLM response: {response[:200]}...")

        return {
            "success": True,
            "response": response,
            "actions_taken": [{"action": "news_search", "topic": topic,
                               "provider": provider}],
            "spoken": False
        }

    def get_config_schema(self) -> dict:
        return {
            "fields": [
                {"key": "provider", "label": "News Provider",
                 "type": "choice", "default": "DuckDuckGo",
                 "choices": ["DuckDuckGo", "NewsAPI", "GNews"]},
                {"key": "api_key", "label": "API Key",
                 "type": "password", "default": ""},
                {"key": "max_results", "label": "Max News Articles",
                 "type": "int", "default": 5, "min": 1, "max": 20},
            ]
        }

    # ── Provider dispatch ──────────────────────────────────────────

    def _get_news(self, topic="general", max_results=5, provider="DuckDuckGo"):
        """Get news headlines from the configured provider."""
        providers = {
            "DuckDuckGo": self._news_duckduckgo,
            "NewsAPI": self._news_newsapi,
            "GNews": self._news_gnews,
        }
        news_fn = providers.get(provider, self._news_duckduckgo)
        return news_fn(topic, max_results)

    # ── DuckDuckGo (default, no key) ──────────────────────────────

    def _news_duckduckgo(self, topic, max_results):
        """Get news via DuckDuckGo (free, no API key)."""
        try:
            print(f"   -> Fetching news from DuckDuckGo: '{topic}'")
            with DDGS() as ddgs:
                results = list(ddgs.news(topic, max_results=max_results))

            if not results:
                print("   -> WARNING: DuckDuckGo news returned zero results!")
                return "No news found."

            print(f"   -> Got {len(results)} news articles from DuckDuckGo")

            formatted_news = ""
            for i, article in enumerate(results, 1):
                formatted_news += f"Article {i}:\n"
                formatted_news += f"  Headline: {article['title']}\n"
                formatted_news += f"  Summary: {article['body']}\n"
                formatted_news += f"  Source: {article['source']}\n"
                formatted_news += f"  Date: {article['date']}\n\n"

            return formatted_news

        except Exception as e:
            print(f"   -> ERROR in DuckDuckGo news: {e}")
            return f"Error fetching news: {str(e)}"

    # ── NewsAPI.org ────────────────────────────────────────────────

    def _news_newsapi(self, topic, max_results):
        """Get news via NewsAPI.org (requires API key).

        Free tier: 100 requests/day.
        Get a key at https://newsapi.org/register
        """
        api_key = self._config.get("api_key", "")
        if not api_key:
            return ("NewsAPI requires an API key.\n"
                    "Get one at https://newsapi.org/register")

        try:
            print(f"   -> Fetching news from NewsAPI: '{topic}'")

            if topic == "general":
                url = "https://newsapi.org/v2/top-headlines"
                params = {
                    "apiKey": api_key,
                    "language": "en",
                    "pageSize": max_results,
                }
            else:
                url = "https://newsapi.org/v2/everything"
                params = {
                    "apiKey": api_key,
                    "q": topic,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": max_results,
                }

            resp = requests.get(url, params=params, timeout=15)

            if resp.status_code == 401:
                return "NewsAPI key is invalid."
            if resp.status_code == 429:
                return "NewsAPI rate limit reached. Try again later."
            resp.raise_for_status()

            data = resp.json()
            articles = data.get("articles", [])

            if not articles:
                return "No news found."

            print(f"   -> Got {len(articles)} articles from NewsAPI")

            formatted_news = ""
            for i, article in enumerate(articles, 1):
                formatted_news += f"Article {i}:\n"
                formatted_news += f"  Headline: {article.get('title', '')}\n"
                formatted_news += f"  Summary: {article.get('description', '')}\n"
                formatted_news += f"  Source: {article.get('source', {}).get('name', '')}\n"
                formatted_news += f"  Date: {article.get('publishedAt', '')}\n\n"

            return formatted_news

        except Exception as e:
            print(f"   -> ERROR in NewsAPI: {e}")
            return f"Error fetching news: {str(e)}"

    # ── GNews ─────────────────────────────────────────────────────

    def _news_gnews(self, topic, max_results):
        """Get news via GNews API (requires API key).

        Free tier: 100 requests/day.
        Get a key at https://gnews.io/
        """
        api_key = self._config.get("api_key", "")
        if not api_key:
            return ("GNews requires an API key.\n"
                    "Get one at https://gnews.io/")

        try:
            print(f"   -> Fetching news from GNews: '{topic}'")

            if topic == "general":
                url = "https://gnews.io/api/v4/top-headlines"
                params = {
                    "token": api_key,
                    "lang": "en",
                    "max": max_results,
                }
            else:
                url = "https://gnews.io/api/v4/search"
                params = {
                    "token": api_key,
                    "q": topic,
                    "lang": "en",
                    "max": max_results,
                }

            resp = requests.get(url, params=params, timeout=15)

            if resp.status_code == 403:
                return "GNews API key is invalid or quota exceeded."
            resp.raise_for_status()

            data = resp.json()
            articles = data.get("articles", [])

            if not articles:
                return "No news found."

            print(f"   -> Got {len(articles)} articles from GNews")

            formatted_news = ""
            for i, article in enumerate(articles, 1):
                formatted_news += f"Article {i}:\n"
                formatted_news += f"  Headline: {article.get('title', '')}\n"
                formatted_news += f"  Summary: {article.get('description', '')}\n"
                formatted_news += f"  Source: {article.get('source', {}).get('name', '')}\n"
                formatted_news += f"  Date: {article.get('publishedAt', '')}\n\n"

            return formatted_news

        except Exception as e:
            print(f"   -> ERROR in GNews: {e}")
            return f"Error fetching news: {str(e)}"
