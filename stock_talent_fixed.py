"""StockTalent — look up stock prices and market data via Yahoo Finance.

Uses yfinance for real-time and historical stock data.

Examples:
    "stock price of AAPL"
    "how is TSLA doing"
    "check MSFT stock"
    "stock info for GOOGL"
    "compare AAPL and MSFT"
"""

import re
from talents.base import BaseTalent


class StockTalent(BaseTalent):
    name = "stock"
    subprocess_isolated = True
    description = "Look up stock prices, ticker info, and basic market data"
    keywords = [
        "stock", "stock price", "ticker", "shares", "market",
        "nasdaq", "nyse", "s&p", "dow jones",
    ]
    priority = 50

    _EXCLUSIONS = [
        "remind", "timer", "email", "note", "weather", "hue",
        "light", "search", "news", "todo", "task", "pomodoro",
        "docker", "github", "regex", "json", "snippet",
        "crypto", "bitcoin", "ethereum",
    ]

    def get_config_schema(self) -> dict:
        return {
            "fields": [
                {"key": "default_tickers", "label": "Watchlist (comma-separated tickers)",
                 "type": "string", "default": ""},
            ]
        }

    def can_handle(self, command: str) -> bool:
        cmd = command.lower()
        if any(ex in cmd for ex in self._EXCLUSIONS):
            return False
        if any(kw in cmd for kw in self.keywords):
            return True
        if re.search(r'\b[A-Z]{1,5}\b', command) and ("price" in cmd or "how is" in cmd or "check" in cmd):
            return True
        return False

    def _get_yf(self):
        """Lazy-import yfinance to keep it out of the main process."""
        import yfinance as yf
        return yf

    # Major indices — always included for general market queries
    _INDICES = {
        "^DJI":  "Dow Jones",
        "^IXIC": "NASDAQ",
        "^GSPC": "S&P 500",
    }

    _MARKET_PHRASES = [
        "stock market", "the market", "how's the market",
        "how is the market", "market today", "market check",
        "check the market", "market update", "market summary",
    ]

    def execute(self, command: str, context: dict) -> dict:
        try:
            self._get_yf()
        except ImportError:
            return self._fail("yfinance is not installed. Run: pip install yfinance")

        cmd = command.lower().strip()

        # General market query → show major indices + watchlist
        if any(phrase in cmd for phrase in self._MARKET_PHRASES) or cmd in ("check stock", "stocks"):
            return self._market_overview()

        tickers = self._extract_tickers(command)

        if not tickers:
            watchlist = self._config.get("default_tickers", "")
            if watchlist:
                tickers = [t.strip().upper() for t in watchlist.split(",") if t.strip()]
            if not tickers:
                # No specific ticker and no watchlist → show indices
                return self._market_overview()

        if len(tickers) > 1:
            return self._compare_stocks(tickers)

        ticker = tickers[0]

        if any(p in cmd for p in ["info", "detail", "about", "tell me about"]):
            return self._stock_info(ticker)

        return self._stock_price(ticker)

    # ── Market Overview ──────────────────────────────────────────

    def _market_overview(self):
        yf = self._get_yf()
        lines = ["Market Overview:\n"]

        for symbol, name in self._INDICES.items():
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                price = info.get("regularMarketPrice") or info.get("currentPrice")
                prev = info.get("regularMarketPreviousClose") or info.get("previousClose")

                if price and prev:
                    diff = price - prev
                    pct = (diff / prev) * 100
                    arrow = "\u2b06\ufe0f" if diff >= 0 else "\u2b07\ufe0f"
                    sign = "+" if diff >= 0 else ""
                    lines.append(
                        f"  {name:<12} {price:>10,.2f}  "
                        f"{arrow} {sign}{diff:,.2f} ({sign}{pct:.2f}%)")
                elif price:
                    lines.append(f"  {name:<12} {price:>10,.2f}")
                else:
                    lines.append(f"  {name:<12} {'N/A':>10}")
            except Exception:
                lines.append(f"  {name:<12} {'Error':>10}")

        # Append watchlist tickers if configured
        watchlist = self._config.get("default_tickers", "")
        if watchlist:
            extras = [t.strip().upper() for t in watchlist.split(",") if t.strip()]
            if extras:
                lines.append("")
                for ticker in extras[:5]:
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        name = info.get("shortName", ticker)
                        price = info.get("currentPrice") or info.get("regularMarketPrice")
                        prev = info.get("previousClose") or info.get("regularMarketPreviousClose")
                        if price and prev:
                            diff = price - prev
                            pct = (diff / prev) * 100
                            arrow = "\u2b06\ufe0f" if diff >= 0 else "\u2b07\ufe0f"
                            sign = "+" if diff >= 0 else ""
                            lines.append(
                                f"  {ticker:<8} ${price:>9,.2f}  "
                                f"{arrow} {sign}{diff:,.2f} ({sign}{pct:.2f}%)")
                        elif price:
                            lines.append(f"  {ticker:<8} ${price:>9,.2f}")
                    except Exception:
                        lines.append(f"  {ticker:<8} {'Error':>10}")

        return self._ok("\n".join(lines))

    # ── Price ────────────────────────────────────────────────────

    def _stock_price(self, ticker):
        yf = self._get_yf()
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            name = info.get("shortName", ticker)
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
            currency = info.get("currency", "USD")

            if price is None:
                return self._fail(f"Could not fetch price for {ticker}. Is the ticker correct?")

            change = ""
            if prev_close and price:
                diff = price - prev_close
                pct = (diff / prev_close) * 100
                arrow = "\u2b06\ufe0f" if diff >= 0 else "\u2b07\ufe0f"
                sign = "+" if diff >= 0 else ""
                change = f"  {arrow} {sign}{diff:.2f} ({sign}{pct:.2f}%)"

            mkt_cap = info.get("marketCap")
            cap_str = ""
            if mkt_cap:
                if mkt_cap >= 1e12:
                    cap_str = f"  Market Cap: ${mkt_cap/1e12:.2f}T"
                elif mkt_cap >= 1e9:
                    cap_str = f"  Market Cap: ${mkt_cap/1e9:.2f}B"
                elif mkt_cap >= 1e6:
                    cap_str = f"  Market Cap: ${mkt_cap/1e6:.2f}M"

            high = info.get("dayHigh", info.get("regularMarketDayHigh"))
            low = info.get("dayLow", info.get("regularMarketDayLow"))
            range_str = ""
            if high and low:
                range_str = f"  Day Range: ${low:.2f} - ${high:.2f}"

            lines = [f"{name} ({ticker})"]
            lines.append(f"  Price: ${price:.2f} {currency}{change}")
            if range_str:
                lines.append(range_str)
            if cap_str:
                lines.append(cap_str)

            volume = info.get("volume", info.get("regularMarketVolume"))
            if volume:
                lines.append(f"  Volume: {volume:,.0f}")

            return self._ok("\n".join(lines))

        except Exception as e:
            return self._fail(f"Error fetching {ticker}: {e}")

    # ── Info ─────────────────────────────────────────────────────

    def _stock_info(self, ticker):
        yf = self._get_yf()
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            name = info.get("shortName", ticker)
            lines = [f"{name} ({ticker})\n"]

            fields = [
                ("Sector", "sector"),
                ("Industry", "industry"),
                ("Country", "country"),
                ("Employees", "fullTimeEmployees"),
                ("Website", "website"),
                ("P/E Ratio", "trailingPE"),
                ("EPS", "trailingEps"),
                ("Dividend Yield", "dividendYield"),
                ("52-Week High", "fiftyTwoWeekHigh"),
                ("52-Week Low", "fiftyTwoWeekLow"),
            ]

            for label, key in fields:
                val = info.get(key)
                if val is not None:
                    if key == "dividendYield" and val:
                        val = f"{val * 100:.2f}%"
                    elif key == "fullTimeEmployees":
                        val = f"{val:,}"
                    elif isinstance(val, float):
                        val = f"${val:.2f}" if "High" in label or "Low" in label else f"{val:.2f}"
                    lines.append(f"  {label}: {val}")

            summary = info.get("longBusinessSummary", "")
            if summary:
                lines.append(f"\n{summary[:200]}...")

            return self._ok("\n".join(lines))

        except Exception as e:
            return self._fail(f"Error fetching info for {ticker}: {e}")

    # ── Compare ──────────────────────────────────────────────────

    def _compare_stocks(self, tickers):
        yf = self._get_yf()
        lines = ["Stock Comparison:\n"]
        lines.append(f"  {'Ticker':<8} {'Price':>10} {'Change':>10} {'Mkt Cap':>12}")
        lines.append("  " + "-" * 44)

        for ticker in tickers[:5]:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
                prev = info.get("previousClose") or info.get("regularMarketPreviousClose", 0)
                cap = info.get("marketCap", 0)

                if price and prev:
                    pct = ((price - prev) / prev) * 100
                    change_str = f"{pct:+.2f}%"
                else:
                    change_str = "N/A"

                if cap >= 1e12:
                    cap_str = f"${cap/1e12:.1f}T"
                elif cap >= 1e9:
                    cap_str = f"${cap/1e9:.1f}B"
                elif cap >= 1e6:
                    cap_str = f"${cap/1e6:.1f}M"
                else:
                    cap_str = "N/A"

                price_str = f"${price:.2f}" if price else "N/A"
                lines.append(f"  {ticker:<8} {price_str:>10} {change_str:>10} {cap_str:>12}")

            except Exception:
                lines.append(f"  {ticker:<8} {'Error':>10}")

        return self._ok("\n".join(lines))

    # ── Helpers ──────────────────────────────────────────────────

    def _extract_tickers(self, command):
        candidates = re.findall(r'\b([A-Z]{1,5})\b', command)
        noise = {"I", "A", "AND", "OR", "THE", "FOR", "OF", "IN", "IS",
                 "IT", "TO", "MY", "HOW", "VS", "NYSE", "NASDAQ", "SP"}
        tickers = [t for t in candidates if t not in noise]
        return tickers[:5]

    def _ok(self, msg):
        return {"success": True, "response": msg,
                "actions_taken": [{"action": "stock"}], "spoken": False}

    def _fail(self, msg):
        return {"success": False, "response": msg,
                "actions_taken": [], "spoken": False}
