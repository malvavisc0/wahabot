"""Yahoo Finance-backed market data tools.

Ported from aria-ai's ``aria.tools.search.finance``. Pure Python via the
``yfinance`` library; no API key. Tools follow whabot conventions: they
return a human-readable status string and never raise (failures become
an explanatory message fed back to the model).
"""

import math
import re
from datetime import UTC, datetime
from typing import Any

import yfinance
from loguru import logger

__all__ = [
    "fetch_company_information",
    "fetch_current_stock_price",
    "fetch_ticker_news",
]

# News article limits
_MIN_ARTICLES = 1
_MAX_ARTICLES = 50

# Valid ticker symbol pattern (letters, numbers, dots, hyphens)
_TICKER_PATTERN = re.compile(r"^[A-Z0-9.-]{1,10}$")

# Common quote currencies for crypto pairs where users often omit the dash.
_CRYPTO_QUOTE_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD"}


class YFinanceError(Exception):
    """Custom exception for YFinance-related errors."""


class YFinanceValidationError(YFinanceError):
    """Exception for input validation errors."""


class YFinanceDataError(YFinanceError):
    """Exception for data retrieval errors."""


def fetch_current_stock_price(ticker: str) -> str:
    """Fetch the current price for a stock, ETF, or crypto ticker.

    Returns:
        A short status line with the price, currency and day change, or
        an explanatory error message.
    """
    raw_ticker = ticker
    try:
        ticker = _validate_ticker(ticker)
        info = _get_ticker_info(_get_ticker(ticker), ticker)

        current_price = _finite_price(
            info.get("regularMarketPrice")
            or info.get("currentPrice")
            or info.get("previousClose")
        )
        if current_price is None:
            raise YFinanceDataError(f"No price data available for {ticker}")

        currency = info.get("currency", "USD")
        market_state = info.get("marketState", "UNKNOWN")

        prev_close = _finite_price(info.get("previousClose"))
        result = {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "currency": currency,
            "market_state": market_state,
            "previous_close": prev_close,
        }
        if prev_close:
            day_change = current_price - prev_close
            result["day_change"] = round(day_change, 2)
            result["day_change_percent"] = round((day_change / prev_close) * 100, 2)
        return _format_price(result)
    except YFinanceValidationError as exc:
        logger.warning("Validation error for {ticker}: {exc}", ticker=raw_ticker, exc=exc)
        return f"Stock price error: {exc}"
    except YFinanceDataError as exc:
        logger.warning("Data error for {ticker}: {exc}", ticker=raw_ticker, exc=exc)
        return f"Stock price error: {exc}"
    except Exception as exc:
        logger.exception(
            "Unexpected error fetching price for {ticker}", ticker=raw_ticker
        )
        return f"Stock price error: {exc}"


def fetch_company_information(ticker: str) -> str:
    """Fetch company fundamentals and metadata for a ticker.

    Returns:
        A compact summary of company basics, price data and financial
        health, or an explanatory error message.
    """
    raw_ticker = ticker
    try:
        ticker = _validate_ticker(ticker)
        info = _get_ticker_info(_get_ticker(ticker), ticker)
        return _format_company(ticker, info)
    except YFinanceValidationError as exc:
        logger.warning("Validation error for {ticker}: {exc}", ticker=raw_ticker, exc=exc)
        return f"Company info error: {exc}"
    except YFinanceDataError as exc:
        logger.warning("Data error for {ticker}: {exc}", ticker=raw_ticker, exc=exc)
        return f"Company info error: {exc}"
    except Exception as exc:
        logger.exception(
            "Unexpected error fetching company info for {ticker}",
            ticker=raw_ticker,
        )
        return f"Company info error: {exc}"


def _format_price(result: dict[str, Any]) -> str:
    """Render the price result as a compact status string."""
    lines = [
        f"{result['ticker']}: {result['current_price']} {result['currency']}"
        f" ({result['market_state']})"
    ]
    if result.get("day_change") is not None:
        lines.append(
            f"day change: {result['day_change']} ({result['day_change_percent']}%)"
        )
    if result.get("previous_close") is not None:
        lines.append(f"previous close: {result['previous_close']}")
    return "\n".join(lines)


def _format_company(ticker: str, info: dict[str, Any]) -> str:
    """Render company info as a compact, model-friendly summary."""
    lines: list[str] = []
    name = info.get("shortName") or info.get("longName")
    if name:
        lines.append(f"{name} ({ticker})")
    for label, key in (
        ("sector", "sector"),
        ("industry", "industry"),
        ("market cap", "marketCap"),
        ("trailing PE", "trailingPE"),
        ("forward PE", "forwardPE"),
        ("52w low", "fiftyTwoWeekLow"),
        ("52w high", "fiftyTwoWeekHigh"),
        ("employees", "fullTimeEmployees"),
    ):
        value = info.get(key)
        if value is not None:
            lines.append(f"{label}: {value}")
    summary = info.get("longBusinessSummary")
    if summary:
        lines.append(f"summary: {str(summary)[:400]}")
    price = info.get("regularMarketPrice") or info.get("currentPrice")
    if price is not None:
        currency = info.get("currency", "USD")
        lines.append(f"current price: {price} {currency}")
    return "\n".join(lines)


def fetch_ticker_news(ticker: str, max_articles: int = 10) -> str:
    """Fetch recent news for a stock or crypto ticker.

    Returns:
        One line per recent article (title, source, url), or an
        explanatory error message.
    """
    raw_ticker = ticker
    try:
        ticker = _validate_ticker(ticker)
        if not isinstance(max_articles, int):
            raise YFinanceValidationError("max_articles must be an integer")
        max_articles = max(_MIN_ARTICLES, min(_MAX_ARTICLES, max_articles))

        try:
            news_data = _get_ticker(ticker).news
        except Exception as exc:
            raise YFinanceDataError(f"Failed to fetch news data: {exc}") from exc

        articles = []
        for article in news_data[:max_articles]:
            processed = _process_news_article(article)
            if processed:
                articles.append(processed)
        if not articles:
            return f"No news articles found for {ticker}."
        return "\n".join(_article_line(a) for a in articles)
    except YFinanceValidationError as exc:
        logger.warning(
            "Validation error for {ticker} news: {exc}",
            ticker=raw_ticker,
            exc=exc,
        )
        return f"News error: {exc}"
    except YFinanceDataError as exc:
        logger.warning(
            "Data error for {ticker} news: {exc}",
            ticker=raw_ticker,
            exc=exc,
        )
        return f"News error: {exc}"
    except Exception as exc:
        logger.exception("Unexpected error fetching news for {ticker}", ticker=raw_ticker)
        return f"News error: {exc}"


def _article_line(article: dict[str, Any]) -> str:
    """Render one news article as a compact line."""
    line = article.get("title", "Untitled")
    source = article.get("source")
    if source:
        line += f" ({source})"
    url = article.get("url")
    if url:
        line += f" — {url}"
    published = article.get("published_date")
    if published:
        line += f" [{published}]"
    return line


def _process_news_article(article: Any) -> dict[str, Any] | None:
    """Process a raw yfinance news article into a standard format."""
    try:
        if not isinstance(article, dict):
            return None
        if "content" in article:
            content = article["content"]
            canonical_url = content.get("canonicalUrl", {})
            url = canonical_url.get("url", "") if isinstance(canonical_url, dict) else ""
            provider = content.get("provider", {})
            source = (
                provider.get("displayName", "Unknown")
                if isinstance(provider, dict)
                else "Unknown"
            )
            return {
                "title": content.get("title", "No title"),
                "summary": content.get("summary", ""),
                "published_date": content.get("pubDate", ""),
                "url": url,
                "source": source,
            }
        return {
            "title": article.get("title", "No title"),
            "summary": article.get("summary", ""),
            "published_date": _format_epoch(article.get("providerPublishTime")),
            "url": article.get("link", ""),
            "source": article.get("publisher", "Unknown"),
        }
    except KeyError, TypeError, ValueError:
        return None


def _format_epoch(value: Any) -> str:
    """Render a unix epoch as an ISO date; pass through anything else."""
    if isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(value, tz=UTC).date().isoformat()
    return str(value or "")


def _finite_price(value: Any) -> float | None:
    """Return value as a finite float, or None if not a usable price.

    Rejects None, NaN and inf, which would otherwise serialise to invalid
    JSON tokens.
    """
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _validate_ticker(ticker: Any) -> str:
    """Validate and normalize a ticker symbol."""
    if ticker is None:
        raise YFinanceValidationError("Ticker symbol cannot be empty")
    if not isinstance(ticker, str):
        raise YFinanceValidationError("Ticker symbol must be a string")
    ticker = _normalize_ticker(ticker)
    if not ticker:
        raise YFinanceValidationError("Ticker symbol cannot be empty after normalization")
    if not _TICKER_PATTERN.match(ticker):
        raise YFinanceValidationError(f"Invalid ticker symbol format: {ticker}")
    return ticker


def _normalize_ticker(ticker: str) -> str:
    """Normalize common user-entered ticker variants."""
    normalized = str(ticker).strip().upper()
    if "/" in normalized:
        normalized = normalized.replace("/", "-")
    if "-" not in normalized:
        for quote in _CRYPTO_QUOTE_CURRENCIES:
            if normalized.endswith(quote) and len(normalized) > len(quote):
                base = normalized[: -len(quote)]
                if 2 <= len(base) <= 6:
                    normalized = f"{base}-{quote}"
                break
    return normalized


def _get_ticker(ticker: str) -> Any:
    """Get yfinance Ticker object."""
    return yfinance.Ticker(ticker)


def _get_ticker_info(ticker_obj: Any, ticker: str) -> dict[str, Any]:
    """Get ticker info with error handling."""
    try:
        info = ticker_obj.info
        if not info or not isinstance(info, dict):
            raise YFinanceDataError(f"No information available for {ticker}")
        return info
    except Exception as exc:
        raise YFinanceDataError(f"Failed to get ticker info for {ticker}: {exc}") from exc
