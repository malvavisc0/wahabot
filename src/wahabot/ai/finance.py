"""Yahoo Finance-backed market data tool.

Ported from aria-ai's ``aria.tools.search.finance``. Pure Python via the
``yfinance`` library; no API key. The tool follows wahabot conventions:
it returns a human-readable status string and never raises (failures
become an explanatory message fed back to the model).
"""

import math
import re
from typing import Any

import yfinance
from loguru import logger

__all__ = [
    "fetch_current_stock_price",
]

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
