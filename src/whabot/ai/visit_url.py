"""Website fetching tool using ``curl_cffi`` with browser TLS impersonation.

Plain tightly-scoped HTTP clients are often blocked by anti-bot checks.
``curl_cffi`` drives libcurl with a real Chrome TLS/JA3 fingerprint, so a
``visit_url`` tool gets a much better signal on many news/retail/blog
sites than a bare ``httpx`` client could.

The response body is returned inline (HTML stripped, truncated), since
whabot has no file tools. Tools follow whabot conventions: they return a
status string and never raise (failures become an explanatory message).
"""

import re
from typing import Any

from curl_cffi import requests as cffi_requests
from loguru import logger

from whabot.settings import Settings

__all__ = ["visit_url"]

_IMPERSONATE = "chrome"
_MAX_CHARS = 4000

# Strip common non-content tags in one pass, cheaply.
_TAG_RE = re.compile(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>|<[^>]+>", re.I)
_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]{2,}| *\n *|\n{3,}")


def visit_url(settings: Settings, url: str) -> str:
    """Fetch a web page and return its visible text.

    Args:
        url: The web page URL to visit.

    Returns:
        Up to ~4000 chars of readable page text, or an explanatory error
        message if the page could not be fetched.
    """
    if not url.strip():
        return "Error: url cannot be empty."
    try:
        response = _fetch(url, settings)
    except Exception as exc:
        logger.warning("visit_url failed for {url}: {exc}", url=url, exc=exc)
        return f"visit_url failed: {exc}"

    text = _to_text(response)
    header = f"{response.url} [{response.status_code}]\n"
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + "\n...[truncated]"
    return header + text


def _fetch(url: str, settings: Settings) -> Any:
    """Fetch *url* with Chrome impersonation, raising on HTTP errors."""
    kwargs: dict[str, Any] = {"impersonate": _IMPERSONATE}
    if settings.web_search_proxy:
        proxies = {
            "http": settings.web_search_proxy,
            "https": settings.web_search_proxy,
        }
        kwargs["proxies"] = proxies
    timeout = max(settings.web_search_timeout, 2.0)
    response = cffi_requests.get(url, timeout=timeout, **kwargs)
    response.raise_for_status()
    return response


def _to_text(response: Any) -> str:
    """Return the response body as stripped readable text."""
    try:
        text = response.text or ""
    except Exception:
        return "(no readable body)"

    content_type = response.headers.get("content-type", "")
    stripped = text if "json" in content_type else _TAG_RE.sub(" ", text)

    collapsed = _WHITESPACE_RE.sub(" ", stripped)
    return collapsed.strip() or "(no readable text on page)"
