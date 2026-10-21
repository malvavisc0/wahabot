from html import unescape
from typing import Any, cast

import yt_dlp
from curl_cffi import requests as cffi_requests
from loguru import logger

from wahabot.ai.tools.envelope import error, ok
from wahabot.ai.tools.outfile import write_text_output
from wahabot.ai.tools.visit_url.constants import (
    DESCRIPTION_CHARS,
    IMPERSONATE,
    MAX_CHARS,
    MAX_META_CHARS,
    META_CONTENT_RE,
    META_KEY_RE,
    META_KEYS,
    META_TAG_RE,
    TAG_RE,
    WHITESPACE_RE,
)
from wahabot.settings import Settings


def page_envelope(url: str, settings: Settings) -> str:
    """The fetched page's envelope, or an error envelope on failure."""
    try:
        response = fetch_page(url, settings)
        meta = meta_description(response.text)
        text = to_text(response)
        if meta and len(text) < len(meta):
            # A JS-shell page (Instagram, X…) strips down to its site
            # name while the meta tags carry the real caption: the
            # page's own description of itself beats the empty render.
            text = meta
        return ok(
            url=str(response.url),
            status=response.status_code,
            **page_fields(text),
        )
    except Exception as exc:
        logger.warning("visit_url failed for {url}: {exc}", url=url, exc=exc)
        return error(f"visit_url failed: {exc}")


def meta_description(body: str) -> str:
    """OpenGraph/Apple description text from a page's meta tags, or "".

    Title and description are concatenated (title first), both capped,
    with HTML-escaped entities decoded and whitespace collapsed to one
    readable line.
    """
    parts: dict[str, str] = {}
    for tag in META_TAG_RE.findall(body):
        key_match = META_KEY_RE.search(tag)
        if key_match is None:
            continue
        field = META_KEYS.get(key_match.group(1).lower())
        content = META_CONTENT_RE.search(tag)
        if field is None or content is None or not content.group(1).strip():
            continue
        parts.setdefault(field, unescape(content.group(1)).strip())
    if not parts:
        return ""
    joined = " ".join(parts.get(key, "") for key in ("title", "description"))
    return WHITESPACE_RE.sub(" ", joined[:MAX_META_CHARS]).strip()


def page_fields(text: str) -> dict[str, Any]:
    """Envelope fields for a fetched page body: preview + optional spill.

    Short bodies ride the envelope whole; long ones keep a bounded
    ``text`` preview, flag ``truncated``, and spill the full body to
    ``file`` when writing succeeds. The spill is fail-soft: on a write
    error the envelope just keeps the preview and the flag. This character
    preview limit does not cap the HTTP response download.
    """
    if len(text) <= MAX_CHARS:
        return {"text": text, "truncated": False}
    fields: dict[str, Any] = {"text": text[:MAX_CHARS], "truncated": True}
    try:
        fields["file"] = write_text_output("page", text)
    except Exception as exc:
        logger.warning("could not spill page text to file: {exc}", exc=exc)
    return fields


def extract_safely(url: str, opts: dict[str, Any]) -> dict[str, Any] | None:
    """The extracted info dict, or None when yt-dlp cannot resolve it."""
    try:
        with yt_dlp.YoutubeDL(cast(Any, opts)) as ydl:
            return extract_page(ydl, url)
    except Exception as exc:
        logger.info("Video metadata unavailable for {url}: {exc}", url=url, exc=exc)
        return None


def meta_fields(info: dict[str, Any]) -> dict[str, Any]:
    """The envelope's scalar metadata fields from a resolved info dict."""
    description = str(info.get("description") or "")
    fields: dict[str, Any] = {
        "kind": "post" if "entries" in info else "video",
        "title": info.get("title"),
        "description": description[:DESCRIPTION_CHARS] or None,
        "uploader": info.get("uploader") or info.get("channel"),
        "duration_s": info.get("duration"),
        "view_count": info.get("view_count"),
        "id": info.get("id"),
    }
    if "entries" in info:
        fields["item_count"] = info["item_count"]
        if info.get("item_count_truncated"):
            fields["item_count_truncated"] = True
    if len(description) > DESCRIPTION_CHARS:
        fields["description_truncated"] = True
    return fields


def extract_page(ydl: Any, url: str) -> dict[str, Any] | None:
    """Raw extractor dict, or a processed retry when it came back bare.

    A bare dict (url/id only, no descriptive keys) means the extractor
    returned an unresolved placeholder — Facebook ``/share/r/`` links
    do — so one processed pass is the only way to reach its metadata.
    Retry failure or another undescribed placeholder returns None so
    ``visit_url`` can attempt HTML rather than claim metadata success.
    """
    info = cast(
        dict[str, Any] | None,
        cast(object, ydl.extract_info(url, download=False, process=False)),
    )
    if not info or info.get("title") or info.get("description") or "entries" in info:
        return info
    try:
        processed = cast(
            dict[str, Any] | None,
            cast(object, ydl.extract_info(url, download=False)),
        )
    except Exception as exc:
        logger.info("Processed retry failed for {url}: {exc}", url=url, exc=exc)
        return None
    if processed and (
        processed.get("title") or processed.get("description") or "entries" in processed
    ):
        return processed
    return None


def fetch_page(url: str, settings: Settings) -> Any:
    """Fetch *url* with Chrome impersonation, raising on HTTP errors."""
    kwargs: dict[str, Any] = {"impersonate": IMPERSONATE}
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


def to_text(response: Any) -> str:
    """Preserve non-HTML decoded bodies; best-effort tag stripping for HTML.

    HTML conversion does not render CSS/JavaScript or establish visibility.
    """
    try:
        text = response.text or ""
    except Exception:
        return "(no readable body)"

    content_type = (
        response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    )
    if content_type not in ("text/html", "application/xhtml+xml"):
        return text
    stripped = TAG_RE.sub(" ", text)

    collapsed = WHITESPACE_RE.sub(" ", stripped)
    return collapsed.strip() or "(no readable text on page)"
