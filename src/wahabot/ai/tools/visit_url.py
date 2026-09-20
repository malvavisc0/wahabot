"""Website fetching tool using ``curl_cffi`` with browser TLS impersonation.

Plain tightly-scoped HTTP clients are often blocked by anti-bot checks.
``curl_cffi`` drives libcurl with a real Chrome TLS/JA3 fingerprint, so a
``visit_url`` tool gets a much better signal on many news/retail/blog
sites than a bare ``httpx`` client could.

Media hosts are the exception: Instagram, Facebook, TikTok and friends
serve unauthenticated fetchers a login/consent wall, so the HTML path
returns chrome instead of content. For those, yt-dlp's per-site
extractors provide the video's metadata (title, description, uploader,
duration, views) without downloading anything — see the inbound-link
pipeline in ``url_videos`` for the heavyweight download path.

The response body is returned inline (HTML stripped, truncated), since
wahabot has no file tools. Tools follow wahabot conventions: they return
the shared JSON envelope and never raise (failures become an ``error``
envelope fed back to the model).
"""

import re
from typing import Any, cast

import yt_dlp
from curl_cffi import requests as cffi_requests
from loguru import logger

from wahabot.ai.tools.envelope import error, ok
from wahabot.settings import Settings

__all__ = ["visit_url"]

_IMPERSONATE = "chrome"
_MAX_CHARS = 4000
_DESCRIPTION_CHARS = 800

# Strip common non-content tags in one pass, cheaply.
_TAG_RE = re.compile(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>|<[^>]+>", re.I)
_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]{2,}| *\n *|\n{3,}")

# Hosts where an HTML fetch is known to lose (login walls) or where
# video metadata beats page text. Scoped list, not a yt-dlp support probe
# — same matching style as ``url_videos._YOUTUBE_HOST_RE``.
_MEDIA_HOSTS = (
    "instagram.com",
    "facebook.com",
    "fb.watch",
    "tiktok.com",
    "youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
    "twitter.com",
    "x.com",
    "twitch.tv",
)
_MEDIA_HOST_RE = re.compile(
    r"^https?://(?:[\w-]+\.)?(?:"
    + r"|".join(re.escape(host) for host in _MEDIA_HOSTS)
    + r")/",
    re.IGNORECASE,
)


def visit_url(settings: Settings, url: str) -> str:
    """Fetch a web page and return its visible text or video metadata.

    Media-host URLs resolve to the video's yt-dlp metadata (no download);
    every other page takes the HTML path. If the metadata extraction
    fails, the HTML path still runs — the tool never dead-ends.

    Args:
        url: The web page URL to visit.

    Returns:
        A JSON envelope with the page ``text`` (up to ~4000 chars), its
        final ``url``, HTTP ``status`` and a ``truncated`` flag — or, for
        a resolved video link, its ``title``/``description``/``uploader``
        metadata. An ``error`` envelope is returned if the page could
        not be fetched at all.
    """
    if not url.strip():
        return error("url cannot be empty")
    if _MEDIA_HOST_RE.match(url):
        meta = video_meta(url, settings)
        if meta is not None:
            return ok(source="yt-dlp", kind="video", url=url, **meta)
        # Extractor failed → the page might still be readable (e.g. a
        # private/removed post), so fall through to the HTML path.
        logger.info("yt-dlp could not resolve {url}; falling back to HTML", url=url)
    try:
        response = _fetch(url, settings)
    except Exception as exc:
        logger.warning("visit_url failed for {url}: {exc}", url=url, exc=exc)
        return error(f"visit_url failed: {exc}")

    text = _to_text(response)
    truncated = len(text) > _MAX_CHARS
    if truncated:
        text = text[:_MAX_CHARS]
    return ok(
        url=str(response.url),
        status=response.status_code,
        truncated=truncated,
        text=text,
    )


def video_meta(url: str, settings: Settings) -> dict[str, Any] | None:
    """The video's yt-dlp metadata for *url*, or None when unresolvable.

    Two tiers, cheapest reliable first: the raw extractor dict
    (``process=False``) skips format resolution — the step that needs a
    JS runtime for YouTube and dies on Instagram multi-item posts ("No
    video formats found") — and carries the post's caption, uploader
    and id where the processed one raises. Some links (Facebook share
    redirects) come back as a bare unresolved dict, so a metadata-less
    raw result retries once with format processing before giving up.
    """
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "socket_timeout": max(settings.web_search_timeout, 2.0),
    }
    if settings.web_search_proxy:
        opts["proxy"] = settings.web_search_proxy
    try:
        with yt_dlp.YoutubeDL(cast(Any, opts)) as ydl:
            info = _extract(ydl, url)
    except Exception as exc:
        logger.info("Video metadata unavailable for {url}: {exc}", url=url, exc=exc)
        return None
    if not info:
        return None
    if "entries" in info:
        info = _post_info(info)
    description = str(info.get("description") or "")[:_DESCRIPTION_CHARS]
    return {
        "title": info.get("title"),
        "description": description or None,
        "uploader": info.get("uploader") or info.get("channel"),
        "duration_s": info.get("duration"),
        "view_count": info.get("view_count"),
        "id": info.get("id"),
    }


def _extract(ydl: Any, url: str) -> dict[str, Any] | None:
    """Raw extractor dict, or a processed retry when it came back bare.

    A bare dict (url/id only, no descriptive keys) means the extractor
    returned an unresolved placeholder — Facebook ``/share/r/`` links
    do — so one processed pass is the only way to reach its metadata.
    That retry is where format errors raise ("No video formats
    found"), so it fails soft back to the bare dict: its url/id still
    identify the link, and the caller decides between a sparse
    envelope and the HTML fallback.
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
        return info
    return processed if processed and "entries" not in processed else info


def _post_info(info: dict[str, Any]) -> dict[str, Any]:
    """A multi-item post's own info: the parent dict carries the caption.

    Format processing dies on multi-item Instagram posts ("No video
    formats found"), but the raw extractor dict still describes the
    *post* — caption, uploader, id — which is what the model needs,
    plus the item count and summed duration.
    """
    entries: list[dict[str, Any]] = [e for e in (info.get("entries") or []) if e]
    total = sum(int(e.get("duration") or 0) for e in entries)
    return info | {
        "title": info.get("title") or f"post with {len(entries)} items",
        "duration": info.get("duration") or (total or None),
    }


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
