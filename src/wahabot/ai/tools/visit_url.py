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
pipeline in ``url_videos`` for the heavyweight download path. YouTube
links additionally carry their captions inline (``transcript`` in the
envelope): yt-dlp's extractor exposes the caption track URLs, so the
spoken content arrives without the dropped ``get_youtube_transcript``
tool (docs/bug-report-2c665d8.md, bug 7b) or its
``youtube-transcript-api`` dependency.

The response body is returned inline (HTML stripped, truncated), since
wahabot has no file tools. Tools follow wahabot conventions: they return
the shared JSON envelope and never raise (failures become an ``error``
envelope fed back to the model).
"""

import json
import re
from typing import Any, cast

import httpx
import yt_dlp
from curl_cffi import requests as cffi_requests
from loguru import logger

from wahabot.ai.tools.envelope import error, ok
from wahabot.settings import Settings

__all__ = ["visit_url"]

_IMPERSONATE = "chrome"
_MAX_CHARS = 4000
_DESCRIPTION_CHARS = 800
_MAX_TRANSCRIPT_CHARS = 6000

# Strip common non-content tags in one pass, cheaply.
_TAG_RE = re.compile(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>|<[^>]+>", re.I)
_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]{2,}| *\n *|\n{3,}")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SENTENCES = 3

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

#: YouTube links get their captions inlined alongside the metadata.
_YOUTUBE_HOST_RE = re.compile(
    r"^https?://(?:[\w-]+\.)?(?:youtube\.com|youtu\.be|youtube-nocookie\.com)/",
    re.IGNORECASE,
)

#: The yt-dlp player client that resolves YouTube without a JS runtime:
#: the default ``web`` client dies on format resolution ("No video
#: formats found") — the same external constraint ``video_meta``'s
#: raw-extract path works around — and ``android`` still returns caption
#: track URLs with it. Verified empirically: manual captions arrive as
#: plain ``json3`` files, auto-captions as an HLS playlist of ``vtt``
#: segments.
_YOUTUBE_CLIENT = "android"


def visit_url(settings: Settings, url: str) -> str:
    """Fetch a web page and return its visible text or video metadata.

    Media-host URLs resolve to the video's yt-dlp metadata (no
    download); YouTube links additionally carry the video's captions
    as ``transcript`` when they exist. Every other page takes the HTML
    path. If the metadata extraction fails, the HTML path still runs —
    the tool never dead-ends.

    Args:
        url: The web page URL to visit.

    Returns:
        A JSON envelope with the page ``text`` (up to ~4000 chars), its
        final ``url``, HTTP ``status`` and a ``truncated`` flag — or, for
        a resolved video link, its ``title``/``description``/``uploader``
        metadata (plus ``transcript``/``transcript_truncated`` for
        captioned YouTube videos). An ``error`` envelope is returned if
        the page could not be fetched at all.
    """
    if not url.strip():
        return error("url cannot be empty")
    if _MEDIA_HOST_RE.match(url):
        meta = video_meta(url, settings)
        if meta is not None:
            tracks = cast(
                "dict[str, list[dict[str, Any]]] | None",
                meta.pop("_caption_tracks", None),
            )
            if tracks and _YOUTUBE_HOST_RE.match(url):
                transcript = youtube_transcript(url, tracks)
                if transcript:
                    meta["transcript"] = transcript
                    if len(transcript) >= _MAX_TRANSCRIPT_CHARS:
                        meta["transcript_truncated"] = True
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

    YouTube is resolved with the ``android`` player client — the
    default ``web`` client needs a JS runtime for format resolution —
    and its caption tracks ride along under ``_caption_tracks`` for
    :func:`youtube_transcript` to fetch.
    """
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "socket_timeout": max(settings.web_search_timeout, 2.0),
    }
    if _YOUTUBE_HOST_RE.match(url):
        opts["extractor_args"] = {"youtube": {"player_client": [_YOUTUBE_CLIENT]}}
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
    meta = {
        "title": info.get("title"),
        "description": description or None,
        "uploader": info.get("uploader") or info.get("channel"),
        "duration_s": info.get("duration"),
        "view_count": info.get("view_count"),
        "id": info.get("id"),
    }
    # Caption tracks come from the processed dict (the raw one lacks
    # per-format caption URLs); both live on ``info`` after _extract's
    # retry, and videos without captions carry empty dicts — drop those.
    if _YOUTUBE_HOST_RE.match(url):
        tracks = cast(
            "dict[str, list[dict[str, Any]]]",
            info.get("subtitles") or {},
        )
        if tracks:
            meta["_caption_tracks"] = tracks
    return meta


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


def youtube_transcript(
    url: str, tracks: dict[str, list[dict[str, Any]]], language: str = "en"
) -> str:
    """The video's captions as paragraphed prose, or "" when unavailable.

    *tracks* is yt-dlp's ``subtitles`` dict (manual caption tracks —
    the ``android`` client exposes them as plain ``json3``/``vtt``/
    ``srt`` files at fetchable URLs). A missing *language* track falls
    back to any single available language, so non-English videos still
    transcribe. Auto-generated captions are not in ``subtitles`` and
    cost an HLS playlist hop; the model can still read the description,
    so "" (not an error) is the honest answer for a caption-less video.
    """
    lang = next(
        (t for name, t in sorted(tracks.items()) if name.startswith(language)),
        None,
    )
    if lang is None and len(tracks) == 1:
        lang = next(iter(tracks.values()))
    if lang is None:
        logger.info("No usable caption track for {url}", url=url)
        return ""
    ordered = [lang] + [t for t in tracks.values() if t is not lang]
    for formats in ordered:
        for wanted in ("json3", "srt", "vtt"):
            track = next((f for f in formats if f.get("ext") == wanted), None)
            if track is None:
                continue
            text = _fetch_captions(str(track.get("url") or ""))
            if text:
                return _paragraph(text if wanted == "json3" else _strip_vtt(text))
    logger.info("Caption tracks for {url} had no fetchable format", url=url)
    return ""


def _fetch_captions(track_url: str) -> str:
    """Download one caption track, "" on any failure (fail-soft)."""
    if not track_url:
        return ""
    try:
        response = httpx.get(
            track_url,
            timeout=10.0,
            headers={"User-Agent": "Mozilla/5.0"},
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.text
    except Exception as exc:
        logger.info("Caption fetch failed: {exc}", exc=exc)
        return ""


def _strip_vtt(vtt: str) -> str:
    """Cue text lines from a WebVTT/SRT body, one cue per line."""
    lines: list[str] = []
    for line in vtt.splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith(("WEBVTT", "#"))
            or "-->" in stripped
            or re.fullmatch(r"\d{1,2}:\d{2}:\d{2}[.,]\d{3}", stripped)
            or stripped.isdigit()
        ):
            continue
        lines.append(stripped)
    return " ".join(lines)


def _paragraph(caption_text: str) -> str:
    """Caption text as readable paragraphed prose, truncated.

    Caption fragments arrive as ~3-second snippets; joining them
    verbatim yields mid-sentence breaks every few words. Join, collapse
    whitespace, and break into paragraphs at sentence boundaries —
    the same formatting the dropped transcript tool used.
    """
    if caption_text.lstrip().startswith("{"):
        # json3: event objects with utf8 segments.
        try:
            data: dict[str, Any] = json.loads(caption_text)
            events = data.get("events") or []
            parts = [
                "".join(
                    seg.get("utf8", "") for seg in cast("list[Any]", ev.get("segs")) or []
                )
                for ev in cast("list[Any]", events)
            ]
            joined = " ".join(part.strip() for part in parts if part.strip())
        except json.JSONDecodeError:
            return ""
    else:
        joined = caption_text
    joined = re.sub(r"\s+", " ", joined).strip()
    sentences = _SENTENCE_END_RE.split(joined)
    paragraphs = [
        " ".join(sentences[i : i + _PARAGRAPH_SENTENCES]).strip()
        for i in range(0, len(sentences), _PARAGRAPH_SENTENCES)
    ]
    text = "\n\n".join(p for p in paragraphs if p)
    return text[:_MAX_TRANSCRIPT_CHARS]


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
