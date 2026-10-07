"""Website fetching tool using ``curl_cffi`` with browser TLS impersonation.

Plain tightly-scoped HTTP clients are often blocked by anti-bot checks.
``curl_cffi`` drives libcurl with a real Chrome TLS/JA3 fingerprint, so a
``visit_url`` tool gets a much better signal on many news/retail/blog
sites than a bare ``httpx`` client could.

Media hosts are the exception: Instagram, Facebook, TikTok and friends
serve unauthenticated fetchers a login/consent wall, so the HTML path
returns chrome instead of content. For those, yt-dlp's per-site
extractors provide available metadata (title, description, uploader,
duration, views) without downloading the video — see the inbound-link
pipeline in ``url_videos`` for the heavyweight download path. YouTube
links attempt caption retrieval from direct json3/SRT/VTT tracks;
missing transcripts do not establish that no captions exist.

Only HTML content types undergo best-effort tag stripping; other decoded
bodies retain their whitespace and tags. This is not browser-rendered or
verified visible text. Bodies are fetched in full before preview limits
are applied; these limits are not download byte caps. Long text attempts
a full-text spill (``file.path``), readable by the model only when the
shell tool is enabled (otherwise the operator can open the path).
Ordinary metadata/caption processing exceptions fall back to HTTP;
HTTP fetch/response-processing/envelope exceptions become an ``error``
envelope; undecodable bodies yield a placeholder and failed spills keep
the preview. Process-level interrupts are not caught.
"""

import json
import re
from html import unescape
from itertools import islice
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

import httpx
import yt_dlp
from curl_cffi import requests as cffi_requests
from loguru import logger

from wahabot.ai.tools.envelope import error, ok
from wahabot.ai.tools.outfile import write_text_output
from wahabot.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["visit_url"]

_IMPERSONATE = "chrome"
_MAX_CHARS = 4000
_DESCRIPTION_CHARS = 800
_MAX_TRANSCRIPT_CHARS = 6000
_MAX_POST_ITEMS = 100

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
#: track URLs with it. Either manual or automatic tracks can use direct
#: json3/SRT/VTT URLs; HLS playlists require unsupported segment retrieval.
_YOUTUBE_CLIENT = "android"


def visit_url(settings: Settings, url: str) -> str:
    """Fetch HTTP(S) text or available video/post metadata, without rendering.

    Media-host URLs attempt yt-dlp metadata without downloading video;
    YouTube links attempt captions as ``transcript``. Extraction failures,
    unresolved placeholders, and ordinary exceptions during metadata or
    caption processing fall back to an HTTP page fetch. Unavailable or
    malformed caption tracks normally just omit ``transcript``.

    Returns:
        A JSON envelope with the page ``text`` (a bounded preview of up
        to ~4000 chars), its final ``url``, HTTP ``status`` and a
        ``truncated`` flag — and, when the body was cut, a ``file``
        (metadata containing the extracted text's file.path) if writing succeeds.
        A metadata result has ``kind="video"`` or ``kind="post"`` and
        ``title``/``description``/``uploader`` fields. Posts carry
        ``item_count``; ``item_count_truncated`` marks counts and summed
        durations limited to the first 100 inspected entries (unless
        the parent supplies its own duration). Captions add ``transcript``/
        ``transcript_truncated`` and (on long captions) a ``transcript_file`` with
        the fetched caption text, which can be incomplete or generated.
        Descriptions cap at 800 characters with description_truncated.
        HTML text is best-effort tag stripping, not verified visible text;
        other decoded bodies are preserved. Preview limits are not download
        caps. For string inputs, ordinary HTTP fetch/response-processing/
        envelope failures return an ``error`` envelope; unreadable bodies
        yield a placeholder and spill failures keep the preview.
        Process-level interrupts propagate.
    """
    url = url.strip()
    if not url:
        return error("url cannot be empty")
    try:
        parts = urlsplit(url)
    except ValueError:
        return error("url must be a valid HTTP(S) URL")
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return error("url must be a valid HTTP(S) URL")
    if _MEDIA_HOST_RE.match(url):
        try:
            meta = video_meta(url, settings)
            if meta is not None:
                tracks = cast(
                    "dict[str, list[dict[str, Any]]] | None",
                    meta.pop("_caption_tracks", None),
                )
                if tracks and _YOUTUBE_HOST_RE.match(url):
                    transcript = youtube_transcript(url, tracks, settings=settings)
                    if transcript:
                        meta.update(transcript_fields(transcript))
                return ok(source="yt-dlp", url=url, **meta)
        except Exception as exc:
            logger.info("Media processing failed for {url}: {exc}", url=url, exc=exc)
        # Extractor failed → the page might still be readable (e.g. a
        # private/removed post), so fall through to the HTML path.
        logger.info("yt-dlp could not resolve {url}; falling back to HTML", url=url)
    try:
        response = _fetch(url, settings)
        text = _to_text(response)
        return ok(
            url=str(response.url),
            status=response.status_code,
            **page_fields(text),
        )
    except Exception as exc:
        logger.warning("visit_url failed for {url}: {exc}", url=url, exc=exc)
        return error(f"visit_url failed: {exc}")


def page_fields(text: str) -> dict[str, Any]:
    """Envelope fields for a fetched page body: preview + optional spill.

    Short bodies ride the envelope whole; long ones keep a bounded
    ``text`` preview, flag ``truncated``, and spill the full body to
    ``file`` when writing succeeds. The spill is fail-soft: on a write
    error the envelope just keeps the preview and the flag. This character
    preview limit does not cap the HTTP response download.
    """
    if len(text) <= _MAX_CHARS:
        return {"text": text, "truncated": False}
    fields: dict[str, Any] = {"text": text[:_MAX_CHARS], "truncated": True}
    try:
        fields["file"] = write_text_output("page", text)
    except Exception as exc:
        logger.warning("could not spill page text to file: {exc}", exc=exc)
    return fields


def transcript_fields(transcript: str) -> dict[str, Any]:
    """Envelope fields for a transcript: preview + optional spill file.

    Short transcripts ride the envelope as ``transcript``; long ones
    keep a ``transcript`` preview, flag ``transcript_truncated``, and
    attempt to spill all fetched caption text to ``transcript_file``.
    Shell-disabled models cannot read the file; failed writes omit it.
    """
    if len(transcript) <= _MAX_TRANSCRIPT_CHARS:
        return {"transcript": transcript}
    fields: dict[str, Any] = {
        "transcript": transcript[:_MAX_TRANSCRIPT_CHARS],
        "transcript_truncated": True,
    }
    try:
        fields["transcript_file"] = write_text_output("transcript", transcript)
    except Exception as exc:
        logger.warning("could not spill transcript to file: {exc}", exc=exc)
    return fields


def video_meta(url: str, settings: Settings) -> dict[str, Any] | None:
    """Video/post yt-dlp metadata for *url*, or None when unresolvable.

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
        "lazy_playlist": True,
        "playlistend": _MAX_POST_ITEMS + 1,
        "socket_timeout": max(settings.web_search_timeout, 2.0),
    }
    if _YOUTUBE_HOST_RE.match(url):
        opts["extractor_args"] = {"youtube": {"player_client": [_YOUTUBE_CLIENT]}}
    if settings.web_search_proxy:
        opts["proxy"] = settings.web_search_proxy
    info = _extract_safely(url, opts)
    if not info:
        return None
    if "entries" in info:
        info = _post_info(info)
    meta = _meta_fields(info)
    _attach_caption_tracks(meta, info, url)
    return meta


def _extract_safely(url: str, opts: dict[str, Any]) -> dict[str, Any] | None:
    """The extracted info dict, or None when yt-dlp cannot resolve it."""
    try:
        with yt_dlp.YoutubeDL(cast(Any, opts)) as ydl:
            return _extract(ydl, url)
    except Exception as exc:
        logger.info("Video metadata unavailable for {url}: {exc}", url=url, exc=exc)
        return None


def _meta_fields(info: dict[str, Any]) -> dict[str, Any]:
    """The envelope's scalar metadata fields from a resolved info dict."""
    description = str(info.get("description") or "")
    fields: dict[str, Any] = {
        "kind": "post" if "entries" in info else "video",
        "title": info.get("title"),
        "description": description[:_DESCRIPTION_CHARS] or None,
        "uploader": info.get("uploader") or info.get("channel"),
        "duration_s": info.get("duration"),
        "view_count": info.get("view_count"),
        "id": info.get("id"),
    }
    if "entries" in info:
        fields["item_count"] = info["item_count"]
        if info.get("item_count_truncated"):
            fields["item_count_truncated"] = True
    if len(description) > _DESCRIPTION_CHARS:
        fields["description_truncated"] = True
    return fields


def _attach_caption_tracks(meta: dict[str, Any], info: dict[str, Any], url: str) -> None:
    """Stash YouTube's caption tracks on *meta* for the transcript path.

    Merge available manual and automatic tracks, preferring manual for
    the same language. Availability and fetchability are not guaranteed.
    """
    if not _YOUTUBE_HOST_RE.match(url):
        return
    manual = cast("dict[str, list[dict[str, Any]]]", info.get("subtitles") or {})
    automatic = cast(
        "dict[str, list[dict[str, Any]]]", info.get("automatic_captions") or {}
    )
    tracks = automatic | {
        language: formats for language, formats in manual.items() if formats
    }
    if tracks:
        meta["_caption_tracks"] = tracks


def _extract(ydl: Any, url: str) -> dict[str, Any] | None:
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


def _post_info(info: dict[str, Any]) -> dict[str, Any]:
    """A multi-item post's own info: the parent dict carries the caption.

    Format processing dies on multi-item Instagram posts ("No video
    formats found"), but the raw extractor dict still describes the
    *post* — caption, uploader, id — which is what the model needs,
    plus the item count and summed available duration. Inspect at most
    100 entries plus one lookahead, never exhaust an arbitrary playlist.
    A truncated count/sum covers only inspected dict entries; preserve
    a supplied parent duration and the raw parent metadata.
    """
    source = cast("Iterable[Any] | None", info.get("entries"))
    inspected = list(islice(source if source is not None else (), _MAX_POST_ITEMS + 1))
    truncated = len(inspected) > _MAX_POST_ITEMS
    entries = [
        cast("dict[str, Any]", e)
        for e in inspected[:_MAX_POST_ITEMS]
        if isinstance(e, dict) and e
    ]
    durations = [e["duration"] for e in entries if e.get("duration") is not None]
    total = sum(durations) if durations else None
    post = info | {
        "entries": entries,
        "title": info.get("title")
        or f"post with {len(entries)}{'+' if truncated else ''} items",
        "duration": info.get("duration") if info.get("duration") is not None else total,
        "item_count": len(entries),
    }
    if truncated:
        post["item_count_truncated"] = True
    return post


def youtube_transcript(
    url: str,
    tracks: dict[str, list[dict[str, Any]]],
    language: str = "en",
    settings: Settings | None = None,
) -> str:
    """The video's captions as paragraphed prose, or "" when unavailable.

    *tracks* combines available manual and automatic tracks. Try the
    requested language first, then other languages, using direct
    json3/SRT/VTT formats. HLS playlists are not transcribed. Empty
    output means no fetched usable text, not proof of absent captions.
    When supplied, settings control the proxy and effective HTTP timeout.
    Caption HTTPX requests do not impersonate browser TLS and can be refused.
    """
    ordered = _ordered_caption_tracks(tracks, language)
    if not ordered:
        logger.info("No usable caption track for {url}", url=url)
        return ""
    for formats in ordered:
        text = _fetch_track_text(formats, settings)
        if text:
            return text
    logger.info("Caption tracks for {url} had no fetchable format", url=url)
    return ""


def _ordered_caption_tracks(
    tracks: dict[str, list[dict[str, Any]]], language: str
) -> list[list[dict[str, Any]]]:
    """Caption format lists, the *language* track first, others after."""
    ordered = sorted(
        tracks.items(), key=lambda entry: (not entry[0].startswith(language), entry[0])
    )
    return [formats for _, formats in ordered]


def _fetch_track_text(
    formats: list[dict[str, Any]], settings: Settings | None = None
) -> str:
    """One track list's captions as prose, trying json3, srt, then vtt."""
    for wanted in ("json3", "srt", "vtt"):
        for track in (format for format in formats if format.get("ext") == wanted):
            track_url = str(track.get("url") or "")
            text = _fetch_captions(track_url, settings)
            if text and not text.lstrip().startswith("#EXTM3U"):
                parsed = _paragraph(
                    _join_json3(text) if wanted == "json3" else _strip_vtt(text)
                )
                if parsed:
                    return parsed
    return ""


def _fetch_captions(track_url: str, settings: Settings | None = None) -> str:
    """Fetch a caption body, "" on ordinary failures; no browser impersonation.

    Supplied settings use the page/extractor proxy and timeout floor (2s).
    Without settings, uses a 10s HTTPX timeout and no proxy.
    The body is downloaded in full; transcript preview limits are not byte caps.
    """
    if not track_url:
        return ""
    try:
        response = httpx.get(
            track_url,
            timeout=max(settings.web_search_timeout, 2.0) if settings else 10.0,
            proxy=settings.web_search_proxy if settings else None,
            headers={"User-Agent": "Mozilla/5.0"},
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.text
    except Exception as exc:
        logger.info("Caption fetch failed: {exc}", exc=exc)
        return ""


def _strip_vtt(vtt: str) -> str:
    """Best-effort WebVTT/SRT cue text, without metadata blocks or tags."""
    lines: list[str] = []
    normalized = vtt.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    for block in re.split(r"\n[ \t]*\n", normalized):
        if re.match(r"(?:NOTE|STYLE|REGION)(?:\s|$)", block.lstrip()):
            continue
        cue_lines = block.splitlines()
        cue_start = next((i + 1 for i, line in enumerate(cue_lines) if "-->" in line), 0)
        if not cue_start and block.lstrip().startswith("WEBVTT"):
            continue
        for line in cue_lines[cue_start:]:
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith(("WEBVTT", "#"))
                or "-->" in stripped
                or re.fullmatch(r"\d{1,2}:\d{2}:\d{2}[.,]\d{3}", stripped)
                or stripped.isdigit()
            ):
                continue
            lines.append(unescape(_TAG_RE.sub("", stripped)))
    return " ".join(lines)


def _paragraph(caption_text: str) -> str:
    """Caption text as readable paragraphed prose.

    Caption fragments arrive as ~3-second snippets; joining them
    verbatim yields mid-sentence breaks every few words. Join, collapse
    whitespace, and break into paragraphs at sentence boundaries —
    the same formatting the dropped transcript tool used. The result
    stays whole: truncation/spill is the envelope's job
    (:func:`transcript_fields`).
    """
    joined = re.sub(r"\s+", " ", caption_text).strip()
    sentences = _SENTENCE_END_RE.split(joined)
    paragraphs = [
        " ".join(sentences[i : i + _PARAGRAPH_SENTENCES]).strip()
        for i in range(0, len(sentences), _PARAGRAPH_SENTENCES)
    ]
    return "\n\n".join(p for p in paragraphs if p)


def _join_json3(caption_text: str) -> str:
    """Join json3 utf8 segments, or "" for malformed JSON/data shapes."""
    try:
        data = json.loads(caption_text)
    except json.JSONDecodeError:
        return ""
    if not isinstance(data, dict):
        return ""
    events = cast("dict[str, Any]", data).get("events")
    if not isinstance(events, list):
        return ""
    parts: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            return ""
        segments = cast("dict[str, Any]", event).get("segs", [])
        if not isinstance(segments, list):
            return ""
        chunks: list[str] = []
        for segment in segments:
            if not isinstance(segment, dict):
                return ""
            text = cast("dict[str, Any]", segment).get("utf8", "")
            if not isinstance(text, str):
                return ""
            chunks.append(text)
        part = "".join(chunks).strip()
        if part:
            parts.append(part)
    return " ".join(parts)


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
    stripped = _TAG_RE.sub(" ", text)

    collapsed = _WHITESPACE_RE.sub(" ", stripped)
    return collapsed.strip() or "(no readable text on page)"
