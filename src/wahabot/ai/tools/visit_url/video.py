from collections.abc import Iterable
from itertools import islice
from typing import Any, cast

from loguru import logger

from wahabot.ai.tools.envelope import ok
from wahabot.ai.tools.outfile import write_text_output
from wahabot.ai.tools.visit_url.captions import (
    attach_caption_tracks,
    fetch_track_text,
    ordered_caption_tracks,
)
from wahabot.ai.tools.visit_url.constants import (
    MAX_POST_ITEMS,
    MAX_TRANSCRIPT_CHARS,
    MEDIA_HOST_RE,
    YOUTUBE_CLIENT,
    YOUTUBE_HOST_RE,
)
from wahabot.ai.tools.visit_url.page import extract_safely, meta_fields
from wahabot.settings import Settings


def media_envelope(url: str, settings: Settings) -> str | None:
    """A media host's yt-dlp metadata envelope, or None to take the HTML path.

    Extraction failures and unresolved placeholders mean the page may
    still be readable (e.g. a private or removed post), so every miss
    falls back to HTML instead of returning an error envelope.
    """
    if not MEDIA_HOST_RE.match(url):
        return None
    try:
        meta = video_meta(url, settings)
        if meta is not None:
            fields = with_transcript(url, meta, settings)
            if "transcript" in fields:
                fields["content_kind"] = "transcript"
                fields["note"] = "Transcript only; the video itself was not watched."
            else:
                fields["content_kind"] = "caption"
                fields["metadata_only"] = True
                fields["note"] = (
                    "Caption/metadata only; the video itself was not watched. "
                    "Do not claim to have watched or verified the video."
                )
            return ok(source="yt-dlp", url=url, **fields)
    except Exception as exc:
        logger.info("Media processing failed for {url}: {exc}", url=url, exc=exc)
    logger.info("yt-dlp could not resolve {url}; falling back to HTML", url=url)
    return None


def with_transcript(url: str, meta: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Metadata fields with a YouTube transcript inlined when tracks resolve."""
    tracks = cast(
        "dict[str, list[dict[str, Any]]] | None", meta.pop("_caption_tracks", None)
    )
    if tracks and YOUTUBE_HOST_RE.match(url):
        transcript = youtube_transcript(url, tracks, settings=settings)
        if transcript:
            meta.update(transcript_fields(transcript))
    return meta


def transcript_fields(transcript: str) -> dict[str, Any]:
    """Envelope fields for a transcript: preview + optional spill file.

    Short transcripts ride the envelope as ``transcript``; long ones
    keep a ``transcript`` preview, flag ``transcript_truncated``, and
    attempt to spill all fetched caption text to ``transcript_file``.
    Shell-disabled models cannot read the file; failed writes omit it.
    """
    if len(transcript) <= MAX_TRANSCRIPT_CHARS:
        return {"transcript": transcript}
    fields: dict[str, Any] = {
        "transcript": transcript[:MAX_TRANSCRIPT_CHARS],
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
        "playlistend": MAX_POST_ITEMS + 1,
        "socket_timeout": max(settings.web_search_timeout, 2.0),
    }
    if YOUTUBE_HOST_RE.match(url):
        opts["extractor_args"] = {"youtube": {"player_client": [YOUTUBE_CLIENT]}}
    if settings.web_search_proxy:
        opts["proxy"] = settings.web_search_proxy
    info = extract_safely(url, opts)
    if not info:
        return None
    if "entries" in info:
        info = post_info(info)
    meta = meta_fields(info)
    attach_caption_tracks(meta, info, url)
    return meta


def post_info(info: dict[str, Any]) -> dict[str, Any]:
    """A multi-item post's own info: the parent dict carries the caption.

    Format processing dies on multi-item Instagram posts ("No video
    formats found"), but the raw extractor dict still describes the
    *post* — caption, uploader, id — which is what the model needs,
    plus the item count and summed available duration. Inspect at most
    100 entries plus one lookahead, never exhaust an arbitrary playlist.
    A truncated count/sum covers only inspected dict entries; preserve
    a supplied parent duration and the raw parent metadata.
    """
    source = cast(Iterable[Any] | None, info.get("entries"))
    inspected = list(islice(source if source is not None else (), MAX_POST_ITEMS + 1))
    truncated = len(inspected) > MAX_POST_ITEMS
    entries = post_entries(inspected[:MAX_POST_ITEMS])
    post = info | {
        "entries": entries,
        "title": info.get("title")
        or f"post with {len(entries)}{'+' if truncated else ''} items",
        "duration": info.get("duration")
        if info.get("duration") is not None
        else entries_duration(entries),
        "item_count": len(entries),
    }
    if truncated:
        post["item_count_truncated"] = True
    return post


def post_entries(inspected: list[Any]) -> list[dict[str, Any]]:
    """The non-empty dict entries among a post's inspected items."""
    return [cast("dict[str, Any]", e) for e in inspected if isinstance(e, dict) and e]


def entries_duration(entries: list[dict[str, Any]]) -> Any:
    """Summed duration of the entries that carry one; None when none do."""
    durations = [e["duration"] for e in entries if e.get("duration") is not None]
    return sum(durations) if durations else None


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
    ordered = ordered_caption_tracks(tracks, language)
    if not ordered:
        logger.info("No usable caption track for {url}", url=url)
        return ""
    for formats in ordered:
        text = fetch_track_text(formats, settings)
        if text:
            return text
    logger.info("Caption tracks for {url} had no fetchable format", url=url)
    return ""
