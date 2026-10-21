import json
import re
from html import unescape
from typing import Any, cast

import httpx
from loguru import logger

from wahabot.ai.tools.visit_url.constants import (
    PARAGRAPH_SENTENCES,
    SENTENCE_END_RE,
    TAG_RE,
    YOUTUBE_HOST_RE,
)
from wahabot.settings import Settings


def attach_caption_tracks(meta: dict[str, Any], info: dict[str, Any], url: str) -> None:
    """Stash YouTube's caption tracks on *meta* for the transcript path.

    Merge available manual and automatic tracks, preferring manual for
    the same language. Availability and fetchability are not guaranteed.
    """
    if not YOUTUBE_HOST_RE.match(url):
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


def ordered_caption_tracks(
    tracks: dict[str, list[dict[str, Any]]], language: str
) -> list[list[dict[str, Any]]]:
    """Caption format lists, the *language* track first, others after."""
    ordered = sorted(
        tracks.items(), key=lambda entry: (not entry[0].startswith(language), entry[0])
    )
    return [formats for _, formats in ordered]


def fetch_track_text(
    formats: list[dict[str, Any]], settings: Settings | None = None
) -> str:
    """One track list's captions as prose, trying json3, srt, then vtt."""
    for wanted in ("json3", "srt", "vtt"):
        for track in (format for format in formats if format.get("ext") == wanted):
            track_url = str(track.get("url") or "")
            text = fetch_captions(track_url, settings)
            if text and not text.lstrip().startswith("#EXTM3U"):
                parsed = paragraph(
                    join_json3(text) if wanted == "json3" else strip_vtt(text)
                )
                if parsed:
                    return parsed
    return ""


def fetch_captions(track_url: str, settings: Settings | None = None) -> str:
    """Fetch a caption body, "" on ordinary failures; no browser impersonation.

    Supplied settings use the page/extractor proxy and timeout floor (2s).
    Without settings, uses a 10s HTTPX timeout and no proxy.
    The body is downloaded in full; transcript preview limits are not byte caps.
    """
    if not track_url:
        return ""
    proxy = (settings.web_search_proxy or None) if settings else None
    timeout = max(settings.web_search_timeout, 2.0) if settings else 10.0
    try:
        response = httpx.get(
            track_url,
            timeout=timeout,
            proxy=proxy,
            headers={"User-Agent": "Mozilla/5.0"},
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.text
    except Exception as exc:
        logger.info("Caption fetch failed: {exc}", exc=exc)
        return ""


def strip_vtt(vtt: str) -> str:
    """Best-effort WebVTT/SRT cue text, without metadata blocks or tags."""
    normalized = vtt.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for block in re.split(r"\n[ \t]*\n", normalized):
        lines.extend(vtt_block_lines(block))
    return " ".join(lines)


def vtt_block_lines(block: str) -> list[str]:
    """One cue block's spoken lines; metadata and header blocks yield none."""
    if re.match(r"(?:NOTE|STYLE|REGION)(?:\s|$)", block.lstrip()):
        return []
    cue_lines = block.splitlines()
    cue_start = next((i + 1 for i, line in enumerate(cue_lines) if "-->" in line), 0)
    if not cue_start and block.lstrip().startswith("WEBVTT"):
        return []
    return [
        unescape(TAG_RE.sub("", stripped))
        for stripped in (line.strip() for line in cue_lines[cue_start:])
        if not is_cue_noise(stripped)
    ]


def is_cue_noise(stripped: str) -> bool:
    """Whether a stripped cue line is timing, numbering, or file markup."""
    return (
        not stripped
        or stripped.startswith(("WEBVTT", "#"))
        or "-->" in stripped
        or stripped.isdigit()
        or re.fullmatch(r"\d{1,2}:\d{2}:\d{2}[.,]\d{3}", stripped) is not None
    )


def paragraph(caption_text: str) -> str:
    """Caption text as readable paragraphed prose.

    Caption fragments arrive as ~3-second snippets; joining them
    verbatim yields mid-sentence breaks every few words. Join, collapse
    whitespace, and break into paragraphs at sentence boundaries —
    the same formatting the dropped transcript tool used. The result
    stays whole: truncation/spill is the envelope's job
    (:func:`transcript_fields`).
    """
    joined = re.sub(r"\s+", " ", caption_text).strip()
    sentences = SENTENCE_END_RE.split(joined)
    paragraphs = [
        " ".join(sentences[i : i + PARAGRAPH_SENTENCES]).strip()
        for i in range(0, len(sentences), PARAGRAPH_SENTENCES)
    ]
    return "\n\n".join(p for p in paragraphs if p)


def join_json3(caption_text: str) -> str:
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
        part = json3_event_text(event)
        if part is None:
            return ""
        if part:
            parts.append(part)
    return " ".join(parts)


def json3_event_text(event: Any) -> str | None:
    """One caption event's joined utf8 text; None marks a malformed shape.

    An event without segments yields "" (nothing to say); a malformed
    event or segment yields None so the whole body fails soft.
    """
    if not isinstance(event, dict):
        return None
    segments = cast("dict[str, Any]", event).get("segs", [])
    if not isinstance(segments, list):
        return None
    chunks: list[str] = []
    for segment in segments:
        text = json3_segment_text(segment)
        if text is None:
            return None
        chunks.append(text)
    return "".join(chunks).strip()


def json3_segment_text(segment: Any) -> str | None:
    """One segment's utf8 text, or None when the shape is malformed."""
    if not isinstance(segment, dict):
        return None
    text = cast("dict[str, Any]", segment).get("utf8", "")
    if not isinstance(text, str):
        return None
    return text
