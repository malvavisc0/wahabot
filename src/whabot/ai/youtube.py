"""YouTube transcript extraction tool.

Ported from aria-ai's ``aria.tools.search.youtube``, but returns the
transcript text inline (truncated) instead of writing it to disk, since
whabot has no file tools. Pure Python via ``youtube-transcript-api``; no
API key. The tool returns a status string and never raises.
"""

import re
from typing import Any

from loguru import logger
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

__all__ = ["get_youtube_transcript"]

_MAX_TRANSCRIPT_CHARS = 6000

# Split the joined transcript into paragraphs roughly every N sentences so
# the output is readable prose, not hundreds of caption-fragment lines.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SENTENCES = 3

_TRANSCRIPT_ERRORS = (NoTranscriptFound, TranscriptsDisabled)


def get_youtube_transcript(url: str) -> str:
    """Fetch and format a YouTube video's captions/transcript.

    Args:
        url: A YouTube video URL, e.g. ``https://youtube.com/watch?v=...``.

    Returns:
        Readable paragraphed prose (truncated), or an explanatory error
        message. Only works for videos that have captions available.
    """
    video_id = _extract_video_id(url)
    if not video_id:
        return "Could not extract YouTube video ID from URL."

    try:
        transcript_text, segments, duration = _get_youtube_transcript(video_id)
    except _TRANSCRIPT_ERRORS:
        return (
            f"No captions available for video {video_id}. The video may "
            "lack subtitles or have them disabled."
        )
    except Exception as exc:
        logger.exception(
            "Failed to get YouTube transcript for {video_id}", video_id=video_id
        )
        return f"Failed to get YouTube transcript: {exc}"

    header = (
        f"YouTube transcript for {video_id} ({segments} segments, ~{duration:.0f}s):\n"
    )
    if len(transcript_text) > _MAX_TRANSCRIPT_CHARS:
        transcript_text = transcript_text[:_MAX_TRANSCRIPT_CHARS] + "\n...[truncated]"
    return header + transcript_text


_VIDEO_ID_PATTERN = (
    r"(?:youtube\.com(?:/watch\?[^#]*\bv=|/shorts/|/embed/|/live/|/v/)"
    r"|youtu\.be/)"
    r"([0-9A-Za-z_-]{11})(?![0-9A-Za-z_-])"
)
_VIDEO_ID_RE = re.compile(_VIDEO_ID_PATTERN)


def _extract_video_id(url: str) -> str | None:
    """Extract a YouTube video ID from a YouTube URL.

    Matches ``youtube.com`` watch/shorts/embed/live/v URLs and
    ``youtu.be`` short links. Bare IDs and non-YouTube URLs return None.
    """
    match = _VIDEO_ID_RE.search(url)
    return match.group(1) if match else None


def _get_youtube_transcript(
    video_id: str, languages: list[str] | None = None
) -> tuple[str, int, float]:
    """Fetch and format a YouTube transcript.

    Returns ``(transcript_text, segment_count, estimated_duration)``.
    """
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=languages or ["en"])
    text = _format_transcript_text(transcript.snippets)
    duration = sum(snippet.duration for snippet in transcript.snippets)
    return text, len(transcript.snippets), duration


def _format_transcript_text(snippets: list[Any]) -> str:
    """Join caption snippets into readable paragraphed prose.

    YouTube splits captions into ~3-second fragments, so joining them
    verbatim yields hundreds of mid-sentence line breaks. Instead, join
    snippet text with spaces, collapse whitespace, and break into
    paragraphs at sentence boundaries.
    """
    joined = " ".join(snippet.text for snippet in snippets)
    joined = re.sub(r"\s+", " ", joined).strip()
    sentences = _SENTENCE_END_RE.split(joined)
    paragraphs: list[str] = []
    for i in range(0, len(sentences), _PARAGRAPH_SENTENCES):
        paragraphs.append(" ".join(sentences[i : i + _PARAGRAPH_SENTENCES]).strip())
    return "\n\n".join(paragraphs)
