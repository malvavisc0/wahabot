"""Sniff video URLs out of message text and download them via yt-dlp.

A bare media link ("watch this https://www.instagram.com/reel/...") is
just text to the handler. This module finds those URLs, resolves and
downloads them with yt-dlp — whose per-site extractors return a direct
media URL without hand-rolled HTML parsing, and which covers Instagram
and Facebook Reels, TikTok, YouTube and ~1900 other sites with one
code path. The bytes come back to the video pipeline (frames + WhisperX
transcript), the same one WAHA video messages use.

Unsupported URLs fail soft: yt-dlp raises "Unsupported URL" quickly and
the link simply stays text. The download runs yt-dlp's own
extractor + downloader to a temp file, which is read back and discarded.
"""

import re
import tempfile
from pathlib import Path
from typing import Any, cast

import yt_dlp
from loguru import logger

from wahabot.settings import Settings

__all__ = ["fetch_url_video", "video_urls"]

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

#: Hosts excluded from sniffing: YouTube is long-form — the
#: ``get_youtube_transcript`` tool reads its captions directly, which
#: beats downloading an hour of video to sample six frames. Its formats
#: also need a JS runtime to resolve, so ``best`` often fails outright.
_YOUTUBE_HOST_RE = re.compile(
    r"^https?://(?:[\w-]+\.)?(?:youtube\.com|youtu\.be|youtube-nocookie\.com)/",
    re.IGNORECASE,
)

#: Files to skip when picking the downloaded artifact out of the temp dir.
_SKIP_SUFFIXES = (".part", ".ytdl", ".temp")


def video_urls(text: str, limit: int) -> list[str]:
    """Extract up to *limit* candidate media URLs from *text*.

    Any http(s) URL qualifies except YouTube (see ``_YOUTUBE_HOST_RE``);
    yt-dlp decides whether the rest is a downloadable video. Trailing
    punctuation/quote chars are stripped (the same trim ``url_images``
    does) so a period after a pasted link does not break the extractor.
    """
    urls: list[str] = []
    for match in _URL_RE.finditer(text):
        url = match.group().rstrip(").,;:!?\"'>]}")
        if _YOUTUBE_HOST_RE.match(url):
            continue
        if url not in urls:
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def fetch_url_video(settings: Settings, url: str) -> dict[str, Any] | None:
    """Download *url* as video bytes via yt-dlp, or None on failure.

    Returns ``{"data": bytes, "filename": str}`` for a resolved, in-cap
    video; an unsupported URL, a download error or an oversized file
    costs nothing but a log line. ``filename`` keeps the media's real
    extension so the transcript upload names the file correctly.
    """
    try:
        data, filename = download_video(settings, url)
    except Exception as exc:
        logger.info("Video URL {url} resolved to nothing: {exc}", url=url, exc=exc)
        return None
    if data is None:
        return None
    logger.info("Fetched video URL {url} ({size} B)", url=url, size=len(data))
    return {"data": data, "filename": filename}


def download_video(settings: Settings, url: str) -> tuple[bytes | None, str]:
    """Resolve and download *url* to a temp file; ``(bytes, name)`` or ``(None, "")``."""
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "format": "best",
        "socket_timeout": max(settings.web_search_timeout, 2.0),
        "max_filesize": settings.max_video_bytes,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        opts["outtmpl"] = str(Path(tmpdir) / "%(id)s.%(ext)s")
        with yt_dlp.YoutubeDL(cast(Any, opts)) as ydl:
            ydl.download([url])
        for path in sorted(Path(tmpdir).iterdir()):
            if not path.is_file() or path.name.endswith(_SKIP_SUFFIXES):
                continue
            if path.stat().st_size > settings.max_video_bytes:
                return None, ""
            return path.read_bytes(), path.name
        return None, ""
