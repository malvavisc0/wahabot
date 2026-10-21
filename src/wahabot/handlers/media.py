import asyncio
import io
from typing import Any, cast

from llama_index.core.llms.function_calling import FunctionCallingLLM
from loguru import logger
from PIL import Image

from wahabot.ai.messages import (
    video_media,
)
from wahabot.ai.tools.url_videos import fetch_url_video, video_urls
from wahabot.ai.video import caption_video, extract_frames, video_marker
from wahabot.core.models import WahaEvent
from wahabot.core.transcribe import fetch_transcript
from wahabot.core.waha import MediaTooLargeError, WahaClient
from wahabot.settings import Settings


def album_message_id(event: WahaEvent) -> str:
    """The album container's message id — the anchor its reply quotes."""
    return str(event.payload.get("id", ""))


def download_image(
    waha: WahaClient, media: dict[str, Any], message_id: str, max_bytes: int
) -> dict[str, Any] | None:
    """Download an image's bytes; None keeps the turn text-only on failure."""
    url = str(media.get("url", ""))
    if not url:
        return None
    try:
        data = waha.download_media(url, max_bytes=max_bytes)
    except MediaTooLargeError:
        logger.info(
            "Skipping image over {max_bytes} B in message {id}",
            max_bytes=max_bytes,
            id=message_id,
        )
        return None
    except Exception as exc:
        logger.warning(
            "Image download failed for message {id}: {exc}", id=message_id, exc=exc
        )
        return None
    mimetype = str(media.get("mimetype") or "image/jpeg")
    logger.info("Downloaded image ({mime}, {size} B)", mime=mimetype, size=len(data))
    if mimetype == "image/webp":
        try:
            data = first_frame_png(data)
            mimetype = "image/png"
        except Exception as exc:
            logger.warning(
                "WebP first-frame conversion failed for {id}: {exc}",
                id=message_id,
                exc=exc,
            )
            return None
    return {"data": data, "mimetype": mimetype}


def first_frame_png(data: bytes) -> bytes:
    """The first frame of an animated image (webp/gif) as PNG bytes.

    Vision models accept stills, not animations — a 23-frame sticker
    webp would be rejected or misread. Callers decide by mimetype
    whether to convert; static webp converts losslessly too.
    """
    with Image.open(io.BytesIO(data)) as img:
        img.seek(0)
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()


async def video_transcript(settings: Settings, data: bytes, filename: str) -> str:
    """The video's spoken track via WhisperX, or "" when unavailable.

    Posts the whole video file — the service demuxes audio itself, so
    this deliberately
    bypasses ``is_transcribable_mimetype``: that gate exists for the
    voice-note path. Audio-less videos (loops, screen recordings) get
    a 500 from the service every time — expected traffic, not an
    error, hence the debug log — and any other failure drops the
    transcript part of the marker.
    """
    if not settings.transcribe_url:
        return ""
    try:
        transcript = await asyncio.to_thread(fetch_transcript, settings, data, filename)
    except Exception as exc:
        logger.debug("Video transcript unavailable: {exc}", exc=exc)
        return ""
    return transcript.strip()


async def prepare_video(
    event: WahaEvent,
    waha: WahaClient,
    settings: Settings,
    llm: FunctionCallingLLM,
    semaphore: asyncio.Semaphore | None = None,
) -> dict[str, Any] | None:
    """Download + frames + caption + transcript for a video turn.

    Returns ``{"frames": [...], "marker": "(video shows: …) …"}``, or
    None when the download failed (the turn degrades exactly like a
    failed image download: sender text runs, a bare video stays
    silent). Every stage is fail-soft — a dead ffmpeg, a failed
    caption call or a failed transcription each drop their marker
    part and the turn still runs. Runs before the chat's run lock so
    no network or vision call extends the serialized agent-run section.
    """
    media = video_media(event)
    if media is None:
        return None
    url = str(media["url"])
    message_id = str(event.payload.get("id", ""))
    try:
        data = await asyncio.to_thread(waha.download_media, url, settings.max_video_bytes)
    except MediaTooLargeError:
        logger.info(
            "Skipping video over {max} B in message {id}",
            max=settings.max_video_bytes,
            id=message_id,
        )
        return None
    except Exception as exc:
        logger.warning(
            "Video download failed for message {id}: {exc}", id=message_id, exc=exc
        )
        return None
    logger.info(
        "Downloaded video ({size} B) from message {id}", size=len(data), id=message_id
    )
    filename = str(media.get("filename") or "") or "video.mp4"
    return await build_video(settings, llm, semaphore, data, filename)


async def build_video(
    settings: Settings,
    llm: FunctionCallingLLM,
    semaphore: asyncio.Semaphore | None,
    data: bytes,
    filename: str,
) -> dict[str, Any]:
    """Frames + caption + transcript for already-downloaded video bytes.

    Shared by the WAHA video path (:func:`prepare_video`) and the URL
    path (:func:`prepare_url_video`): identical fail-soft behavior —
    a dead ffmpeg, caption call or transcription each drop their marker
    part and still return a (possibly bare) result.
    """
    frames = await asyncio.to_thread(extract_frames, data, settings.video_frames)
    caption = await caption_video(llm, frames, semaphore=semaphore) if frames else ""
    transcript = await video_transcript(settings, data, filename)
    return {
        "frames": [{"data": frame, "mimetype": "image/jpeg"} for frame in frames],
        "marker": video_marker(caption, transcript),
    }


async def prepare_url_video(
    settings: Settings,
    llm: FunctionCallingLLM,
    semaphore: asyncio.Semaphore | None,
    body: str,
) -> dict[str, Any] | None:
    """Resolve + download video URLs in *body*, then frames + transcript.

    Returns the same ``{"frames": [...], "marker": ...}`` shape as
    :func:`prepare_video`, or None when no URL resolved to a video. Only
    the first ``settings.max_url_videos`` links are tried; each failure
    is a log line and the link stays ordinary text. Runs before the
    chat's run lock (like ``prepare_video``) so the yt-dlp download and
    the vision/transcription calls don't extend the serialized
    agent-run section.
    """
    urls = video_urls(body, settings.max_url_videos)
    for url in urls:
        result = await asyncio.to_thread(fetch_url_video, settings, url)
        if result is None:
            continue
        return await build_video(
            settings, llm, semaphore, result["data"], result["filename"]
        )
    return None


def video_frames(
    video: dict[str, Any] | None, url_video: dict[str, Any] | None
) -> list[dict[str, Any]] | None:
    """The frames of whichever video source resolved, or None.

    The URL path only runs when no WAHA video was attached, so at most
    one of the two is ever non-None — this is a pick, not a merge.
    """
    for source in (video, url_video):
        if source is not None:
            return cast(list[dict[str, Any]], source["frames"]) or None
    return None
