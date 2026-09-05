"""Voice-note transcription via the WhisperX service."""

import asyncio
from typing import Any

import httpx
from loguru import logger

from wahabot.core.models import WahaEvent
from wahabot.core.waha import MediaTooLargeError, WahaClient
from wahabot.settings import Settings


def is_transcribable_mimetype(mimetype: str) -> bool:
    """True for audio mimes we can hand to the service (audio/...), else False."""
    return mimetype.startswith("audio/")


def audio_media(event: WahaEvent) -> dict[str, Any] | None:
    """The media dict of an audio message, or None.

    Voice notes carry ``media`` at the top level or inside ``_data``
    (engine-dependent); only entries with a URL qualify.
    """
    payload: dict[str, Any] = event.payload
    media: Any = payload.get("media")
    if not isinstance(media, dict):
        data: Any = payload.get("_data")
        if isinstance(data, dict):
            data_dict: dict[str, Any] = data
            media = data_dict.get("media")
    if not isinstance(media, dict):
        return None
    media_dict: dict[str, Any] = media
    if not media_dict.get("url"):
        return None
    return media_dict


def fetch_transcript(settings: Settings, audio: bytes, filename: str) -> str:
    """POST audio to settings.transcribe_url + '/transcribe' and return text.

    Multipart form: ``file`` plus the configured ``language``. Raises on
    non-2xx / timeout. Joins the response ``segments``' ``text`` in
    order, stripped and space-separated (drops per-segment leading-space
    artifacts). Diarization fields are left to the service's defaults.
    """
    url = f"{settings.transcribe_url.rstrip('/')}/transcribe"
    files = {"file": (filename, audio, "application/octet-stream")}
    data = {"language": settings.transcribe_language}
    with httpx.Client(timeout=settings.transcribe_timeout) as client:
        response = client.post(url, files=files, data=data)
        response.raise_for_status()
        segments = response.json().get("segments", [])
    return " ".join(str(s.get("text", "")).strip() for s in segments).strip()


async def transcribe_voice_note(
    event: WahaEvent, waha: WahaClient, settings: Settings
) -> str | None:
    """Transcribe a voice note's audio; None on any failure.

    Fetches the note's media (bounded by ``max_audio_bytes``), sends it
    to the WhisperX service and returns the transcript (stripped, None
    when empty). Expected failures — download errors, oversized media,
    bad mimetype, service errors, empty/blank transcripts — return None
    after logging so the handler drops the message like today. Network
    runs on threads so notes transcribe in parallel.
    """
    media = audio_media(event)
    if media is None:
        return None
    url = str(media["url"])
    if not is_transcribable_mimetype(str(media.get("mimetype") or "")):
        logger.debug(
            "Skipping non-audio mimetype in message {id}", id=event.payload.get("id")
        )
        return None
    try:
        data = await asyncio.to_thread(waha.download_media, url, settings.max_audio_bytes)
    except MediaTooLargeError:
        logger.info(
            "Skipping voice note over {max} B in message {id}",
            max=settings.max_audio_bytes,
            id=event.payload.get("id"),
        )
        return None
    except Exception as exc:
        logger.warning(
            "Voice-note download failed for message {id}: {exc}",
            id=event.payload.get("id"),
            exc=exc,
        )
        return None
    filename = str(media.get("filename") or "") or "voice-note.oga"
    try:
        text = await asyncio.to_thread(fetch_transcript, settings, data, filename)
    except Exception as exc:
        logger.warning(
            "Transcription failed for message {id}: {exc}",
            id=event.payload.get("id"),
            exc=exc,
        )
        return None
    return text.strip() or None
