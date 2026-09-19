"""Voice synthesis via the OpenAI-compatible TTS service.

Outbound counterpart to :mod:`wahabot.core.transcribe` — perception is
preprocessing (WhisperX on the way in), expression is a tool source
(:func:`wahabot.ai.tools.whatsapp.send_voice` with ``text``). One
sync function, called from the tool fn's worker thread like every
other network call, fail-soft like presence: a synthesis failure must
degrade to a text reply, never crash the run.
"""

import httpx
from loguru import logger

from wahabot.settings import Settings

__all__ = ["synthesize"]


def synthesize(settings: Settings, text: str, language: str) -> bytes | None:
    """Speak *text* as mp3 bytes via the TTS service; None on any failure.

    ``POST {tts_url}/v1/audio/speech`` with the OpenAI speech shape:
    ``voice`` comes from ``settings.tts_voices`` for *language*
    (``tts_default_language`` when unmapped), and the language's frozen
    ``tts_instruct`` entry rides the request when non-empty — never a
    model-controlled knob. Any failure (timeout, non-200, empty body)
    logs a one-line warning and returns None so the caller can fall
    back to sending plain text.
    """
    if not settings.tts_url:
        return None
    voices = settings.tts_voices
    voice = voices.get(language) or voices.get(settings.tts_default_language)
    if not voice:
        logger.warning("TTS: no voice mapped for {lang!r}", lang=language)
        return None
    body: dict[str, str] = {
        "model": "tts-1",
        "input": text,
        "voice": voice,
        "response_format": "mp3",
    }
    instruct = settings.tts_instruct.get(language, "")
    if instruct:
        body["instruct"] = instruct
    url = f"{settings.tts_url.rstrip('/')}/v1/audio/speech"
    try:
        with httpx.Client(timeout=settings.tts_timeout) as client:
            response = client.post(url, json=body)
            response.raise_for_status()
            audio = response.content
    except httpx.HTTPError as exc:
        logger.warning("TTS synthesis failed for {lang!r}: {exc}", lang=language, exc=exc)
        return None
    if not audio:
        logger.warning("TTS synthesis returned empty body for {lang!r}", lang=language)
        return None
    return audio
