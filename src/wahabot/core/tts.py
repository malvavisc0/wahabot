"""Voice synthesis via the OpenAI-compatible TTS service.

Outbound counterpart to :mod:`wahabot.core.transcribe` — perception is
preprocessing (WhisperX on the way in), expression is a tool source
(:func:`wahabot.ai.tools.whatsapp.send_media` kind=voice with ``text``). One
sync function, called from the tool fn's worker thread like every
other network call, fail-soft like presence: a synthesis failure must
degrade to a text reply, never crash the run.

The service caps one request's audio around a minute and a half; a
longer script would be cut mid-sentence. Long text is therefore spoken
in sentence-sized chunks and the MP3 frames are concatenated, so the
voice note finishes the thought.
"""

from __future__ import annotations

import re

import httpx
from loguru import logger

from wahabot.settings import Settings

__all__ = ["split_for_speech", "synthesize"]

#: Conservative per-request script size. The service's own audio cap
#: lands around 100 s; ~900 characters stays well below it at its
#: observed speaking rate.
SPEECH_CHUNK_CHARS = 900

#: Sentence ends, keeping the punctuation in the preceding chunk.
SENTENCE_END_RE = re.compile(r"(?<=[.!?…])\s+")


def split_for_speech(text: str, limit: int = SPEECH_CHUNK_CHARS) -> list[str]:
    """Split *text* into sentence-sized chunks that fit one TTS request."""
    chunks: list[str] = []
    current = ""
    for sentence in SENTENCE_END_RE.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long_sentence(sentence, limit))
            continue
        if current and len(current) + 1 + len(sentence) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks or ([text.strip()] if text.strip() else [])


def split_long_sentence(sentence: str, limit: int) -> list[str]:
    """Break one over-long sentence at whitespace, never dropping text."""
    words = sentence.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > limit:
            chunks.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        chunks.append(current)
    return chunks


def synthesize(settings: Settings, text: str, language: str) -> bytes | None:
    """Speak *text* as mp3 bytes via the TTS service; None on any failure.

    ``POST {tts_url}/v1/audio/speech`` with the OpenAI speech shape:
    ``voice`` comes from ``settings.tts_voices`` for *language*
    (``tts_default_language`` when unmapped), and the language's frozen
    ``tts_instruct`` entry rides the request when non-empty — never a
    model-controlled knob. Any failure (timeout, non-200, empty body)
    logs a one-line warning and returns None so the caller can fall
    back to sending plain text.

    Text longer than one request is split at sentence boundaries and the
    MP3 bodies are concatenated (later ID3 headers stripped), because the
    service cuts a single request at roughly 100 s.
    """
    chunks = split_for_speech(text)
    if not chunks:
        return None
    audio: list[bytes] = []
    for index, chunk in enumerate(chunks):
        part = synthesize_chunk(settings, chunk, language)
        if part is None:
            return None
        audio.append(part if index == 0 else strip_id3(part))
    return b"".join(audio)


def synthesize_chunk(settings: Settings, text: str, language: str) -> bytes | None:
    """One TTS request; the configured voice/instruct for *language*."""
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


def strip_id3(audio: bytes) -> bytes:
    """Drop a leading ID3v2 tag so concatenated MP3 frames keep playing."""
    if not audio.startswith(b"ID3") or len(audio) < 10:
        return audio
    size = (
        (audio[6] & 0x7F) << 21
        | (audio[7] & 0x7F) << 14
        | (audio[8] & 0x7F) << 7
        | (audio[9] & 0x7F)
    )
    end = 10 + size
    return audio[end:] if end < len(audio) else b""
