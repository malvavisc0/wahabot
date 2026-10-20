"""TTS chunking: the service cuts one request around 100 s."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from wahabot.core.tts import (
    SPEECH_CHUNK_CHARS,
    split_for_speech,
    strip_id3,
    synthesize,
)
from wahabot.settings import Settings


def test_split_for_speech_keeps_sentences_and_the_limit() -> None:
    text = " ".join(f"Sentence {i} is short and complete." for i in range(80))
    chunks = split_for_speech(text, limit=200)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert " ".join(chunks).split() == text.split()


def test_split_for_speech_splits_an_overlong_sentence() -> None:
    text = " ".join("word" for _ in range(100))
    chunks = split_for_speech(text, limit=40)
    assert len(chunks) > 1
    assert all(len(chunk) <= 40 for chunk in chunks)
    assert " ".join(chunks).split() == text.split()


def test_split_for_speech_empty() -> None:
    assert split_for_speech("   ") == []


def test_strip_id3() -> None:
    tag = b"ID3" + b"\x03\x00\x00" + b"\x00\x00\x00\x05" + b"abcde"
    assert strip_id3(tag + b"AUDIO") == b"AUDIO"
    assert strip_id3(b"AUDIO") == b"AUDIO"


def _settings(**updates: Any) -> Settings:
    return Settings(
        webhook_hmac_key="k",
        waha_url="http://waha.invalid",
        waha_api_key="k",
        llm_api_base="http://llm.invalid",
        llm_api_key="k",
        triage_gate=False,
        langfuse_public_key="",
        langfuse_secret_key="",
        tts_url="http://tts.invalid",
        tts_voices={"es": "voice-es"},
        _env_file=None,
        **updates,
    )


class _Response:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


class _Client:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.posts: list[dict[str, Any]] = []

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def post(self, url: str, json: dict[str, Any]) -> _Response:
        self.posts.append(json)
        return _Response(b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"FRAME")


def test_long_text_is_synthesized_in_chunks_and_joined() -> None:
    text = " ".join(f"Sentence number {i} carries a thought." for i in range(80))
    clients: list[_Client] = []

    def factory(*args: Any, **kwargs: Any) -> _Client:
        client = _Client()
        clients.append(client)
        return client

    with patch("wahabot.core.tts.httpx.Client", side_effect=factory):
        audio = synthesize(_settings(), text, "es")

    assert audio is not None
    assert len(clients) > 1
    assert all(len(client.posts) == 1 for client in clients)
    assert all(len(client.posts[0]["input"]) <= SPEECH_CHUNK_CHARS for client in clients)
    assert audio.endswith(b"FRAME")
    assert audio.count(b"ID3") == 1
