import asyncio
import datetime
import json
import shutil
import tempfile
import unittest.mock
from pathlib import Path
from typing import Any, ClassVar

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from tests.harness import (
    CHAT_ID,
    SESSION,
    smoke_video_bytes,
)
from tests.unit.support import (
    _fake_ydl,
)
from wahabot.ai.messages import (
    message_kind,
)
from wahabot.ai.tools.url_videos import video_urls
from wahabot.ai.video import extract_frames, join_anchor, probe_duration, video_marker
from wahabot.core.models import WahaEvent
from wahabot.core.transcribe import is_transcribable_mimetype
from wahabot.settings import Settings


def test_degrade_message_squeezes_old_tool_call_kwargs() -> None:
    """Old assistant tool calls keep every argument key, values trimmed.

    The meme-script incident: a
    3k-char ``run_shell_command`` body rode every later prompt. Keeping
    only ``reason`` fixed the bulk but taught the model to send
    ``{"reason": …}`` alone; now each long value is cut to a preview
    and every key survives. Already-squeezed and small calls pass
    through unchanged (idempotence + no pointless swaps).
    """
    from llama_index.core.base.llms.types import TextBlock, ToolCallBlock

    from wahabot.ai.history import CALL_ARG_PREVIEW, degrade_message

    big = ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[
            TextBlock(text=""),
            ToolCallBlock(
                tool_call_id="c-meme",
                tool_name="run_shell_command",
                tool_kwargs={
                    "command": "mkdir -p /tmp/meme && cat > gen.py <<'EOF'\n"
                    + "from PIL import Image\n" * 120,
                    "reason": "Dibujar el meme pedido",
                },
            ),
        ],
    )
    degraded = degrade_message(big)
    (block,) = [b for b in degraded.blocks if isinstance(b, ToolCallBlock)]
    assert block.tool_name == "run_shell_command"
    assert block.tool_call_id == "c-meme"
    assert set(block.tool_kwargs) == {"command", "reason"}
    assert block.tool_kwargs["reason"] == "Dibujar el meme pedido"
    assert len(block.tool_kwargs["command"]) == CALL_ARG_PREVIEW + 1
    assert "from PIL import Image\n" * 10 not in str(block.tool_kwargs)

    no_reason = ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[
            ToolCallBlock(
                tool_call_id="c-2",
                tool_name="web_search",
                tool_kwargs={"query": "q" * 2000},
            )
        ],
    )
    degraded_call = degrade_message(no_reason)
    (call,) = [b for b in degraded_call.blocks if isinstance(b, ToolCallBlock)]
    assert call.tool_name == "web_search"
    assert call.tool_kwargs == {"query": "q" * CALL_ARG_PREVIEW + "…"}

    small = ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[
            ToolCallBlock(
                tool_call_id="c-3",
                tool_name="read_chat",
                tool_kwargs={"mode": "list", "reason": "leer hilo"},
            )
        ],
    )
    assert degrade_message(small).blocks == small.blocks

    # Idempotence: a second pass changes nothing.
    assert degrade_message(degraded).blocks == degraded.blocks


def test_degrade_message_keeps_delivered_text() -> None:
    """Degradation never eats the words the chat actually saw."""
    from llama_index.core.base.llms.types import TextBlock

    from wahabot.ai.history import degrade_message

    spoken = ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[TextBlock(text="el puente está cerrado, avisé dos veces")],
    )
    assert degrade_message(spoken).content == spoken.content


def test_burst_refuses_unresolvable_sender() -> None:
    """A group message with no participant/author never buffers.

    An empty sender key would pool unrelated participants into one
    shared buffer — a silent isolation break. The caller runs the
    event through the normal single-message path instead.
    """
    from wahabot.ai import bursts

    async def scenario() -> None:
        async def complete(_buffer: bursts.BurstBuffer) -> None:
            return None

        buffers = bursts.Bursts(0.2, 5.0, complete)
        event = WahaEvent(
            id="e-no-sender",
            timestamp=1,
            event="message",
            session=SESSION,
            me={},
            payload={"id": "m1", "from": CHAT_ID, "body": "leak"},
        )
        assert buffers.add("m1", event) is False
        assert not buffers.pending(CHAT_ID)
        await asyncio.sleep(0.3)
        assert not buffers.pending(CHAT_ID)
        buffers.reset()

    asyncio.run(scenario())


def test_escalation_cli_rows(tmp_path: Path) -> None:
    """escalation_entries filters by kind and date, newest first."""
    from wahabot.core.audit import save_action
    from wahabot.core.cli import escalation_entries

    # The writer stamps day files in UTC, so the since boundary is a
    # UTC date too.
    today = datetime.datetime.now(tz=datetime.UTC).date()
    save_action(tmp_path, SESSION, "escalation", chat_id="chat-a", report="first")
    save_action(tmp_path, SESSION, "reply", chat_id="chat-b", reply="not an escalation")
    save_action(tmp_path, SESSION, "escalation", chat_id="chat-c", report="second")
    entries = escalation_entries(tmp_path / "audit" / SESSION, today)
    assert [entry["chat_id"] for entry in entries] == ["chat-c", "chat-a"]


def test_escalation_cli_rows_across_days(tmp_path: Path) -> None:
    """Newest first across day files, not only within one.

    Files iterate newest-first, appends oldest-first — a single
    end-flip interleaves days wrongly (yesterday's last entry would
    outrank today's). Each file's entries must be reversed before
    joining the newest-first file order.
    """
    from wahabot.core.cli import escalation_entries

    directory = tmp_path / "audit" / SESSION
    directory.mkdir(parents=True)

    def day_file(name: str, rows: list[tuple[str, str]]) -> None:
        lines = [
            json.dumps({"kind": "escalation", "at": f"{name}T{hour}:00", "chat_id": chat})
            for hour, chat in rows
        ]
        (directory / f"{name}.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    yesterday = (
        datetime.datetime.now(tz=datetime.UTC).date() - datetime.timedelta(days=1)
    ).isoformat()
    today = datetime.datetime.now(tz=datetime.UTC).date().isoformat()
    day_file(yesterday, [("08", "old-early"), ("10", "old-late")])
    day_file(today, [("09", "new-early"), ("11", "new-late")])
    # A corrupt line (half-written tail) and a stray non-date file
    # must not hide the readable rows or crash the listing.
    with (directory / f"{yesterday}.jsonl").open("a", encoding="utf-8") as tail:
        tail.write('{"kind": "escalation", "at": "trunc')
    (directory / "stray-notes.jsonl").write_text(
        json.dumps({"kind": "escalation", "chat_id": "stray"}) + "\n",
        encoding="utf-8",
    )
    since = datetime.datetime.now(tz=datetime.UTC).date() - datetime.timedelta(days=7)
    entries = escalation_entries(directory, since)
    assert [entry["chat_id"] for entry in entries] == [
        "new-late",
        "new-early",
        "old-late",
        "old-early",
    ]


def test_message_kind_audio_and_mimetype_guard() -> None:
    ptt = WahaEvent(
        id="e5",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={
            "id": f"false_{CHAT_ID}_PTT",
            "from": CHAT_ID,
            "body": "",
            "_data": {"type": "ptt"},
        },
    )
    assert message_kind(ptt) == "audio"
    audio = WahaEvent(
        id="e6",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={
            "id": f"false_{CHAT_ID}_AUDIO",
            "from": CHAT_ID,
            "body": "",
            "_data": {"type": "audio"},
        },
    )
    assert message_kind(audio) == "audio"
    assert is_transcribable_mimetype("audio/ogg; codecs=opus")
    assert is_transcribable_mimetype("audio/mpeg")
    assert not is_transcribable_mimetype("video/mp4")
    assert not is_transcribable_mimetype("application/json")


def test_video_markers() -> None:
    assert video_marker("a dog runs", "hello there") == (
        '(video shows: a dog runs) [audio: "hello there"]'
    )
    assert video_marker("a dog runs", "") == "(video shows: a dog runs)"
    assert video_marker("", "hello there") == '[audio: "hello there"]'
    assert video_marker("", "") == "(video)"
    assert len(video_marker("", "x" * 2500)) < 1020
    assert join_anchor("", "(video)") == "(video)"
    assert join_anchor("look at this", "(video)") == "look at this (video)"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_video_frame_extraction() -> None:
    video_bytes = smoke_video_bytes()
    with tempfile.NamedTemporaryFile(suffix=".mp4") as probe_file:
        probe_file.write(video_bytes)
        probe_file.flush()
        duration = probe_duration(probe_file.name)
    assert 1.5 < duration < 2.5
    frames = extract_frames(video_bytes, 4)
    assert len(frames) == 4
    assert all(frame[:2] == b"\xff\xd8" for frame in frames)
    many = extract_frames(video_bytes, 99)
    assert len(many) >= 1
    assert extract_frames(b"not a video at all", 4) == []


def test_video_urls() -> None:
    assert video_urls("watch this https://insta.example/reel/abc.", 2) == [
        "https://insta.example/reel/abc"
    ]
    assert video_urls('see "https://a.example/v" and https://b.example/v)!', 2) == [
        "https://a.example/v",
        "https://b.example/v",
    ]
    # Duplicates collapse; the limit caps how many are tried.
    assert video_urls("https://a.example/v https://a.example/v", 2) == [
        "https://a.example/v"
    ]
    assert video_urls("https://a.example/v https://b.example/v", 1) == [
        "https://a.example/v"
    ]
    assert video_urls("no links here", 2) == []


def test_video_urls_skip_youtube() -> None:
    # YouTube stays out of the download pipeline (captions + metadata via
    # visit_url beat six sampled frames on long-form), so sniffing skips it.
    assert video_urls("watch https://www.youtube.com/watch?v=dQw4w9WgXcQ", 2) == []
    assert video_urls("https://youtu.be/dQw4w9WgXcQ nice", 2) == []
    assert video_urls(
        "https://youtu.be/dQw4w9WgXcQ then https://insta.example/reel/abc", 2
    ) == ["https://insta.example/reel/abc"]


def test_visit_url_media_returns_video_meta(unit_settings: Settings) -> None:
    """A media-host URL resolves to the yt-dlp metadata envelope."""
    from wahabot.ai.tools.visit_url import visit_url

    info = {
        "title": "dog steals taco",
        "description": "a heist in three acts",
        "uploader": "camiloromero",
        "duration": 32,
        "view_count": 1_200_000,
        "id": "DdXprrXGx5e",
    }
    with unittest.mock.patch(
        "wahabot.ai.tools.visit_url.page.yt_dlp.YoutubeDL", _fake_ydl(info)
    ):
        result = json.loads(visit_url(unit_settings, "https://www.instagram.com/p/abc/"))
    assert result == {
        "ok": True,
        "source": "yt-dlp",
        "kind": "video",
        "url": "https://www.instagram.com/p/abc/",
        "title": "dog steals taco",
        "description": "a heist in three acts",
        "uploader": "camiloromero",
        "duration_s": 32,
        "view_count": 1_200_000,
        "id": "DdXprrXGx5e",
        "content_kind": "caption",
        "metadata_only": True,
        "note": (
            "Caption/metadata only; the video itself was not watched. "
            + "Do not claim to have watched or verified the video."
        ),
    }


def test_visit_url_youtube_inlines_transcript(unit_settings: Settings) -> None:
    """A YouTube URL with captions carries the spoken content inline.

    The dropped get_youtube_transcript affordance lives in visit_url
    now (7b merge): a manual caption track rides the envelope as
    `transcript`, and a track that exceeds the char cap is flagged
    `transcript_truncated` — but no content is lost, the full text
    spills to `transcript_file` for the agent to read in parts.
    """
    from wahabot.ai.tools.visit_url import (
        MAX_TRANSCRIPT_CHARS,
        visit_url,
    )

    long_cue = " ".join(f"Sentence {i} keeps going." for i in range(800))
    info = {
        "title": "a very long talk",
        "description": "",
        "uploader": "speaker",
        "duration": 3600,
        "view_count": 9,
        "id": "long000001",
        "subtitles": {
            "en": [
                {
                    "ext": "vtt",
                    "url": "https://captions.invalid/en.vtt",
                }
            ]
        },
    }
    fetched: list[str] = []

    def fake_captions(track_url: str, settings: Settings | None = None) -> str:
        fetched.append(track_url)
        return f"WEBVTT\n\n00:00:00.000 --> 00:00:03.000\n{long_cue}"

    with (
        unittest.mock.patch(
            "wahabot.ai.tools.visit_url.page.yt_dlp.YoutubeDL", _fake_ydl(info)
        ),
        unittest.mock.patch(
            "wahabot.ai.tools.visit_url.captions.fetch_captions", fake_captions
        ),
    ):
        result = json.loads(
            visit_url(unit_settings, "https://www.youtube.com/watch?v=long000001")
        )
    assert fetched == ["https://captions.invalid/en.vtt"]
    assert result["ok"] is True
    assert result["transcript"].startswith("Sentence 0 keeps going.")
    assert result["transcript_truncated"] is True
    assert len(result["transcript"]) == MAX_TRANSCRIPT_CHARS
    assert "transcript_file" in result
    assert result["content_kind"] == "transcript"
    assert result["note"] == "Transcript only; the video itself was not watched."
    assert "metadata_only" not in result
    spill = Path(result["transcript_file"]["path"]).read_text(encoding="utf-8")
    assert len(spill) > MAX_TRANSCRIPT_CHARS
    assert "Sentence 500 keeps going." in spill


def test_visit_url_media_falls_back_to_html(unit_settings: Settings) -> None:
    """An unresolvable media URL still takes the HTML path."""
    from wahabot.ai.tools.visit_url import visit_url

    failing = _fake_ydl(None)
    fetched: list[str] = []

    class FakeResponse:
        url = "https://www.instagram.com/p/abc/"
        status_code = 200
        text = "<html>Log In Sign Up</html>"
        headers: ClassVar[dict[str, str]] = {"content-type": "text/html"}

    def fake_fetch(url: str, settings: Settings) -> FakeResponse:
        fetched.append(url)
        return FakeResponse()

    with (
        unittest.mock.patch("wahabot.ai.tools.visit_url.page.yt_dlp.YoutubeDL", failing),
        unittest.mock.patch("wahabot.ai.tools.visit_url.page.fetch_page", fake_fetch),
    ):
        result = json.loads(visit_url(unit_settings, "https://www.instagram.com/p/abc/"))
    assert fetched == ["https://www.instagram.com/p/abc/"]
    assert result["ok"] is True
    assert result["status"] == 200
    assert "Log In Sign Up" in result["text"]


def test_visit_url_media_skips_playlist_info(unit_settings: Settings) -> None:
    """A multi-item post describes itself: caption, uploader, item count.

    The trace URL (d60f06bb…) is an Instagram post with three videos;
    format processing dies on it ("No video formats found") but the raw
    extractor dict still carries the post's caption — so a playlist
    shape must resolve, not refuse.
    """
    from wahabot.ai.tools.visit_url import visit_url

    info = {
        "title": "Post by camiloromero",
        "description": "a political caption",
        "uploader": "camiloromero",
        "entries": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        "id": "DdXprrXGx5e",
    }
    with unittest.mock.patch(
        "wahabot.ai.tools.visit_url.page.yt_dlp.YoutubeDL", _fake_ydl(info)
    ):
        result = json.loads(visit_url(unit_settings, "https://www.instagram.com/p/abc/"))
    assert result == {
        "ok": True,
        "source": "yt-dlp",
        "kind": "post",
        "url": "https://www.instagram.com/p/abc/",
        "title": "Post by camiloromero",
        "description": "a political caption",
        "uploader": "camiloromero",
        "duration_s": None,
        "view_count": None,
        "id": "DdXprrXGx5e",
        "item_count": 3,
        "content_kind": "caption",
        "metadata_only": True,
        "note": (
            "Caption/metadata only; the video itself was not watched. "
            + "Do not claim to have watched or verified the video."
        ),
    }


def test_visit_url_plain_page_skips_yt_dlp(unit_settings: Settings) -> None:
    """A non-media URL never touches yt-dlp; the HTML path runs."""
    from wahabot.ai.tools.visit_url import visit_url

    def explode(opts: Any) -> Any:
        raise AssertionError("yt-dlp must not be constructed for a plain page")

    class FakeResponse:
        url = "https://example.com/a"
        status_code = 200
        text = "just words"
        headers: ClassVar[dict[str, str]] = {"content-type": "text/html"}

    def fake_fetch(url: str, settings: Settings) -> Any:
        return FakeResponse()

    with (
        unittest.mock.patch("wahabot.ai.tools.visit_url.page.yt_dlp.YoutubeDL", explode),
        unittest.mock.patch("wahabot.ai.tools.visit_url.page.fetch_page", fake_fetch),
    ):
        result = json.loads(visit_url(unit_settings, "https://example.com/a"))
    assert result["ok"] is True
    assert result["text"] == "just words"
