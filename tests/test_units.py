"""Unit-level checks ported from ``scripts/smoke_test.py``'s ``check_units``.

Pure-function assertions (JID shapes, mimetypes, fences, video markers,
echo cache) plus the WAHA wire-shape block, split into plain test
functions. No servers needed except the wire block, which uses a mock
transport, not a port.
"""

import base64
import json
import shutil
import tempfile
import time
import unittest.mock
from pathlib import Path
from typing import Any, cast, override

import httpx
import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer

from tests.harness import (
    CHAT_ID,
    FOREIGN_JID,
    SESSION,
    smoke_video_bytes,
)
from wahabot.ai.context import (
    is_silence_narration,
    is_single_emoji,
    participant_names,
    quoted_participant,
    render_system_prompt,
    roster_cache,
    sender_tag,
)
from wahabot.ai.history import chat_visible_text
from wahabot.ai.messages import (
    bot_jids,
    bot_mentioned,
    is_group_addressed,
    is_replyable,
    jid_string,
    message_kind,
    replies_to_bot,
    self_command_instruction,
    video_media,
)
from wahabot.ai.observability import _mask_value  # pyright: ignore[reportPrivateUsage]
from wahabot.ai.tools.url_videos import video_urls
from wahabot.ai.tools.whatsapp import (
    _DOC_MIME_BY_EXT as DOC_MIME_BY_EXT,  # pyright: ignore[reportPrivateUsage]
)
from wahabot.ai.tools.whatsapp import (
    _FENCE_ERROR as FENCE_ERROR_TEXT,  # pyright: ignore[reportPrivateUsage]
)
from wahabot.ai.tools.whatsapp import (
    _VIDEO_MIME_BY_EXT as VIDEO_MIME_BY_EXT,  # pyright: ignore[reportPrivateUsage]
)
from wahabot.ai.tools.whatsapp import (
    OPERATOR_ARMED,
    OPERATOR_KEY,
    chat_jid,
    deliver_chat_text,
    fenced_chat,
    fenced_message_id,
    infer_image_mimetype,
    infer_mimetype,
    local_file,
    mention_tokens,
    ordered_merge,
    participant_jid,
    probe_media_url,
    remote_file,
    resolve_mentions,
    roster_entries,
    search_matches,
    sender_names,
    summarize_chat,
    video_file,
)
from wahabot.ai.video import extract_frames, join_anchor, probe_duration, video_marker
from wahabot.cli import build_forget_event
from wahabot.commands import build_command_event
from wahabot.core.cache import TtlCache
from wahabot.core.echoes import is_self_echo, remember_self_echo
from wahabot.core.filters import chat_allowed
from wahabot.core.host import host_context
from wahabot.core.models import WahaEvent
from wahabot.core.persistence import (
    forget_memory,
    load_memory,
    memory_file,
    persistable,
    save_memory,
)
from wahabot.core.transcribe import fetch_transcript, is_transcribable_mimetype
from wahabot.core.waha import WahaClient
from wahabot.reactions import is_own_message_id
from wahabot.settings import Settings
from wahabot.status import session_healthy, set_session_health


@pytest.fixture()
def unit_settings() -> Settings:
    return Settings(
        webhook_hmac_key="k",
        waha_url="http://waha.invalid",
        waha_api_key="k",
        llm_api_base="http://llm.invalid",
        llm_api_key="k",
        _env_file=None,
    )


def test_jid_string() -> None:
    jid_obj = {"_serialized": "146406912311368@lid", "user": "146406912311368"}
    assert jid_string(jid_obj) == "146406912311368@lid"
    assert jid_string({"user": "1464", "server": "lid"}) == "1464@lid"
    assert jid_string("x@c.us") == "x@c.us"
    assert jid_string(None) == ""


def test_pill_mention_wakes_bot() -> None:
    event = WahaEvent(
        id="e1",
        timestamp=1,
        event="message",
        session=SESSION,
        me={"id": "491555000000@c.us", "lid": "491555000000@lid"},
        payload={
            "from": CHAT_ID,
            "participant": "491555000001@c.us",
            "body": "hi",
            "_data": {"mentionedJidList": [{"_serialized": "491555000000@lid"}]},
        },
    )
    assert bot_mentioned(event)


@pytest.mark.parametrize("source", ["status@broadcast", "channel@newsletter"])
def test_broadcast_and_newsletter_not_replyable(source: str) -> None:
    assert not is_replyable(
        WahaEvent(
            id="broadcast",
            timestamp=1,
            event="message",
            session=SESSION,
            me={},
            payload={"from": source},
        )
    )


def _addressed_event() -> WahaEvent:
    return WahaEvent(
        id="e-unaddressed",
        timestamp=1,
        event="message",
        session=SESSION,
        me={"id": "491555000000@c.us", "lid": "491555000000@lid"},
        payload={
            "from": CHAT_ID,
            "participant": "491555000001@c.us",
            "body": "hello",
        },
    )


def test_group_addressing_participation() -> None:
    unaddressed = _addressed_event()
    assert not is_group_addressed(unaddressed, bot_name="kai", participation="mentioned")
    assert is_group_addressed(unaddressed, bot_name="kai", participation="judicious")
    assert not is_group_addressed(unaddressed, bot_name="kai", participation="never")


def test_dm_bypasses_participation() -> None:
    dm = WahaEvent(
        id="dm",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={"from": "491555000001@c.us", "body": "hello"},
    )
    assert is_group_addressed(dm, participation="never")


def _chat_event() -> WahaEvent:
    return WahaEvent(
        id="e1",
        timestamp=1,
        event="message",
        session=SESSION,
        me={"id": "491555000000@c.us", "lid": "491555000000@lid"},
        payload={
            "from": CHAT_ID,
            "participant": "491555000001@c.us",
            "body": "hi",
            "_data": {"mentionedJidList": [{"_serialized": "491555000000@lid"}]},
        },
    )


def test_chat_allowed() -> None:
    event = _chat_event()
    assert not chat_allowed(event, {"someone-else@c.us"}, set())
    assert chat_allowed(event, {CHAT_ID}, set())
    assert not chat_allowed(event, {CHAT_ID}, {"491555000001@c.us"})


def test_replies_to_bot_via_jid_object() -> None:
    quoting = WahaEvent(
        id="e2",
        timestamp=1,
        event="message",
        session=SESSION,
        me={"id": "491555000000@c.us", "lid": "491555000000@lid"},
        payload={
            "from": CHAT_ID,
            "body": "ok",
            "replyTo": {"participant": {"_serialized": "491555000000@lid"}},
        },
    )
    assert replies_to_bot(quoting)


def test_roster_and_summarize_chat() -> None:
    overview = {
        "id": CHAT_ID,
        "_chat": {
            "groupMetadata": {
                "participants": [
                    {"id": {"_serialized": "491555000001@lid"}},
                    "491555000002@lid",
                ]
            }
        },
    }
    roster = roster_entries(overview)
    assert len(roster) == 2
    assert participant_jid(roster[0]) == "491555000001@lid"
    assert participant_jid(roster[1]) == "491555000002@lid"
    summary = summarize_chat(CHAT_ID, overview, {"491555000001@lid": "Smoke Sender"})
    assert summary["participant_list"] == [
        {"id": "491555000001@lid", "name": "Smoke Sender"},
        {"id": "491555000002@lid"},
    ]
    assert summary["participants"] == 2


def test_chat_jid_resolution() -> None:
    holder = {"chat_id": CHAT_ID}
    assert chat_jid(None, holder) == CHAT_ID
    assert chat_jid(f"false_{CHAT_ID}_ABCDEF", holder) == CHAT_ID
    assert chat_jid(CHAT_ID, holder) == CHAT_ID


def test_fenced_chat() -> None:
    holder = {"chat_id": CHAT_ID}
    jid, err = fenced_chat(None, holder)
    assert jid == CHAT_ID and err is None
    jid, err = fenced_chat(CHAT_ID, holder)
    assert jid == CHAT_ID and err is None
    jid, err = fenced_chat(f"false_{CHAT_ID}_ABC", holder)
    assert jid == CHAT_ID and err is None
    jid, err = fenced_chat(FOREIGN_JID, holder)
    assert jid is None and err == FENCE_ERROR_TEXT
    jid, err = fenced_chat("1234", holder)
    assert jid is None and err is not None and "not a valid chat id" in err
    op_holder = {**holder, OPERATOR_KEY: OPERATOR_ARMED}
    jid, err = fenced_chat(FOREIGN_JID, op_holder)
    assert jid == FOREIGN_JID and err is None


def test_fenced_message_id() -> None:
    holder = {"chat_id": CHAT_ID}
    op_holder = {**holder, OPERATOR_KEY: OPERATOR_ARMED}
    mid, _ = fenced_message_id(f"false_{CHAT_ID}_XYZ", holder)
    assert mid == f"false_{CHAT_ID}_XYZ"
    mid, _ = fenced_message_id(f"false_{FOREIGN_JID}_XYZ", holder)
    assert mid is None
    mid, _ = fenced_message_id("odd-shape-no-at", holder)
    assert mid == "odd-shape-no-at"
    mid, _ = fenced_message_id(f"false_{FOREIGN_JID}_XYZ", op_holder)
    assert mid == f"false_{FOREIGN_JID}_XYZ"


def test_album_reassembly() -> None:
    from wahabot.ai.albums import add_album_image, is_album_container, reset, start_album

    reset()
    container = WahaEvent(
        id="e3",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={
            "id": f"false_{CHAT_ID}_ALBUM",
            "from": CHAT_ID,
            "body": "",
            "_data": {"type": "album", "expectedImageCount": 2},
        },
    )
    assert is_album_container(container)
    start_album(container)
    image_event = WahaEvent(
        id="e4",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={
            "id": f"false_{CHAT_ID}_IMG1",
            "from": CHAT_ID,
            "body": "",
            "_data": {"type": "image"},
        },
    )
    assert add_album_image(image_event)
    assert add_album_image(image_event)
    assert not add_album_image(image_event)


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
    # YouTube stays with the get_youtube_transcript tool (full captions
    # beat six sampled frames on long-form), so sniffing skips it.
    assert video_urls("watch https://www.youtube.com/watch?v=dQw4w9WgXcQ", 2) == []
    assert video_urls("https://youtu.be/dQw4w9WgXcQ nice", 2) == []
    assert video_urls(
        "https://youtu.be/dQw4w9WgXcQ then https://insta.example/reel/abc", 2
    ) == ["https://insta.example/reel/abc"]


def test_video_media_kind_guard() -> None:
    video_evt = WahaEvent(
        id="e7",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={
            "id": f"false_{CHAT_ID}_VID",
            "from": CHAT_ID,
            "body": "",
            "hasMedia": True,
            "media": {"url": "http://waha.invalid/v.mp4", "mimetype": "video/mp4"},
            "_data": {"type": "video"},
        },
    )
    assert video_media(video_evt) is not None
    ptv = WahaEvent(
        id="e8",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={
            "id": f"false_{CHAT_ID}_PTV",
            "from": CHAT_ID,
            "body": "",
            "media": {"url": "http://waha.invalid/p.mp4", "mimetype": "video/mp4"},
            "_data": {"type": "ptv"},
        },
    )
    assert video_media(ptv) is not None
    text_evt = WahaEvent(
        id="e1",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={"from": CHAT_ID, "body": "hi"},
    )
    assert video_media(text_evt) is None
    no_url = WahaEvent(
        id="e9",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={"id": "x", "from": CHAT_ID, "body": "", "_data": {"type": "video"}},
    )
    assert video_media(no_url) is None


def test_fetch_transcript_joins_segments(unit_settings: Settings) -> None:
    def _fake_post(_self: httpx.Client, *_args: object, **_kwargs: object) -> Any:
        resp = unittest.mock.Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "segments": [
                {"text": " hello "},
                {"text": " world"},
                {"text": ""},
            ]
        }
        return resp

    with unittest.mock.patch("httpx.Client.post", _fake_post):
        transcript = fetch_transcript(unit_settings, b"audio", "n.oga")
    assert transcript == "hello world"


def test_search_matches_ranking() -> None:
    roster = [
        {"id": "1@g.us", "name": "Familia"},
        {"id": "2@c.us", "name": "familia domingo"},
        {"id": "3@c.us", "name": "Ana"},
        {"id": "4@g.us", "name": "Work"},
        {
            "id": {
                "server": "g.us",
                "user": "1809-1373",
                "_serialized": "1809-1373@g.us",
            },
            "name": "Familia",
        },
        {"id": "", "name": "no id, dropped"},
    ]
    matches = search_matches(roster, "FAMILIA")
    assert [m["id"] for m in matches] == ["1@g.us", "1809-1373@g.us", "2@c.us"]
    assert all(set(m) == {"id", "name"} for m in matches)


def test_command_event_shape() -> None:
    command = build_command_event(SESSION, "send the plan to Familia")
    command_payload = cast(dict[str, Any], command["payload"])
    assert command["event"] == "command"
    assert command_payload["body"] == "[operator command] send the plan to Familia"
    assert cast(dict[str, Any], command_payload["_data"])["notifyName"] == "operator"


def test_own_message_id_detection() -> None:
    assert is_own_message_id(f"true_{CHAT_ID}_X")
    assert not is_own_message_id(f"false_{CHAT_ID}_X")


def test_send_file_payloads() -> None:
    remote = remote_file("http://files.invalid/q3/report.pdf")
    assert (
        remote["mimetype"] == "application/pdf"
        and remote["url"] == "http://files.invalid/q3/report.pdf"
        and remote["filename"] == "report.pdf"
    )
    assert (
        infer_mimetype(
            "http://x.invalid/noext", DOC_MIME_BY_EXT, "application/octet-stream"
        )
        == "application/octet-stream"
    )
    assert infer_image_mimetype("http://x.invalid/pic.jpg") == "image/jpeg"
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf = Path(tmpdir) / "out.pdf"
        pdf.write_bytes(b"%PDF-1.4 smoke bytes")
        local = local_file(str(pdf), max_file_bytes=1024)
        assert not isinstance(local, str)
        assert local["mimetype"] == "application/pdf"
        assert local["filename"] == "out.pdf"
        assert base64.b64decode(local["data"]) == b"%PDF-1.4 smoke bytes"
        assert isinstance(local_file(str(pdf), max_file_bytes=4), str)
        assert isinstance(local_file(str(Path(tmpdir) / "missing.pdf"), 1024), str)


def test_send_video_payloads() -> None:
    """The video payloads for both send_video sources.

    A URL must carry the video MIME (not a guessed type) and the path's
    basename; a local file must be re-typed from the document default
    to ``video/mp4`` — a shell-tool ``.mp4`` must not ride the wire
    stamped ``application/octet-stream``. Oversize/missing paths come
    back as the shared error-string shape, same as ``send_file``.
    """
    remote = video_file("http://files.invalid/q4/clip.webm", max_file_bytes=1024)
    assert not isinstance(remote, str)
    assert remote == {
        "mimetype": "video/webm",
        "url": "http://files.invalid/q4/clip.webm",
        "filename": "clip.webm",
    }
    assert (
        infer_mimetype("http://x.invalid/noext", VIDEO_MIME_BY_EXT, "video/mp4")
        == "video/mp4"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        mp4 = Path(tmpdir) / "out.mp4"
        mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42 smoke bytes")
        local = video_file(str(mp4), max_file_bytes=1024)
        assert not isinstance(local, str)
        assert local["mimetype"] == "video/mp4"
        assert local["filename"] == "out.mp4"
        assert base64.b64decode(local["data"]).startswith(b"\x00\x00\x00\x18ftyp")
        assert isinstance(video_file(str(mp4), max_file_bytes=4), str)
        assert isinstance(video_file(str(Path(tmpdir) / "missing.mp4"), 1024), str)


def test_probe_media_url_refuses_malformed() -> None:
    assert "not an http(s) URL" in (probe_media_url("not a url") or "")
    assert "not an http(s) URL" in (probe_media_url("ftp://x/y.png") or "")


def test_probe_media_url_refuses_missing() -> None:
    """A definitive 404 refuses the send — the URL was likely invented."""
    gone = unittest.mock.Mock(status_code=404)
    with (
        unittest.mock.patch.object(httpx, "head", return_value=gone),
        unittest.mock.patch.object(httpx, "get", return_value=gone),
    ):
        error = probe_media_url("https://x.invalid/pic.png")
    assert error is not None and "does not exist" in error


def test_probe_media_url_soft_fails_offline() -> None:
    """Connection errors leave the send to WAHA (soft fail), never refuse."""
    with unittest.mock.patch.object(
        httpx, "head", side_effect=httpx.ConnectError("dns gone")
    ):
        assert probe_media_url("http://files.invalid/q3/report.pdf") is None


def test_probe_media_url_allows_live() -> None:
    probe_ok = unittest.mock.Mock(status_code=200)
    with unittest.mock.patch.object(httpx, "head", return_value=probe_ok):
        assert probe_media_url("https://cdn.example.org/pic.png") is None


def test_mask_value_redacts_all_jid_forms() -> None:
    """c.us, g.us and lid addresses must all be masked — lid is PII too."""
    assert _mask_value("491555000000@c.us") == "[jid redacted]"
    assert _mask_value("491555000000@lid") == "[jid redacted]"
    assert _mask_value("120363000000000000@g.us") == "[jid redacted]"
    assert _mask_value("4915151503271-1630682381@g.us") == "[jid redacted]"
    assert _mask_value("status@broadcast") == "[jid redacted]"
    # Bare mention tokens (the @<lid-number> shape group chats show).
    assert _mask_value("Para @111222333444555") == "Para [jid redacted]"
    masked = _mask_value("chat with 491555000000@lid about 491555000001@c.us")
    assert "491555000000" not in masked and "491555000001" not in masked
    assert _mask_value("no identifiers here") == "no identifiers here"
    assert _mask_value("invoice 123456 stays; @1234567890 goes") == (
        "invoice 123456 stays; [jid redacted] goes"
    )
    assert _mask_value(["491555000002@lid", {"id": "491555000003@lid"}]) == [
        "[jid redacted]",
        {"id": "[jid redacted]"},
    ]


def test_command_holder_bare_send_target() -> None:
    command_payload = cast(dict[str, Any], build_command_event(SESSION, "x")["payload"])
    holder = {"chat_id": str(command_payload["from"])}
    assert chat_jid(None, holder) == "operator"
    command_payload["from"] = "491555000001@c.us"
    assert (
        chat_jid(None, {"chat_id": str(command_payload["from"])}) == "491555000001@c.us"
    )


def test_waha_wire_shapes() -> None:
    """The WAHA client's wire contract via an in-process mock transport."""
    waha_requests: list[httpx.Request] = []

    def waha_wire(request: httpx.Request) -> httpx.Response:
        waha_requests.append(request)
        return httpx.Response(200, json=[])

    wire_waha = WahaClient(
        "http://waha.invalid",
        "wire-key",
        transport=httpx.MockTransport(waha_wire),
    )
    wire_waha.send_text(
        SESSION,
        CHAT_ID,
        "@Smoke Sender hello",
        reply_to=f"false_{CHAT_ID}_QUOTE",
        mentions=["491555000001@c.us"],
    )
    wire_waha.fetch_chat_messages(SESSION, "123 456@g.us", limit=7)
    wire_waha.send_image(SESSION, CHAT_ID, {"url": "https://x.invalid/a.png"}, "image")
    wire_waha.send_file(SESSION, CHAT_ID, {"url": "https://x.invalid/a.pdf"}, "file")
    wire_waha.send_video(
        SESSION, CHAT_ID, {"url": "https://x.invalid/a.mp4"}, "video", convert=False
    )
    wire_waha.forward_message(SESSION, CHAT_ID, "false_message")
    wire_waha.send_reaction(SESSION, "false_message", "")

    (
        send_request,
        read_request,
        image_request,
        file_request,
        video_request,
        forward_request,
        reaction_request,
    ) = waha_requests
    assert (
        send_request.url.path == "/api/sendText"
        and send_request.headers["X-Api-Key"] == "wire-key"
        and json.loads(send_request.content)
        == {
            "session": SESSION,
            "chatId": CHAT_ID,
            "text": "@Smoke Sender hello",
            "reply_to": f"false_{CHAT_ID}_QUOTE",
            "mentions": ["491555000001@c.us"],
        }
    )
    assert read_request.url.raw_path.startswith(
        b"/api/default/chats/123%20456%40g.us/messages?"
    ) and dict(read_request.url.params) == {"limit": "7", "downloadMedia": "false"}
    assert (
        image_request.url.path == "/api/sendImage"
        and json.loads(image_request.content)["caption"] == "image"
        and file_request.url.path == "/api/sendFile"
        and json.loads(file_request.content)["caption"] == "file"
        and video_request.url.path == "/api/sendVideo"
        and json.loads(video_request.content)
        == {
            "session": SESSION,
            "chatId": CHAT_ID,
            "file": {"url": "https://x.invalid/a.mp4"},
            "convert": False,
            "caption": "video",
        }
        and forward_request.url.path == "/api/forwardMessage"
        and json.loads(forward_request.content)["messageId"] == "false_message"
        and reaction_request.url.path == "/api/reaction"
        and json.loads(reaction_request.content)["reaction"] == ""
    )
    wire_waha._client.close()  # pyright: ignore[reportPrivateUsage]


def test_forget_event_shape() -> None:
    forget = build_forget_event(SESSION, "123@g.us")
    forget_payload = cast(dict[str, Any], forget["payload"])
    assert forget["event"] == "forget"
    assert forget_payload["chat_id"] == "123@g.us"


def test_persistent_memory_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as memdir:
        data_dir = Path(memdir)
        chat = "1234567890-1234567890@g.us"
        memory = ChatMemoryBuffer.from_defaults(token_limit=8000)  # pyright: ignore[reportUnknownMemberType]
        memory.put(ChatMessage(role=MessageRole.USER, content="[Ana] hai"))
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content="hey"))
        save_memory(data_dir, SESSION, chat, memory)
        path = memory_file(data_dir, SESSION, chat)
        assert path.exists()
        restored = load_memory(data_dir, SESSION, chat)
        assert restored is not None
        assert isinstance(restored, ChatMemoryBuffer)
        assert [str(m.content) for m in restored.get_all()] == ["[Ana] hai", "hey"]


def test_memory_envelope_mismatch_ignored() -> None:
    with tempfile.TemporaryDirectory() as memdir:
        data_dir = Path(memdir)

        def write_envelope(chat_id: str, session: str, version: int = 1) -> Path:
            p = memory_file(data_dir, SESSION, chat_id)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps(
                    {
                        "version": version,
                        "session": session,
                        "chat_id": chat_id,
                        "saved_at": 1,
                        "memory": {},
                    }
                )
            )
            return p

        mismatch = write_envelope("mismatch@g.us", "other-session")
        assert load_memory(data_dir, SESSION, "mismatch@g.us") is None
        assert mismatch.exists()

        oldversion = write_envelope("old@g.us", SESSION, version=99)
        assert load_memory(data_dir, SESSION, "old@g.us") is None
        assert oldversion.exists()

        corrupt = write_envelope("corrupt@g.us", SESSION)
        corrupt.write_text("{ this is not json")
        assert load_memory(data_dir, SESSION, "corrupt@g.us") is None
        assert not corrupt.exists()
        assert Path(str(corrupt) + ".bad").exists()


def test_memory_atomic_and_forget() -> None:
    with tempfile.TemporaryDirectory() as memdir:
        data_dir = Path(memdir)
        atomic_path = memory_file(data_dir, SESSION, "atomic@g.us")
        atomic_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_path.write_text("PREVIOUS")
        atomic_mem = ChatMemoryBuffer.from_defaults(token_limit=8000)  # pyright: ignore[reportUnknownMemberType]
        atomic_mem.put(ChatMessage(role=MessageRole.USER, content="new"))
        with unittest.mock.patch(
            "wahabot.core.persistence.os.replace", side_effect=OSError("disk full")
        ):
            save_memory(data_dir, SESSION, "atomic@g.us", atomic_mem)
        assert atomic_path.read_text() == "PREVIOUS"
        assert forget_memory(data_dir, SESSION, "atomic@g.us")
        assert not atomic_path.exists()
        assert not forget_memory(data_dir, SESSION, "atomic@g.us")


def test_memory_serialization_swallowed() -> None:
    with tempfile.TemporaryDirectory() as memdir:
        data_dir = Path(memdir)
        alien = ChatMemoryBuffer.from_defaults(token_limit=8000)  # pyright: ignore[reportUnknownMemberType]
        alien.put(
            ChatMessage(
                role=MessageRole.USER, content="x", additional_kwargs={"o": object()}
            )
        )
        save_memory(data_dir, SESSION, "alien@g.us", alien)


def test_persistable_guards_paths() -> None:
    with tempfile.TemporaryDirectory() as memdir:
        data_dir = Path(memdir)
        atomic_mem = ChatMemoryBuffer.from_defaults(token_limit=8000)  # pyright: ignore[reportUnknownMemberType]
        atomic_mem.put(ChatMessage(role=MessageRole.USER, content="x"))
        assert not persistable("../../etc/passwd") and not persistable("a/b/c")
        assert persistable("123@g.us") and persistable("1809-1@g.us")
        save_memory(data_dir, SESSION, "a/b/c", atomic_mem)
        assert not (data_dir / "memory" / SESSION / "a").exists()
        assert load_memory(data_dir, SESSION, "a/b/c") is None
        assert not forget_memory(data_dir, SESSION, "a/b/c")


def test_health_gate() -> None:
    set_session_health("SCAN_QR_CODE")
    assert not session_healthy()
    set_session_health("WORKING")
    assert session_healthy()


def test_self_chat_command_classification() -> None:
    ME_JID = "491555000000@c.us"
    self_cmd = WahaEvent(
        id="sc1",
        timestamp=1,
        event="message",
        session=SESSION,
        me={"id": ME_JID, "lid": "491555000000@lid"},
        payload={"from": ME_JID, "fromMe": True, "body": "kai do the thing"},
    )
    assert self_command_instruction(self_cmd, bot_name="kai") == "do the thing"
    assert self_command_instruction(self_cmd, bot_mention_regex="@?kay") is None
    assert (
        self_command_instruction(self_cmd, bot_name="kai", bot_mention_regex="@?kay")
        is None
    )
    other_chat = WahaEvent(
        id="sc2",
        timestamp=1,
        event="message",
        session=SESSION,
        me=self_cmd.me,
        payload={"from": CHAT_ID, "fromMe": True, "body": "kai do the thing"},
    )
    assert self_command_instruction(other_chat, bot_name="kai") is None
    not_mine = WahaEvent(
        id="sc3",
        timestamp=1,
        event="message",
        session=SESSION,
        me=self_cmd.me,
        payload={"from": ME_JID, "fromMe": False, "body": "kai do the thing"},
    )
    assert self_command_instruction(not_mine, bot_name="kai") is None
    no_me = WahaEvent(
        id="sc4",
        timestamp=1,
        event="message",
        session=SESSION,
        me=None,
        payload={"from": ME_JID, "fromMe": True, "body": "kai do the thing"},
    )
    assert self_command_instruction(no_me, bot_name="kai") is None
    bare = WahaEvent(
        id="sc5",
        timestamp=1,
        event="message",
        session=SESSION,
        me=self_cmd.me,
        payload={"from": ME_JID, "fromMe": True, "body": "kai"},
    )
    assert self_command_instruction(bare, bot_name="kai") is None
    lid_self = WahaEvent(
        id="sc6",
        timestamp=1,
        event="message",
        session=SESSION,
        me=self_cmd.me,
        payload={"from": "491555000000@lid", "fromMe": True, "body": "kai x"},
    )
    assert self_command_instruction(lid_self, bot_name="kai") == "x"
    mid_text = WahaEvent(
        id="sc7",
        timestamp=1,
        event="message",
        session=SESSION,
        me=self_cmd.me,
        payload={"from": ME_JID, "fromMe": True, "body": "note: kai do the thing"},
    )
    assert self_command_instruction(mid_text, bot_name="kai") is None
    lid_linked = WahaEvent(
        id="sc8",
        timestamp=1,
        event="message",
        session=SESSION,
        me=self_cmd.me,
        payload={
            "from": ME_JID,
            "to": "491555000000@lid",
            "fromMe": True,
            "body": "kai do the thing",
        },
    )
    assert self_command_instruction(lid_linked, bot_name="kai") == "do the thing"
    to_other = WahaEvent(
        id="sc9",
        timestamp=1,
        event="message",
        session=SESSION,
        me=self_cmd.me,
        payload={
            "from": "491555000001@c.us",
            "to": "491555000002@c.us",
            "fromMe": True,
            "body": "kai do the thing",
        },
    )
    assert self_command_instruction(to_other, bot_name="kai") is None


def test_bot_jids_ignores_captured_identity() -> None:
    """Events without ``me`` yield no identity — gates fail closed."""
    from wahabot.status import state as status_state

    status_state.operator_jid = "491555000000@c.us"
    status_state.operator_lid = "491555000000@lid"
    thin = WahaEvent(
        id="bj1",
        timestamp=1,
        event="message",
        session=SESSION,
        me=None,
        payload={
            "from": "491555000000@c.us",
            "to": "491555000000@lid",
            "fromMe": True,
            "body": "kai do the thing",
        },
    )
    assert bot_jids(thin) == set()
    assert self_command_instruction(thin, bot_name="kai") is None


def test_echo_tracking() -> None:
    from wahabot.core.echoes import _echoes  # pyright: ignore[reportPrivateUsage]

    _echoes.clear()
    remember_self_echo("true_x_ECHO")
    assert is_self_echo("true_x_ECHO")
    assert not is_self_echo("true_x_OTHER")
    remember_self_echo("")
    assert not is_self_echo("")
    stale = TtlCache[str, bool](300, 1000)
    stale.put("true_x_ECHO2", True)
    stale._entries["true_x_ECHO2"] = (time.monotonic() - 400, True)  # pyright: ignore[reportPrivateUsage]
    assert stale.get("true_x_ECHO2") is None


def test_sender_names_reads_notify_name() -> None:
    """``sender_names`` extracts, strips, dedups and drops nameless jids.

    A stub feeds messages with the real production shapes; the loop must
    strip whitespace, collapse duplicate senders, and ignore entries with
    no ``notifyName``.
    """

    class StubWaha:
        def fetch_chat_messages(
            self,
            _session: str,
            _chat_id: str,
            limit: int = 100,  # pyright: ignore[reportUnusedParameter]
        ) -> list[dict[str, Any]]:
            return [
                {
                    "participant": "491555000001@c.us",
                    "_data": {"notifyName": " Smoke "},
                },
                {
                    "participant": "491555000001@c.us",
                    "_data": {"notifyName": " Smoke "},
                },
                {"participant": "491555000002@c.us", "_data": {"notifyName": ""}},
                {
                    "participant": {"_serialized": "491555000003@lid"},
                    "_data": {"notifyName": "Lidler"},
                },
            ]

    names = sender_names(cast(Any, StubWaha()), SESSION, CHAT_ID)
    assert names == {  # stripped, no dup, nameless dropped, jid object resolved
        "491555000001@c.us": "Smoke",
        "491555000003@lid": "Lidler",
    }


def test_host_placeholder() -> None:
    """``{{host}}`` resolves to the machine snapshot in the prompt."""
    rendered = render_system_prompt("Host info:\n{{host}}")
    assert "{{host}}" not in rendered
    assert rendered == f"Host info:\n{host_context()}"
    assert "Host: " in rendered
    assert "- Python: " in rendered


def test_remember_strips_thinking_separator(unit_settings: Settings) -> None:
    """``remember`` stores the reply text without the thinking separator.

    Reasoning models split thinking from text with a leading blank line
    inside the text block; memory must keep the words, not the
    separator — the stored prefix costs tokens every turn and teaches
    the model to keep emitting it.
    """
    import asyncio

    from llama_index.core.base.llms.types import (
        ChatMessage,
        ChatResponse,
        MessageRole,
        TextBlock,
        ThinkingBlock,
    )
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.core.workflow import Context

    from wahabot.ai.workflow import FunctionCallingAgentWorkflow, load_llm

    async def stored_texts() -> list[str]:
        wf = FunctionCallingAgentWorkflow(llm=load_llm(unit_settings))
        ctx = Context(wf)
        await ctx.store.set("memory", ChatMemoryBuffer.from_defaults())
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ThinkingBlock(content="reasoning"),
                TextBlock(text="\n\njaj real reply"),
            ],
        )
        await FunctionCallingAgentWorkflow.remember(
            wf, ctx, ChatResponse(message=message), []
        )
        memory = await ctx.store.get("memory")
        return [
            block.text or ""
            for m in await memory.aget_all()
            for block in m.blocks
            if isinstance(block, TextBlock)
        ]

    assert asyncio.run(stored_texts()) == ["jaj real reply"]


def test_remember_drops_lone_emoji_reply(unit_settings: Settings) -> None:
    """``remember`` never stores a lone-emoji reply as the bot's text.

    The handler converts a lone emoji into a reaction (~50 %) or drops
    it as silence — the chat never sees it as a text reply, so memory
    must not record it (memory mirrors the chat). Delivery still
    receives it: the conversion happens handler-side. Multi-emoji
    strings are real chat text and must stay.
    """
    import asyncio

    from llama_index.core.base.llms.types import (
        ChatMessage,
        ChatResponse,
        MessageRole,
        TextBlock,
    )
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.core.workflow import Context

    from wahabot.ai.workflow import FunctionCallingAgentWorkflow, load_llm

    async def stored_count(reply: str) -> int:
        wf = FunctionCallingAgentWorkflow(llm=load_llm(unit_settings))
        ctx = Context(wf)
        await ctx.store.set("memory", ChatMemoryBuffer.from_defaults())
        message = ChatMessage(role=MessageRole.ASSISTANT, blocks=[TextBlock(text=reply)])
        await FunctionCallingAgentWorkflow.remember(
            wf, ctx, ChatResponse(message=message), []
        )
        memory = await ctx.store.get("memory")
        return len(await memory.aget_all())

    assert asyncio.run(stored_count("👋")) == 0
    assert asyncio.run(stored_count("🤣🤣🤣")) == 1


def test_warn_accidental_silence() -> None:
    """``warn_accidental_silence`` fires only on the accidental-empty shape.

    The trace-audit case: a reasoning model spends its tokens thinking,
    returns an empty final answer after tool rounds, nothing is
    delivered — indistinguishable at run time from chosen silence, so
    this warning is the only signal in the logs. Chosen silence (first
    round, or a delivery already made) must stay unlogged.
    """
    from collections.abc import Iterator
    from contextlib import contextmanager

    from llama_index.core.base.llms.types import ChatResponse
    from loguru import logger

    from wahabot.ai.tools.whatsapp import RunTarget, bind_target, reset_target
    from wahabot.ai.workflow import FunctionCallingAgentWorkflow

    wf = FunctionCallingAgentWorkflow.__new__(FunctionCallingAgentWorkflow)
    empty = ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""))
    text = ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content="hi"))
    quiet = RunTarget(session=SESSION, chat_id=CHAT_ID)
    delivered = RunTarget(session=SESSION, chat_id=CHAT_ID, sent=CHAT_ID)

    @contextmanager
    def bound(target: RunTarget) -> Iterator[None]:
        token = bind_target(target)
        try:
            yield
        finally:
            reset_target(token)

    logs: list[str] = []

    def record(message: str, **_: Any) -> None:
        logs.append(message)

    with (
        unittest.mock.patch.object(logger, "warning", record),
        bound(quiet),
    ):
        wf.warn_accidental_silence(empty, rounds=8)  # the audit shape: warns
        wf.warn_accidental_silence(empty, rounds=1)  # first-round silence: no
        wf.warn_accidental_silence(text, rounds=8)  # visible answer: no
    with unittest.mock.patch.object(logger, "warning", record), bound(delivered):
        wf.warn_accidental_silence(empty, rounds=8)  # post-delivery quiet: no
    assert logs == [
        "".join(
            (
                "Stopping run: empty final answer after {rounds} rounds ",
                "(nothing delivered, no stay_silent)",
            )
        )
    ]


def test_mention_tokens() -> None:
    """``mention_tokens`` extracts only user-part and JID-shaped tokens."""
    text = "Para @111222333444555 y @491555000001@c.us, no @ana ni mail@x.com"
    assert mention_tokens(text) == ["111222333444555", "491555000001@c.us"]
    assert mention_tokens("no ats here") == []


def test_resolve_mentions() -> None:
    """Tokens resolve against roster user parts; non-members stay out."""
    roster = [
        "111222333444555@lid",
        "491555000001@c.us",
        "666777888999000@lid",
    ]
    assert resolve_mentions("Para @111222333444555", roster) == ["111222333444555@lid"]
    # Full-JID token resolves to the roster's canonical form.
    assert resolve_mentions("hi @491555000001@c.us", roster) == ["491555000001@c.us"]
    # Multiple tokens, roster order, no duplicates.
    assert resolve_mentions(
        "@666777888999000 mira @111222333444555 y @111222333444555", roster
    ) == ["666777888999000@lid", "111222333444555@lid"]
    # Token order wins over roster order; duplicates collapse.
    assert resolve_mentions("@111222333444555 luego @666777888999000", roster) == [
        "111222333444555@lid",
        "666777888999000@lid",
    ]
    # A token naming nobody on the roster invents no mention.
    assert resolve_mentions("cc @999999999999", roster) == []


def test_ordered_merge() -> None:
    """Explicit mentions keep their order; resolved additions follow."""
    assert ordered_merge(
        ["491555000001@c.us"], ["111222333444555@lid", "491555000001@c.us"]
    ) == ["491555000001@c.us", "111222333444555@lid"]
    assert ordered_merge([], []) == []


def test_deliver_chat_text_resolves_mentions() -> None:
    """Text delivery resolves ``@``-tokens into real mention JIDs.

    The handler's final-reply send and the ``send_message`` tool share
    one delivery core: a ``@<number>`` token naming a roster member
    rides the send as a mention JID (highlight + push), and text
    naming nobody mentions nobody. A roster outage fails soft — the
    text still goes out, unmentioned.
    """

    class RosterWaha(WahaClient):
        """A WAHA recording sends, whose overview serves a LID roster."""

        def __init__(self) -> None:
            super().__init__(base_url="http://waha.invalid", api_key="k")
            self.sent: list[tuple[str, str, str, list[str] | None]] = []

        @override
        def get_chat_overview(self, session: str, chat_id: str) -> Any:
            return {
                "id": chat_id,
                "participants": [
                    {"id": {"_serialized": "111222333444555@lid"}},
                    {"id": {"_serialized": "491555000000@c.us"}},
                ],
            }

        @override
        def send_text(
            self,
            session: str,
            chat_id: str,
            text: str,
            reply_to: str | None = None,
            mentions: list[str] | None = None,
        ) -> str:
            self.sent.append((session, chat_id, text, mentions))
            return f"true_{chat_id}_SENT{len(self.sent)}"

    class BrokenRosterWaha(RosterWaha):
        @override
        def get_chat_overview(self, session: str, chat_id: str) -> Any:
            raise RuntimeError("roster unavailable")

    waha = RosterWaha()
    text = "listado vacío, @111222333444555, pero documentado"
    sent_id, mentions = deliver_chat_text(waha, "default", "123@g.us", text)
    assert mentions == ["111222333444555@lid"]
    assert waha.sent == [("default", "123@g.us", text, ["111222333444555@lid"])]
    assert sent_id == "true_123@g.us_SENT1"
    # No tokens naming members: nothing to mention.
    _, mentions = deliver_chat_text(waha, "default", "123@g.us", "sin menciones aquí")
    assert mentions == []
    # A roster outage fails soft: the text still goes out.
    _, mentions = deliver_chat_text(BrokenRosterWaha(), "default", "123@g.us", text)
    assert mentions == []


def test_log_action_reason() -> None:
    """``log_action_reason`` logs the justification, warns when absent.

    The reason is capped at 200 chars and rides the log kwargs; a
    missing/blank reason is the model acting without articulating why
    — worth a WARNING so the operator sees it in the audit trail.
    """
    from loguru import logger

    from wahabot.ai.tools.whatsapp import log_action_reason

    infos: list[tuple[str, dict[str, Any]]] = []
    warnings: list[tuple[str, dict[str, Any]]] = []

    def record(message: str, **kwargs: Any) -> None:
        (warnings if "carried no reason" in message else infos).append((message, kwargs))

    with (
        unittest.mock.patch.object(logger, "info", record),
        unittest.mock.patch.object(logger, "warning", record),
    ):
        log_action_reason("send_message", "directly asked by name", chat=CHAT_ID)
        log_action_reason("stay_silent", "  ")
        log_action_reason("react_to_message", "x" * 500)
    assert [tool for _, kw in infos for tool in [kw["tool"]]] == [
        "send_message",
        "react_to_message",
    ]
    assert infos[0][1]["reason"] == "directly asked by name"
    assert infos[0][1]["suffix"] == f" {{'chat': '{CHAT_ID}'}}"
    assert len(infos[1][1]["reason"]) == 200  # the 500-char reason was capped
    assert warnings[0][0] == "Tool call {tool} carried no reason{suffix}"
    assert warnings[0][1] == {"tool": "stay_silent", "suffix": ""}


def test_is_silence_narration_catches_leaked_tool_token() -> None:
    """A leaked ``stay_silent`` tool token is silence chatter, not an answer.

    The trace shows the model emitting the literal tool name ``stay_silent``
    as a plain-text reply instead of calling the tool; those go straight to
    the chat, so ``is_silence_narration`` must treat the token as silence.
    It stays anchored to the *whole* reply — a real sentence containing the
    word must still go through.
    """
    assert is_silence_narration("stay_silent")
    assert is_silence_narration(" stay_silent ")
    assert not is_silence_narration(
        "I could say stay_silent here but I'll answer anyway."
    )


def test_is_single_emoji_catches_lone_emoji() -> None:
    """A lone emoji reply is intercepted — it's a reaction, not a message.

    The model sometimes outputs a single emoji as text instead of calling
    ``react_to_message``.  ``is_single_emoji`` catches these so the handler
    can convert them to reactions or silence.
    """
    assert is_single_emoji("👋")
    assert is_single_emoji("🤣")
    assert is_single_emoji("😂")
    assert is_single_emoji("👍")
    assert is_single_emoji("❤")
    assert is_single_emoji("🔥")
    # Surrounding whitespace is tolerated.
    assert is_single_emoji(" 👋 ")
    assert is_single_emoji("  🤣\n")


def test_is_single_emoji_passes_multi_emoji() -> None:
    """Multi-emoji strings are real messages — they must NOT be intercepted."""
    assert not is_single_emoji("🤣🤣🤣")
    assert not is_single_emoji("😂👍")
    assert not is_single_emoji("🤣😂")
    assert not is_single_emoji("👋🔥❤")


def test_is_single_emoji_passes_text() -> None:
    """Plain text and text-with-emoji must pass through as normal messages."""
    assert not is_single_emoji("hola 👋")
    assert not is_single_emoji("me sirve pa' reírme 🤣")
    assert not is_single_emoji("hello")
    assert not is_single_emoji("stay_silent")
    assert not is_single_emoji("")
    assert not is_single_emoji("   ")


# ---------------------------------------------------------------------------
# Semantic identity: sender tags, resolver backfill, quoting, own identity
# ---------------------------------------------------------------------------


def _group_waha() -> Any:
    """A stub WAHA with a bare LID roster, whose recent messages carry
    ``notifyName`` for the LID participants."""

    class StubWaha:
        def get_chat_overview(self, _session: str, chat_id: str) -> dict[str, Any]:
            return {
                "id": chat_id,
                "participants": [{"id": "132469693124738@lid"}],
            }

        def fetch_chat_messages(
            self,
            _session: str,
            _chat_id: str,
            limit: int = 100,  # pyright: ignore[reportUnusedParameter]
        ) -> list[dict[str, Any]]:
            return [
                {
                    "participant": {"_serialized": "132469693124738@lid"},
                    "_data": {"notifyName": "Mikhail Polozhaev"},
                },
                {
                    "participant": {"_serialized": "74943001800935@lid"},
                    "_data": {"notifyName": "Troche"},
                },
            ]

    return StubWaha()


def test_participant_names_backfills_from_messages() -> None:
    """A bare LID roster gains names from recent messages' notifyName."""
    roster_cache.clear()
    names = participant_names(_group_waha(), SESSION, CHAT_ID)
    assert names == {
        "132469693124738@lid": "Mikhail Polozhaev",
        "74943001800935@lid": "Troche",
    }


def _conflicting_waha() -> Any:
    """A stub WAHA whose roster and messages disagree on one JID's name.

    Only the roster's entry may survive — roster names win.
    """

    class ConflictingWaha:
        def get_chat_overview(self, _session: str, chat_id: str) -> dict[str, Any]:
            return {
                "id": chat_id,
                "participants": [{"id": "132469693124738@lid", "name": "Roster Name"}],
            }

        def fetch_chat_messages(
            self,
            _session: str,
            _chat_id: str,
            limit: int = 100,  # pyright: ignore[reportUnusedParameter]
        ) -> list[dict[str, Any]]:
            return [
                {
                    "participant": {"_serialized": "132469693124738@lid"},
                    "_data": {"notifyName": "Message Name"},
                }
            ]

    return ConflictingWaha()


def test_participant_names_roster_wins_over_backfill() -> None:
    """Roster names are authoritative; the message walk only backfills."""
    roster_cache.clear()
    names = participant_names(_conflicting_waha(), SESSION, CHAT_ID)
    assert names == {"132469693124738@lid": "Roster Name"}


def _group_event(
    participant: Any = "132469693124738@lid", name: str | None = None
) -> WahaEvent:
    payload: dict[str, Any] = {
        "from": CHAT_ID,
        "participant": participant,
        "body": "hola",
    }
    if name is not None:
        payload["_data"] = {"notifyName": name}
    return WahaEvent(
        id="tag1",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload=payload,
    )


def test_sender_tag_group_renders_name_and_jid() -> None:
    event = _group_event(name="Mikhail Polozhaev")
    assert sender_tag(event) == "[Mikhail Polozhaev <132469693124738@lid>]"


def test_sender_tag_group_resolves_name_from_roster() -> None:
    event = _group_event()  # no notifyName on the event itself
    names = {"132469693124738@lid": "Mikhail Polozhaev"}
    assert sender_tag(event, names) == "[Mikhail Polozhaev <132469693124738@lid>]"


def test_sender_tag_group_unknown_name_falls_back_to_jid() -> None:
    assert sender_tag(_group_event()) == "[132469693124738@lid]"


def test_sender_tag_dm_keeps_bare_name() -> None:
    dm = WahaEvent(
        id="dm-tag",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={
            "from": "491555000001@c.us",
            "body": "hi",
            "_data": {"notifyName": "Smoke Sender"},
        },
    )
    assert sender_tag(dm) == "[Smoke Sender]"


def test_sender_tag_dm_without_name_yields_empty() -> None:
    """A DM with neither notifyName nor participant has nothing to show.

    Today's documented fallback: an empty tag (should not happen —
    real DM events always carry one of the two).
    """
    dm = WahaEvent(
        id="dm-tag2",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={"from": "491555000001@c.us", "body": "hi"},
    )
    assert sender_tag(dm) == ""


def test_sender_tag_normalizes_jid_object() -> None:
    event = _group_event(participant={"_serialized": "132469693124738@lid"})
    assert sender_tag(event) == "[132469693124738@lid]"


def test_quoted_participant_renders_name_and_jid() -> None:
    reply = {
        "participant": {"_serialized": "132469693124738@lid"},
    }
    names = {"132469693124738@lid": "Mikhail Polozhaev"}
    assert quoted_participant(reply, names) == "Mikhail Polozhaev <132469693124738@lid>"


def test_quoted_participant_notify_name_wins() -> None:
    reply = {
        "participant": "74943001800935@lid",
        "_data": {"notifyName": "Troche"},
    }
    assert quoted_participant(reply, {"74943001800935@lid": "wrong"}) == (
        "Troche <74943001800935@lid>"
    )


def test_quoted_participant_unknown_falls_back_to_bare_id() -> None:
    assert quoted_participant({"participant": "74943001800935@lid"}) == "74943001800935"


def test_render_system_prompt_substitutes_own_identities() -> None:
    prompt = "You are {{own_identities}} — a message that names you.\n{{date}}"
    out = render_system_prompt(prompt, own_jid="4915@c.us", own_lid="4915@lid")
    assert "You are `4915@c.us` or `4915@lid`" in out
    assert "{{own" not in out


def test_render_system_prompt_drops_identity_lines_when_unknown() -> None:
    prompt = "# Groups\nYou are {{own_identities}} — addresses you.\nRule stays.\n"
    out = render_system_prompt(prompt)
    assert "{{own" not in out
    assert "addresses you" not in out
    assert "Rule stays." in out


def test_render_system_prompt_drops_own_jid_line_when_unknown() -> None:
    out = render_system_prompt("id: {{own_jid}}\nkeep")
    assert "{{own_jid}}" not in out
    assert "id:" not in out
    assert "keep" in out


def test_chat_visible_text_filters_leaked_tokens() -> None:
    assert chat_visible_text("stay_silent") == ""
    assert chat_visible_text('{"error": {"message": "boom"}}') == ""
    assert chat_visible_text("real answer") == "real answer"
    assert chat_visible_text("") == ""


def test_remember_filters_undelivered_leak(unit_settings: Settings) -> None:
    """``remember`` stores nothing when the final text is a leaked token."""
    import asyncio

    from llama_index.core.base.llms.types import ChatResponse
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.core.workflow import Context

    from wahabot.ai.workflow import FunctionCallingAgentWorkflow, load_llm

    async def stored() -> list[Any]:
        wf = FunctionCallingAgentWorkflow(llm=load_llm(unit_settings))
        ctx = Context(wf)
        await ctx.store.set("memory", ChatMemoryBuffer.from_defaults())
        for text in ("stay_silent", "I'll stay silent here"):
            message = ChatMessage(role=MessageRole.ASSISTANT, content=text)
            await FunctionCallingAgentWorkflow.remember(
                wf, ctx, ChatResponse(message=message), []
            )
        real = ChatMessage(role=MessageRole.ASSISTANT, content="real words")
        await FunctionCallingAgentWorkflow.remember(
            wf, ctx, ChatResponse(message=real), []
        )
        memory = await ctx.store.get("memory")
        return [str(m.content) for m in await memory.aget_all()]

    assert asyncio.run(stored()) == ["real words"]


def test_purge_script_drops_leaked_tokens() -> None:
    """The purge script produces sanitized, alternating history."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "purge_leaked_silence",
        Path("scripts/purge_leaked_silence.py"),
    )
    assert spec is not None and spec.loader is not None
    purge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(purge)

    memory = ChatMemoryBuffer.from_defaults(token_limit=8000)  # pyright: ignore[reportUnknownMemberType]

    def U(c: str) -> ChatMessage:
        return ChatMessage(role=MessageRole.USER, content=c)

    def A(c: str) -> ChatMessage:
        return ChatMessage(role=MessageRole.ASSISTANT, content=c)

    memory.put(U("[Ana] q1"))
    memory.put(A("stay_silent"))
    memory.put(U("[Ana] q2"))
    memory.put(A("real answer"))
    memory.put(A("stay_silent"))
    kept, dropped = purge.purge_buffer(memory)
    assert dropped == 2
    # The purge removed the assistant turns between/before user turns;
    # the re-sanitize merges the now-consecutive user messages (every
    # word kept, kwargs merged) so alternation holds.
    assert [str(m.content) for m in kept] == ["[Ana] q1\n[Ana] q2", "real answer"]
    roles = [m.role for m in kept]
    assert roles[0] == MessageRole.USER and roles[-1] == MessageRole.ASSISTANT
    # Idempotent: a second pass over already-clean history drops nothing
    # and keeps the merged shape.
    again = ChatMemoryBuffer.from_defaults(token_limit=8000)  # pyright: ignore[reportUnknownMemberType]
    for message in kept:
        again.put(message)
    kept2, dropped2 = purge.purge_buffer(again)
    assert dropped2 == 0
    assert [str(m.content) for m in kept2] == ["[Ana] q1\n[Ana] q2", "real answer"]
