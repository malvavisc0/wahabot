import json
import tempfile
import time
import unittest.mock
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer

from tests.harness import (
    CHAT_ID,
    SESSION,
)
from wahabot.ai.messages import (
    bot_jids,
    self_command_instruction,
)
from wahabot.ai.observability import (
    _mask_value,  # pyright: ignore[reportPrivateUsage]
    chat_trace_attributes,
)
from wahabot.ai.tools.whatsapp import (
    chat_jid,
    probe_media_url,
)
from wahabot.commands import build_command_event
from wahabot.core.cache import TtlCache
from wahabot.core.cli import build_forget_event
from wahabot.core.echoes import is_self_echo, remember_self_echo
from wahabot.core.models import WahaEvent
from wahabot.core.persistence import (
    forget_memory,
    load_memory,
    memory_file,
    persistable,
    save_memory,
)
from wahabot.core.waha import WahaClient, response_json
from wahabot.status import (
    session_healthy,
    set_session_health,
)


def test_probe_media_url_retries_head_method_not_allowed() -> None:
    with (
        unittest.mock.patch.object(
            httpx, "head", return_value=unittest.mock.Mock(status_code=405)
        ),
        unittest.mock.patch.object(
            httpx, "get", return_value=unittest.mock.Mock(status_code=404)
        ),
    ):
        error = probe_media_url("https://media.invalid/missing.webp")
    assert error is not None and "HTTP 404" in error


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
    assert _mask_value("491555000009-123456789@g.us") == "[jid redacted]"
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


def test_trace_attributes_accept_ids_without_a_client() -> None:
    """Tracing disabled is a no-op even with per-turn audit ids."""
    with chat_trace_attributes("491555000002@g.us", message_ids=["m1"]):
        pass


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
    wire_waha.send_voice(
        SESSION, CHAT_ID, {"mimetype": "audio/mpeg", "url": "https://x.invalid/a.mp3"}
    )
    wire_waha.send_sticker(SESSION, CHAT_ID, {"mimetype": "image/webp", "data": "AAAA"})
    wire_waha.set_typing(SESSION, CHAT_ID, True)
    wire_waha.set_typing(SESSION, CHAT_ID, False)
    wire_waha.send_seen(SESSION, CHAT_ID)

    (
        send_request,
        read_request,
        image_request,
        file_request,
        video_request,
        forward_request,
        reaction_request,
        voice_request,
        sticker_request,
        typing_on_request,
        typing_off_request,
        seen_request,
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
        and voice_request.url.path == "/api/sendVoice"
        and json.loads(voice_request.content)
        == {
            "session": SESSION,
            "chatId": CHAT_ID,
            "file": {"mimetype": "audio/mpeg", "url": "https://x.invalid/a.mp3"},
            "convert": True,
        }
        and sticker_request.url.path == "/api/sendSticker"
        and json.loads(sticker_request.content)
        == {
            "session": SESSION,
            "chatId": CHAT_ID,
            "file": {"mimetype": "image/webp", "data": "AAAA"},
        }
        and typing_on_request.url.path == "/api/startTyping"
        and json.loads(typing_on_request.content)
        == {"session": SESSION, "chatId": CHAT_ID}
        and typing_off_request.url.path == "/api/stopTyping"
        and seen_request.url.path == "/api/sendSeen"
        and json.loads(seen_request.content) == {"session": SESSION, "chatId": CHAT_ID}
    )
    wire_waha._client.close()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("body", [b"", b"not json", b"\xff"])
def test_waha_invalid_json_is_http_error(body: bytes) -> None:
    request = httpx.Request("GET", "http://waha.invalid/api/sessions/default/me")
    response = httpx.Response(200, content=body, request=request)
    with pytest.raises(httpx.HTTPStatusError) as failure:
        response_json(response)
    assert str(failure.value) == f"Empty or non-JSON response from {request.url}"
    assert failure.value.request is request
    assert failure.value.response is response
    assert isinstance(failure.value.__cause__, ValueError)


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
    # The console: a mention anywhere is an instruction, not a quote.
    assert self_command_instruction(mid_text, bot_name="kai") == "note: kai do the thing"
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
