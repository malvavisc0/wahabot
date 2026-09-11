"""Wire scenarios ported from ``scripts/smoke_test.py``'s ``main``.

Each scenario posts a genuine event over HTTP (real HMAC signature)
against the booted stack and asserts the bot's behavior. The ``bot``
fixture provides an isolated agent + settings per test; ``bot.post``
handles signing and status assertion.
"""

import asyncio
import base64
import json
import shutil
import threading
import time
import unittest.mock
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from tests.conftest import Bot
from tests.harness import (
    CHAT_ID,
    FOREIGN_JID,
    ME_JID,
    SESSION,
    SMOKE_PNG,
    EscalateResponse,
    FenceRefusalResponse,
    FileResponse,
    FirstResponse,
    RecordingWaha,
    SecondResponse,
    album_events,
    chat_event,
    data_url_payload,
    image_event,
    image_part_count,
    is_caption_request,
    sign,
    smoke_video_bytes,
    tool_call_response,
    video_event,
    voice_event,
    waha_event,
)
from wahabot.ai.history import sanitize_chat_history as _sanitize
from wahabot.ai.tools.whatsapp import (
    EscalationChannel,
    bind_target,
    reset_target,
)
from wahabot.ai.tools.whatsapp import escalate as build_escalate
from wahabot.cli import build_forget_event
from wahabot.commands import build_command_event
from wahabot.core.echoes import remember_self_echo
from wahabot.core.persistence import load_memory, memory_file
from wahabot.core.runs import contexts as handlers_contexts
from wahabot.handlers import seen_recently
from wahabot.reactions import forget_reaction_notes as _forget_notes
from wahabot.status import state as status_state

# Zero-seconds polling cap for the async runs: un-comment to tighten.
POLL_TIMEOUT = 15


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


#: Video wire tests need ffmpeg (frames ride a real MP4); skip cleanly
#: so a contributor without it still runs the rest of the suite.
requires_ffmpeg = pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not installed")


def _wait(pred: Callable[[], bool], timeout: float = POLL_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


# ---------------------------------------------------------------------------
# Boot + first reply
# ---------------------------------------------------------------------------


def test_agent_replies_via_send_message(bot: Bot) -> None:
    llm = bot.stack.llm
    bot.post(waha_event())
    assert bot.waha.sent == [(SESSION, CHAT_ID, "smoke reply one", None)]
    assert len(llm.requests) == 2
    body = llm.requests[0]
    assert body.get("top_p") == 0.95 and body.get("top_k") == 20
    assert body.get("tools") is not None
    assert body.get("parallel_tool_calls") is True


def test_event_journal_exact_bytes(bot: Bot) -> None:
    event = waha_event()
    body = json.dumps(event).encode()
    bot.post(event)
    journals = sorted((bot.settings.data_dir / "events" / SESSION).glob("*.jsonl"))
    assert len(journals) == 1
    assert journals[0].read_bytes().strip() == body


def test_foreign_session_event_rejected(bot: Bot) -> None:
    event = waha_event()
    event["id"] = "evt-smoke-foreign-session"
    event["session"] = "other"
    body = json.dumps(event).encode()
    bot.post_raw(
        body,
        headers={
            "X-Webhook-Hmac": sign(body, bot.settings.webhook_hmac_key),
        },
        expect=404,
    )
    journals = (bot.settings.data_dir / "events" / SESSION).glob("*.jsonl")
    assert len(list(journals)) == 0


def test_malformed_event_rejected(bot: Bot) -> None:
    body = b"{ not json"
    bot.post_raw(
        body,
        headers={"X-Webhook-Hmac": sign(body, bot.settings.webhook_hmac_key)},
        expect=400,
    )
    assert len(list((bot.settings.data_dir / "events" / SESSION).glob("*.jsonl"))) == 0


def test_missing_hmac_rejected(bot: Bot) -> None:
    event = waha_event()
    body = json.dumps(event).encode()
    bot.post_raw(body, headers={"Content-Type": "application/json"}, expect=401)


def test_broadcast_does_not_wake_llm(bot: Bot) -> None:
    llm = bot.stack.llm
    broadcast = waha_event()
    broadcast["id"] = "evt-smoke-broadcast"
    broadcast["payload"]["id"] = "false_status@broadcast_STATUS"
    broadcast["payload"]["from"] = "status@broadcast"
    bot.post(broadcast)
    assert len(llm.requests) == 0
    assert len(bot.waha.sent) == 0


def test_image_blocks_ride_first_llm_call(bot: Bot) -> None:
    llm = bot.stack.llm
    bot.post(image_event())
    image_caption_reqs = [r for r in llm.requests if is_caption_request(r)]
    image_agent_reqs = [r for r in llm.requests if not is_caption_request(r)]
    assert len(image_caption_reqs) == 1
    assert len(image_agent_reqs) >= 1
    assert image_part_count(image_agent_reqs[0]) == 1
    assert any(
        isinstance(m.get("content"), list)
        and any(
            part.get("type") == "text"
            and "(image shows: two smoke-test squares on a white background)"
            in part.get("text", "")
            for part in m["content"]
        )
        for m in image_agent_reqs[0]["messages"]
        if m.get("role") == "user"
    )


def test_voice_note_transcription(bot: Bot) -> None:
    llm = bot.stack.llm
    with unittest.mock.patch(
        "wahabot.handlers.transcribe_voice_note", return_value="hello from voice"
    ):
        bot.post(voice_event())
    assert _wait(lambda: len(llm.requests) >= 1)
    voiced_turns = [m for m in llm.requests[0]["messages"] if m.get("role") == "user"]
    assert any(
        "[voice note] hello from voice" in str(m.get("content", "")) for m in voiced_turns
    )


def test_self_chat_voice_command(bot: Bot) -> None:
    """A spoken 'kai …' in the self-chat runs as an operator command."""
    llm = bot.stack.llm
    voice = voice_event()
    voice["payload"]["id"] = f"true_{ME_JID}_SELFCMDVOICE"
    voice["payload"]["from"] = ME_JID
    voice["payload"]["to"] = "491555000000@lid"
    voice["payload"]["fromMe"] = True
    with unittest.mock.patch(
        "wahabot.handlers.transcribe_voice_note",
        return_value="kai do the vocal thing",
    ):
        bot.post(voice)
    assert _wait(lambda: len(llm.requests) >= 1)
    turns = [
        str(m.get("content", ""))
        for m in llm.requests[0]["messages"]
        if m.get("role") == "user"
    ]
    assert any("[operator command] do the vocal thing" in t for t in turns)
    assert bot.waha.sent == [(SESSION, "operator", "smoke reply one", None)]


def test_voice_note_in_mentioned_group_skipped(bot: Bot) -> None:
    group_voice = voice_event()
    group_voice["payload"]["id"] = f"false_{CHAT_ID}_GROUPVOICE"
    group_voice["payload"]["from"] = CHAT_ID
    group_voice["payload"]["participant"] = "491555000001@c.us"
    transcriber = unittest.mock.Mock(
        side_effect=AssertionError("must not transcribe in mentioned group")
    )
    with unittest.mock.patch("wahabot.handlers.transcribe_voice_note", transcriber):
        bot.post(group_voice)
    transcriber.assert_not_called()


def test_silent_run_message_stays_in_context(bot: Bot) -> None:
    """A message the bot read but stayed silent on survives into history.

    The judicious-mode bug this pins: a silent run's user turn was
    trailing and unstamped, so the next run's ``repair_memory``
    dropped it as scaffolding — every message the bot chose not to
    answer vanished from context, and replies came out "factually
    correct but out of context". The run-end ``turn_handled`` stamp
    keeps it; the next run sees the whole conversation.
    """
    llm = bot.stack.llm
    # Message 1: the model stays silent on it.
    llm.override = tool_call_response(
        "stay_silent",
        {"reason": "banter between others"},
        call_id="call_silent_m1",
        response_id="chatcmpl-silent-m1",
        created=1788525845,
    )
    silent_event = waha_event()
    silent_event["payload"]["id"] = f"false_{CHAT_ID}_SILENT1"
    silent_event["payload"]["body"] = "kai mira el partido de anoche"
    bot.post(silent_event)
    assert _wait(lambda: len(llm.requests) >= 1)
    assert bot.waha.sent == []

    # Message 2: a fresh run must see message 1 in its history.
    llm.clear()
    bot.waha.sent.clear()
    llm.override = None
    follow_up = waha_event()
    follow_up["payload"]["id"] = f"false_{CHAT_ID}_FOLLOW1"
    follow_up["payload"]["body"] = "kai y tu que viste el partido?"
    bot.post(follow_up)
    assert _wait(lambda: len(llm.requests) >= 1)
    follow_up_turns = [
        str(m.get("content", ""))
        for m in llm.requests[0]["messages"]
        if m.get("role") == "user"
    ]
    assert any("el partido de anoche" in turn for turn in follow_up_turns), (
        "the silent-run message must ride the next run's history"
    )


def test_fromMe_own_message_folded(bot: Bot) -> None:
    llm = bot.stack.llm
    # Establish a context first (the fold needs prior memory).
    bot.post(waha_event())
    llm.requests.clear()
    bot.waha.sent.clear()
    own = waha_event()
    own["id"] = "evt-smoke-own"
    own["payload"]["id"] = f"true_{CHAT_ID}_OWNMSG"
    own["payload"]["fromMe"] = True
    own["payload"]["body"] = "operator typed this from the app"
    bot.post(own)
    assert len(llm.requests) == 0
    assert len(bot.waha.sent) == 0

    async def remembered() -> list[str]:
        ctx = handlers_contexts[(SESSION, CHAT_ID)]
        memory = await ctx.store.get("memory")
        messages = await memory.aget_all()
        return [
            str(m.content) for m in messages if str(m.role) == "MessageRole.ASSISTANT"
        ]

    assistant_turns = asyncio.run(remembered())
    assert any(
        turn.startswith("[operator message] ")
        and "operator typed this from the app" in turn
        for turn in assistant_turns
    )
    assert "smoke reply one" in assistant_turns


def test_fromMe_orphan_folds_nothing(bot: Bot) -> None:
    orphan_chat = "9998887777-9998887777@g.us"
    own_orphan = waha_event()
    own_orphan["id"] = "evt-smoke-own-orphan"
    own_orphan["payload"]["id"] = f"true_{orphan_chat}_ORPHAN"
    own_orphan["payload"]["from"] = orphan_chat
    own_orphan["payload"]["fromMe"] = True
    own_orphan["payload"]["body"] = "operator text in a never-seen chat"
    bot.post(own_orphan)
    assert (SESSION, orphan_chat) not in handlers_contexts
    assert not memory_file(bot.settings.data_dir, SESSION, orphan_chat).exists()


def test_history_repair_merges_assistant_turns() -> None:
    from llama_index.core.base.llms.types import ChatMessage

    def U(c: str) -> ChatMessage:
        return ChatMessage(role=MessageRole.USER, content=c)

    def A(c: str) -> ChatMessage:
        return ChatMessage(role=MessageRole.ASSISTANT, content=c)

    merged = _sanitize(
        [U("[Ana] q"), A("model reply"), A("[operator message] addendum")],
        drop_trailing_user=False,
    )
    assert len(merged) == 2
    assert "model reply" in str(merged[-1].content)
    assert "addendum" in str(merged[-1].content)

    noted = _sanitize(
        [
            U("[Ana] q"),
            A("model reply"),
            U('[reaction 👍 from Ana to your message: "model reply"]'),
            U("[Ana] next"),
        ],
        drop_trailing_user=False,
    )
    assert len(noted) == 3 and "reaction" in str(noted[-1].content)

    trailing = _sanitize(
        [
            U("[Ana] q"),
            A("model reply"),
            ChatMessage(
                role=MessageRole.USER,
                content='[reaction 👍 from Ana to your message: "model reply"]',
                additional_kwargs={"reaction_target_id": "true_C_R0"},
            ),
        ],
        drop_trailing_user=True,
    )
    assert len(trailing) == 3

    repaired = _sanitize(
        [U("[Ana] q"), A("model reply"), U("[Ana] unanswered")],
        drop_trailing_user=True,
    )
    assert len(repaired) == 2

    merged = _sanitize(
        [
            U("[Ana] q"),
            A("model reply"),
            ChatMessage(
                role=MessageRole.USER,
                content='[reaction 👍 from Ana to your message: "model reply"]',
                additional_kwargs={"reaction_target_id": "true_C_R"},
            ),
            U("[Ana] next"),
        ],
        drop_trailing_user=False,
    )
    assert merged[-1].additional_kwargs.get("reaction_target_id") == "true_C_R"


def test_tagged_reaction_removal(bot: Bot) -> None:
    # Establish the chat context first.
    bot.post(waha_event())

    async def tagged_removal() -> tuple[list[str], list[str]]:
        from llama_index.core.base.llms.types import ChatMessage

        memory = await handlers_contexts[(SESSION, CHAT_ID)].store.get("memory")
        merged_list = [
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    '[reaction 👍 from Ana to your message: "model reply"]\n[Ana] next'
                ),
                additional_kwargs={"reaction_target_id": "true_C_R2"},
            )
        ]
        rest = [
            ChatMessage(role=MessageRole.USER, content="[Ana] q"),
            ChatMessage(role=MessageRole.ASSISTANT, content="model reply"),
        ]
        await memory.aset([*rest, *merged_list])
        await _forget_notes(SESSION, CHAT_ID, "true_C_R2", bot.agent, bot.settings)
        after_merge = [str(m.content) for m in await memory.aget_all()]

        await memory.aset(
            [
                ChatMessage(
                    role=MessageRole.USER,
                    content="[reaction leading]",
                    additional_kwargs={"reaction_target_id": "true_C_R3"},
                ),
                ChatMessage(role=MessageRole.ASSISTANT, content="later"),
            ]
        )
        await _forget_notes(SESSION, CHAT_ID, "true_C_R3", bot.agent, bot.settings)
        after_leading = [str(m.content) for m in await memory.aget_all()]
        return after_merge, after_leading

    after_merge, after_leading = asyncio.run(tagged_removal())
    assert after_merge == ["[Ana] q", "model reply"]
    assert after_leading == ["[reaction leading]", "later"]


def test_no_send_tool_scaffolding_in_memory(bot: Bot) -> None:
    bot.post(waha_event())

    async def no_send_tool_group() -> bool:
        ctx = handlers_contexts[(SESSION, CHAT_ID)]
        memory = await ctx.store.get("memory")
        messages = await memory.aget_all()
        return not any(
            "send_message" in str(m.additional_kwargs.get("tool_calls", ""))
            for m in messages
        )

    assert asyncio.run(no_send_tool_group())


def test_album_vision(bot: Bot) -> None:
    llm = bot.stack.llm
    for album_event in album_events():
        bot.post(album_event)
    assert _wait(lambda: len(llm.requests) >= 1)

    caption_requests = [r for r in llm.requests if is_caption_request(r)]
    agent_requests = [r for r in llm.requests if not is_caption_request(r)]
    assert len(caption_requests) == 2
    assert all(image_part_count(r) == 1 for r in caption_requests)
    assert 1 <= len(agent_requests) <= 2
    album_request = agent_requests[0]
    assert image_part_count(album_request) == 2
    if len(agent_requests) > 1:
        assert image_part_count(agent_requests[-1]) == 0

    expected_b64 = base64.b64encode(SMOKE_PNG).decode()
    expected_url = f"data:image/png;base64,{expected_b64}"
    assert all(url == expected_url for url in data_url_payload(album_request))

    user_turns = [m for m in album_request["messages"] if m.get("role") == "user"]
    album_caption = (
        "two smoke-test squares on a white background; "
        "two smoke-test squares on a white background"
    )
    assert any(
        isinstance(m.get("content"), list)
        and any(
            p.get("type") == "text"
            and f"(images show: {album_caption})" in p.get("text", "")
            for p in m["content"]
        )
        for m in user_turns
    )
    assert all(
        "(images show:" not in str(m.get("content", ""))
        for r in caption_requests
        for m in r["messages"]
    )


@requires_ffmpeg
def test_video_understanding(bot: Bot) -> None:
    llm = bot.stack.llm
    bot.waha.video_bytes = smoke_video_bytes()
    llm.requests.clear()
    with unittest.mock.patch(
        "wahabot.handlers.fetch_transcript", return_value="spoken words here"
    ):
        bot.post(video_event())
    assert _wait(lambda: len(llm.requests) >= 1)
    video_caption_reqs = [r for r in llm.requests if is_caption_request(r)]
    video_agent_reqs = [r for r in llm.requests if not is_caption_request(r)]
    assert len(video_caption_reqs) == 1
    if video_caption_reqs:
        assert image_part_count(video_caption_reqs[0]) == bot.settings.video_frames
    assert len(video_agent_reqs) >= 1
    if video_agent_reqs:
        video_turns = [
            m for m in video_agent_reqs[0]["messages"] if m.get("role") == "user"
        ]
        assert any(
            "(video shows: two smoke-test squares on a white background)"
            in str(m.get("content", ""))
            and '[audio: "spoken words here"]' in str(m.get("content", ""))
            for m in video_turns
        )
        assert image_part_count(video_agent_reqs[0]) == bot.settings.video_frames


def test_video_disabled(bot: Bot) -> None:
    bot.settings.video = False
    llm = bot.stack.llm
    bot.post(video_event(mid="VIDOFF"))
    time.sleep(0.5)
    assert len(llm.requests) == 0


@requires_ffmpeg
def test_video_without_transcriber(bot: Bot) -> None:
    llm = bot.stack.llm
    bot.waha.video_bytes = smoke_video_bytes()
    llm.requests.clear()
    saved_url = bot.settings.transcribe_url
    bot.settings.transcribe_url = ""
    try:
        bot.post(video_event(mid="VIDNT"))
        assert _wait(lambda: len(llm.requests) >= 1)
    finally:
        bot.settings.transcribe_url = saved_url
    nt_agent_reqs = [r for r in llm.requests if not is_caption_request(r)]
    assert len(nt_agent_reqs) >= 1
    if nt_agent_reqs:
        nt_turns = [m for m in nt_agent_reqs[0]["messages"] if m.get("role") == "user"]
        assert any(
            "(video shows:" in str(m.get("content", ""))
            and "[audio:" not in str(m.get("content", ""))
            for m in nt_turns
        )


@requires_ffmpeg
def test_video_transcript_failure(bot: Bot) -> None:
    llm = bot.stack.llm
    bot.waha.video_bytes = smoke_video_bytes()
    llm.requests.clear()
    with unittest.mock.patch(
        "wahabot.handlers.fetch_transcript",
        side_effect=RuntimeError("500: no audio stream"),
    ):
        bot.post(video_event(mid="VIDTF"))
    assert _wait(lambda: len(llm.requests) >= 1)
    tf_agent_reqs = [r for r in llm.requests if not is_caption_request(r)]
    assert len(tf_agent_reqs) >= 1
    if tf_agent_reqs:
        tf_turns = [m for m in tf_agent_reqs[0]["messages"] if m.get("role") == "user"]
        assert any("(video shows:" in str(m.get("content", "")) for m in tf_turns)


def test_redelivery_dedup(bot: Bot) -> None:
    llm = bot.stack.llm
    # First delivery wakes the agent.
    bot.post(waha_event())
    llm.requests.clear()
    bot.waha.sent.clear()
    # Redelivery 150s later must not wake again.
    redelivery = waha_event()
    redelivery["payload"]["timestamp"] = int(time.time()) - 150
    bot.post(redelivery)
    assert len(llm.requests) == 0
    assert len(bot.waha.sent) == 0


def test_fresh_message_passes(bot: Bot) -> None:
    llm = bot.stack.llm
    bot.post(waha_event())
    llm.requests.clear()
    fresh = waha_event()
    fresh["payload"]["id"] = f"false_{CHAT_ID}_FRESH"
    fresh["payload"]["body"] = "kai fresh turn"
    bot.post(fresh)
    assert _wait(lambda: len(llm.requests) >= 1)


def test_operator_command(bot: Bot) -> None:
    llm = bot.stack.llm
    llm.override = tool_call_response(
        "send_message",
        {"text": "smoke reply one", "chat": CHAT_ID},
        call_id="call_cmd_1",
        response_id="chatcmpl-smoke-cmd",
        created=1788525834,
    )
    command = build_command_event(SESSION, "send the plan to Familia")
    bot.post(command)
    assert _wait(lambda: len(llm.requests) >= 1)
    command_turns = [m for m in llm.requests[0]["messages"] if m.get("role") == "user"]
    assert any(
        "[operator command] send the plan to Familia" in str(m.get("content", ""))
        for m in command_turns
    )
    assert len(llm.requests[0]["messages"]) <= 3
    assert bot.waha.sent == [(SESSION, CHAT_ID, "smoke reply one", None)]


def test_operator_history_shared_across_commands(bot: Bot) -> None:
    """Commands share one rolling history: turn two sees turn one."""
    llm = bot.stack.llm
    llm.override = None
    bot.post(build_command_event(SESSION, "first command context"))
    assert _wait(lambda: len(llm.requests) >= 1)
    llm.requests.clear()
    bot.post(build_command_event(SESSION, "second command"))
    assert _wait(lambda: len(llm.requests) >= 1)
    first_turns = [
        str(m.get("content", ""))
        for m in llm.requests[0]["messages"]
        if m.get("role") == "user"
    ]
    assert any("[operator command] first command context" in t for t in first_turns), (
        "second command must see the first turn in its history"
    )
    # One shared context key, persisted like a chat's memory.
    assert (SESSION, "operator") in handlers_contexts
    assert memory_file(bot.settings.data_dir, SESSION, "operator").exists()


def test_operator_history_survives_eviction(bot: Bot) -> None:
    """An LRU-evicted operator context reloads from disk, not blank."""
    llm = bot.stack.llm
    llm.override = None
    bot.post(build_command_event(SESSION, "persisted operator turn"))
    assert _wait(lambda: len(llm.requests) >= 1)
    handlers_contexts.pop((SESSION, "operator"), None)
    llm.requests.clear()
    bot.post(build_command_event(SESSION, "after eviction"))
    assert _wait(lambda: len(llm.requests) >= 1)
    restored_users = [
        str(m.get("content", ""))
        for m in llm.requests[0]["messages"]
        if m.get("role") == "user"
    ]
    assert any("persisted operator turn" in t for t in restored_users)


def test_operator_forget(bot: Bot) -> None:
    """``wahabot forget operator`` wipes the shared command history."""
    llm = bot.stack.llm
    llm.override = None
    bot.post(build_command_event(SESSION, "forgettable operator turn"))
    assert _wait(lambda: len(llm.requests) >= 1)
    mem_path = memory_file(bot.settings.data_dir, SESSION, "operator")
    assert mem_path.exists()
    bot.post(build_forget_event(SESSION, "operator"))
    assert (SESSION, "operator") not in handlers_contexts
    assert not mem_path.exists()
    llm.requests.clear()
    bot.post(build_command_event(SESSION, "after forget"))
    assert _wait(lambda: len(llm.requests) >= 1)
    post_forget_users = [
        str(m.get("content", ""))
        for m in llm.requests[0]["messages"]
        if m.get("role") == "user"
    ]
    assert not any("forgettable operator turn" in t for t in post_forget_users)


def test_cross_chat_fence_refusal(bot: Bot) -> None:
    llm = bot.stack.llm
    llm.override = FenceRefusalResponse
    fence_event = waha_event()
    fence_event["payload"]["id"] = f"false_{CHAT_ID}_FENCE"
    fence_event["payload"]["body"] = "kai send this to that guy you know"
    bot.post(fence_event)
    assert _wait(lambda: len(llm.requests) >= 2)
    assert not any(chat == FOREIGN_JID for _, chat, _, _ in bot.waha.sent)
    assert all(chat == CHAT_ID for _, chat, _, _ in bot.waha.sent)
    fence_feedback = str(llm.requests[1]["messages"] if len(llm.requests) > 1 else [])
    assert "cross-chat reach is reserved for operator commands" in fence_feedback


def test_message_id_fence_refusal(bot: Bot) -> None:
    llm = bot.stack.llm
    llm.override = tool_call_response(
        "react_to_message",
        {"message_id": f"false_{FOREIGN_JID}_X", "reaction": "👍"},
        call_id="call_idfence_1",
        response_id="chatcmpl-smoke-idfence",
        created=1788525835,
    )
    idfence_event = waha_event()
    idfence_event["payload"]["id"] = f"false_{CHAT_ID}_IDFENCE"
    idfence_event["payload"]["body"] = "kai react to that other message"
    bot.post(idfence_event)
    assert _wait(lambda: len(llm.requests) >= 2)
    assert not any(FOREIGN_JID in mid for mid, _ in bot.waha.reactions)
    idfence_feedback = str(llm.requests[1]["messages"] if len(llm.requests) > 1 else [])
    assert "cross-chat reach is reserved for operator commands" in idfence_feedback


def test_operator_run_reaches_foreign_chat(bot: Bot) -> None:
    llm = bot.stack.llm
    llm.override = tool_call_response(
        "send_message",
        {"text": "operator says hi", "chat": FOREIGN_JID},
        call_id="call_fence_ok_1",
        response_id="chatcmpl-smoke-fence-ok",
        created=1788525833,
    )
    command = build_command_event(SESSION, "send a message to that guy")
    bot.post(command)
    assert _wait(lambda: len(bot.waha.sent) >= 1)
    assert bot.waha.sent == [(SESSION, FOREIGN_JID, "operator says hi", None)]


def test_text_token_becomes_real_mention(bot: Bot) -> None:
    """An ``@<lid-number>`` token in the text tags the roster member.

    The trace-audit failure: the model copies the chat's own mention
    shape ("Para @111222333444555") into its reply but passes no
    ``mentions`` — WAHA then sends plain text and nobody is notified.
    The tool now resolves text tokens against the chat roster, so the
    identical call delivers a real mention.
    """
    llm = bot.stack.llm
    # RecordingWaha's roster: 491555000001@c.us ("Smoke Sender") + the bot.
    llm.override = tool_call_response(
        "send_message",
        {"text": "Para @491555000001"},
        call_id="call_mention_1",
        response_id="chatcmpl-smoke-mention",
        created=1788525836,
    )
    event = waha_event()
    event["payload"]["body"] = "kai dale para smoke sender"
    bot.post(event)
    assert _wait(lambda: len(bot.waha.sent) >= 1)
    assert bot.waha.sent == [
        (SESSION, CHAT_ID, "Para @491555000001", ["491555000001@c.us"])
    ]


def test_stay_silent_reason_logged(bot: Bot) -> None:
    """The ``reason`` of a stay_silent call reaches the log, not the chat.

    ``stay_silent`` is terminal — the workflow stops before executing
    it — so the workflow itself must surface the reason; the message
    here names another member, and the model's justification ("not
    addressed to me") is exactly the audit line judicious mode needs.
    """
    from loguru import logger

    llm = bot.stack.llm
    llm.override = tool_call_response(
        "stay_silent",
        {"reason": "question addressed to @222333444555666, not to me"},
        call_id="call_quiet_1",
        response_id="chatcmpl-smoke-quiet",
        created=1788525837,
    )
    quiet_event = waha_event()
    quiet_event["payload"]["id"] = f"false_{CHAT_ID}_QUIET"
    # The harness config runs "mentioned" mode, so the body must name
    # the bot for the run to wake; the model then judges the *quoted*
    # question as aimed at @222333444555666 and stays silent.
    quiet_event["payload"]["body"] = (
        "kai mira @222333444555666 que haces para ser millonario"
    )
    infos: list[tuple[str, dict[str, Any]]] = []

    def record(message: str, **kwargs: Any) -> None:
        infos.append((message, kwargs))

    with unittest.mock.patch.object(logger, "info", record):
        bot.post(quiet_event)
        assert _wait(lambda: any(kw.get("tool") == "stay_silent" for _, kw in infos))
    assert any(
        kw["reason"] == "question addressed to @222333444555666, not to me"
        for _, kw in infos
        if kw.get("tool") == "stay_silent"
    )
    assert bot.waha.sent == []


def test_send_file_wire(bot: Bot) -> None:
    llm = bot.stack.llm
    llm.override = FileResponse
    file_event = waha_event()
    file_event["payload"]["id"] = f"false_{CHAT_ID}_SENDFILE"
    file_event["payload"]["body"] = "kai send me the report"
    bot.post(file_event)
    assert _wait(lambda: len(bot.waha.sent_files) >= 1)
    assert len(bot.waha.sent_files) == 1
    f_session, f_chat, f_file, f_caption = bot.waha.sent_files[0]
    assert (
        f_session == SESSION
        and f_chat == CHAT_ID
        and f_file
        == {
            "mimetype": "application/pdf",
            "url": "http://files.invalid/q3/report.pdf",
            "filename": "report.pdf",
        }
        and f_caption == "the report"
    )


def test_reaction_fold_in(bot: Bot) -> None:
    llm = bot.stack.llm
    bot.post(waha_event())  # establish context + delivered reply
    llm.requests.clear()
    bot.waha.sent.clear()
    reaction = {
        "id": "evt-smoke-reaction",
        "timestamp": int(time.time()),
        "event": "message.reaction",
        "session": SESSION,
        "me": None,
        "payload": {
            "id": "evt-smoke-reaction-payload",
            "from": CHAT_ID,
            "participant": "491555000001@c.us",
            "fromMe": False,
            "reaction": {"text": "👍", "messageId": f"true_{CHAT_ID}_SMOKEREPLY"},
        },
    }
    bot.post(reaction)
    time.sleep(0.5)
    assert len(llm.requests) == 0
    assert len(bot.waha.sent) == 0

    async def reaction_notes() -> list[str]:
        ctx = handlers_contexts[(SESSION, CHAT_ID)]
        memory = await ctx.store.get("memory")
        messages = await memory.aget_all()
        return [str(m.content) for m in messages if "[reaction" in str(m.content)]

    notes = asyncio.run(reaction_notes())
    assert any("👍" in note and "smoke reply one" in note for note in notes)
    assert any("491555000001@c.us" in note for note in notes)

    # Reaction to someone else's message stays ignored.
    foreign = {
        "id": "evt-smoke-reaction-foreign",
        "timestamp": int(time.time()),
        "event": "message.reaction",
        "session": SESSION,
        "me": None,
        "payload": {
            "id": "evt-smoke-reaction-foreign-payload",
            "from": CHAT_ID,
            "reaction": {"text": "👎", "messageId": f"false_{CHAT_ID}_NOTOURS"},
        },
    }
    bot.post(foreign)
    time.sleep(0.5)
    foreign_notes = asyncio.run(reaction_notes())
    assert not any("👎" in note for note in foreign_notes)


def test_session_health_mute(bot: Bot) -> None:
    llm = bot.stack.llm
    status_down = {
        "id": "evt-smoke-status-down",
        "timestamp": int(time.time()),
        "event": "session.status",
        "session": SESSION,
        "me": None,
        "payload": {"name": SESSION, "status": "FAILED"},
    }
    bot.post(status_down)
    time.sleep(0.2)
    from wahabot.status import session_healthy

    assert not session_healthy()

    muted = waha_event()
    muted["payload"]["id"] = f"false_{CHAT_ID}_MUTED"
    muted["payload"]["body"] = "kai while muted"
    bot.post(muted)
    time.sleep(0.5)
    assert len(llm.requests) == 0
    assert not seen_recently(f"false_{CHAT_ID}_MUTED")

    status_up = dict(status_down)
    status_up["id"] = "evt-smoke-status-up"
    status_up["payload"] = {"name": SESSION, "status": "WORKING"}
    bot.post(status_up)
    time.sleep(0.2)
    assert session_healthy()


def test_recovery_recaptures_operator_jid(bot: Bot) -> None:
    status_state.operator_jid = ""

    def post_status(status: str, event_id: str) -> None:
        status_event = {
            "id": event_id,
            "timestamp": int(time.time()),
            "event": "session.status",
            "session": SESSION,
            "me": None,
            "payload": {"name": SESSION, "status": status},
        }
        bot.post(status_event)

    post_status("FAILED", "evt-smoke-status-down2")
    time.sleep(0.2)
    post_status("WORKING", "evt-smoke-status-up3")
    time.sleep(0.2)
    from wahabot.status import session_healthy

    assert session_healthy()
    assert status_state.operator_jid == ME_JID
    assert status_state.operator_lid == "491555000000@lid"


def test_persistent_memory_roundtrip_wire(bot: Bot) -> None:
    llm = bot.stack.llm
    PC = "5553333333-5553333333@g.us"

    def pc_event(
        body: str, mid: str, from_me: bool = False, chat: str = PC
    ) -> dict[str, Any]:
        return chat_event(
            body=body, mid=mid, chat=chat, from_me=from_me, include_serialized=True
        )

    def persisted_contents() -> list[str]:
        memory = load_memory(bot.settings.data_dir, SESSION, PC)
        return [] if memory is None else [str(m.content) for m in memory.get_all()]

    # Delivered reply persists.
    bot.post(pc_event("kai pc first", "PC1"))
    assert len(bot.waha.sent) == 1
    mem_path = memory_file(bot.settings.data_dir, SESSION, PC)
    assert mem_path.exists()
    envelope = json.loads(mem_path.read_text())
    assert (
        envelope["version"] == 1
        and envelope["session"] == SESSION
        and envelope["chat_id"] == PC
    )
    assert "smoke reply one" in persisted_contents()

    # Restore-on-miss.
    from wahabot.core.runs import contexts as ctxs

    ctxs.pop((SESSION, PC), None)
    assert (SESSION, PC) not in ctxs
    llm.requests.clear()
    bot.waha.sent.clear()
    bot.post(pc_event("kai pc second", "PC2"))
    assert (SESSION, PC) in ctxs
    assert len(llm.requests) >= 1
    first_request_users = [
        str(m.get("content", ""))
        for m in llm.requests[0]["messages"]
        if m.get("role") == "user"
    ]
    assert any("pc first" in text for text in first_request_users)

    # fromMe fold persists.
    llm.requests.clear()
    bot.waha.sent.clear()
    bot.post(pc_event("operator typed pc", "PCOWN", from_me=True))
    assert len(bot.waha.sent) == 0
    assert any(
        c.startswith("[operator message] ") and "operator typed pc" in c
        for c in persisted_contents()
    )

    # Reaction fold persists.
    llm.requests.clear()
    bot.post(
        {
            "id": "evt-smoke-pc-reaction",
            "timestamp": int(time.time()),
            "event": "message.reaction",
            "session": SESSION,
            "me": None,
            "payload": {
                "id": "evt-smoke-pc-reaction-payload",
                "from": PC,
                "participant": "491555000001@c.us",
                "fromMe": False,
                "reaction": {"text": "👍", "messageId": f"true_{PC}_SMOKEREPLY"},
            },
        }
    )
    assert len(llm.requests) == 0
    assert any("[reaction" in c and "smoke reply one" in c for c in persisted_contents())


def test_persistence_corrupt_and_forget(bot: Bot) -> None:
    llm = bot.stack.llm
    PC = "5553333333-5553333333@g.us"

    def pc_event(body: str, mid: str, chat: str = PC) -> dict[str, Any]:
        return chat_event(body=body, mid=mid, chat=chat, include_serialized=True)

    llm.override = None
    bot.post(pc_event("kai pc first", "PC1"))
    mem_path = memory_file(bot.settings.data_dir, SESSION, PC)
    llm.requests.clear()
    bot.waha.sent.clear()
    ctxs = handlers_contexts
    ctxs.pop((SESSION, PC), None)
    mem_path.write_text("{ truncated json")
    bot.post(pc_event("kai pc third", "PC3"))
    assert len(bot.waha.sent) == 1
    assert Path(str(mem_path) + ".bad").exists()
    assert mem_path.exists()

    # forget: context dropped, file deleted, next turn blank.
    llm.requests.clear()
    bot.waha.sent.clear()
    bot.post(build_forget_event(SESSION, PC))
    assert (SESSION, PC) not in ctxs
    assert not mem_path.exists()
    llm.requests.clear()
    bot.waha.sent.clear()
    bot.post(pc_event("kai pc fourth", "PC4"))
    assert len(bot.waha.sent) == 1
    post_forget_users = [
        str(m.get("content", ""))
        for request in llm.requests
        for m in request["messages"]
        if m.get("role") == "user"
    ]
    assert not any("pc first" in t or "pc second" in t for t in post_forget_users)


def test_memory_persist_disabled(bot: Bot) -> None:
    NPC = "8887777777-8887777777@g.us"
    no_persist_bot = bot.rebuild(memory_persist=False)

    def npc_event(body: str, chat: str = NPC) -> dict[str, Any]:
        ev = waha_event()
        ev["id"] = "evt-smoke-npc-1"
        ev["payload"]["id"] = f"false_{chat}_NPC1"
        ev["payload"]["from"] = chat
        ev["payload"]["body"] = body
        return ev

    llm = no_persist_bot.stack.llm
    with unittest.mock.patch(
        "wahabot.settings.get_settings", return_value=no_persist_bot.settings
    ):
        no_persist_bot.post(npc_event("kai npc turn"))
    assert _wait(lambda: len(llm.requests) >= 1)
    assert not memory_file(no_persist_bot.settings.data_dir, SESSION, NPC).exists()


def test_self_chat_command(bot: Bot) -> None:
    llm = bot.stack.llm
    llm.clear()

    def self_event(body: str, mid: str) -> dict[str, Any]:
        ev = waha_event()
        ev["id"] = f"evt-smoke-self-{mid}"
        ev["payload"]["id"] = f"true_{ME_JID}_{mid}"
        ev["payload"]["from"] = ME_JID
        ev["payload"]["fromMe"] = True
        ev["payload"]["body"] = body
        return ev

    bot.post(self_event("kai do the thing", "SELF1"))
    assert _wait(lambda: len(llm.requests) >= 1)
    self_turns = [
        str(m.get("content", ""))
        for m in llm.requests[0]["messages"]
        if m.get("role") == "user"
    ]
    assert any("[operator command] do the thing" in t for t in self_turns)
    assert len(llm.requests[0]["messages"]) <= 3
    assert bot.waha.sent == [(SESSION, "operator", "smoke reply one", None)]

    # Research-only answer: final text quote-replied to the self-chat.
    llm.clear()
    bot.waha.sent.clear()
    llm.override = SecondResponse
    bot.post(self_event("kai just answer me", "SELF3"))
    assert _wait(lambda: len(bot.waha.sent) >= 1)
    assert bot.waha.sent == [(SESSION, ME_JID, "smoke final answer", None)]

    # Non-matching fromMe in the self-chat wakes nothing.
    llm.clear()
    bot.waha.sent.clear()
    bot.post(self_event("just a note to myself", "SELF2"))
    time.sleep(0.5)
    assert len(llm.requests) == 0
    assert (SESSION, ME_JID) not in handlers_contexts

    # Echo-tracked self-chat message never re-triggers.
    llm.clear()
    echo_mid = f"true_{ME_JID}_SELFECHO"
    remember_self_echo(echo_mid)
    bot.post(self_event("kai run again", "SELFECHO"))
    time.sleep(0.5)
    assert len(llm.requests) == 0

    # LID-linked self-chat: WAHA reports from=<phone>@c.us, to=<lid>@lid.
    llm.clear()
    bot.waha.sent.clear()
    llm.override = None
    lid_event = self_event("kai lid linked command", "SELF4")
    lid_event["payload"]["to"] = "491555000000@lid"
    bot.post(lid_event)
    assert _wait(lambda: len(llm.requests) >= 1)
    lid_turns = [
        str(m.get("content", ""))
        for m in llm.requests[0]["messages"]
        if m.get("role") == "user"
    ]
    assert any("[operator command] lid linked command" in t for t in lid_turns)

    # A command that delivers via a tool still acknowledges in the
    # self-chat — the console is never silent about where it went.
    llm.clear()
    bot.waha.sent.clear()
    llm.override = tool_call_response(
        "send_message",
        {"text": "smoke reply one", "chat": CHAT_ID},
        call_id="call_cmd_self5",
        response_id="chatcmpl-smoke-cmd-self5",
        created=1788525841,
    )
    bot.post(self_event("kai send the plan to the group", "SELF5"))
    assert _wait(lambda: len(bot.waha.sent) >= 2)
    assert (SESSION, CHAT_ID, "smoke reply one", None) in bot.waha.sent
    notice = [s for s in bot.waha.sent if s[1] == ME_JID]
    assert len(notice) == 1 and "delivered to" in notice[0][2] and CHAT_ID in notice[0][2]


def test_escalate_delivery(bot: Bot) -> None:
    llm = bot.stack.llm
    llm.override = EscalateResponse
    PC = "5553333333-5553333333@g.us"
    pc_event = waha_event()
    pc_event["id"] = "evt-smoke-escalate-1"
    pc_event["payload"]["id"] = f"false_{PC}_ESC1"
    pc_event["payload"]["from"] = PC
    pc_event["payload"]["body"] = "kai I want to speak to a human"
    bot.post(pc_event)
    assert _wait(lambda: len(llm.requests) >= 2)
    assert any(
        chat == ME_JID and "Smoke Sender wants a human" in text
        for _, chat, text, _ in bot.waha.sent
    )
    assert any(
        f"escalation from {PC}" in text
        for _, chat, text, _ in bot.waha.sent
        if chat == ME_JID
    )


def test_escalate_cooldown(bot: Bot) -> None:
    llm = bot.stack.llm
    llm.override = EscalateResponse
    # Trigger the first escalation so the channel is stamped.
    PC = "5553333333-5553333333@g.us"
    pc_event = waha_event()
    pc_event["id"] = "evt-smoke-esc-1"
    pc_event["payload"]["id"] = f"false_{PC}_ESC1"
    pc_event["payload"]["from"] = PC
    pc_event["payload"]["body"] = "kai I want to speak to a human"
    bot.post(pc_event)
    assert _wait(lambda: len(llm.requests) >= 2)
    self_sent_before = sum(1 for _, chat, _, _ in bot.waha.sent if chat == ME_JID)
    llm.requests.clear()
    # Second escalate: cooldown refuses, no new self-chat delivery.
    pc_event2 = waha_event()
    pc_event2["id"] = "evt-smoke-esc-2"
    pc_event2["payload"]["id"] = f"false_{PC}_ESC2"
    pc_event2["payload"]["from"] = PC
    pc_event2["payload"]["body"] = "kai please escalate again"
    bot.post(pc_event2)
    assert _wait(lambda: len(llm.requests) >= 2)
    self_sent_after = sum(1 for _, chat, _, _ in bot.waha.sent if chat == ME_JID)
    assert self_sent_after == self_sent_before
    cooldown_feedback = str(llm.requests[1]["messages"] if len(llm.requests) > 1 else [])
    assert "cooldown" in cooldown_feedback


def test_escalate_fail_soft() -> None:
    ME_JID = "491555000000@c.us"
    status_state.operator_jid = ME_JID
    failing_waha = RecordingWaha()

    def boom(*_args: object, **_kwargs: object) -> str:
        raise httpx.HTTPError("WAHA is down")

    failing_waha.send_text = boom  # type: ignore[method-assign]
    down_channel = EscalationChannel()
    assert down_channel.operator_jid == ME_JID

    esc_tool = build_escalate(failing_waha, down_channel)
    esc_fn = cast(Any, esc_tool).fn
    PC = "5553333333-5553333333@g.us"
    token = bind_target({"session": SESSION, "chat_id": PC, "sent": "", "reacted": ""})
    try:
        outcome = str(esc_fn(report="help"))
    finally:
        reset_target(token)
    assert '"ok": false' in outcome and "did NOT go through" in outcome
    assert down_channel.cooldown_refusal(PC) is None
    down_channel.stamp(PC)
    assert (
        down_channel.cooldown_refusal(PC) is not None
        and down_channel.cooldown_refusal(CHAT_ID) is None
    )
    status_state.operator_jid = "491555999999@c.us"
    assert down_channel.operator_jid == "491555999999@c.us"
    status_state.operator_jid = ME_JID


def test_concurrent_chats_overlap(bot: Bot) -> None:
    llm = bot.stack.llm
    PC = "5553333333-5553333333@g.us"
    DM = "491555000009@c.us"
    G2 = "7776665555-7776665555@g.us"

    def pc_event(body: str, mid: str, chat: str) -> dict[str, Any]:
        return chat_event(body=body, mid=mid, chat=chat)

    def concurrent_llm(body: dict[str, Any]) -> dict[str, Any]:
        text = json.dumps(body.get("messages", []))
        if '"ok"' in text:
            return SecondResponse
        return FirstResponse

    llm.selector = concurrent_llm
    events = [
        pc_event("kai concurrent one", "CONC1", PC),
        pc_event("kai concurrent dm", "CONC2", DM),
        pc_event("kai concurrent two", "CONC3", G2),
    ]

    # Genuinely overlap the runs with an LLM gate.
    gate = asyncio.Barrier(3)
    gate_timeout = time.monotonic() + 60
    gate_broken = threading.Event()

    async def gated_concurrent_llm(body: dict[str, Any]) -> dict[str, Any]:
        text = json.dumps(body.get("messages", []))
        if '"ok"' not in text:
            if time.monotonic() > gate_timeout or gate_broken.is_set():
                return concurrent_llm(body)
            try:
                await asyncio.wait_for(
                    gate.wait(), timeout=max(0.1, gate_timeout - time.monotonic())
                )
            except TimeoutError, asyncio.BrokenBarrierError:
                gate_broken.set()
        return concurrent_llm(body)

    llm.selector = gated_concurrent_llm

    responses: list[httpx.Response] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(bot.post, ev) for ev in events]
        responses = [f.result(timeout=120) for f in futures]
    for response in responses:
        assert response.status_code == 200

    assert _wait(lambda: len(bot.waha.sent) >= 3)
    assert not gate_broken.is_set()
    assert len(bot.waha.sent) == 3
    replied_chats = sorted(chat for _, chat, _, _ in bot.waha.sent)
    assert replied_chats == sorted([PC, DM, G2])


def test_fence_isolation_under_concurrency(bot: Bot) -> None:
    llm = bot.stack.llm
    G2 = "7776665555-7776665555@g.us"
    fence_event = waha_event()
    fence_event["id"] = "evt-smoke-concf"
    fence_event["payload"]["id"] = f"false_{G2}_CONCF"
    fence_event["payload"]["from"] = G2
    fence_event["payload"]["body"] = "kai leak it"
    command = build_command_event(SESSION, "send a message to that guy")
    command_send = tool_call_response(
        "send_message",
        {"text": "operator hi", "chat": FOREIGN_JID},
        call_id="call_conc_cmd",
        response_id="chatcmpl-conc-cmd",
        created=1788525840,
        usage=(5, 5),
    )

    def selective_llm(body: dict[str, Any]) -> dict[str, Any]:
        text = json.dumps(body.get("messages", []))
        if '"ok"' in text:
            return SecondResponse
        if "[operator command]" in text:
            return command_send
        return FenceRefusalResponse

    gate = asyncio.Barrier(2)
    gate_timeout = time.monotonic() + 60
    gate_broken = threading.Event()

    async def gated_selective_llm(body: dict[str, Any]) -> dict[str, Any]:
        text = json.dumps(body.get("messages", []))
        if '"ok"' not in text:
            if time.monotonic() > gate_timeout or gate_broken.is_set():
                return selective_llm(body)
            try:
                await asyncio.wait_for(
                    gate.wait(), timeout=max(0.1, gate_timeout - time.monotonic())
                )
            except TimeoutError, asyncio.BrokenBarrierError:
                gate_broken.set()
        return selective_llm(body)

    llm.selector = gated_selective_llm
    responses: list[httpx.Response] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(bot.post, ev) for ev in (fence_event, command)]
        responses = [f.result(timeout=120) for f in futures]
    for response in responses:
        assert response.status_code == 200

    assert _wait(lambda: len(bot.waha.sent) >= 1)
    assert not gate_broken.is_set()
    assert bot.waha.sent == [(SESSION, FOREIGN_JID, "operator hi", None)]
    assert not any(text == "leak" for _, _, text, _ in bot.waha.sent)
    fence_feedback = json.dumps([m for r in llm.requests for m in r.get("messages", [])])
    assert "cross-chat reach is reserved for operator commands" in fence_feedback


def test_same_chat_runs_serialize(bot: Bot) -> None:
    llm = bot.stack.llm
    PC = "5553333333-5553333333@g.us"
    llm.clear()

    def pc_event(body: str, mid: str) -> dict[str, Any]:
        return chat_event(body=body, mid=mid, chat=PC)

    def serialize_llm(body: dict[str, Any]) -> dict[str, Any]:
        text = json.dumps(body.get("messages", []))
        if '"ok"' in text:
            return SecondResponse
        return FirstResponse

    llm.selector = serialize_llm
    events = [
        pc_event("kai serial one", "SER1"),
        pc_event("kai serial two", "SER2"),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(bot.post, ev) for ev in events]
        responses = [f.result(timeout=120) for f in futures]
    for response in responses:
        assert response.status_code == 200

    assert _wait(lambda: len(bot.waha.sent) >= 2)

    def first_round_of(turn_text: str) -> dict[str, Any] | None:
        for request in llm.requests:
            if turn_text in json.dumps(request.get("messages", [])):
                return request
        return None

    assert len(bot.waha.sent) == 2
    run_one = first_round_of("kai serial one")
    run_two = first_round_of("kai serial two")
    assert run_one is not None
    assert run_two is not None
    order = [run_one, run_two]
    order.sort(key=lambda r: llm.requests.index(r))
    earlier, later = order
    earlier_text = next(
        t
        for t in ("kai serial one", "kai serial two")
        if t in json.dumps(earlier.get("messages", []))
    )
    later_text = (
        "kai serial two" if earlier_text == "kai serial one" else "kai serial one"
    )
    earlier_dump = json.dumps(earlier.get("messages", []))
    later_dump = json.dumps(later.get("messages", []))
    assert later_text not in earlier_dump
    assert earlier_text in later_dump and "smoke reply one" in later_dump
