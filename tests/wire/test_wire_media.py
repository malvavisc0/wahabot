import base64
import time
import unittest.mock
from typing import Any

import httpx

from tests.conftest import Bot
from tests.harness import (
    CHAT_ID,
    FOREIGN_JID,
    SESSION,
    TTS_MP3,
    FenceRefusalResponse,
    FileResponse,
    VideoResponse,
    VoiceResponse,
    VoiceTextResponse,
    is_caption_request,
    smoke_video_bytes,
    tool_call_response,
    video_event,
    waha_event,
)
from tests.wire.support import (
    _wait,
    requires_ffmpeg,
)
from wahabot.commands import build_command_event
from wahabot.core.cli import build_forget_event
from wahabot.core.persistence import memory_file
from wahabot.core.runs import contexts as handlers_contexts


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
        "wahabot.handlers.media.fetch_transcript",
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
    fresh = waha_event("FRESH")
    fresh["payload"]["body"] = "kai fresh turn"
    bot.post(fresh)
    assert _wait(lambda: len(llm.requests) >= 1)
    fresh_turns = [
        str(m.get("content", ""))
        for m in llm.requests[-1]["messages"]
        if m.get("role") == "user"
    ]
    assert any("kai fresh turn" in t for t in fresh_turns), (
        "the fresh message must ride its own run's history"
    )


def test_operator_command(bot: Bot) -> None:
    llm = bot.stack.llm
    llm.override = tool_call_response(
        "send_message",
        {"text": "smoke reply one", "chat": CHAT_ID},
        call_id="call_cmd_1",
        response_id="chatcmpl-smoke-cmd",
        created=1788525834,
    )
    command = build_command_event(SESSION, f"send the plan to Family {CHAT_ID}")
    bot.post(command)
    assert _wait(lambda: len(llm.requests) >= 1)
    command_turns = [m for m in llm.requests[0]["messages"] if m.get("role") == "user"]
    assert any(
        "[operator command] send the plan to Family" in str(m.get("content", ""))
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
    command = build_command_event(SESSION, f"send a message to that guy {FOREIGN_JID}")
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
    quiet_event = waha_event("QUIET")
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


def test_send_video_wire(bot: Bot) -> None:
    llm = bot.stack.llm
    llm.override = VideoResponse
    video_event = waha_event()
    video_event["payload"]["id"] = f"false_{CHAT_ID}_SENDVIDEO"
    video_event["payload"]["body"] = "kai send me the clip"
    bot.post(video_event)
    assert _wait(lambda: len(bot.waha.sent_videos) >= 1)
    assert len(bot.waha.sent_videos) == 1
    v_session, v_chat, v_file, v_caption, v_convert = bot.waha.sent_videos[0]
    assert (
        v_session == SESSION
        and v_chat == CHAT_ID
        and v_file
        == {
            "mimetype": "video/mp4",
            "url": "http://files.invalid/q4/clip.mp4",
            "filename": "clip.mp4",
        }
        and v_caption == "the clip"
        and v_convert is True
    )


def test_send_voice_wire(bot: Bot) -> None:
    llm = bot.stack.llm
    llm.override = VoiceResponse
    voice_reply_event = waha_event()
    voice_reply_event["payload"]["id"] = f"false_{CHAT_ID}_SENDVOICE"
    voice_reply_event["payload"]["body"] = "kai answer this with a voice note"
    bot.post(voice_reply_event)
    assert _wait(lambda: len(bot.waha.sent_voices) >= 1)
    assert len(bot.waha.sent_voices) == 1
    v_session, v_chat, v_file, v_convert = bot.waha.sent_voices[0]
    assert (
        v_session == SESSION
        and v_chat == CHAT_ID
        and v_file
        == {
            "mimetype": "audio/mpeg",
            "url": "http://files.invalid/q5/note.mp3",
            "filename": "note.mp3",
        }
        and v_convert is True
    )
    assert len(bot.waha.sent) == 0  # the latch: no text rode along too


def test_send_voice_speaks_text(bot: Bot) -> None:
    """send_voice(text=…) synthesizes and delivers a playable voice note.

    The model writes text; the tool speaks it through the configured
    TTS voice (never a model-picked one) and the bytes ride the send
    as a VoiceBinaryFile without touching disk. Fail-soft contract: a
    synthesis failure degrades to an error envelope, no send — the
    model recovers in-run with plain text.
    """
    tts_bot = bot.rebuild(tts_url="http://tts.invalid")
    llm = tts_bot.stack.llm
    llm.override = VoiceTextResponse
    spoken: list[tuple[str, str]] = []

    def fake_synthesize(settings: Any, text: str, language: str) -> bytes:
        spoken.append((text, language))
        return TTS_MP3

    with unittest.mock.patch(
        "wahabot.ai.tools.whatsapp.media_send.synthesize", fake_synthesize
    ):
        voice_event = waha_event("VOICETEXT")
        voice_event["payload"]["body"] = "kai answer with your voice"
        tts_bot.post(voice_event)
    assert _wait(lambda: len(tts_bot.waha.sent_voices) >= 1)
    assert spoken == [("ya voy, un momento", "es")]
    v_session, v_chat, v_file, v_convert = tts_bot.waha.sent_voices[0]
    assert v_session == SESSION and v_chat == CHAT_ID and v_convert is True
    assert v_file["mimetype"] == "audio/mpeg"
    assert v_file["filename"] == "voice-note.mp3"  # required by the wire schema
    assert base64.b64decode(v_file["data"]) == TTS_MP3
    assert len(tts_bot.waha.sent) == 0  # the latch: no text rode along


def test_send_voice_text_requires_configured_tts(bot: Bot) -> None:
    """TTS off: the text form errors — no voice is sent, the run survives.

    The gate is the settings URL: the error envelope tells the model to
    fall back, and its text reply lands normally.
    """
    llm = bot.stack.llm
    # Pin the gate explicitly: pydantic-settings still reads the OS
    # environment even with ``_env_file=None``, so a developer shell
    # exporting a TTS URL would silently arm synthesis and ship a
    # voice — the exact opposite of what this test asserts.
    bot.settings.tts_url = ""
    llm.requests.clear()
    llm.override = VoiceTextResponse
    bot.post(waha_event("VOICETEXTOFF"))
    # The error envelope loops the run back; it survives, no voice is
    # synthesized, and the model's fallback text lands normally.
    assert _wait(lambda: len(bot.waha.sent) >= 1)
    assert bot.waha.sent_voices == []  # the gate: no synthesis, no voice send
    assert bot.waha.sent[0][2] == "smoke final answer"


def test_send_voice_text_synthesis_failure_degrades_to_text(bot: Bot) -> None:
    """A synthesis failure returns an error envelope; the run survives.

    Fail-soft: the tool must degrade to a text reply, never crash the
    run — the model sees the envelope and answers in writing.
    """
    tts_bot = bot.rebuild(tts_url="http://tts.invalid")
    llm = tts_bot.stack.llm
    llm.override = VoiceTextResponse
    with unittest.mock.patch(
        "wahabot.ai.tools.whatsapp.media_send.synthesize", return_value=None
    ):
        voice_event = waha_event("VOICETEXTFAIL")
        voice_event["payload"]["body"] = "kai answer with your voice"
        tts_bot.post(voice_event)
    assert _wait(lambda: len(llm.requests) >= 2)  # the envelope looped back
    assert tts_bot.waha.sent_voices == []
    # Fail-soft: the run survives and the model's fallback text still lands.
    assert _wait(lambda: len(tts_bot.waha.sent) >= 1)
    assert tts_bot.waha.sent[0][2] == "smoke final answer"


def test_send_voice_send_failure_degrades_to_text(bot: Bot) -> None:
    """A failed WAHA send returns an error envelope; the run survives.

    Fail-soft contract for the delivery hop itself: a WAHA HTTP error
    (the 500 the log shows) must not crash the tool — the model gets
    the fall-back envelope and answers in writing, and the delivery
    latch stays open so the text reply can land.
    """
    tts_bot = bot.rebuild(tts_url="http://tts.invalid")
    llm = tts_bot.stack.llm
    llm.override = VoiceTextResponse
    request = httpx.Request("POST", "http://waha.invalid/api/sendVoice")
    failure = unittest.mock.patch.object(
        tts_bot.waha,
        "send_voice",
        side_effect=httpx.HTTPStatusError(
            "Server error '500 Internal Server Error'",
            request=request,
            response=httpx.Response(500, request=request),
        ),
    )
    synthesis = unittest.mock.patch(
        "wahabot.ai.tools.whatsapp.media_send.synthesize", return_value=TTS_MP3
    )
    with failure, synthesis:
        voice_event = waha_event("VOICESENDFAIL")
        voice_event["payload"]["body"] = "kai answer with your voice"
        tts_bot.post(voice_event)
    assert _wait(lambda: len(llm.requests) >= 2)  # the envelope looped back
    assert tts_bot.waha.sent_voices == []  # the gate: no voice landed
    # Fail-soft: the run survives and the model's fallback text still lands.
    assert _wait(lambda: len(tts_bot.waha.sent) >= 1)
    assert tts_bot.waha.sent[0][2] == "smoke final answer"
