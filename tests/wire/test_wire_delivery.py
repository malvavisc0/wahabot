import asyncio
import base64
import tempfile
import time
import unittest.mock
from pathlib import Path
from typing import Any

import httpx
from llama_index.core.base.llms.types import MessageRole

from tests.conftest import Bot
from tests.harness import (
    CHAT_ID,
    ME_JID,
    SESSION,
    SMOKE_PNG,
    FirstResponse,
    SecondResponse,
    StickerResponse,
    VoiceResponse,
    tool_call_response,
    waha_event,
)
from tests.wire.support import (
    _run_failing_delivery_case,
    _wait,
)
from wahabot.commands import build_command_event
from wahabot.core.runs import contexts as handlers_contexts
from wahabot.handlers import seen_recently
from wahabot.status import state as status_state


def test_send_sticker_wire(bot: Bot) -> None:
    llm = bot.stack.llm
    llm.override = StickerResponse
    sticker_reply_event = waha_event()
    sticker_reply_event["payload"]["id"] = f"false_{CHAT_ID}_SENDSTICKER"
    sticker_reply_event["payload"]["body"] = "kai reply with a sticker"
    bot.post(sticker_reply_event)
    assert _wait(lambda: len(bot.waha.sent_stickers) >= 1)
    assert len(bot.waha.sent_stickers) == 1
    s_session, s_chat, s_file = bot.waha.sent_stickers[0]
    assert (
        s_session == SESSION
        and s_chat == CHAT_ID
        and s_file
        == {
            "mimetype": "image/webp",
            "url": "http://files.invalid/q6/laugh.webp",
            "filename": "laugh.webp",
        }
    )
    assert len(bot.waha.sent) == 0


def test_send_media_local_image_replay(bot: Bot) -> None:
    """The incident's exact failing call, replayed over the full stack.

    Run 2c665d8's send_image(path=…) crashed on a missing parameter;
    the merged send_media must deliver the local PNG as a proper image
    (bug 1's fix, keyed through kind=image), with the correct MIME
    stamped from the extension — not application/octet-stream.
    """
    with tempfile.TemporaryDirectory(prefix="meme-") as tmpdir:
        meme = Path(tmpdir) / "millennial.png"
        meme.write_bytes(SMOKE_PNG)
        llm = bot.stack.llm
        llm.override = tool_call_response(
            "send_media",
            {"kind": "image", "path": str(meme), "reason": "the meme request"},
            call_id="call_meme_1",
        )
        meme_event = waha_event("MEMEREPLAY")
        meme_event["payload"]["body"] = "kai make me a millennial starter pack meme"
        bot.post(meme_event)
        assert _wait(lambda: len(bot.waha.sent_images) >= 1)
        assert len(bot.waha.sent_images) == 1
        i_session, i_chat, i_file, _i_caption = bot.waha.sent_images[0]
        assert i_session == SESSION and i_chat == CHAT_ID
        assert i_file["mimetype"] == "image/png"  # re-typed, not octet-stream
        assert i_file["filename"] == "millennial.png"
        assert base64.b64decode(i_file["data"]) == SMOKE_PNG
        assert len(bot.waha.sent) == 0  # the image was the delivery
        assert len(bot.waha.sent_stickers) == 0  # not the sticker detour


def test_send_media_delivery_latches_across_kinds(bot: Bot) -> None:
    """One delivery per run holds across the merged kinds and tools.

    The latch used to be per-tool-name; the merge must not loosen it —
    after send_media succeeds, a follow-up send_message in the same
    run gets the already-sent refusal envelope, never a second send.
    """
    llm = bot.stack.llm
    llm.override = tool_call_response(
        "send_media",
        {"kind": "file", "url": "http://files.invalid/q3/report.pdf", "reason": "r"},
        call_id="call_latch_1",
    )
    latch_event = waha_event("LATCHCHECK")
    latch_event["payload"]["body"] = "kai send me the report"
    bot.post(latch_event)
    assert _wait(lambda: len(bot.waha.sent_files) >= 1)
    # The tool registry is the merged 7b set: no per-media senders left.
    tools = {t.metadata.get_name() for t in bot.agent.tools}
    assert "send_media" in tools and "read_chat" in tools
    old_senders = {"send_image", "send_video", "send_file", "send_voice", "send_sticker"}
    assert not tools & old_senders
    # The latch: no text send rode along with the delivered file.
    assert bot.waha.sent == []


def test_delivery_send_failures_degrade_to_text(bot: Bot) -> None:
    """Every delivery tool survives a failed WAHA send.

    The fail-soft contract is a property of the whole delivery set:
    whichever tool the model picked, a WAHA error (the 500 the log
    shows) must degrade to the fall-back envelope — the model then
    answers in writing — never crash the run. One representative send
    per tool; the mock fails the first call only, so the fallback
    text's own send lands and proves the latch stayed open.
    """
    cases: list[tuple[str, str, dict[str, Any], str | None]] = [
        ("send_message", "send_text", {"text": "tool text that must not land"}, None),
        (
            "send_media",
            "send_image",
            {"kind": "image", "url": "https://x.invalid/a.png"},
            None,
        ),
        (
            "send_media",
            "send_video",
            {"kind": "video", "url": "https://x.invalid/a.mp4"},
            "sent_videos",
        ),
        (
            "send_media",
            "send_file",
            {"kind": "file", "url": "https://x.invalid/a.pdf"},
            "sent_files",
        ),
        (
            "send_media",
            "send_sticker",
            {"kind": "sticker", "url": "https://x.invalid/a.webp"},
            "sent_stickers",
        ),
        (
            "send_media",
            "send_voice",
            {"kind": "voice", "url": "https://x.invalid/a.mp3"},
            "sent_voices",
        ),
        (
            "forward_message",
            "forward_message",
            {"message_id": f"false_{CHAT_ID}_SMOKEREPLY"},
            None,
        ),
        (
            "react_to_message",
            "send_reaction",
            {"message_id": f"false_{CHAT_ID}_SMOKEREPLY", "reaction": "👍"},
            "reactions",
        ),
    ]
    for tool, waha_method, args, record in cases:
        _run_failing_delivery_case(bot, tool, waha_method, args, record)


def test_voice_delivery_collapses_in_memory(bot: Bot) -> None:
    """A voice send's tool group folds into a plain assistant turn.

    ``send_voice``/``send_sticker``/``send_file`` are delivery tools:
    after the send, memory must mirror the chat — one plain assistant
    message, no raw tool-call scaffolding left for the next run.
    """
    llm = bot.stack.llm
    llm.override = VoiceResponse
    voice_reply_event = waha_event("VOICECOLLAPSE")
    voice_reply_event["payload"]["body"] = "kai answer this with a voice note"
    bot.post(voice_reply_event)
    assert _wait(lambda: len(bot.waha.sent_voices) >= 1)

    async def memory_contents() -> list[Any]:
        ctx = handlers_contexts[(SESSION, CHAT_ID)]
        memory = await ctx.store.get("memory")
        return await memory.aget_all()

    messages = asyncio.run(memory_contents())
    assert any(
        m.role == MessageRole.ASSISTANT and "[voice note: audio/mpeg]" in str(m.content)
        for m in messages
    ), [str(m.content) for m in messages]
    assert not any(m.role == MessageRole.TOOL for m in messages), (
        "delivery scaffolding must not survive the run"
    )


def test_seen_and_typing_presence(bot: Bot) -> None:
    """An addressed message is marked seen; a reply types before it lands.

    The seen receipt goes out for *their* message regardless of the
    run's outcome. FirstResponse delivers via the send_message tool,
    so this also proves the typing prelude rides the *tool* path —
    both delivery paths must type alike. The indicator is left on;
    the send itself clears it.
    """
    typing_bot = bot.rebuild(typing_presence_min_s=0.05, typing_presence_max_s=0.1)
    llm = typing_bot.stack.llm
    llm.override = FirstResponse  # send_message tool delivery
    seen_event = waha_event()
    seen_event["payload"]["id"] = f"false_{CHAT_ID}_SEEN"
    seen_event["payload"]["body"] = "kai que tal"
    typing_bot.post(seen_event)
    assert _wait(lambda: len(typing_bot.waha.sent) >= 1)
    assert typing_bot.waha.seen_chats == [(SESSION, CHAT_ID)]
    assert typing_bot.waha.typing_calls == [(SESSION, CHAT_ID, True)]


def test_seen_off_when_disabled(bot: Bot) -> None:
    quiet_bot = bot.rebuild(send_seen=False, typing_presence_min_s=0.0)
    llm = quiet_bot.stack.llm
    llm.override = FirstResponse
    quiet_bot.post(waha_event())
    assert _wait(lambda: len(quiet_bot.waha.sent) >= 1)
    assert quiet_bot.waha.seen_chats == []
    assert quiet_bot.waha.typing_calls == []


def test_final_text_fallback_types_before_sending(bot: Bot) -> None:
    """The handler's final-text fallback types like the tool path does.

    A model answering in plain text (no ``send_message`` call) is the
    other delivery path; without the pause there it goes out
    sub-second — the machine tell this feature exists to kill. The
    indicator is left ON for the send to clear.
    """
    typing_bot = bot.rebuild(typing_presence_min_s=0.05, typing_presence_max_s=0.1)
    llm = typing_bot.stack.llm
    llm.override = SecondResponse  # plain final text, no tool call
    typing_bot.post(waha_event())
    assert _wait(lambda: len(typing_bot.waha.sent) >= 1)
    assert typing_bot.waha.typing_calls == [(SESSION, CHAT_ID, True)]
    assert typing_bot.waha.sent[0][2] == "smoke final answer"


def test_typing_cleared_when_fallback_send_fails(bot: Bot) -> None:
    """A failed fallback send clears the typing indicator it lit.

    The send itself clears the indicator on success; when the send
    raises, nothing lands and nothing clears it — the handler's
    generic failure path must not strand "typing…" on forever.
    """
    typing_bot = bot.rebuild(typing_presence_min_s=0.05, typing_presence_max_s=0.1)
    llm = typing_bot.stack.llm
    llm.override = SecondResponse
    with unittest.mock.patch.object(
        type(typing_bot.waha), "send_text", side_effect=httpx.ConnectError("waha gone")
    ) as broken_send:
        typing_bot.post(waha_event())
    assert _wait(lambda: broken_send.call_count >= 1)
    assert _wait(
        lambda: (
            typing_bot.waha.typing_calls
            == [(SESSION, CHAT_ID, True), (SESSION, CHAT_ID, False)]
        )
    )


def test_silent_run_still_marks_seen(bot: Bot) -> None:
    """A run that ends silent still read the message — seen goes out.

    The receipt answers *their* message, not the bot's reply; a
    ``stay_silent`` outcome is the headline case (the bot "saw it and
    chose not to speak", not "never read it").
    """
    llm = bot.stack.llm
    llm.override = tool_call_response(
        "stay_silent",
        {"reason": "banter between others"},
        call_id="call_seen_silent",
        response_id="chatcmpl-seen-silent",
    )
    silent_event = waha_event("SEENSILENT")
    bot.post(silent_event)
    assert _wait(lambda: bot.waha.seen_chats == [(SESSION, CHAT_ID)])
    assert bot.waha.sent == []


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
        return [str(m.content) for m in messages if "reacted" in str(m.content)]

    notes = asyncio.run(reaction_notes())
    assert any("👍" in note and 'to: you "smoke reply one"' in note for note in notes)
    assert any("Smoke Sender" in note and "491555000001" not in note for note in notes)

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
    # WAHA's API must agree the session is down from the start: the
    # FAILED event arms the recovery poller, whose first probe runs
    # immediately and would otherwise see WORKING and unmute.
    bot.waha.session_status = "FAILED"
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

    # WAHA's API still agrees the session is down: a muted message
    # probes recovery, and a WORKING answer would unmute mid-test.
    bot.waha.session_status = "FAILED"
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
    bot.waha.session_status = "WORKING"
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


def test_muted_message_probes_recovery(bot: Bot) -> None:
    """A message arriving while muted probes WAHA once — no poller needed.

    Webhook delivery is itself evidence the session may be back; the
    probe (GET /api/sessions) decides. Here the API says WORKING while
    no status event ever arrives — the message-triggered probe alone
    unmutes the bot and notifies.
    """
    from wahabot.status import session_healthy

    bot.waha.session_status = "FAILED"  # the poller's first probe must miss
    bot.post(
        {
            "id": "evt-probe-down",
            "timestamp": int(time.time()),
            "event": "session.status",
            "session": SESSION,
            "me": None,
            "payload": {"name": SESSION, "status": "FAILED"},
        }
    )
    assert _wait(lambda: not session_healthy())
    # The status event went missing; WAHA's API now answers WORKING.
    bot.waha.session_status = "WORKING"
    bot.post(waha_event("MUTEDPROBE"))
    assert _wait(lambda: session_healthy(), timeout=5)
    assert _wait(
        lambda: any(
            entry[1] == ME_JID and "WhatsApp session recovered" in entry[2]
            for entry in bot.waha.sent
        ),
        timeout=5,
    )


def test_muted_command_probes_recovery(bot: Bot) -> None:
    """A command arriving while muted probes WAHA, like a message does.

    Commands carry no seen marker, so the mute branch has nothing to
    drop — but it must still run the same recovery probe as the
    message path, or an operator command during a silent-WAHA hang
    would be dropped where a chat message would have unmuted the bot.
    """
    from wahabot.status import session_healthy

    bot.waha.session_status = "FAILED"  # the poller's first probe must miss
    bot.post(
        {
            "id": "evt-cmd-probe-down",
            "timestamp": int(time.time()),
            "event": "session.status",
            "session": SESSION,
            "me": None,
            "payload": {"name": SESSION, "status": "FAILED"},
        }
    )
    assert _wait(lambda: not session_healthy())
    # The status event went missing; WAHA's API now answers WORKING.
    bot.waha.session_status = "WORKING"
    bot.post(build_command_event(SESSION, "kai are you there"))
    assert _wait(lambda: session_healthy(), timeout=5)
    assert _wait(
        lambda: any(
            entry[1] == ME_JID and "WhatsApp session recovered" in entry[2]
            for entry in bot.waha.sent
        ),
        timeout=5,
    )
