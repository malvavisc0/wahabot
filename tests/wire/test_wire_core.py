import asyncio
import json
import time
import unittest.mock
from typing import Any

import pytest

from tests.conftest import Bot
from tests.harness import (
    CHAT_ID,
    ME_JID,
    OWN_LID,
    SESSION,
    chat_event,
    image_event,
    image_part_count,
    is_caption_request,
    make_operator_typed,
    sign,
    tool_call_response,
    voice_event,
    waha_event,
)
from tests.wire.support import (
    _wait,
    user_turns_of,
)
from wahabot.core.notebook import SESSION_NOTEBOOK_CHAT, Notebook
from wahabot.core.runs import contexts as handlers_contexts


def test_agent_replies_via_send_message(bot: Bot) -> None:
    llm = bot.stack.llm
    bot.post(waha_event())
    assert bot.waha.sent == [(SESSION, CHAT_ID, "smoke reply one", None)]
    assert len(llm.requests) == 2
    body = llm.requests[0]
    assert body.get("top_p") == 0.95 and body.get("top_k") == 20
    assert body.get("tools") is not None
    assert body.get("parallel_tool_calls") is True


def test_wrap_up_note_never_rides_the_next_prompt(bot: Bot) -> None:
    """A delivered run's private note is a notebook fact, not a chat turn.

    Before the fix the note was stored as an assistant turn and merged
    into the delivered reply, so the next run read ``reply + note`` as
    its own style. The note may reach the
    next prompt only inside the notebook (the system message), never as
    a user or assistant message.
    """
    llm = bot.stack.llm
    chat = "5551112222-5551112222@g.us"
    bot.post(chat_event(body="kai, please help with the form", mid="WN1", chat=chat))
    # A delivered run: the tool round, then the wrap-up round whose final
    # text is the private note.
    assert _wait(lambda: len(llm.requests) >= 2)
    note = "smoke final answer"
    llm.requests.clear()
    bot.waha.sent.clear()

    bot.post(chat_event(body="kai, any update?", mid="WN2", chat=chat))
    assert _wait(lambda: len(llm.requests) >= 1)
    messages = llm.requests[0]["messages"]
    assert not any(
        note in str(m.get("content", ""))
        for m in messages
        if m.get("role") in ("user", "assistant")
    )
    # The delivered reply is still the model's own turn.
    assert any(
        "smoke reply one" in str(m.get("content", ""))
        for m in messages
        if m.get("role") == "assistant"
    )


def test_pinned_state_block_rides_the_system_message(bot: Bot) -> None:
    """Notebook facts, policies, and rules ride the system turn, never a chat turn.

    The pinned block is assembled by code at the start of the run and
    injected system-side, so history trimming can never evict it and no
    user or assistant turn ever carries it.
    """
    llm = bot.stack.llm
    notebook = Notebook(bot.settings.data_dir)
    notebook.write(SESSION, CHAT_ID, "customer-email", "ana@example.com")
    notebook.write(SESSION, SESSION_NOTEBOOK_CHAT, "reply-style", "Use concise replies.")
    llm.requests.clear()
    bot.post(waha_event("PIN1"))

    def pinned_request() -> dict[str, Any] | None:
        """This run's request, found by its pinned-state marker.

        The fake LLM is session-scoped, and the post-reply memory jobs
        run in the background: a request from an earlier test can still
        land here, so ``requests[0]`` is not necessarily this run's.
        """
        for request in llm.requests:
            for message in request.get("messages", []):
                if message.get("role") == "system" and "# Pinned State" in str(
                    message.get("content", "")
                ):
                    return request
        return None

    assert _wait(lambda: pinned_request() is not None)
    request = pinned_request()
    assert request is not None
    messages = request["messages"]
    system = [str(m.get("content", "")) for m in messages if m.get("role") == "system"]
    assert system, "no system message in the run's request"
    content = system[0]
    assert "# Pinned State" in content
    assert "## Notebook facts\n\n- customer-email: ana@example.com" in content
    assert "## Operator policies\n\n- reply-style: Use concise replies." in content
    assert "## Session rules" in content
    assert "Timezone: UTC" in content
    assert not any(
        "# Pinned State" in str(m.get("content", ""))
        for m in messages
        if m.get("role") in ("user", "assistant")
    )
    assert not any(
        "ana@example.com" in str(m.get("content", ""))
        for m in messages
        if m.get("role") in ("user", "assistant")
    )


def test_final_text_reply_carries_resolved_mentions(bot: Bot) -> None:
    """A final-text reply (no tool call) still tags roster members.

    The model answered in plain text instead of calling ``send_message``
    — exactly the production incident: the reply contained a bare
    ``@<number>`` token and WhatsApp rendered it literally because no
    mention JID rode the send. The handler fallback must resolve the
    token against the chat roster and attach the mention.
    """
    llm = bot.stack.llm
    llm.override = {
        "id": "chatcmpl-smoke-mention",
        "object": "chat.completion",
        "created": 1788525840,
        "model": "smoke-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "listo @491555000001, la captura salió vacía",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    try:
        bot.post(waha_event())
        assert _wait(lambda: len(bot.waha.sent) >= 1)
        session_id, chat_id, text, mentions = bot.waha.sent[-1]
        assert (session_id, chat_id) == (SESSION, CHAT_ID)
        assert "listo @491555000001" in text
        # The roster's canonical JID for the tokened participant rides
        # the send as a real mention.
        assert mentions == ["491555000001@c.us"]
    finally:
        llm.override = None


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


@pytest.mark.parametrize("body", [b"{ not json", b"\xff", b"{}"])
def test_malformed_event_rejected(bot: Bot, body: bytes) -> None:
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
        "wahabot.handlers.burst_mixin.transcribe_voice_note",
        return_value="hello from voice",
    ):
        bot.post(voice_event())
    assert _wait(lambda: len(llm.requests) >= 1)
    voiced_turns = [m for m in llm.requests[0]["messages"] if m.get("role") == "user"]
    assert any(
        '[audio: "hello from voice"]' in str(m.get("content", "")) for m in voiced_turns
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
        "wahabot.handlers.burst_mixin.transcribe_voice_note",
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
    # No default chat in an operator run: the chat-less send_message is
    # refused, never posted to the synthetic "operator" id.
    assert _wait(lambda: len(bot.waha.sent) >= 1)
    assert all(sent[1] != "operator" for sent in bot.waha.sent)


def test_voice_note_in_mentioned_group_skipped(bot: Bot) -> None:
    group_voice = voice_event()
    group_voice["payload"]["id"] = f"false_{CHAT_ID}_GROUPVOICE"
    group_voice["payload"]["from"] = CHAT_ID
    group_voice["payload"]["participant"] = "491555000001@c.us"
    transcriber = unittest.mock.Mock(
        side_effect=AssertionError("must not transcribe in mentioned group")
    )
    with unittest.mock.patch(
        "wahabot.handlers.burst_mixin.transcribe_voice_note", transcriber
    ):
        bot.post(group_voice)
    transcriber.assert_not_called()
    assert (SESSION, CHAT_ID) in bot.waha.seen_chats


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
    silent_event = waha_event("SILENT1")
    silent_event["payload"]["body"] = "kai mira el partido de anoche"
    bot.post(silent_event)
    assert _wait(lambda: len(llm.requests) >= 1)
    assert bot.waha.sent == []

    # Message 2: a fresh run must see message 1 in its history.
    llm.clear()
    bot.waha.sent.clear()
    llm.override = None
    follow_up = waha_event("FOLLOW1")
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
    assert any(
        "el partido de anoche" in line and " — #m" in line
        for turn in follow_up_turns
        for line in turn.splitlines()
    ), "the silent run must be a ledger line"
    assert any("viste el partido" in turn for turn in follow_up_turns), (
        "the new message itself must ride its own run"
    )


def test_redelivery_does_not_duplicate_message(bot: Bot) -> None:
    """A WAHA redelivery of a crashed run never duplicates the message.

    The handler drops the seen marker on failure so WAHA retries. The
    first attempt's turn survives the repair (an earlier completed run
    stamped the merged turn it rides), so a plain re-append would copy
    the message once per retry. ``already_in_buffer`` keys on the
    serialized id: the redelivered event's id is already in the newest
    user turn, so the run proceeds without appending again — the agent
    still gets its fresh run. Posting the same event twice stands in
    for the crash+redelivery pair (identical wire shape; the second
    post is exactly what WAHA re-sends).
    """
    llm = bot.stack.llm
    # Message 1 completes and is stamped.
    bot.post(waha_event())
    assert _wait(lambda: len(llm.requests) >= 1)
    llm.clear()
    bot.waha.sent.clear()

    # Message 2 arrives but its run "crashes": simulate the marker
    # drop by entering the turn directly — then WAHA redelivers.
    probe = chat_event(body="kai redelivery probe", mid="REDEL1")
    bot.post(probe)
    assert _wait(lambda: len(llm.requests) >= 1)
    turns_after_first = user_turns_of(llm.requests[-1])
    assert any("redelivery probe" in t for t in turns_after_first)

    # Redelivery: same wire event, the seen marker was dropped by the
    # failed run (the handler's except path forgets it).
    from wahabot.handlers import forget_seen

    forget_seen(f"false_{CHAT_ID}_REDEL1")
    bot.post(probe)
    assert _wait(lambda: len(bot.waha.sent) >= 1)

    turns_after_redelivery = user_turns_of(llm.requests[-1])
    probe_turn = next((t for t in turns_after_redelivery if "redelivery probe" in t), "")
    assert probe_turn.count("redelivery probe") == 1, (
        f"the redelivered message must appear exactly once, got: {probe_turn!r}"
    )
    assert bot.waha.sent, "the redelivery run must still deliver its reply"


def test_fromMe_own_message_folded(bot: Bot) -> None:
    """An operator-typed message is remembered as the operator's, not the bot's.

    Real WAHA shape: ``from`` is the account, ``to`` the chat,
    ``source="app"``. Before the fix the whitelist checked ``from`` and
    dropped every such message, so the bot never saw its operator.
    """
    llm = bot.stack.llm
    bot.post(waha_event())
    llm.requests.clear()
    bot.waha.sent.clear()
    own = make_operator_typed(waha_event("OWNMSG"), CHAT_ID)
    own["id"] = "evt-smoke-own"
    own["payload"]["id"] = f"true_{CHAT_ID}_OWNMSG_{OWN_LID}"
    own["payload"]["body"] = "operator typed this from the app"
    bot.post(own)
    assert len(llm.requests) == 0
    assert len(bot.waha.sent) == 0

    async def remembered() -> list[tuple[str, str]]:
        ctx = handlers_contexts[(SESSION, CHAT_ID)]
        memory = await ctx.store.get("memory")
        messages = await memory.aget_all()
        return [(str(m.role), str(m.content)) for m in messages]

    turns = asyncio.run(remembered())
    folded = [c for r, c in turns if "operator typed this from the app" in c]
    assert folded, turns
    # The fold is a ledger line, not a live record.
    assert not folded[0].startswith("[[") and "<operator>" in folded[0]
    assert "operator typed this from the app" in folded[0]
    assert " — #m" in folded[0]
    # A teammate's words are never stored as the bot's own.
    assert all(
        r != "MessageRole.ASSISTANT"
        for r, c in turns
        if "operator typed this from the app" in c
    )
    from wahabot.core.identity import authors

    assert authors.author(own["payload"]["id"]) == "operator"


def test_bot_api_echo_not_folded(bot: Bot) -> None:
    """The echo of the bot's own send (source=api) is never re-remembered."""
    llm = bot.stack.llm
    bot.post(waha_event())
    llm.requests.clear()
    echo = make_operator_typed(waha_event("APIECHO"), CHAT_ID)
    echo["payload"]["source"] = "api"
    echo["payload"]["id"] = f"true_{CHAT_ID}_APIECHO_{OWN_LID}"
    echo["payload"]["body"] = "kai says something with its own name"
    bot.post(echo)
    time.sleep(0.3)
    assert len(llm.requests) == 0

    async def contents() -> list[str]:
        ctx = handlers_contexts[(SESSION, CHAT_ID)]
        memory = await ctx.store.get("memory")
        return [str(m.content) for m in await memory.aget_all()]

    assert not any("its own name" in c for c in asyncio.run(contents()))
    from wahabot.core.identity import authors

    assert authors.author(echo["payload"]["id"]) == "bot"


def test_operator_mention_wakes_bot(bot: Bot) -> None:
    """The operator naming the bot in a group runs it, tagged as the operator."""
    llm = bot.stack.llm
    bot.post(waha_event())
    llm.requests.clear()
    bot.waha.sent.clear()
    own = make_operator_typed(waha_event("OPWAKE"), CHAT_ID)
    own["payload"]["id"] = f"true_{CHAT_ID}_OPWAKE_{OWN_LID}"
    own["payload"]["body"] = "kai, resume el grupo"
    bot.post(own)
    assert _wait(lambda: len(llm.requests) >= 1)
    users = [
        str(m.get("content", ""))
        for m in llm.requests[0]["messages"]
        if m.get("role") == "user"
    ]
    turn = next(t for t in users if "resume el grupo" in t)
    assert turn.startswith("[turn: 1 message | newest #m")
    assert "<operator>" in turn and ":: kai, resume el grupo" in turn
    assert "491555000000" not in turn
    # Same chat fence as any member's turn: the reply lands in this group.
    assert _wait(lambda: len(bot.waha.sent) >= 1)
    assert bot.waha.sent[0][1] == CHAT_ID
