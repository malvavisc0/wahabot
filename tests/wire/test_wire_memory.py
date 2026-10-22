import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from tests.conftest import Bot
from tests.harness import (
    CHAT_ID,
    ME_JID,
    SESSION,
    EscalateResponse,
    FirstResponse,
    RecordingWaha,
    SecondResponse,
    album_events,
    chat_event,
    tool_call_response,
    waha_event,
)
from tests.wire.support import (
    _audit_entries,
    _wait,
    user_turns_of,
)
from wahabot.ai.history import ContextBudgetError
from wahabot.ai.tools.whatsapp import (
    EscalationChannel,
    bind_target,
    reset_target,
)
from wahabot.ai.tools.whatsapp import escalate as build_escalate
from wahabot.core.runs import contexts as handlers_contexts
from wahabot.handlers import seen_recently
from wahabot.status import state as status_state


def test_self_chat_command_over_budget_reports_once_without_retry(
    bot: Bot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A command that can never fit the budget reports and never retries.

    The failure must be visible in all the places the operator looks:
    its own audit kind (not borrowed silence), the self-chat reply on
    their phone, and no seen-marker drop — a redelivery would rebuild
    the same oversized turn and loop forever.
    """

    def self_event(body: str, mid: str) -> dict[str, Any]:
        ev = waha_event(mid)
        ev["id"] = f"evt-smoke-self-{mid}"
        ev["payload"]["id"] = f"true_{ME_JID}_{mid}"
        ev["payload"]["_data"]["id"] = {"_serialized": f"true_{ME_JID}_{mid}"}
        ev["payload"]["from"] = ME_JID
        ev["payload"]["fromMe"] = True
        ev["payload"]["body"] = body
        return ev

    llm = bot.stack.llm
    llm.clear()
    bot.waha.sent.clear()
    agent = bot.agent
    real_history = agent.chat_history

    async def cramped_history(ctx: Any) -> list[Any]:
        raise ContextBudgetError("operator history exceeds the budget")

    monkeypatch.setattr(agent, "chat_history", cramped_history)
    bot.post(self_event("kai summarize this enormous transcript", "SELFOVER"))

    # The failure reply lands in the self-chat, quote-replying nothing.
    assert _wait(lambda: bot.waha.sent != [])
    assert all(chat == ME_JID for _, chat, _, _ in bot.waha.sent)
    assert "Command rejected" in bot.waha.sent[0][2]
    # Its own audit kind — the durable record, not operator_silence.
    assert _wait(lambda: _audit_entries(bot, "operator_budget_error", "operator"))
    assert _audit_entries(bot, "operator_silence", "operator") == []
    # No retry: the seen marker held, so a redelivery is deduplicated.
    assert not llm.requests

    # Recovering the budget restores normal commands end to end.
    monkeypatch.setattr(agent, "chat_history", real_history)
    seen_id = f"true_{ME_JID}_SELFOVER"

    assert seen_recently(seen_id) is True  # still marked: dedup window
    monkeypatch.setattr(agent, "chat_history", cramped_history)
    bot.post(self_event("kai summarize this enormous transcript", "SELFOVER"))
    time.sleep(0.5)
    assert len(bot.waha.sent) == 1  # deduplicated, no second failure reply
    monkeypatch.setattr(agent, "chat_history", real_history)


def test_escalate_delivery(bot: Bot) -> None:
    llm = bot.stack.llm
    llm.override = EscalateResponse
    PC = "5553333333-5553333333@g.us"
    pc_event = waha_event("ESC1")
    pc_event["payload"]["id"] = f"false_{PC}_ESC1"
    pc_event["payload"]["_data"]["id"] = {"_serialized": f"false_{PC}_ESC1"}
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
    # The confirmed escalation left a durable record in the audit
    # journal — the row the CLI's `escalations` command reads.
    assert _wait(lambda: _audit_entries(bot, "escalation", PC) != [])
    entries = _audit_entries(bot, "escalation", PC)
    assert entries[0]["status"] == "open"
    assert "human" in entries[0]["report"]


def test_escalate_cooldown(bot: Bot) -> None:
    llm = bot.stack.llm
    llm.override = EscalateResponse
    # Trigger the first escalation so the channel is stamped.
    PC = "5553333333-5553333333@g.us"
    pc_event = waha_event("ESC1")
    pc_event["payload"]["id"] = f"false_{PC}_ESC1"
    pc_event["payload"]["_data"]["id"] = {"_serialized": f"false_{PC}_ESC1"}
    pc_event["payload"]["from"] = PC
    pc_event["payload"]["body"] = "kai I want to speak to a human"
    bot.post(pc_event)
    assert _wait(lambda: len(llm.requests) >= 2)
    self_sent_before = sum(1 for _, chat, _, _ in bot.waha.sent if chat == ME_JID)
    llm.requests.clear()
    # Second escalate: cooldown refuses, no new self-chat delivery.
    pc_event2 = waha_event("ESC2")
    pc_event2["payload"]["id"] = f"false_{PC}_ESC2"
    pc_event2["payload"]["_data"]["id"] = {"_serialized": f"false_{PC}_ESC2"}
    pc_event2["payload"]["from"] = PC
    pc_event2["payload"]["body"] = "kai please escalate again"
    bot.post(pc_event2)
    assert _wait(lambda: len(llm.requests) >= 2)
    self_sent_after = sum(1 for _, chat, _, _ in bot.waha.sent if chat == ME_JID)
    assert self_sent_after == self_sent_before
    cooldown_feedback = str(llm.requests[1]["messages"] if len(llm.requests) > 1 else [])
    assert "cooldown" in cooldown_feedback
    assert "an earlier report from this chat was forwarded" in cooldown_feedback
    assert "this new report was not forwarded" in cooldown_feedback


def test_journal_covers_every_path(bot: Bot) -> None:
    """Album, silence and command decisions all reach the audit journal.

    Every decision path: an album reply (previously inline,
    unjournaled), a tool-delivered run, and an operator
    command's reply each leave their audit row.
    """
    # Album reply via the shared finish path: SecondResponse makes the
    # final text the answer, quote-replied to the album container.
    album_bot = bot.rebuild(typing_presence_min_s=0.0, typing_presence_max_s=0.0)
    album_bot.stack.llm.override = SecondResponse
    for album_event in album_events():
        album_bot.post(album_event)
    assert _wait(lambda: len(album_bot.waha.sent) >= 1)
    assert _wait(lambda: _audit_entries(album_bot, "reply", CHAT_ID) != [])
    reply_row = _audit_entries(album_bot, "reply", CHAT_ID)[0]
    assert reply_row["outbound_id"].startswith("true_")
    assert reply_row["reply_to"] == reply_row["message_id"]
    assert reply_row["duration_ms"] >= 0
    assert reply_row["message_ids"]
    # The album path journals the same kinds as the single path —
    # a delivered_via_tool album run must not leave a "reply" row too.
    assert _audit_entries(album_bot, "silence", CHAT_ID) == []

    # Operator command reply journaled as its own kind.
    cmd_bot = album_bot.rebuild()
    cmd_bot.stack.llm.clear()
    cmd_bot.stack.llm.override = SecondResponse
    self_event = waha_event("CMDFINAL")
    self_event["payload"]["id"] = f"true_{ME_JID}_CMDFINAL"
    self_event["payload"]["_data"]["id"] = {"_serialized": f"true_{ME_JID}_CMDFINAL"}
    self_event["payload"]["from"] = ME_JID
    self_event["payload"]["fromMe"] = True
    self_event["payload"]["body"] = "kai just answer me"
    cmd_bot.post(self_event)
    assert _wait(lambda: _audit_entries(cmd_bot, "operator_reply", "operator") != [])


def test_journal_records_send_failure_not_fake_reply(bot: Bot) -> None:
    """A failed final-reply send leaves a send_failed row, not a reply row.

    The audit record follows the confirmed send (the escalation
    standard): a WAHA failure during the final-text send must not
    journal a "reply" the chat never saw.
    """
    failing_bot = bot.rebuild()
    failing_bot.stack.llm.override = SecondResponse

    def boom(*_args: object, **_kwargs: object) -> str:
        raise httpx.HTTPError("WAHA is down")

    failing_bot.waha.send_text = boom  # type: ignore[method-assign]
    failing_bot.post(waha_event())
    assert _wait(lambda: _audit_entries(failing_bot, "send_failed", CHAT_ID) != [])
    assert _audit_entries(failing_bot, "reply", CHAT_ID) == []


def test_escalate_fail_soft(tmp_path: Path) -> None:
    from wahabot.settings import Settings

    ME_JID = "491555000000@c.us"
    status_state.operator_jid = ME_JID
    failing_waha = RecordingWaha()

    def boom(*_args: object, **_kwargs: object) -> str:
        raise httpx.HTTPError("WAHA is down")

    failing_waha.send_text = boom  # type: ignore[method-assign]
    down_channel = EscalationChannel()
    assert down_channel.operator_jid == ME_JID

    # The send fails before any audit write, so a throwaway settings
    # (data_dir under tmp_path) satisfies the now-required parameter.
    settings = Settings(
        webhook_hmac_key="k",
        waha_url="http://waha.invalid",
        waha_api_key="k",
        llm_api_base="http://llm.invalid",
        llm_api_key="k",
        data_dir=tmp_path,
        _env_file=None,
    )
    esc_tool = build_escalate(failing_waha, down_channel, settings)
    esc_fn = cast(Any, esc_tool).fn
    PC = "5553333333-5553333333@g.us"
    token = bind_target({"session": SESSION, "chat_id": PC, "sent": "", "reacted": ""})
    try:
        outcome = str(esc_fn(report="help"))
    finally:
        reset_target(token)
    assert '"ok": false' in outcome and "delivery was not confirmed" in outcome
    assert down_channel.cooldown_refusal(PC) is None
    down_channel.stamp(PC)
    assert (
        down_channel.cooldown_refusal(PC) is not None
        and down_channel.cooldown_refusal(CHAT_ID) is None
    )
    status_state.operator_jid = "491555999999@c.us"
    assert down_channel.operator_jid == "491555999999@c.us"
    status_state.operator_jid = ME_JID


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


def test_self_reaction_folds_nothing(bot: Bot) -> None:
    """A reaction whose reactor is the bot itself never becomes a note.

    The audit case: the bot's own LID tapped (operator's phone or the
    echo of our own react_to_message) and folded back as external
    praise. The guard skips the fold — and the WAHA target fetch —
    before it happens; the debug log line is the only trace.
    """
    llm = bot.stack.llm
    bot.post(waha_event())  # establish context + delivered reply
    llm.requests.clear()
    bot.waha.sent.clear()
    self_reaction = {
        "id": "evt-smoke-self-reaction",
        "timestamp": int(time.time()),
        "event": "message.reaction",
        "session": SESSION,
        "me": {"id": ME_JID, "lid": "491555000000@lid"},
        "payload": {
            "id": "evt-smoke-self-reaction-payload",
            "from": CHAT_ID,
            "participant": "491555000000@lid",  # the bot's own LID
            "fromMe": False,
            "reaction": {"text": "🙏", "messageId": f"true_{CHAT_ID}_SMOKEREPLY"},
        },
    }
    bot.post(self_reaction)
    time.sleep(0.5)
    assert len(llm.requests) == 0

    async def reaction_notes() -> list[str]:
        ctx = handlers_contexts[(SESSION, CHAT_ID)]
        memory = await ctx.store.get("memory")
        messages = await memory.aget_all()
        return [str(m.content) for m in messages if "🙏" in str(m.content)]

    assert asyncio.run(reaction_notes()) == []


def test_foreign_reaction_folds_name_and_jid(bot: Bot) -> None:
    """Another member's reaction folds with the ``Name <jid>`` rendering."""
    bot.post(waha_event())  # establish context + delivered reply
    bot.waha.sent.clear()
    reaction = {
        "id": "evt-smoke-reaction-named",
        "timestamp": int(time.time()),
        "event": "message.reaction",
        "session": SESSION,
        "me": None,
        "payload": {
            "id": "evt-smoke-reaction-named-payload",
            "from": CHAT_ID,
            "participant": "491555000001@c.us",
            "fromMe": False,
            "reaction": {"text": "😂", "messageId": f"true_{CHAT_ID}_SMOKEREPLY"},
        },
    }
    bot.post(reaction)
    time.sleep(0.5)

    async def reaction_notes() -> list[str]:
        ctx = handlers_contexts[(SESSION, CHAT_ID)]
        memory = await ctx.store.get("memory")
        messages = await memory.aget_all()
        return [str(m.content) for m in messages if "reacted" in str(m.content)]

    notes = asyncio.run(reaction_notes())
    assert any("😂" in note and "Smoke Sender" in note for note in notes), notes
    assert all("491555000001" not in note for note in notes), notes


def test_group_turn_sender_tag_carries_jid(bot: Bot) -> None:
    """A group turn's prompt opens with the header, then the record."""
    llm = bot.stack.llm
    event = waha_event()
    bot.post(event)
    assert _wait(lambda: len(llm.requests) >= 1)
    first_user = user_turns_of(llm.requests[0])[0]
    assert first_user.startswith("[turn: 1 message | newest #m"), first_user
    assert "]] text addressed :: kai hola, smoke check" in first_user
    assert "491555000001@c.us" not in first_user


def test_tag_copied_mention_resolves_end_to_end(bot: Bot) -> None:
    """An ``@<user-part>`` copied from a sender tag becomes a real mention.

    The bridge already exists (`resolve_mentions` matches text tokens
    against roster user parts); the sender tag putting the user part in
    reach is what makes the model copy it. Wire-verify the full path:
    tag in the history → model copies the bare id → send carries
    ``mentions``.
    """
    llm = bot.stack.llm
    llm.override = tool_call_response(
        "send_message",
        {"text": "Para @491555000001 dile hola"},
        call_id="call_tag_mention_1",
        response_id="chatcmpl-smoke-tag-mention",
        created=1788525841,
    )
    bot.post(waha_event())
    assert _wait(lambda: len(bot.waha.sent) >= 1)
    session_id, chat_id, _text, mentions = bot.waha.sent[0]
    assert (session_id, chat_id) == (SESSION, CHAT_ID)
    assert mentions == ["491555000001@c.us"]


def test_system_prompt_states_own_identity(bot: Bot) -> None:
    """The first captured LLM request contains the own-identity sentence.

    The wiring under test is ``status.state`` (captured at boot from
    ``get_me``) → ``render_prompt`` → the rendered system message; the
    template sentence is operator data, so the test injects it into
    the session config the same way an operator's edit would (hot
    reload picks it up on the next run).
    """
    llm = bot.stack.llm
    config_path = bot.settings.access_config
    config = json.loads(config_path.read_text())
    config["system_prompt"] += (
        "\n\n# Identity\n\nYou are {{own_identities}} — a message that "
        "names or quotes those ids addresses you.\n"
    )
    config_path.write_text(json.dumps(config, indent=2))
    bot.post(waha_event())
    assert _wait(lambda: len(llm.requests) >= 1)
    system = [
        str(m.get("content", ""))
        for m in llm.requests[0]["messages"]
        if m.get("role") == "system"
    ]
    assert system, "no system message in the first request"
    assert f"You are `{ME_JID}` or `491555000000@lid`" in system[0], system[0][:400]


def test_unknown_identity_drops_the_sentence(bot: Bot) -> None:
    """Without a captured identity the placeholder line is dropped wholesale."""
    llm = bot.stack.llm
    config_path = bot.settings.access_config
    config = json.loads(config_path.read_text())
    config["system_prompt"] += (
        "\n\n# Identity\n\nYou are {{own_identities}} — addresses you.\n"
    )
    config_path.write_text(json.dumps(config, indent=2))
    saved = (status_state.operator_jid, status_state.operator_lid)
    status_state.operator_jid = ""
    status_state.operator_lid = ""
    try:
        bot.post(waha_event())
        assert _wait(lambda: len(llm.requests) >= 1)
        system = [
            str(m.get("content", ""))
            for m in llm.requests[0]["messages"]
            if m.get("role") == "system"
        ]
        assert system
        assert "{{own" not in system[0]
        assert "addresses you" not in system[0]
    finally:
        status_state.operator_jid, status_state.operator_lid = saved
