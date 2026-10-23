import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from tests.conftest import Bot
from tests.harness import (
    CHAT_ID,
    ME_JID,
    OWN_LID,
    FirstResponse,
    SecondResponse,
    album_events,
    tool_call_response,
    waha_event,
)
from tests.wire.support import _audit_entries, _wait


def _turn_event(mid: str, body: str) -> dict[str, Any]:
    event = waha_event(mid)
    event["payload"]["body"] = body
    return event


def _text_response(text: str) -> dict[str, Any]:
    """A plain final-text completion for a one-shot reply run."""
    return {
        "id": "chatcmpl-smoke-text",
        "object": "chat.completion",
        "created": 1788525832,
        "model": "smoke-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }


def _run_outcomes(bot: Bot) -> list[dict[str, Any]]:
    return _audit_entries(bot, "run_outcome", CHAT_ID)


def _post_while_running(
    bot: Bot,
    first_event: dict[str, Any],
    second_event: dict[str, Any],
    a_started: threading.Event,
) -> None:
    """Post A, wait until its first LLM call, post B, then join both."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(bot.post, first_event)
        assert a_started.wait(timeout=15)
        second = pool.submit(bot.post, second_event)
        first.result(timeout=120)
        second.result(timeout=120)


def test_slow_run_refused_send_answers_the_newest_record_once(bot: Bot) -> None:
    """A's stale tool send is refused; the claimed newer message rides A's
    next round, and B's queued run never starts."""
    llm = bot.stack.llm
    a_started = threading.Event()

    async def selector(body: dict[str, Any]) -> dict[str, Any]:
        if any(message.get("role") == "tool" for message in body["messages"]):
            return SecondResponse
        if not a_started.is_set():
            a_started.set()
            while len(bot.waha.seen_chats) < 2:
                await asyncio.sleep(0.02)
        return FirstResponse

    llm.selector = selector
    _post_while_running(
        bot,
        _turn_event("SLOWA", "kai first question"),
        _turn_event("SLOWB", "kai newest question"),
        a_started,
    )

    assert _wait(lambda: len(bot.waha.sent) == 1)
    assert bot.waha.sent[0][2] == "smoke final answer"
    assert len(llm.requests) == 2
    tool_results = [
        str(message.get("content", ""))
        for message in llm.requests[1]["messages"]
        if message.get("role") == "tool"
    ]
    assert any(
        "Read the newest record" in result and "newest question" in result
        for result in tool_results
    )
    outcomes = _run_outcomes(bot)
    assert any(row["outcome"] == "reply" for row in outcomes)
    assert any(row["outcome"] == "superseded" for row in outcomes)
    replied = next(row for row in outcomes if row["outcome"] == "reply")
    assert set(replied["message_ids"]) == {
        f"false_{CHAT_ID}_SLOWA",
        f"false_{CHAT_ID}_SLOWB",
    }
    assert replied["outbound_id"].startswith("true_")
    reply_rows = _audit_entries(bot, "reply", CHAT_ID)
    assert reply_rows[-1]["reply_to"] == f"false_{CHAT_ID}_SLOWB"


def test_final_text_superseded_by_newer_intake_is_not_sent(bot: Bot) -> None:
    """A plain final text is revalidated too: B arrives first, A stays silent,
    and B's own run sends the only reply."""
    llm = bot.stack.llm
    a_started = threading.Event()

    async def selector(body: dict[str, Any]) -> dict[str, Any]:
        if not a_started.is_set():
            a_started.set()
            while len(bot.waha.seen_chats) < 2:
                await asyncio.sleep(0.02)
            return SecondResponse
        if any(message.get("role") == "tool" for message in body["messages"]):
            return SecondResponse
        return FirstResponse

    llm.selector = selector
    _post_while_running(
        bot,
        _turn_event("TEXTA", "kai answer this"),
        _turn_event("TEXTB", "kai newer answer"),
        a_started,
    )

    assert _wait(lambda: len(bot.waha.sent) == 1)
    assert bot.waha.sent[0][2] == "smoke reply one"
    outcomes = _run_outcomes(bot)
    assert {row["outcome"] for row in outcomes} == {"superseded", "delivered_via_tool"}
    assert not any(
        "smoke final answer" in json.dumps(request) for request in llm.requests
    )


def test_silent_run_releases_the_newer_message_for_its_own_run(bot: Bot) -> None:
    """A silent A releases B; B runs once and is the only reply."""
    llm = bot.stack.llm
    a_started = threading.Event()

    async def selector(body: dict[str, Any]) -> dict[str, Any]:
        if not a_started.is_set():
            a_started.set()
            while len(bot.waha.seen_chats) < 2:
                await asyncio.sleep(0.02)
            return tool_call_response(
                "stay_silent",
                {"reason": "banter between others"},
                call_id="call_slow_silent",
            )
        if any(message.get("role") == "tool" for message in body["messages"]):
            return SecondResponse
        return FirstResponse

    llm.selector = selector
    _post_while_running(
        bot,
        _turn_event("SILENTA", "kai first message"),
        _turn_event("SILENTB", "kai newest message"),
        a_started,
    )

    assert _wait(lambda: len(bot.waha.sent) == 1)
    assert bot.waha.sent[0][2] == "smoke reply one"
    outcomes = _run_outcomes(bot)
    assert {row["outcome"] for row in outcomes} == {
        "silence",
        "delivered_via_tool",
    }


def test_message_after_a_successful_send_gets_its_own_run(bot: Bot) -> None:
    """A later message is not a duplicate: it is pending, not newer than
    its own run start, and answers independently."""
    llm = bot.stack.llm
    llm.override = FirstResponse
    bot.post(_turn_event("AFTERA", "kai first question"))
    assert _wait(lambda: len(bot.waha.sent) == 1)

    llm.clear()
    llm.override = FirstResponse
    bot.post(_turn_event("AFTERB", "kai second question"))

    assert _wait(lambda: len(bot.waha.sent) == 2)
    outcomes = _run_outcomes(bot)
    assert {row["outcome"] for row in outcomes} == {"delivered_via_tool"}
    assert all(row["outbound_id"].startswith("true_") for row in outcomes)
    assert all(row["duration_ms"] >= 0 for row in outcomes)
    assert all(row["message_ids"] for row in outcomes)


def test_operator_command_writes_a_run_outcome(bot: Bot) -> None:
    """The operator path records its outcome too, so an empty slow run has a class."""
    llm = bot.stack.llm
    llm.override = SecondResponse
    event = waha_event("CMD_OUTCOME")
    event["payload"]["id"] = f"true_{ME_JID}_CMD_OUTCOME"
    event["payload"]["_data"]["id"] = {"_serialized": f"true_{ME_JID}_CMD_OUTCOME"}
    event["payload"]["from"] = ME_JID
    event["payload"]["fromMe"] = True
    event["payload"]["participant"] = OWN_LID
    event["payload"]["source"] = "app"
    event["payload"]["body"] = "kai just answer"
    bot.post(event)

    assert _wait(lambda: _audit_entries(bot, "operator_reply", "operator") != [])
    assert _wait(lambda: _audit_entries(bot, "run_outcome", "operator") != [])
    outcome = _audit_entries(bot, "run_outcome", "operator")[-1]
    assert outcome["outcome"] == "reply"
    assert outcome["duration_ms"] >= 0


def test_repeated_tool_round_records_an_empty_outcome(bot: Bot) -> None:
    """A loop that cannot advance stops empty, not as a chosen silence."""
    llm = bot.stack.llm

    async def selector(_body: dict[str, Any]) -> dict[str, Any]:
        return tool_call_response(
            "read_chat", {"mode": "metadata"}, call_id="call_repeat_meta"
        )

    llm.selector = selector
    event = _turn_event("REPEAT", "kai check the metadata")
    bot.post(event)

    assert _wait(lambda: _run_outcomes(bot) != [])
    assert bot.waha.sent == []
    outcome = _run_outcomes(bot)[-1]
    assert outcome["outcome"] == "empty"
    assert outcome["delivered_via"] == "silence"


def test_unaddressed_group_banter_does_not_supersede_a_reply(bot: Bot) -> None:
    """Banter that would never wake a run must not stall the answer.

    A message without the bot's name in a `mentioned` group gets no run
    of its own, so it must also never trip the delivery guard: the
    addressed turn's reply goes out on the first attempt.
    """
    llm = bot.stack.llm
    a_started = threading.Event()

    async def selector(body: dict[str, Any]) -> dict[str, Any]:
        if any(message.get("role") == "tool" for message in body["messages"]):
            return SecondResponse
        if not a_started.is_set():
            a_started.set()
            while len(bot.waha.seen_chats) < 2:
                await asyncio.sleep(0.02)
        return FirstResponse

    llm.selector = selector
    _post_while_running(
        bot,
        _turn_event("ADDRA", "kai first question"),
        _turn_event("ADDRB", "plain banter between others"),
        a_started,
    )

    assert _wait(lambda: len(bot.waha.sent) == 1)
    assert bot.waha.sent[0][2] == "smoke reply one"
    tool_results = [
        str(message.get("content", ""))
        for message in llm.requests[1]["messages"]
        if message.get("role") == "tool"
    ]
    assert all("Read the newest record" not in result for result in tool_results)
    outcomes = _run_outcomes(bot)
    assert {row["outcome"] for row in outcomes} == {"delivered_via_tool"}


def test_lone_emoji_reply_records_a_reaction_outcome(
    bot: Bot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delivered lone-emoji reaction is an outcome of its own.

    The reaction reaches the chat, so the run's outcome row must not
    read as a silence an incident review would chase as a lost answer.
    """
    monkeypatch.setattr("wahabot.handlers.replies.random.random", lambda: 0.0)
    llm = bot.stack.llm
    llm.override = _text_response("👍")
    bot.post(_turn_event("EMOJIA", "kai say hi"))

    assert _wait(lambda: len(bot.waha.reactions) == 1)
    assert bot.waha.sent == []
    outcomes = _run_outcomes(bot)
    assert [row["outcome"] for row in outcomes] == ["reaction"]
    assert outcomes[-1]["delivered_via"] == "reaction"
    assert outcomes[-1]["outbound_id"] == ""


def _double_send_response() -> dict[str, Any]:
    """One model turn that requests two sends in the same tool batch."""
    response = json.loads(json.dumps(FirstResponse))
    response["choices"][0]["message"]["tool_calls"].append(
        {
            "id": "call_send_2",
            "type": "function",
            "function": {
                "name": "send_message",
                "arguments": json.dumps(
                    {"text": "second stale reply", "reason": "keep sending"}
                ),
            },
        }
    )
    return response


def test_one_refusal_blocks_the_rest_of_the_tool_batch(bot: Bot) -> None:
    """A second delivery call in the same batch cannot bypass the guard."""
    llm = bot.stack.llm
    a_started = threading.Event()

    async def selector(body: dict[str, Any]) -> dict[str, Any]:
        if any(message.get("role") == "tool" for message in body["messages"]):
            return SecondResponse
        if not a_started.is_set():
            a_started.set()
            while len(bot.waha.seen_chats) < 2:
                await asyncio.sleep(0.02)
        return _double_send_response()

    llm.selector = selector
    _post_while_running(
        bot,
        _turn_event("BATCHA", "kai first question"),
        _turn_event("BATCHB", "kai newest question"),
        a_started,
    )

    assert _wait(lambda: len(bot.waha.sent) == 1)
    assert bot.waha.sent[0][2] == "smoke final answer"
    tool_results = [
        str(message.get("content", ""))
        for message in llm.requests[-1]["messages"]
        if message.get("role") == "tool"
    ]
    assert any("Read the newest record" in result for result in tool_results)
    assert any("this tool round" in result for result in tool_results)


def test_accepted_message_is_durable_before_the_webhook_returns(
    bot: Bot, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wahabot.webhook as webhook_module

    seen: list[int] = []
    original = webhook_module.dispatch

    async def probe(event: Any) -> None:
        seen.append(len(bot.ledger.rows(event.session, CHAT_ID)))
        await original(event)

    monkeypatch.setattr(webhook_module, "dispatch", probe)
    bot.stack.llm.override = FirstResponse

    bot.post(_turn_event("DURABLE", "kai store this before 200"))

    assert seen == [1]
    row = bot.ledger.row(bot.settings.session, CHAT_ID, f"false_{CHAT_ID}_DURABLE")
    assert row is not None and row.raw_json
    assert row.state == "done"


def test_duplicate_webhook_creates_one_intake_row_and_one_run(bot: Bot) -> None:
    bot.stack.llm.override = FirstResponse
    event = _turn_event("DUPINT", "kai only once")

    bot.post(event)
    bot.post(event)

    rows = bot.ledger.rows(bot.settings.session, CHAT_ID)
    assert len(rows) == 1
    assert len(_run_outcomes(bot)) == 1
    assert len(bot.waha.sent) == 1


def test_non_message_events_create_no_intake_rows(bot: Bot) -> None:
    from tests.harness import sign

    body = json.dumps(
        {
            "event": "forget",
            "session": bot.settings.session,
            "payload": {"chat_id": CHAT_ID},
        }
    ).encode()
    bot.post_raw(
        body,
        headers={"X-Webhook-Hmac": sign(body, bot.settings.webhook_hmac_key)},
    )

    assert bot.ledger.rows(bot.settings.session, CHAT_ID) == []


def test_intake_write_failure_returns_500_and_does_not_dispatch(
    bot: Bot, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wahabot.ai.intake import IntakeLedger

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("disk unavailable")

    monkeypatch.setattr(IntakeLedger, "record", fail)
    bot.stack.llm.override = FirstResponse

    bot.post(_turn_event("NOWRITE", "kai must not run"), expect=500)

    assert bot.stack.llm.requests == []
    assert bot.waha.sent == []


def test_unaddressed_group_banter_is_settled_silent(bot: Bot) -> None:
    bot.stack.llm.override = FirstResponse
    event = _turn_event("BANTERFIN", "plain banter with no address")

    bot.post(event)

    row = bot.ledger.row(bot.settings.session, CHAT_ID, f"false_{CHAT_ID}_BANTERFIN")
    assert row is not None
    assert row.state == "done"
    assert row.outcome == "silent"


def test_muted_message_stays_pending_until_recovery(bot: Bot) -> None:
    from wahabot.status.state import set_session_health

    bot.waha.session_status = "STOPPED"
    set_session_health("STOPPED")
    event = _turn_event("MUTEDFORT", "kai while the session is down")

    bot.post(event)

    message_id = str(event["payload"]["id"])
    row = bot.ledger.row(bot.settings.session, CHAT_ID, message_id)
    assert row is not None and row.state == "pending"
    assert bot.waha.sent == []

    bot.waha.session_status = "WORKING"
    set_session_health("WORKING")
    bot.stack.llm.override = SecondResponse
    asyncio.run(bot.recovery.startup())

    assert len([entry for entry in bot.waha.sent if entry[1] == CHAT_ID]) == 1
    assert bot.ledger.row(bot.settings.session, CHAT_ID, message_id).state == "done"


def test_restart_does_not_replay_a_done_message(bot: Bot) -> None:
    bot.stack.llm.override = SecondResponse
    event = _turn_event("NOREPLAY", "kai answer once")
    bot.post(event)

    restarted = bot.restart()
    restarted.stack.llm.override = SecondResponse
    restarted.post(event)
    asyncio.run(restarted.recovery.startup())

    assert len([entry for entry in bot.waha.sent if entry[1] == CHAT_ID]) == 1
    assert restarted.waha.sent == []


def test_burst_survives_restart_and_flushes_once(bot: Bot) -> None:
    running = bot.rebuild(burst_enabled=True, burst_inactivity_s=60.0)
    running.stack.llm.override = SecondResponse
    events = [_turn_event(f"BURSTLIVE{i}", f"kai burst part {i}") for i in range(3)]
    for event in events:
        running.post(event)
    assert running.waha.sent == []
    running.bursts.reset()

    restarted = running.restart(burst_enabled=True, burst_inactivity_s=60.0)
    restarted.stack.llm.override = SecondResponse
    asyncio.run(restarted.recovery.startup())

    customer = [entry for entry in restarted.waha.sent if entry[1] == CHAT_ID]
    assert len(customer) == 1
    assert all(
        restarted.ledger.row(
            restarted.settings.session, CHAT_ID, str(event["payload"]["id"])
        ).state
        == "done"
        for event in events
    )
    restarted.bursts.reset()


def test_album_survives_restart_and_flushes_once(bot: Bot) -> None:
    events = album_events()
    events[0]["payload"]["_data"]["expectedImageCount"] = 3
    for event in events:
        bot.post(event)
    assert bot.waha.sent == []
    bot.albums.reset()

    restarted = bot.restart()
    restarted.stack.llm.override = SecondResponse
    asyncio.run(restarted.recovery.startup())

    customer = [entry for entry in restarted.waha.sent if entry[1] == CHAT_ID]
    assert len(customer) == 1
    keys = {
        restarted.ledger.row(
            restarted.settings.session, CHAT_ID, str(event["payload"]["id"])
        ).burst_key
        for event in events
    }
    assert len(keys) == 1 and None not in keys
    restarted.albums.reset()


def test_silent_row_does_not_replay_after_restart(bot: Bot) -> None:
    event = waha_event("SILENTRESTART")
    event["payload"]["body"] = "plain banter without an address"
    bot.post(event)

    restarted = bot.restart()
    restarted.stack.llm.override = SecondResponse
    asyncio.run(restarted.recovery.startup())

    assert restarted.waha.sent == []
