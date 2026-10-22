import asyncio
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
