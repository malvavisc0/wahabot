import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

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
    assert len(llm.requests) == 3


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
    assert json.dumps([row["message_ids"] for row in outcomes])


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
