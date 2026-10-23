import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tests.conftest import Bot
from tests.harness import CHAT_ID, SecondResponse, waha_event
from tests.wire.support import _audit_entries


def _turn_event(mid: str, body: str) -> dict[str, Any]:
    event = waha_event(mid)
    event["payload"]["body"] = body
    return event


def _run_outcomes(bot: Bot) -> list[dict[str, Any]]:
    return _audit_entries(bot, "run_outcome", CHAT_ID)


def test_outbound_intent_records_the_intake_ids_it_answers(bot: Bot) -> None:
    bot.stack.llm.override = SecondResponse
    event = _turn_event("INTENTIDS", "kai record the ids")

    bot.post(event)

    message_id = f"false_{CHAT_ID}_INTENTIDS"
    intent = bot.intents.recent(1)[0]
    assert intent.state == "sent"
    assert intent.intake_ids == [message_id]
    row = bot.ledger.row(bot.settings.session, CHAT_ID, message_id)
    assert row is not None and row.outcome == "delivered"


def test_duplicate_delivery_while_the_first_run_is_processing(bot: Bot) -> None:
    llm = bot.stack.llm
    started = threading.Event()
    release = threading.Event()

    async def selector(body: dict[str, Any]) -> dict[str, Any]:
        if not started.is_set():
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
        return SecondResponse

    llm.selector = selector
    event = _turn_event("DUPLIVE", "kai exactly one run")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(bot.post, event)
        assert started.wait(timeout=15)
        second = pool.submit(bot.post, event)
        second.result(timeout=120)
        release.set()
        first.result(timeout=120)

    rows = bot.ledger.rows(bot.settings.session, CHAT_ID)
    assert len(rows) == 1
    assert len(_run_outcomes(bot)) == 1
    assert len(bot.waha.sent) == 1


def test_restart_answers_a_row_accepted_but_not_processed(bot: Bot) -> None:
    event = _turn_event("ACCEPTEDONLY", "kai accepted then crash")
    message_id = str(event["payload"]["id"])
    bot.ledger.record(
        bot.settings.session,
        CHAT_ID,
        message_id,
        "[[08:00 Ana #m1]] text :: accepted",
        raw_json=json.dumps(event),
        sender="491555000001@c.us",
        received_at=time.time(),
    )

    restarted = bot.restart()
    restarted.stack.llm.override = SecondResponse
    asyncio.run(restarted.recovery.startup())

    assert len([entry for entry in restarted.waha.sent if entry[1] == CHAT_ID]) == 1
    row = restarted.ledger.row(restarted.settings.session, CHAT_ID, message_id)
    assert row is not None and row.outcome == "delivered"


def test_duplicate_webhook_does_not_run_a_delivering_row(bot: Bot) -> None:
    event = _turn_event("DELIVERINGDUP", "kai owned by an intent")
    message_id = str(event["payload"]["id"])
    bot.ledger.record(
        bot.settings.session,
        CHAT_ID,
        message_id,
        "[[08:00 Ana #m1]] text :: owned",
        raw_json=json.dumps(event),
        sender="491555000001@c.us",
        received_at=time.time(),
    )
    bot.ledger.hold(bot.settings.session, CHAT_ID, [message_id], "delivery retrying")
    bot.stack.llm.override = SecondResponse

    bot.post(event)

    assert bot.waha.sent == []
    row = bot.ledger.row(bot.settings.session, CHAT_ID, message_id)
    assert row is not None and row.state == "delivering"
