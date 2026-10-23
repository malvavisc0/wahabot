import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tests.conftest import Bot
from tests.harness import CHAT_ID, FOREIGN_JID, ME_JID, SecondResponse, waha_event
from tests.wire.support import _audit_entries
from wahabot.core.models import WahaEvent


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


def test_recovery_tick_leaves_a_live_inflight_turn_to_its_run(bot: Bot) -> None:
    """A live run claims its turn rows; the 60s tick must not re-run them.

    The tick's due work is pending and failed rows only. A row whose run
    is still answering would be both (it stayed pending until the run
    settled) — before runs claimed their own turn rows, the tick could
    re-run a live turn during its unlocked final phase and deliver the
    same reply twice.
    """
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
    event = _turn_event("LIVETURN", "kai slow in-flight turn")
    message_id = str(event["payload"]["id"])

    async def wait_for_llm() -> None:
        # Poll loop-friendly: a blocking Event.wait would stall the very
        # task that must reach the LLM call.
        for _ in range(3000):
            if started.is_set():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("the live run never reached its LLM call")

    async def scenario() -> None:
        # The writer and the run share one loop with the recovery tick,
        # exactly like the webhook process does.
        message = WahaEvent.model_validate(event)
        await bot.handlers.record_webhook_intake(message)
        run_task = asyncio.create_task(bot.handlers.reply_with_agent(message))
        await wait_for_llm()
        row = bot.ledger.row(bot.settings.session, CHAT_ID, message_id)
        assert row is not None and row.state == "processing"
        await bot.recovery.tick()
        release.set()
        await run_task

    asyncio.run(scenario())

    row = bot.ledger.row(bot.settings.session, CHAT_ID, message_id)
    assert row is not None and row.state == "done" and row.outcome == "delivered"
    assert len(_run_outcomes(bot)) == 1
    assert len([entry for entry in bot.waha.sent if entry[1] == CHAT_ID]) == 1


def test_redelivery_after_restart_leaves_the_durable_row_to_recovery(
    bot: Bot,
) -> None:
    event = _turn_event("BACKLOGREDELIVERY", "kai backlog after restart")
    event["payload"]["timestamp"] = int(time.time() - 600)
    event["timestamp"] = int(time.time() - 600)
    message_id = str(event["payload"]["id"])
    bot.ledger.record(
        bot.settings.session,
        CHAT_ID,
        message_id,
        "[[08:00 Ana #m1]] text :: backlog",
        raw_json=json.dumps(event),
        sender="491555000001@c.us",
        received_at=time.time() - 600,
    )

    restarted = bot.restart()
    restarted.stack.llm.override = SecondResponse
    restarted.post(event)

    row = restarted.ledger.row(restarted.settings.session, CHAT_ID, message_id)
    assert row is not None and row.state == "pending"
    assert row.outcome is None

    asyncio.run(restarted.recovery.startup())

    row = restarted.ledger.row(restarted.settings.session, CHAT_ID, message_id)
    assert row is not None and row.outcome == "delivered"
    assert len([entry for entry in restarted.waha.sent if entry[1] == CHAT_ID]) == 1


def test_message_without_readable_content_is_settled_silent(bot: Bot) -> None:
    event = waha_event("NOCONTENT")
    event["payload"]["from"] = FOREIGN_JID
    event["payload"]["body"] = ""
    event["payload"]["_data"]["type"] = "document"
    message_id = str(event["payload"]["id"])

    bot.post(event)

    row = bot.ledger.row(bot.settings.session, FOREIGN_JID, message_id)
    assert row is not None and row.state == "done"
    assert row.outcome == "silent"
    assert row.reason == "no readable content"

    asyncio.run(bot.recovery.tick())
    assert row.reason == "no readable content"
    assert bot.waha.sent == []


def test_replay_skips_a_turn_a_live_recovery_owner_is_processing(bot: Bot) -> None:
    event = _turn_event("REPLAYPROCESSING", "kai live owner")
    message_id = str(event["payload"]["id"])
    bot.ledger.record(
        bot.settings.session,
        CHAT_ID,
        message_id,
        "[[08:00 Ana #m1]] text :: live",
        raw_json=json.dumps(event),
        sender="491555000001@c.us",
        received_at=time.time(),
    )
    intent = bot.intents.prepare(
        session=bot.settings.session,
        chat_id=CHAT_ID,
        kind="text",
        payload={"text": "stale draft"},
        intake_ids=[message_id],
    )
    bot.ledger.claim(bot.settings.session, CHAT_ID, [message_id], owner="live-run")

    asyncio.run(bot.recovery.tick())

    assert bot.waha.sent == []
    assert bot.intents.get(intent.id).state == "prepared"


def test_suppressed_run_does_not_settle_the_delivering_row(bot: Bot) -> None:
    event = _turn_event("SUPPRESSDELIVER", "kai owned by an intent")
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

    bot.handlers.finish_run(
        bot.settings.session,
        CHAT_ID,
        [message_id],
        None,
        "superseded",
        time.monotonic(),
    )

    row = bot.ledger.row(bot.settings.session, CHAT_ID, message_id)
    assert row is not None and row.state == "delivering"


def test_escalation_intent_does_not_carry_the_customer_turn_ids(bot: Bot) -> None:
    from wahabot.ai.tools.whatsapp.escalation import (
        EscalationChannel,
        forward_escalation,
    )
    from wahabot.core.transport.context import bind_intake_ids, reset_intake_ids

    token = bind_intake_ids(("customer-turn",))
    try:
        outcome = forward_escalation(
            bot.transport,
            EscalationChannel(),
            bot.settings,
            bot.settings.session,
            CHAT_ID,
            "a report",
        )
    finally:
        reset_intake_ids(token)

    assert '"ok": true' in outcome
    intent = bot.intents.recent(1)[0]
    assert intent.chat_id == ME_JID
    assert intent.intake_ids == []
