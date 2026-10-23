import asyncio
import json
import time
from typing import Any

from tests.conftest import Bot
from tests.harness import CHAT_ID, ME_JID, SecondResponse, waha_event


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def seed_row(
    bot: Bot,
    reply_id: str,
    *,
    age_s: float,
    body: str = "kai a delayed question",
    burst_key: str | None = None,
) -> str:
    event = waha_event(reply_id)
    event["payload"]["body"] = body
    event["timestamp"] = int(time.time() - age_s)
    event["payload"]["timestamp"] = int(time.time() - age_s)
    message_id = str(event["payload"]["id"])
    bot.ledger.record(
        bot.settings.session,
        CHAT_ID,
        message_id,
        "[[08:00 Ana #m1]] text :: delayed",
        raw_json=json.dumps(event),
        sender="491555000001@c.us",
        received_at=time.time() - age_s,
        burst_key=burst_key,
    )
    return message_id


def test_tick_answers_a_delayed_row_with_an_apology(bot: Bot) -> None:
    bot.stack.llm.override = SecondResponse
    message_id = seed_row(bot, "RECOVERYLATE", age_s=31 * 60)

    run(bot.recovery.tick())

    customer = [entry for entry in bot.waha.sent if entry[1] == CHAT_ID]
    assert len(customer) == 1
    assert customer[0][2].lower().startswith("sorry for the delay")
    assert len([entry for entry in bot.waha.sent if entry[1] == ME_JID]) == 1
    assert bot.ledger.row(bot.settings.session, CHAT_ID, message_id).outcome == (
        "delivered"
    )


def test_row_outside_the_reply_window_becomes_operator_required(bot: Bot) -> None:
    bot.stack.llm.override = SecondResponse
    message_id = seed_row(bot, "RECOVERYOLD", age_s=9 * 86400)

    run(bot.recovery.tick())

    assert all(chat != CHAT_ID for _session, chat, *_rest in bot.waha.sent)
    row = bot.ledger.row(bot.settings.session, CHAT_ID, message_id)
    assert row.state == "done" and row.outcome == "operator_required"
    notices = [entry for entry in bot.waha.sent if entry[1] == ME_JID]
    assert len(notices) == 1
    assert "operator attention" in notices[0][2]


def test_one_operator_notice_covers_the_whole_batch(bot: Bot) -> None:
    bot.stack.llm.override = SecondResponse
    seed_row(bot, "BATCHNOTE1", age_s=31 * 60)
    seed_row(bot, "BATCHNOTE2", age_s=32 * 60)

    run(bot.recovery.tick())

    notices = [entry for entry in bot.waha.sent if entry[1] == ME_JID]
    assert len(notices) == 1
    assert "2 message(s) delayed by the outage" in notices[0][2]


def test_retry_budget_exhaustion_is_operator_required(bot: Bot) -> None:
    bot.stack.llm.override = SecondResponse
    message_id = seed_row(bot, "EXHAUSTED", age_s=31 * 60)
    for _ in range(bot.settings.intake_max_attempts):
        bot.ledger.fail(
            bot.settings.session,
            CHAT_ID,
            [message_id],
            error="crash",
            next_attempt_at=time.time() - 1,
        )
    bot.ledger.fail(
        bot.settings.session,
        CHAT_ID,
        [message_id],
        error="crash",
        next_attempt_at=time.time() - 1,
    )

    run(bot.recovery.tick())

    row = bot.ledger.row(bot.settings.session, CHAT_ID, message_id)
    assert row.state == "done" and row.outcome == "operator_required"
    assert all(chat != CHAT_ID for _session, chat, *_rest in bot.waha.sent)


def test_stale_processing_claim_is_reset_and_answered(bot: Bot) -> None:
    bot.stack.llm.override = SecondResponse
    message_id = seed_row(bot, "STALECLAIM", age_s=31 * 60)
    bot.ledger.claim_lease_s = 0
    bot.ledger.claim(bot.settings.session, CHAT_ID, [message_id], owner="dead-run")

    run(bot.recovery.tick())

    row = bot.ledger.row(bot.settings.session, CHAT_ID, message_id)
    assert row.state == "done"
    assert len([entry for entry in bot.waha.sent if entry[1] == CHAT_ID]) == 1


def test_startup_is_idempotent(bot: Bot) -> None:
    bot.stack.llm.override = SecondResponse
    seed_row(bot, "STARTUPONCE", age_s=31 * 60)

    run(bot.recovery.startup())
    run(bot.recovery.startup())

    assert len([entry for entry in bot.waha.sent if entry[1] == CHAT_ID]) == 1


def test_burst_rows_recover_as_one_turn(bot: Bot) -> None:
    bot.stack.llm.override = SecondResponse
    key = "BURSTROOT"
    for index in range(3):
        seed_row(bot, f"BURSTREC{index}", age_s=31 * 60, burst_key=key)

    run(bot.recovery.startup())

    assert len([entry for entry in bot.waha.sent if entry[1] == CHAT_ID]) == 1


def test_uncertain_send_is_surfaced_once_to_the_operator(bot: Bot) -> None:
    intent = bot.intents.prepare(
        session=bot.settings.session,
        chat_id=CHAT_ID,
        kind="text",
        payload={"text": "draft that may have landed"},
    )
    bot.intents.sending(intent.id)
    bot.intents.unknown(intent.id, "read timeout")

    run(bot.recovery.tick())
    run(bot.recovery.tick())

    notices = [entry for entry in bot.waha.sent if entry[1] == ME_JID]
    assert len(notices) == 1
    assert "unknown outcome" in notices[0][2]
    assert "draft that may have landed" in notices[0][2]
    assert bot.intents.get(intent.id).notified_at is not None
