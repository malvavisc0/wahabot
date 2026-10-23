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


def test_unknown_send_does_not_rerun_the_intake_row(bot: Bot) -> None:
    message_id = seed_row(bot, "UNKNOWNROW", age_s=60)
    intent = bot.intents.prepare(
        session=bot.settings.session,
        chat_id=CHAT_ID,
        kind="text",
        payload={"text": "draft that may have landed"},
        intake_ids=[message_id],
    )
    bot.intents.sending(intent.id)
    bot.intents.unknown(intent.id, "read timeout")

    run(bot.recovery.tick())

    assert all(chat != CHAT_ID for _session, chat, *_rest in bot.waha.sent)
    row = bot.ledger.row(bot.settings.session, CHAT_ID, message_id)
    assert row.state == "done" and row.outcome == "operator_required"


def test_failed_send_replays_the_recorded_intent_not_a_new_turn(bot: Bot) -> None:
    bot.stack.llm.override = SecondResponse
    message_id = seed_row(bot, "RETRYINTENT", age_s=60)
    intent = bot.intents.prepare(
        session=bot.settings.session,
        chat_id=CHAT_ID,
        kind="text",
        payload={"text": "recorded draft"},
        intake_ids=[message_id],
    )
    bot.intents.sending(intent.id)
    bot.intents.failed(intent.id, "connection refused", next_attempt_at=time.time() + 60)

    run(bot.recovery.tick())
    assert all(chat != CHAT_ID for _session, chat, *_rest in bot.waha.sent)

    run(bot.recovery.tick(now=time.time() + 120))

    customer = [entry for entry in bot.waha.sent if entry[1] == CHAT_ID]
    assert [entry[2] for entry in customer] == ["recorded draft"]
    assert bot.stack.llm.requests == []
    row = bot.ledger.row(bot.settings.session, CHAT_ID, message_id)
    assert row.state == "done" and row.outcome == "delivered"


def test_prepared_intent_outside_the_reply_window_is_not_sent(bot: Bot) -> None:
    message_id = seed_row(bot, "OLDPREP", age_s=9 * 86400)
    intent = bot.intents.prepare(
        session=bot.settings.session,
        chat_id=CHAT_ID,
        kind="text",
        payload={"text": "nine days late"},
        intake_ids=[message_id],
    )

    run(bot.recovery.tick())

    assert all(chat != CHAT_ID for _session, chat, *_rest in bot.waha.sent)
    row = bot.ledger.row(bot.settings.session, CHAT_ID, message_id)
    assert row.state == "done" and row.outcome == "operator_required"
    assert row.notified_at is not None
    assert bot.intents.get(intent.id).state == "cancelled"
    notices = [entry for entry in bot.waha.sent if entry[1] == ME_JID]
    assert len(notices) == 1

    run(bot.recovery.tick())
    assert len([entry for entry in bot.waha.sent if entry[1] == ME_JID]) == 1


def test_prepared_intent_older_than_the_apology_window_gets_one(bot: Bot) -> None:
    message_id = seed_row(bot, "LATEPREP", age_s=31 * 60)
    bot.intents.prepare(
        session=bot.settings.session,
        chat_id=CHAT_ID,
        kind="text",
        payload={"text": "late draft"},
        intake_ids=[message_id],
    )

    run(bot.recovery.tick())

    customer = [entry for entry in bot.waha.sent if entry[1] == CHAT_ID]
    assert len(customer) == 1
    assert customer[0][2].lower().startswith("sorry for the delay")
    assert customer[0][2].endswith("late draft")


def test_unknown_destination_queues_one_notice_and_flushes_it(bot: Bot) -> None:
    from wahabot.status.state import state

    previous = state.operator_jid
    state.operator_jid = ""
    try:
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

        assert all(chat != ME_JID for _session, chat, *_rest in bot.waha.sent)
        assert len(bot.intents.pending_notices()) == 1

        state.operator_jid = ME_JID
        run(bot.recovery.tick())

        notices = [entry for entry in bot.waha.sent if entry[1] == ME_JID]
        assert len(notices) == 1
        assert "unknown outcome" in notices[0][2]
        assert bot.intents.pending_notices() == []
    finally:
        state.operator_jid = previous


def test_startup_resets_a_live_lease_and_answers_once(bot: Bot) -> None:
    bot.stack.llm.override = SecondResponse
    message_id = seed_row(bot, "LIVELEASE", age_s=60)
    bot.ledger.claim(bot.settings.session, CHAT_ID, [message_id], owner="dead-run")

    run(bot.recovery.startup())

    row = bot.ledger.row(bot.settings.session, CHAT_ID, message_id)
    assert row.state == "done" and row.attempts == 0
    assert len([entry for entry in bot.waha.sent if entry[1] == CHAT_ID]) == 1


def test_burst_with_a_photo_and_a_voice_note_recovers_as_one_turn(bot: Bot) -> None:
    from tests.harness import image_event, voice_event

    dm = "491555000001@c.us"
    events = [waha_event("BURSTMEDIA"), image_event(), voice_event()]
    events[0]["payload"]["body"] = "kai three messages"
    for event in events:
        event["payload"]["from"] = dm
        event["payload"]["participant"] = "491555000001@c.us"
        event["timestamp"] = int(time.time() - 60)
        event["payload"]["timestamp"] = int(time.time() - 60)
        message_id = str(event["payload"]["id"])
        bot.ledger.record(
            bot.settings.session,
            dm,
            message_id,
            "[[08:00 Ana #m1]] media",
            raw_json=json.dumps(event),
            sender="491555000001@c.us",
            received_at=time.time() - 60,
            burst_key="BURSTMEDIA",
        )
    bot.stack.llm.override = SecondResponse

    run(bot.recovery.startup())

    sends = [entry for entry in bot.waha.sent if entry[1] == dm]
    assert len(sends) == 1
    assert sends[0][2] == "smoke final answer"


def test_operator_required_rows_are_noticed_after_a_crash(bot: Bot) -> None:
    message_id = seed_row(bot, "NOTICECRASH", age_s=9 * 86400)
    # The crash shape: rows were settled operator-required before the notice
    # was prepared.
    bot.ledger.finish(
        bot.settings.session,
        CHAT_ID,
        [message_id],
        delivered=False,
        outcome="operator_required",
        reason="outside the reply window",
    )

    run(bot.recovery.tick())

    row = bot.ledger.row(bot.settings.session, CHAT_ID, message_id)
    assert row is not None and row.notified_at is not None
    notices = [entry for entry in bot.waha.sent if entry[1] == ME_JID]
    assert len(notices) == 1
    assert "operator attention" in notices[0][2]

    run(bot.recovery.tick())
    assert len([entry for entry in bot.waha.sent if entry[1] == ME_JID]) == 1


def test_startup_finishes_rows_for_an_already_confirmed_send(bot: Bot) -> None:
    message_id = "SENTRECONCILE"
    intent = bot.intents.prepare(
        session=bot.settings.session,
        chat_id=CHAT_ID,
        kind="text",
        payload={"text": "confirmed before the crash"},
        intake_ids=[message_id],
    )
    bot.intents.sending(intent.id)
    bot.intents.sent(intent.id, "wamid-confirmed")
    event = waha_event("SENTRECONCILE")
    bot.ledger.record(
        bot.settings.session,
        CHAT_ID,
        message_id,
        "[[08:00 Ana #m1]] text :: confirmed",
        raw_json=json.dumps(event),
        sender="491555000001@c.us",
        received_at=time.time() - 60,
    )

    run(bot.recovery.startup())

    row = bot.ledger.row(bot.settings.session, CHAT_ID, message_id)
    assert row is not None and row.state == "done" and row.outcome == "delivered"
    assert all(chat != CHAT_ID for _session, chat, *_rest in bot.waha.sent)


def test_startup_requeues_a_delivering_row_without_an_intent(bot: Bot) -> None:
    bot.stack.llm.override = SecondResponse
    message_id = seed_row(bot, "STRANDED", age_s=60)
    bot.ledger.hold(bot.settings.session, CHAT_ID, [message_id], "delivery retrying")

    run(bot.recovery.startup())

    row = bot.ledger.row(bot.settings.session, CHAT_ID, message_id)
    assert row is not None and row.outcome == "delivered"
    assert len([entry for entry in bot.waha.sent if entry[1] == CHAT_ID]) == 1


def test_old_reaction_intent_is_cancelled_by_the_reply_window(bot: Bot) -> None:
    message_id = seed_row(bot, "OLDREACTION", age_s=9 * 86400)
    intent = bot.intents.prepare(
        session=bot.settings.session,
        chat_id=CHAT_ID,
        kind="reaction",
        payload={"message_id": message_id, "reaction": "ok"},
        intake_ids=[message_id],
    )

    run(bot.recovery.tick())

    assert all(chat != CHAT_ID for _session, chat, *_rest in bot.waha.sent)
    assert bot.intents.get(intent.id).state == "cancelled"
    row = bot.ledger.row(bot.settings.session, CHAT_ID, message_id)
    assert row is not None and row.outcome == "operator_required"


def test_failed_delayed_recovery_is_not_announced_every_tick(
    bot: Bot, monkeypatch: Any
) -> None:
    seed_row(bot, "SPAMNOTICE", age_s=31 * 60)

    async def boom(rows: list[Any], *, apology: bool) -> str:
        raise RuntimeError("llm down")

    monkeypatch.setattr(bot.handlers, "recover_rows", boom)
    now = time.time()

    run(bot.recovery.tick(now=now))
    run(bot.recovery.tick(now=now + 61))

    assert bot.waha.sent == []
