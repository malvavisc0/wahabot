import asyncio
from typing import Any

from tests.conftest import Bot
from tests.harness import (
    ME_JID,
    OWN_LID,
    image_event,
    tool_call_response,
    voice_event,
    waha_event,
)
from tests.wire.support import _wait

CUSTOMER = "491555000001@c.us"


def text_response(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-wire-tickets",
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
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def selector_for(kind: str, draft: str):
    """Answer each request by its exact shape, never by arrival order."""

    def selector(body: dict[str, Any]) -> dict[str, Any]:
        text = "\n".join(
            str(message.get("content", "")) for message in body.get("messages", [])
        )
        if "Classify one operator reply" in text:
            return text_response(f'{{"kind": "{kind}"}}')
        if "Draft one WhatsApp message" in text:
            return text_response(draft)
        if any(message.get("role") == "tool" for message in body.get("messages", [])):
            return text_response("the team has this")
        if "HANDOVER" in text:
            return tool_call_response(
                "hand_to_operator",
                {"reason": "the customer needs a person"},
                call_id="call_wire_handover",
            )
        return tool_call_response(
            "send_message",
            {"text": "benign reply", "reason": "wire"},
            call_id="call_wire_benign",
        )

    return selector


def customer_event(mid: str, body: str) -> dict[str, Any]:
    event = waha_event(mid)
    payload = event["payload"]
    payload["id"] = f"false_{CUSTOMER}_{mid}"
    payload["_data"]["id"] = {"_serialized": f"false_{CUSTOMER}_{mid}"}
    payload["from"] = CUSTOMER
    payload["participant"] = CUSTOMER
    payload["body"] = body
    payload["_data"]["notifyName"] = "Smoke Sender"
    return event


def operator_event(mid: str, body: str, *, quote: str = "") -> dict[str, Any]:
    event = waha_event(mid)
    payload = event["payload"]
    payload["id"] = f"true_{ME_JID}_{mid}"
    payload["_data"]["id"] = {"_serialized": f"true_{ME_JID}_{mid}"}
    payload["from"] = ME_JID
    payload["to"] = ME_JID
    payload["fromMe"] = True
    payload["participant"] = OWN_LID
    payload["source"] = "app"
    payload["body"] = body
    if quote:
        payload["replyTo"] = {"id": quote}
    return event


def alerts(bot: Bot) -> list[Any]:
    return [
        entry
        for entry in bot.waha.sent
        if entry[1] == ME_JID and entry[2].startswith(("Ticket ", "URGENT"))
    ]


def alert_id(bot: Bot) -> str:
    entry = alerts(bot)[-1]
    return f"true_{ME_JID}_SENT{bot.waha.sent.index(entry) + 1}"


def handover_bot(bot: Bot, draft: str = "Le devolvimos 10 EUR.") -> Bot:
    bot.stack.llm.selector = selector_for("resolution", draft)
    bot.post(customer_event("HANDOVER", "HANDOVER I need a person about 10 EUR"))
    assert _wait(lambda: len(alerts(bot)) == 1)
    return bot


def test_burst_photo_and_voice_make_one_ticket_and_one_alert(
    bot: Bot, monkeypatch
) -> None:
    async def transcript(_event, _waha, _settings) -> str:
        return "my order never arrived"

    monkeypatch.setattr("wahabot.handlers.burst_mixin.transcribe_voice_note", transcript)
    running = bot.rebuild(burst_enabled=True, burst_inactivity_s=60.0)
    running.stack.llm.selector = selector_for("resolution", "ok")
    image = image_event()
    image["payload"]["from"] = CUSTOMER
    image["payload"]["participant"] = CUSTOMER
    image["payload"]["id"] = f"false_{CUSTOMER}_IMAGE"
    image["payload"]["_data"]["id"] = {"_serialized": f"false_{CUSTOMER}_IMAGE"}
    image["payload"]["body"] = "HANDOVER look at this"
    voice = voice_event()
    running.post(image)
    running.post(voice)
    assert running.waha.sent == []
    running.bursts.reset()

    restarted = running.restart(burst_enabled=True, burst_inactivity_s=60.0)
    restarted.stack.llm.selector = selector_for("resolution", "ok")
    monkeypatch.setattr("wahabot.handlers.burst_mixin.transcribe_voice_note", transcript)
    asyncio.run(restarted.recovery.startup())

    store = restarted.handlers.ticket_store
    assert _wait(lambda: len(store.chat_tickets(CUSTOMER, limit=10)) == 1)
    ticket = store.chat_tickets(CUSTOMER, limit=10)[0]
    assert _wait(lambda: len(alerts(restarted)) == 1)
    assert len(store.attachments(ticket.id)) == 2
    assert "voice note" in alerts(restarted)[0][2].lower()
    restarted.bursts.reset()


def test_quoted_alert_reaches_the_customer_in_spanish_with_the_same_amount(
    bot: Bot,
) -> None:
    handover_bot(bot, "Le devolvimos 10 EUR.")
    bot.handlers.ticket_store.upsert_contact(CUSTOMER, language="es")
    quote = alert_id(bot)
    bot.waha.sent.clear()

    bot.post(operator_event("RESOLVE", "refunded 10 EUR", quote=quote))

    assert _wait(lambda: any(entry[1] == CUSTOMER for entry in bot.waha.sent))
    sent = [entry[2] for entry in bot.waha.sent if entry[1] == CUSTOMER]
    assert sent == ["Le devolvimos 10 EUR."]
    ticket = bot.handlers.ticket_store.chat_tickets(CUSTOMER, limit=10)[0]
    assert ticket.state == "done" and ticket.done_by == "operator"
    decision = [
        event
        for event in bot.handlers.ticket_store.events(ticket.id)
        if event.kind == "operator_reply"
    ]
    assert decision and "refunded 10 EUR" in decision[0].body
    notes = bot.handlers.agent.notebook.read(
        bot.settings.session, CUSTOMER, key=f"ticket-{ticket.number}-decision"
    )
    assert notes and notes[0].kind == "fact"
    assert "refunded 10 EUR" in notes[0].text


def test_wrong_amount_is_not_sent_and_the_draft_is_shown(bot: Bot) -> None:
    handover_bot(bot, "Le devolvimos 20 EUR.")
    quote = alert_id(bot)
    bot.waha.sent.clear()

    bot.post(operator_event("WRONG", "refunded 10 EUR", quote=quote))

    assert _wait(lambda: any(entry[1] == ME_JID for entry in bot.waha.sent))
    assert not any(entry[1] == CUSTOMER for entry in bot.waha.sent)
    assert any("Not sent" in entry[2] and "20 EUR" in entry[2] for entry in bot.waha.sent)


def test_held_reply_goes_out_after_the_next_inbound(bot: Bot) -> None:
    handover_bot(bot, "Le devolvimos 10 EUR.")
    store = bot.handlers.ticket_store
    store.upsert_contact(CUSTOMER, last_inbound_at=store.now() - 8 * 86400)
    quote = alert_id(bot)
    bot.waha.sent.clear()

    bot.post(operator_event("HELD", "refunded 10 EUR", quote=quote))
    assert _wait(lambda: any("Can't send yet" in entry[2] for entry in bot.waha.sent))
    assert not any(entry[1] == CUSTOMER for entry in bot.waha.sent)
    held = store.last_event(store.chat_tickets(CUSTOMER)[0].id, "held")
    assert held is not None

    bot.post(customer_event("AFTER_HELD", "any news?"))

    assert _wait(
        lambda: any(
            entry[1] == CUSTOMER and entry[2] == "Le devolvimos 10 EUR."
            for entry in bot.waha.sent
        )
    )


def test_human_message_pauses_and_resume_restarts(bot: Bot) -> None:
    bot.stack.llm.selector = selector_for("resolution", "ok")
    bot.post(customer_event("BEFORE_PAUSE", "hello"))
    assert _wait(lambda: any(entry[1] == CUSTOMER for entry in bot.waha.sent))

    human = operator_event("HUMAN", "I will handle this one")
    human["payload"]["from"] = OWN_LID
    human["payload"]["to"] = CUSTOMER
    bot.post(human)

    store = bot.handlers.ticket_store
    assert _wait(
        lambda: (
            store.contact(CUSTOMER) is not None
            and store.contact(CUSTOMER).paused_until is not None
        )
    )
    bot.waha.sent.clear()
    bot.post(customer_event("WHILE_PAUSED", "hello?"))
    assert _wait(
        lambda: (
            bot.ledger.row(
                bot.settings.session, CUSTOMER, f"false_{CUSTOMER}_WHILE_PAUSED"
            ).outcome
            == "operator_owned"
        )
    )
    assert bot.waha.sent == []

    restarted = bot.restart()
    restarted.stack.llm.selector = selector_for("resolution", "ok")
    asyncio.run(restarted.recovery.startup())
    assert not any(entry[1] == CUSTOMER for entry in restarted.waha.sent)

    restarted.post(operator_event("RESUME", "resume Smoke Sender"))
    restarted.post(customer_event("AFTER_RESUME", "hello again?"))

    assert _wait(lambda: any(entry[1] == CUSTOMER for entry in restarted.waha.sent))


def test_bot_send_source_api_does_not_pause(bot: Bot) -> None:
    bot.stack.llm.selector = selector_for("resolution", "ok")
    bot.post(customer_event("API_BEFORE", "hello"))
    assert _wait(lambda: any(entry[1] == CUSTOMER for entry in bot.waha.sent))

    own = operator_event("API_SEND", "a bot message")
    own["payload"]["from"] = OWN_LID
    own["payload"]["to"] = CUSTOMER
    own["payload"]["source"] = "api"
    bot.post(own)

    store = bot.handlers.ticket_store
    contact = store.contact(CUSTOMER)
    assert contact is not None and contact.paused_until is None


def test_recovered_run_opens_no_second_ticket(bot: Bot) -> None:
    bot.stack.llm.selector = selector_for("resolution", "ok")
    bot.stack.llm.fail_status = 500
    bot.post(customer_event("RECOVER", "HANDOVER my order is late"))
    store = bot.handlers.ticket_store
    assert _wait(lambda: len(store.chat_tickets(CUSTOMER, limit=10)) == 1)

    bot.stack.llm.fail_status = 0
    restarted = bot.restart()
    restarted.stack.llm.selector = selector_for("resolution", "ok")
    asyncio.run(restarted.recovery.startup())

    assert len(restarted.handlers.ticket_store.chat_tickets(CUSTOMER, limit=10)) == 1


def test_thanks_after_done_reacts_and_keeps_the_ticket_done(bot: Bot) -> None:
    bot.stack.llm.selector = selector_for("resolution", "ok")
    bot.post(customer_event("DONE_FIRST", "hello"))
    assert _wait(lambda: any(entry[1] == CUSTOMER for entry in bot.waha.sent))
    store = bot.handlers.ticket_store
    assert store.chat_tickets(CUSTOMER, limit=10)[0].state == "done"
    bot.waha.sent.clear()

    bot.post(customer_event("THANKS", "thanks!"))

    assert _wait(lambda: len(bot.waha.reactions) == 1)
    assert bot.waha.reactions[0][1] == "👍"
    assert not any(entry[1] == CUSTOMER for entry in bot.waha.sent)
    assert store.chat_tickets(CUSTOMER, limit=10)[0].state == "done"


def test_global_pause_is_set_and_cleared_by_command(bot: Bot) -> None:
    bot.stack.llm.selector = selector_for("resolution", "ok")
    bot.post(customer_event("GLOBAL_BEFORE", "hello"))
    assert _wait(lambda: any(entry[1] == CUSTOMER for entry in bot.waha.sent))

    bot.post(operator_event("GLOBAL_PAUSE", "pause"))
    assert _wait(
        lambda: bot.handlers.ticket_store.session_paused_until("default") is not None
    )
    bot.waha.sent.clear()
    bot.post(customer_event("GLOBAL_PAUSED", "hello?"))
    assert _wait(
        lambda: (
            bot.ledger.row(
                bot.settings.session, CUSTOMER, f"false_{CUSTOMER}_GLOBAL_PAUSED"
            ).outcome
            == "operator_owned"
        )
    )
    assert not any(entry[1] == CUSTOMER for entry in bot.waha.sent)

    bot.post(operator_event("GLOBAL_RESUME", "resume"))
    bot.post(customer_event("GLOBAL_AFTER", "hello again?"))
    assert _wait(lambda: any(entry[1] == CUSTOMER for entry in bot.waha.sent))
