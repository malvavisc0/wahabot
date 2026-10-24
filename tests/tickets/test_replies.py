import asyncio
import json
from collections.abc import Callable

from tests.harness import RecordingWaha
from wahabot.core.tickets.models import (
    Actor,
    Contact,
    EventKind,
    Ticket,
    TicketState,
    TicketUpdate,
)
from wahabot.core.tickets.replies import (
    AMBIGUOUS,
    ANSWER,
    RESOLUTION,
    Remember,
    ReplyOutcome,
    deliver_held,
    fact_tokens,
    handle_operator_reply,
    held_payload,
    mismatch_tokens,
    omitted_tokens,
    parse_kind,
    window_open,
)
from wahabot.core.tickets.store import TicketStore

CUSTOMER = "491555000001@c.us"


def setup_ticket(
    store: TicketStore, *, last_inbound_at: int | None = None
) -> tuple[Contact, Ticket]:
    contact = store.upsert_contact(
        CUSTOMER,
        phone=CUSTOMER,
        display_name="Smoke Sender",
        language="es",
        last_inbound_at=last_inbound_at,
    )
    ticket = store.create_ticket(contact.id, CUSTOMER, summary="refund")
    store.update_ticket(ticket.id, TicketUpdate(state=TicketState.NEEDS_YOU.value))
    store.append_alert_ids(ticket.id, ["true_alert"])
    return store.contact(CUSTOMER), store.ticket(ticket.id)


def completer(kind: str, draft: str) -> Callable[[str], str]:
    def complete(prompt: str) -> str:
        if "Classify one operator reply" in prompt:
            return f'{{"kind": "{kind}"}}'
        if "Draft one WhatsApp message" in prompt:
            return draft
        return "benign"

    return complete


def run(
    store: TicketStore,
    waha: RecordingWaha,
    contact: Contact,
    ticket: Ticket,
    words: str,
    complete: Callable[[str], str],
    *,
    remember: Remember | None = None,
    reply_window_days: int = 7,
) -> ReplyOutcome:
    return asyncio.run(
        handle_operator_reply(
            store,
            waha,
            "default",
            ticket,
            contact,
            words,
            complete=complete,
            reply_window_days=reply_window_days,
            remember=remember,
        )
    )


def test_parse_kind_is_strict() -> None:
    assert parse_kind('{"kind": "resolution"}') == RESOLUTION
    assert parse_kind("not json") == AMBIGUOUS
    assert parse_kind('{"kind": "guess"}') == AMBIGUOUS


def test_fact_tokens_normalize_words_ranges_and_currencies() -> None:
    assert mismatch_tokens("five days", "5 days") == []
    assert mismatch_tokens("10 euros", "10 EUR") == []
    assert mismatch_tokens("5 to 10 days", "5-10 days") == []
    assert "10 eur" in fact_tokens("refunded 10 euros")
    assert "14:30" in fact_tokens("at 14:30")
    assert mismatch_tokens("on 12/05", "on 11/05") != []
    assert mismatch_tokens("20", "10 EUR") != []
    assert mismatch_tokens("5%", "5 EUR") != []
    assert mismatch_tokens("7-10 days", "5-10 days") != []


def test_window_boundaries() -> None:
    assert window_open(0, 7 * 86400, 7)
    assert not window_open(0, 7 * 86400 + 1, 7)
    assert not window_open(None, 0, 7)


def test_resolution_sends_the_draft_and_marks_done_by_operator(
    store: TicketStore,
) -> None:
    contact, ticket = setup_ticket(store, last_inbound_at=store.now())
    waha = RecordingWaha()
    remembered: list[str] = []

    async def remember(text: str) -> None:
        remembered.append(text)

    outcome = run(
        store,
        waha,
        contact,
        ticket,
        "refunded 10 EUR",
        completer(RESOLUTION, "Le devolvimos 10 EUR."),
        remember=remember,
    )

    assert outcome.sent
    assert waha.sent[0][1] == CUSTOMER
    assert "10 EUR" in waha.sent[0][2]
    stored = store.ticket(ticket.id)
    assert stored.state == TicketState.DONE.value
    assert stored.done_by == "operator"
    assert remembered == ["Le devolvimos 10 EUR."]
    assert "Ticket" in outcome.confirmation and "done" in outcome.confirmation
    assert [event.kind for event in store.events(ticket.id)][-2:] == [
        "sent",
        "operator_reply",
    ]


def test_instruction_sets_waiting_for_customer(store: TicketStore) -> None:
    contact, ticket = setup_ticket(store, last_inbound_at=store.now())
    waha = RecordingWaha()

    outcome = run(
        store,
        waha,
        contact,
        ticket,
        "ask for the order number",
        completer("instruction", "¿Puede enviarme el número de pedido?"),
    )

    assert outcome.sent
    assert store.ticket(ticket.id).state == TicketState.WAITING_CUSTOMER.value
    assert "waiting for them" in outcome.confirmation


def test_ambiguous_reply_sends_nothing_and_asks(store: TicketStore) -> None:
    contact, ticket = setup_ticket(store, last_inbound_at=store.now())
    waha = RecordingWaha()

    outcome = run(store, waha, contact, ticket, "mm not sure", completer(AMBIGUOUS, ""))

    assert not outcome.sent
    assert waha.sent == []
    assert "could not tell" in outcome.confirmation
    assert store.ticket(ticket.id).state == TicketState.NEEDS_YOU.value


def test_wrong_amount_is_not_sent_and_the_draft_is_shown(
    store: TicketStore,
) -> None:
    contact, ticket = setup_ticket(store, last_inbound_at=store.now())
    waha = RecordingWaha()

    outcome = run(
        store,
        waha,
        contact,
        ticket,
        "refunded 10 EUR",
        completer(RESOLUTION, "Le devolvimos 20 EUR."),
    )

    assert not outcome.sent
    assert waha.sent == []
    assert outcome.mismatches == ["20 eur"]
    assert "Not sent" in outcome.confirmation and "20 EUR" in outcome.confirmation


def test_closed_window_holds_the_draft(store: TicketStore) -> None:
    contact, ticket = setup_ticket(store, last_inbound_at=0)
    waha = RecordingWaha()

    outcome = run(
        store,
        waha,
        contact,
        ticket,
        "refunded 10 EUR",
        completer(RESOLUTION, "Le devolvimos 10 EUR."),
    )

    assert not outcome.sent
    assert "Can't send yet" in outcome.confirmation
    assert waha.sent == []
    held = store.last_event(ticket.id, "held")
    assert held is not None and "10 EUR" in held.body


def test_held_draft_is_sent_on_the_next_customer_message(store: TicketStore) -> None:
    contact, ticket = setup_ticket(store, last_inbound_at=0)
    waha = RecordingWaha()
    run(
        store,
        waha,
        contact,
        ticket,
        "refunded 10 EUR",
        completer(RESOLUTION, "Le devolvimos 10 EUR."),
    )
    store.upsert_contact(CUSTOMER, last_inbound_at=store.now())

    outcome = asyncio.run(
        deliver_held(
            store,
            waha,
            "default",
            store.ticket(ticket.id),
            store.contact(CUSTOMER),
        )
    )

    assert outcome is not None and outcome.sent
    assert waha.sent[0][2] == "Le devolvimos 10 EUR."
    assert store.ticket(ticket.id).state == TicketState.DONE.value

    again = asyncio.run(
        deliver_held(
            store,
            waha,
            "default",
            store.ticket(ticket.id),
            store.contact(CUSTOMER),
        )
    )

    assert again is None
    assert len(waha.sent) == 1


def test_draft_uses_the_customer_message_when_language_is_unset(
    store: TicketStore,
) -> None:
    customer = "491555000004@c.us"
    contact = store.upsert_contact(
        customer, display_name="Smoke Sender", last_inbound_at=store.now()
    )
    ticket = store.create_ticket(contact.id, customer, summary="refund")
    store.update_ticket(ticket.id, TicketUpdate(state=TicketState.NEEDS_YOU.value))
    store.add_event(ticket.id, "customer", "customer_message", "¿Dónde está mi pedido?")
    prompts: list[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        if "Classify one operator reply" in prompt:
            return f'{{"kind": "{RESOLUTION}"}}'
        return "Le devolvimos 10 EUR."

    run(store, RecordingWaha(), contact, ticket, "refunded 10 EUR", complete)
    draft_prompt = next(prompt for prompt in prompts if "Draft one" in prompt)

    assert "customer's language" in draft_prompt
    assert "¿Dónde está mi pedido?" in draft_prompt


class Refusing(RecordingWaha):
    def send_text(self, *args, **kwargs) -> str:
        raise RuntimeError("network down")


def test_failed_send_is_not_confirmed(store: TicketStore) -> None:
    contact, ticket = setup_ticket(store, last_inbound_at=store.now())

    outcome = run(
        store,
        Refusing(),
        contact,
        ticket,
        "refunded 10 EUR",
        completer(RESOLUTION, "Le devolvimos 10 EUR."),
    )

    assert not outcome.sent
    assert "not confirmed" in outcome.confirmation
    assert store.ticket(ticket.id).state == TicketState.NEEDS_YOU.value


def test_fact_check_composes_and_localizes_numbers() -> None:
    assert mismatch_tokens("200 EUR", "two hundred euros") == []
    assert mismatch_tokens("25 days", "twenty-five days") == []
    assert mismatch_tokens("100 EUR", "one hundred euros") == []
    assert mismatch_tokens("10 EUR", "diez euros") == []
    assert mismatch_tokens("10 eur", "€10") == []
    assert fact_tokens("once") == set()
    assert fact_tokens("neuf") == set()
    assert mismatch_tokens("EUR 10", "10 EUR") == []
    assert mismatch_tokens("10%", "10 %") == []
    assert omitted_tokens("we refunded it", "refunded 10 EUR") == ["10 eur"]


def test_fact_check_rejects_a_draft_that_drops_an_amount(
    store: TicketStore,
) -> None:
    async def run() -> ReplyOutcome:
        contact, ticket = setup_ticket(store, last_inbound_at=store.now())
        return await handle_operator_reply(
            store,
            RecordingWaha(),
            "default",
            ticket,
            contact,
            "refunded 10 EUR",
            complete=completer("resolution", "we refunded it"),
            reply_window_days=7,
        )

    outcome = asyncio.run(run())
    assert not outcome.sent
    assert outcome.missing == ["10 eur"]
    assert "drops values" in outcome.confirmation


def test_deliver_held_sends_every_pending_draft(store: TicketStore) -> None:
    contact, ticket = setup_ticket(store, last_inbound_at=store.now())
    now = store.now()
    store.add_event(
        ticket.id,
        "operator",
        "held",
        json.dumps({"kind": RESOLUTION, "draft": "first answer"}),
        at=now,
    )
    store.add_event(
        ticket.id,
        "operator",
        "held",
        json.dumps({"kind": "instruction", "draft": "second answer"}),
        at=now + 1,
    )

    outcome = asyncio.run(
        deliver_held(store, RecordingWaha(), "default", ticket, contact, now=now + 2)
    )

    assert outcome is not None and outcome.sent
    kinds = [event.kind for event in store.events(ticket.id) if event.kind == "sent"]
    assert len(kinds) == 2
    assert store.ticket(ticket.id).state == TicketState.WAITING_CUSTOMER.value


def test_held_answer_offers_save_when_delivered(store: TicketStore) -> None:
    from wahabot.core.answers.service import AnswerService
    from wahabot.core.answers.store import AnswerStore

    contact, ticket = setup_ticket(store, last_inbound_at=store.now())
    now = store.now()
    store.add_event(
        ticket.id,
        Actor.OPERATOR.value,
        EventKind.HELD.value,
        held_payload(ANSWER, "It shipped yesterday.", "Where is my order?"),
        at=now,
    )
    answers = AnswerService(AnswerStore(store.path))

    outcome = asyncio.run(
        deliver_held(
            store,
            RecordingWaha(),
            "default",
            ticket,
            contact,
            answer_service=answers,
            now=now + 1,
        )
    )

    assert outcome is not None and outcome.sent
    assert outcome.save_ticket == ticket.number
    candidate = answers.latest_candidate(ticket.id)
    assert candidate is not None
    assert candidate.question == "Where is my order?"


def test_held_window_says_when_the_customer_never_wrote(store: TicketStore) -> None:
    async def run() -> ReplyOutcome:
        contact, ticket = setup_ticket(store, last_inbound_at=None)
        return await handle_operator_reply(
            store,
            RecordingWaha(),
            "default",
            ticket,
            contact,
            "refunded",
            complete=completer("resolution", "refunded"),
            reply_window_days=7,
        )

    outcome = asyncio.run(run())
    assert "has not written yet" in outcome.confirmation


def test_answer_reply_offers_one_save_and_stores_a_candidate(
    store: TicketStore,
) -> None:
    from wahabot.core.answers.service import AnswerService
    from wahabot.core.answers.store import AnswerStore
    from wahabot.core.tickets.replies import ANSWER, should_offer_save

    assert should_offer_save(ANSWER)
    contact, ticket = setup_ticket(store, last_inbound_at=store.now())
    answers = AnswerService(AnswerStore(store.path))
    waha = RecordingWaha()
    outcome = asyncio.run(
        handle_operator_reply(
            store,
            waha,
            "default",
            ticket,
            contact,
            "It shipped yesterday",
            complete=completer("answer", "It shipped yesterday."),
            reply_window_days=7,
            answer_service=answers,
        )
    )
    assert outcome.sent
    assert "Reply save 1." in outcome.confirmation
    candidate = answers.latest_candidate()
    assert candidate is not None and candidate.ticket_id == ticket.id
    assert store.ticket(ticket.id).state == TicketState.DONE.value

    saved = answers.save_latest_candidate(ticket.id)
    assert saved is not None
    assert answers.save_latest_candidate(ticket.id) is None


def test_action_reply_never_offers_save(store: TicketStore) -> None:
    from wahabot.core.answers.service import AnswerService
    from wahabot.core.answers.store import AnswerStore

    contact, ticket = setup_ticket(store, last_inbound_at=store.now())
    answers = AnswerService(AnswerStore(store.path))
    outcome = asyncio.run(
        handle_operator_reply(
            store,
            RecordingWaha(),
            "default",
            ticket,
            contact,
            "refunded 10 EUR",
            complete=completer("resolution", "We refunded you 10 EUR."),
            reply_window_days=7,
            answer_service=answers,
        )
    )
    assert outcome.sent
    assert "Reply save." not in outcome.confirmation
    assert answers.latest_candidate() is None
