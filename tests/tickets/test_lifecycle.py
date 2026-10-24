import sqlite3

from tests.harness import chat_event
from wahabot.ai.tools.runcontext import RunTarget
from wahabot.core.models import WahaEvent
from wahabot.core.tickets.lifecycle import (
    Inbound,
    finalize_delivery,
    is_thanks_only,
    open_ticket,
    record_inbound,
    split_ticket,
)
from wahabot.core.tickets.models import TicketState, TicketUpdate
from wahabot.core.tickets.store import TicketStore
from wahabot.handlers.ticket_mixin import TicketMixin

CUSTOMER = "491555000001@c.us"


def inbound(
    store: TicketStore,
    body: str,
    *,
    mid: str = "T1",
    message_ids: list[str] | None = None,
    close_after_days: int = 5,
    at: int | None = None,
) -> Inbound:
    raw = chat_event(body=body, mid=mid, chat=CUSTOMER)
    return record_inbound(
        store,
        WahaEvent.model_validate(raw),
        message_ids=message_ids or [str(raw["payload"]["id"])],
        close_after_days=close_after_days,
        body=body,
        at=at,
    )


def test_first_message_opens_one_silent_ticket(store: TicketStore) -> None:
    result = inbound(store, "where is my order?", mid="FIRST")

    assert result.ticket.state == TicketState.OPEN.value
    assert result.ticket.number == result.ticket.id
    assert not result.reopened
    assert [event.kind for event in store.events(result.ticket.id)] == [
        "opened",
        "customer_message",
        "state",
    ]
    assert store.contact(CUSTOMER).display_name == "Smoke Sender"


def test_burst_rows_share_one_ticket_event(store: TicketStore) -> None:
    first = inbound(store, "part one", mid="B1")
    first_id = f"false_{CUSTOMER}_B1"
    second = inbound(
        store,
        "part two",
        mid="B2",
        message_ids=[first_id, f"false_{CUSTOMER}_B2"],
    )

    events = [
        event
        for event in store.events(second.ticket.id)
        if event.kind == "customer_message"
    ]
    assert second.ticket.id == first.ticket.id
    assert len(events) == 1
    assert events[0].message_ids == [first_id, f"false_{CUSTOMER}_B2"]
    assert events[0].body == "part one"


def test_delivered_reply_finalizes_done_by_assistant(store: TicketStore) -> None:
    ticket = inbound(store, "how late?", mid="DONE").ticket

    done = finalize_delivery(store, ticket.id, TicketUpdate(), delivered=True)

    assert done.state == TicketState.DONE.value
    assert done.done_by == "assistant"
    assert done.done_at is not None
    assert store.events(ticket.id)[-1].kind == "reply"


def test_missing_items_mean_waiting_for_customer(store: TicketStore) -> None:
    ticket = inbound(store, "my order?", mid="WAIT").ticket

    waiting = finalize_delivery(
        store,
        ticket.id,
        TicketUpdate(missing=["order number"]),
        delivered=True,
    )

    assert waiting.state == TicketState.WAITING_CUSTOMER.value
    assert waiting.done_by is None
    assert waiting.missing == ["order number"]


def test_undelivered_reply_does_not_finalize(store: TicketStore) -> None:
    ticket = inbound(store, "hello", mid="UNSENT").ticket

    unchanged = finalize_delivery(store, ticket.id, TicketUpdate(), delivered=False)

    assert unchanged.state == TicketState.OPEN.value
    assert unchanged.done_at is None


def test_reopen_within_window_keeps_ticket_and_flags_operator(store: TicketStore) -> None:
    ticket = inbound(store, "fix this", mid="RE1", at=1000).ticket
    store.update_ticket(
        ticket.id,
        TicketUpdate(state=TicketState.DONE.value, done_by="operator"),
        at=1100,
    )

    reopened = inbound(store, "still broken", mid="RE2", at=1100 + 86400)

    assert reopened.reopened
    assert reopened.operator_solved
    assert reopened.ticket.id == ticket.id
    assert reopened.ticket.state == TicketState.OPEN.value


def test_after_the_window_the_old_ticket_closes_and_a_new_one_links(
    store: TicketStore,
) -> None:
    old = inbound(store, "fix this", mid="OLD", at=1000).ticket
    store.update_ticket(
        old.id,
        TicketUpdate(state=TicketState.DONE.value, done_by="assistant"),
        at=1000,
    )

    fresh = inbound(store, "a new issue", mid="NEW", at=1000 + 6 * 86400)

    assert fresh.ticket.id != old.id
    assert fresh.ticket.previous_ticket_id == old.id
    assert store.ticket(old.id).state == TicketState.CLOSED.value


def test_thanks_detection_is_bare_only() -> None:
    assert is_thanks_only("Thanks!")
    assert is_thanks_only("🙏")
    assert is_thanks_only("👍")
    assert not is_thanks_only("thanks, and when will it arrive?")
    assert not is_thanks_only("")


def test_split_marks_an_open_ticket_done_and_links_it(store: TicketStore) -> None:
    ticket = inbound(store, "two issues", mid="SPLIT").ticket

    split = split_ticket(store, ticket, "the second issue")

    assert store.ticket(ticket.id).state == TicketState.DONE.value
    assert split.previous_ticket_id == ticket.id
    assert split.state == TicketState.OPEN.value


def test_split_refuses_a_needs_you_ticket(store: TicketStore) -> None:
    ticket = inbound(store, "waiting", mid="NO_SPLIT").ticket
    store.update_ticket(ticket.id, TicketUpdate(state=TicketState.NEEDS_YOU.value))

    try:
        split_ticket(store, store.ticket(ticket.id), "other")
    except ValueError as exc:
        assert "open or done" in str(exc)
    else:
        raise AssertionError("a needs_you ticket must not split")


def test_recovered_inbound_reuses_the_ticket(store: TicketStore) -> None:
    first = inbound(store, "needs a person", mid="REC")
    message_ids = [f"false_{CUSTOMER}_REC"]

    again = inbound(store, "needs a person", mid="REC", message_ids=message_ids)

    assert again.ticket.id == first.ticket.id
    assert len(store.chat_tickets(CUSTOMER, limit=10)) == 1


def test_contact_alias_rekeys_the_contact_and_its_ticket(store: TicketStore) -> None:
    contact = store.upsert_contact(
        CUSTOMER, phone=CUSTOMER, lid="lid-1@lid", display_name="Smoke Sender"
    )
    ticket = store.create_ticket(contact.id, CUSTOMER)

    moved = store.upsert_contact("lid-1@lid", lid="lid-1@lid", last_inbound_at=42)

    assert moved.id == contact.id
    assert moved.chat_id == "lid-1@lid"
    assert moved.display_name == "Smoke Sender"
    assert store.contact(CUSTOMER) is None
    assert store.ticket(ticket.id).chat_id == "lid-1@lid"
    active = store.active_ticket("lid-1@lid")
    assert active is not None and active.id == ticket.id


def test_split_refuses_a_closed_ticket(store: TicketStore) -> None:
    contact = store.upsert_contact(CUSTOMER, display_name="Smoke Sender")
    ticket = store.create_ticket(contact.id, CUSTOMER, summary="first")
    store.update_ticket(ticket.id, TicketUpdate(state=TicketState.CLOSED.value))

    try:
        split_ticket(store, store.ticket(ticket.id), "second issue")
    except ValueError as exc:
        assert "open or done" in str(exc)
    else:
        raise AssertionError("closed ticket split was accepted")


def test_finalize_split_marks_the_open_ticket_with_a_note(
    store: TicketStore,
) -> None:
    contact = store.upsert_contact(CUSTOMER, display_name="Smoke Sender")
    ticket = store.create_ticket(contact.id, CUSTOMER, summary="first")
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        ticket_store=store,
        ticket_id=ticket.id,
        ticket_split="second issue",
    )

    TicketMixin().finalize_ticket(target, "delivered_via_tool")

    old = store.ticket(ticket.id)
    assert old is not None and old.state == TicketState.DONE.value
    assert any(
        event.body == "split into a new ticket" for event in store.events(ticket.id)
    )
    new = store.latest_ticket(CUSTOMER)
    assert new is not None and new.id != ticket.id
    assert new.previous_ticket_id == ticket.id


def test_open_ticket_reuses_the_winner_of_a_creation_race(
    store: TicketStore, monkeypatch
) -> None:
    contact = store.upsert_contact(CUSTOMER, display_name="Smoke Sender")
    existing = store.create_ticket(contact.id, CUSTOMER)

    def conflict(*args: object, **kwargs: object) -> None:
        raise sqlite3.IntegrityError("active ticket exists")

    monkeypatch.setattr(store, "create_ticket", conflict)

    assert open_ticket(store, contact, 1).id == existing.id
