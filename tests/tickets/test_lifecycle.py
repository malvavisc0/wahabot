import sqlite3

from tests.harness import chat_event
from wahabot.ai.tools.runcontext import RunTarget
from wahabot.core.models import WahaEvent
from wahabot.core.tickets.identity import customer_details, phone_alias
from wahabot.core.tickets.lifecycle import (
    Inbound,
    close_stale,
    close_ticket,
    finalize_delivery,
    holding_due,
    is_paused,
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


def test_only_appreciative_emoji_count_as_thanks() -> None:
    assert is_thanks_only("👍")
    assert is_thanks_only("👍🏽")
    assert is_thanks_only("🙏🙏")
    assert not is_thanks_only("😡")
    assert not is_thanks_only("❌")
    assert not is_thanks_only("😭")


def test_explicit_done_defaults_done_by_to_assistant(store: TicketStore) -> None:
    ticket = inbound(store, "resolved", mid="EXPLICIT_DONE").ticket

    done = finalize_delivery(
        store,
        ticket.id,
        TicketUpdate(state=TicketState.DONE.value),
        delivered=True,
    )

    assert done is not None
    assert done.state == TicketState.DONE.value
    assert done.done_by == "assistant"


def test_customer_details_reads_phone_domains_and_device_suffixes() -> None:
    raw = chat_event(body="hello", mid="DETAILS", chat="491555000001:12@s.whatsapp.net")
    raw["payload"]["participant"] = "491555000001:12@s.whatsapp.net"
    chat_id, phone, lid, _ = customer_details(WahaEvent.model_validate(raw))

    assert chat_id == "491555000001:12@s.whatsapp.net"
    assert phone == "491555000001@s.whatsapp.net"
    assert lid == ""


def test_last_inbound_uses_the_message_time_and_keeps_the_newest(
    store: TicketStore,
) -> None:
    forwarded = chat_event(body="first", mid="TIME1", chat=CUSTOMER)
    forwarded["payload"]["timestamp"] = 1_000
    record_inbound(
        store,
        WahaEvent.model_validate(forwarded),
        message_ids=[forwarded["payload"]["id"]],
        close_after_days=5,
        body="first",
    )
    assert store.contact(CUSTOMER).last_inbound_at == 1_000

    out_of_order = chat_event(body="old", mid="TIME0", chat=CUSTOMER)
    out_of_order["payload"]["timestamp"] = 500
    record_inbound(
        store,
        WahaEvent.model_validate(out_of_order),
        message_ids=[out_of_order["payload"]["id"]],
        close_after_days=5,
        body="old",
    )
    assert store.contact(CUSTOMER).last_inbound_at == 1_000

    newer = chat_event(body="new", mid="TIME2", chat=CUSTOMER)
    newer["payload"]["timestamp"] = 2_000
    record_inbound(
        store,
        WahaEvent.model_validate(newer),
        message_ids=[newer["payload"]["id"]],
        close_after_days=5,
        body="new",
    )
    assert store.contact(CUSTOMER).last_inbound_at == 2_000


def test_close_stale_closes_untouched_open_and_waiting_tickets(
    store: TicketStore,
) -> None:
    contact = store.upsert_contact(CUSTOMER, display_name="Smoke Sender")
    open_ticket = store.create_ticket(contact.id, CUSTOMER, summary="open", at=1_000)
    waiting = store.create_ticket(contact.id, "491555000002@c.us", at=1_000)
    store.update_ticket(
        waiting.id,
        TicketUpdate(state=TicketState.WAITING_CUSTOMER.value),
        at=1_000,
    )
    needs = store.create_ticket(contact.id, "491555000003@c.us", at=1_000)
    store.update_ticket(
        needs.id, TicketUpdate(state=TicketState.NEEDS_YOU.value), at=1_000
    )

    close_stale(store, 5, at=1_000 + 6 * 86400)

    assert store.ticket(open_ticket.id).state == TicketState.CLOSED.value
    assert store.ticket(waiting.id).state == TicketState.CLOSED.value
    assert store.ticket(needs.id).state == TicketState.NEEDS_YOU.value


def test_closing_an_unresolved_ticket_does_not_claim_done(store: TicketStore) -> None:
    ticket = inbound(store, "still broken", mid="CLOSE_OPEN").ticket

    closed = close_ticket(store, store.ticket(ticket.id))

    assert closed.state == TicketState.CLOSED.value
    assert closed.done_at is None
    assert closed.done_by is None


def test_holding_due_resets_after_a_new_alert_and_cools_down(
    store: TicketStore,
) -> None:
    contact = store.upsert_contact(CUSTOMER, display_name="Smoke Sender")
    ticket = store.create_ticket(contact.id, CUSTOMER, at=1000)

    assert holding_due(store, ticket.id, 1000, 3600)
    store.add_event(ticket.id, "assistant", "holding", "still here", at=1000)
    assert not holding_due(store, ticket.id, 1001, 3600)
    assert holding_due(store, ticket.id, 1000 + 3600, 3600)
    store.add_event(ticket.id, "assistant", "alert", "Ticket 1", at=2000)
    assert holding_due(store, ticket.id, 2001, 3600)


def test_details_merge_across_successive_updates(store: TicketStore) -> None:
    ticket = inbound(store, "details", mid="DETAILS2").ticket

    store.update_ticket(ticket.id, TicketUpdate(details={"order": "A-17"}))
    store.update_ticket(ticket.id, TicketUpdate(details={"address": "Main St"}))

    assert store.ticket(ticket.id).details == {
        "order": "A-17",
        "address": "Main St",
    }


def test_customer_details_reads_alternate_jids_from_the_payload() -> None:
    raw = chat_event(body="hello", mid="ALIAS", chat="999@lid")
    raw["payload"]["participant"] = "999@lid"
    raw["payload"]["_data"]["key"] = {
        "remoteJidAlt": "491555000001@c.us",
        "senderLid": "999@lid",
    }

    chat_id, phone, lid, _ = customer_details(WahaEvent.model_validate(raw))

    assert chat_id == "999@lid"
    assert phone == "491555000001@c.us"
    assert lid == "999@lid"


def test_phone_alias_uses_the_lid_lookup_only_when_needed(
    store: TicketStore,
) -> None:
    raw = chat_event(body="hello", mid="ALIAS2", chat="999@lid")
    raw["payload"]["participant"] = "999@lid"
    event = WahaEvent.model_validate(raw)
    calls: list[str] = []

    def resolve(session: str, lid: str) -> str:
        calls.append(lid)
        return "491555000001@c.us"

    assert phone_alias(store, event, resolve) == "491555000001@c.us"
    assert calls == ["999@lid"]

    store.upsert_contact("999@lid", lid="999@lid", phone="491555000001@c.us")
    assert phone_alias(store, event, resolve) == "491555000001@c.us"
    assert calls == ["999@lid"]


def test_is_paused_matches_any_contact_alias(store: TicketStore) -> None:
    contact = store.upsert_contact(
        "999@lid",
        lid="999@lid",
        phone="491555000001@c.us",
    )
    ticket = store.create_ticket(contact.id, contact.chat_id, at=1000)
    inbound = Inbound(contact=contact, ticket=ticket, body="hello")
    store.set_chat_paused("491555000001@c.us", 2000)

    assert is_paused(store, "default", inbound, 1500)
    assert not is_paused(store, "default", inbound, 2500)

    store.set_chat_paused("491555000001@c.us", None)
    store.set_chat_paused("999@lid", 2000)
    assert is_paused(store, "default", inbound, 1500)


def test_a_new_ticket_inherits_the_contact_external_ref(store: TicketStore) -> None:
    contact = store.upsert_contact(
        CUSTOMER, phone=CUSTOMER, external_user_ref="fictional-user-77"
    )

    ticket = open_ticket(store, contact, moment=100)

    assert ticket.external_user_ref == "fictional-user-77"
