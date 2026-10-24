import time
from typing import override

from tests.harness import ME_JID, RecordingWaha
from wahabot.core.tickets.alerts import (
    ALERT_RECONCILE_GRACE_S,
    alert_intent_key,
    compose_alert,
    send_alert,
)
from wahabot.core.tickets.attachments import persist_bytes
from wahabot.core.tickets.models import Contact, Ticket, TicketState, TicketUpdate
from wahabot.core.tickets.store import TicketStore
from wahabot.settings import Settings

CUSTOMER = "491555000001@c.us"


def ticket_with_request(
    store: TicketStore, *, urgent: bool = False
) -> tuple[Contact, Ticket]:
    contact = store.upsert_contact(
        CUSTOMER,
        phone=CUSTOMER,
        display_name="Smoke Sender",
        language="es",
    )
    ticket = store.create_ticket(
        contact.id, CUSTOMER, summary="late order", urgent=urgent
    )
    store.add_event(ticket.id, "customer", "customer_message", "¿dónde está mi pedido?")
    return contact, store.ticket(ticket.id)


def test_compose_alert_has_every_operator_field(store: TicketStore) -> None:
    contact, ticket = ticket_with_request(store, urgent=True)
    store.update_ticket(
        ticket.id,
        TicketUpdate(details={"order": "A-17"}, missing=["address"]),
    )
    text = compose_alert(store, store.ticket(ticket.id), contact, "en", None)

    assert text.startswith("URGENT")
    assert f"Ticket {ticket.number}:" in text
    assert "Smoke Sender" in text
    assert "491555000001" in text and "@c.us" not in text
    assert "Language: es" in text
    assert "¿dónde está mi pedido?" in text
    assert "order: A-17" in text
    assert "Missing: address" in text
    assert "Nothing was promised." in text
    assert "Reply to this message with what you did." in text


def test_alert_translation_is_cached_by_source_and_language(store: TicketStore) -> None:
    contact, ticket = ticket_with_request(store)
    calls: list[tuple[str, str]] = []

    def translator(text: str, language: str) -> str:
        calls.append((text, language))
        return "TRANSLATED ALERT"

    first = compose_alert(store, ticket, contact, "es", translator)
    second = compose_alert(store, ticket, contact, "es", translator)

    assert first.startswith("TRANSLATED ALERT")
    assert "Original: ¿dónde está mi pedido?" in first
    assert second == first
    assert len(calls) == 1


def test_alert_forwards_stored_attachments_and_records_ids(
    store: TicketStore, settings: Settings
) -> None:
    contact, ticket = ticket_with_request(store)
    waha = RecordingWaha()
    persist_bytes(store, ticket, "msg-photo", "image", "image/png", b"photo", settings)

    result = send_alert(store, waha, "default", ME_JID, ticket, contact)

    stored = store.ticket(ticket.id)
    assert result.delivered
    assert len(waha.forwarded) == 1
    assert "msg-photo" in stored.alert_message_ids
    assert any(item.startswith("true_") for item in stored.alert_message_ids)
    assert stored.state == TicketState.NEEDS_YOU.value
    assert [event.kind for event in store.events(ticket.id)][-2:] == ["alert", "state"]


def test_already_needs_you_sends_only_the_delta(
    store: TicketStore, settings: Settings
) -> None:
    contact, ticket = ticket_with_request(store)
    waha = RecordingWaha()
    persist_bytes(store, ticket, "msg-photo", "image", "image/png", b"photo", settings)
    send_alert(store, waha, "default", ME_JID, ticket, contact)
    forwarded = len(waha.forwarded)
    store.add_event(ticket.id, "customer", "customer_message", "any update?")

    result = send_alert(store, waha, "default", ME_JID, store.ticket(ticket.id), contact)

    assert "update from" in result.text
    assert "any update?" in result.text
    assert len(waha.forwarded) == forwarded


def test_attachment_dedupe_keeps_one_copy(store: TicketStore, settings: Settings) -> None:
    _, ticket = ticket_with_request(store)

    first = persist_bytes(store, ticket, "m1", "image", "image/png", b"same", settings)
    second = persist_bytes(store, ticket, "m2", "image", "image/png", b"same", settings)

    assert len(first) == 1
    assert len(second) == 1
    stored = store.attachments(ticket.id)
    assert len(stored) == 2
    assert stored[0].path == stored[1].path
    assert {item.message_id for item in stored} == {"m1", "m2"}


class NoForward(RecordingWaha):
    def forward_message(self, session: str, chat_id: str, message_id: str) -> str:
        raise RuntimeError("cannot forward")

    def send_image(self, *args, **kwargs) -> str:
        raise RuntimeError("cannot send local")


def test_failed_forward_and_local_send_tells_the_operator(
    store: TicketStore, settings: Settings
) -> None:
    contact, ticket = ticket_with_request(store)
    waha = NoForward()
    persist_bytes(store, ticket, "msg-photo", "image", "image/png", b"photo", settings)

    result = send_alert(store, waha, "default", ME_JID, ticket, contact)

    assert result.failed_attachments == 1
    assert any("Could not attach photo" in entry[2] for entry in waha.sent)
    assert store.ticket(ticket.id).state == TicketState.NEEDS_YOU.value


class NoForwardLocalSend(RecordingWaha):
    def forward_message(self, session: str, chat_id: str, message_id: str) -> str:
        raise RuntimeError("cannot forward")


def test_failed_forward_falls_back_to_the_local_file(
    store: TicketStore, settings: Settings
) -> None:
    contact, ticket = ticket_with_request(store)
    waha = NoForwardLocalSend()
    persist_bytes(store, ticket, "msg-photo", "image", "image/png", b"photo", settings)

    result = send_alert(store, waha, "default", ME_JID, ticket, contact)

    assert result.delivered
    assert result.failed_attachments == 0
    assert len(waha.sent_images) == 1
    assert waha.sent_images[0][2]["data"]


class EmptyAlert(RecordingWaha):
    def send_text(self, *args, **kwargs) -> str:
        return ""


def test_unconfirmed_alert_does_not_change_the_ticket(
    store: TicketStore, settings: Settings
) -> None:
    contact, ticket = ticket_with_request(store)

    result = send_alert(store, EmptyAlert(), "default", ME_JID, ticket, contact)

    assert not result.delivered
    assert store.ticket(ticket.id).state != TicketState.NEEDS_YOU.value
    assert not any(event.kind == "alert" for event in store.events(ticket.id))


def test_a_reopened_ticket_does_not_forward_old_attachments_again(
    store: TicketStore, settings: Settings
) -> None:
    contact, ticket = ticket_with_request(store)
    waha = RecordingWaha()
    persist_bytes(store, ticket, "msg-photo", "image", "image/png", b"photo", settings)

    send_alert(store, waha, "default", ME_JID, ticket, contact)
    assert len(waha.forwarded) == 1

    store.update_ticket(ticket.id, TicketUpdate(state=TicketState.OPEN.value))
    send_alert(store, waha, "default", ME_JID, store.ticket(ticket.id), contact)

    assert len(waha.forwarded) == 1


class HistoryWaha(RecordingWaha):
    """A transport whose chat history can contain a previously sent alert."""

    def __init__(self, history: list[dict[str, object]] | None = None) -> None:
        super().__init__()
        self.history = list(history or [])

    @override
    def fetch_chat_messages(
        self, session: str, chat_id: str, limit: int = 50
    ) -> list[dict[str, object]]:
        return list(self.history)

    @override
    def send_text(
        self,
        session: str,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        mentions: list[str] | None = None,
    ) -> str:
        sent_id = super().send_text(session, chat_id, text, reply_to, mentions)
        self.history.append(
            {
                "id": sent_id,
                "body": text,
                "fromMe": True,
                "timestamp": int(time.time()),
            }
        )
        return sent_id


def seed_alert(
    store: TicketStore, ticket: Ticket, key: str, text: str, at: int = 1000
) -> None:
    store.claim_alert(key, ticket.id, "default", ME_JID, text, at=at)


def test_alert_retry_adopts_a_message_already_in_history(
    store: TicketStore,
) -> None:
    contact, ticket = ticket_with_request(store)
    key = alert_intent_key(store, ticket, "handover")
    text = compose_alert(store, ticket, contact, "en", None)
    seed_alert(store, ticket, key, text)
    waha = HistoryWaha(
        [
            {
                "id": "true_491555000000@c.us_ADOPTED",
                "body": text,
                "fromMe": True,
                "timestamp": 1001,
            }
        ]
    )

    result = send_alert(
        store,
        waha,
        "default",
        ME_JID,
        ticket,
        contact,
        alert_key=key,
        now=1050,
    )

    assert result.delivered and not result.pending
    assert waha.sent == []
    stored = store.ticket(ticket.id)
    assert stored is not None and stored.state == TicketState.NEEDS_YOU.value
    assert "true_491555000000@c.us_ADOPTED" in stored.alert_message_ids
    intent = store.alert_intent(key)
    assert intent is not None and intent.state == "sent"


def test_alert_retry_waits_for_the_grace_before_resending(
    store: TicketStore,
) -> None:
    contact, ticket = ticket_with_request(store)
    key = alert_intent_key(store, ticket, "handover")
    text = compose_alert(store, ticket, contact, "en", None)
    seed_alert(store, ticket, key, text)
    waha = HistoryWaha()

    pending = send_alert(
        store, waha, "default", ME_JID, ticket, contact, alert_key=key, now=1010
    )

    assert pending.pending and not pending.delivered
    assert waha.sent == []


def test_alert_retry_resends_after_the_grace(store: TicketStore) -> None:
    contact, ticket = ticket_with_request(store)
    key = alert_intent_key(store, ticket, "handover")
    text = compose_alert(store, ticket, contact, "en", None)
    seed_alert(store, ticket, key, text)
    waha = HistoryWaha()

    result = send_alert(
        store,
        waha,
        "default",
        ME_JID,
        ticket,
        contact,
        alert_key=key,
        now=1000 + ALERT_RECONCILE_GRACE_S + 1,
    )

    assert result.delivered
    assert len(waha.sent) == 1
    stored = store.ticket(ticket.id)
    assert stored is not None and stored.state == TicketState.NEEDS_YOU.value


def test_alert_is_not_sent_twice_for_the_same_key(store: TicketStore) -> None:
    contact, ticket = ticket_with_request(store)
    key = alert_intent_key(store, ticket, "handover")
    waha = HistoryWaha()

    first = send_alert(
        store, waha, "default", ME_JID, ticket, contact, alert_key=key, now=1000
    )
    second = send_alert(
        store, waha, "default", ME_JID, ticket, contact, alert_key=key, now=1001
    )

    assert first.delivered and second.delivered
    assert len(waha.sent) == 1
    assert second.message_ids == first.message_ids
