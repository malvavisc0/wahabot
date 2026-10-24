from tests.harness import ME_JID, RecordingWaha
from wahabot.core.tickets.alerts import compose_alert, send_alert
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
