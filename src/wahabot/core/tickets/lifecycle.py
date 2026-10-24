"""Ticket lifecycle: opening, reopening, finalizing and closing."""

import re
from dataclasses import dataclass
from typing import Any

from wahabot.core.models import WahaEvent
from wahabot.core.tickets.models import (
    Actor,
    Contact,
    EventKind,
    Ticket,
    TicketEvent,
    TicketState,
    TicketUpdate,
)
from wahabot.core.tickets.store import TicketStore

_THANKS = frozenset(
    {
        "thanks",
        "thankyou",
        "thanku",
        "thx",
        "ty",
        "okthanks",
        "perfectthanks",
        "gracias",
        "danke",
        "obrigado",
    }
)
_THANKS_EMOJI = frozenset({"👍", "🙏", "❤️", "❤", "😊"})
_NON_WORD = re.compile(r"[\s\W_]+", re.UNICODE)
_EMOJI_ONLY = re.compile(r"^[\U0001F300-\U0001FAFF\u2600-\u27BF\s]+$", re.UNICODE)


@dataclass(slots=True)
class Inbound:
    """One inbound customer message's ticket decision."""

    contact: Contact
    ticket: Ticket
    body: str
    reopened: bool = False
    operator_solved: bool = False
    previous_done_by: str | None = None


def customer_details(event: WahaEvent) -> tuple[str, str, str, str]:
    """``(chat_id, phone, lid, display_name)`` from one customer event."""
    chat_id = str(event.payload.get("from", ""))
    sender = str(event.payload.get("participant") or chat_id)
    phone = chat_id if chat_id.endswith("@c.us") else ""
    if sender.endswith("@c.us"):
        phone = phone or sender
    lid = (
        sender
        if sender.endswith("@lid")
        else (chat_id if chat_id.endswith("@lid") else "")
    )
    raw = event.payload.get("_data")
    data: dict[str, Any] = raw if isinstance(raw, dict) else {}
    name = str(data.get("notifyName") or event.payload.get("notifyName") or "")
    return chat_id, phone, lid, name


def contact_language(event: WahaEvent) -> str:
    """The customer's language when the runtime already knows it."""
    raw = event.payload.get("_data")
    data: dict[str, Any] = raw if isinstance(raw, dict) else {}
    return str(data.get("language") or "")


def record_inbound(
    store: TicketStore,
    event: WahaEvent,
    *,
    message_ids: list[str],
    close_after_days: int,
    at: int | None = None,
    body: str | None = None,
) -> Inbound:
    """Resolve the contact, open/reopen its ticket and append the message."""
    moment = store.now() if at is None else at
    chat_id, phone, lid, name = customer_details(event)
    contact = store.upsert_contact(
        chat_id,
        phone=phone,
        lid=lid,
        display_name=name,
        language=contact_language(event),
        last_inbound_at=moment,
    )
    latest_before = store.latest_ticket(contact.chat_id)
    previous_done_by = (
        latest_before.done_by
        if store.active_ticket(contact.chat_id) is None and latest_before is not None
        else None
    )
    ticket, reopened, operator_solved = ticket_for_inbound(
        store, contact, moment, close_after_days
    )
    words = body if body is not None else str(event.payload.get("body", "") or "").strip()
    note_customer_message(store, ticket, words, message_ids, moment)
    return Inbound(
        contact=contact,
        ticket=store.ticket(ticket.id) or ticket,
        body=words,
        reopened=reopened,
        operator_solved=operator_solved,
        previous_done_by=previous_done_by,
    )


def ticket_for_inbound(
    store: TicketStore, contact: Contact, moment: int, close_after_days: int
) -> tuple[Ticket, bool, bool]:
    """The ticket one message belongs to, reopening or creating as needed."""
    active = store.active_ticket(contact.chat_id)
    if active is not None:
        return active, False, False
    latest = store.latest_ticket(contact.chat_id)
    if latest is None:
        return (
            store.create_ticket(contact.id, contact.chat_id, at=moment),
            False,
            False,
        )
    if latest.state == TicketState.DONE.value and within_close_window(
        latest, moment, close_after_days
    ):
        reopened = store.update_ticket(
            latest.id,
            TicketUpdate(state=TicketState.OPEN.value, done_by=None),
            actor=Actor.CUSTOMER.value,
            kind=EventKind.STATE.value,
            body="reopened by a customer message",
            at=moment,
        )
        return reopened, True, latest.done_by == "operator"
    if latest.state == TicketState.DONE.value:
        store.update_ticket(
            latest.id,
            TicketUpdate(state=TicketState.CLOSED.value),
            actor=Actor.SYSTEM.value,
            kind=EventKind.STATE.value,
            body="closed before a new ticket opened",
            at=moment,
        )
    return (
        store.create_ticket(
            contact.id,
            contact.chat_id,
            previous_ticket_id=latest.id,
            at=moment,
        ),
        False,
        False,
    )


def within_close_window(ticket: Ticket, moment: int, close_after_days: int) -> bool:
    """Whether a done ticket is still inside its reopen window."""
    done_at = ticket.done_at or ticket.updated_at
    return moment - done_at <= close_after_days * 86400


def note_customer_message(
    store: TicketStore,
    ticket: Ticket,
    body: str,
    message_ids: list[str],
    moment: int,
) -> None:
    """Append one customer message, folding a burst into a single event."""
    existing = burst_event(store, ticket.id, message_ids)
    if existing is not None:
        store.extend_event_message_ids(existing.id, message_ids)
    else:
        store.add_event(
            ticket.id,
            Actor.CUSTOMER.value,
            EventKind.CUSTOMER_MESSAGE.value,
            body,
            message_ids,
            at=moment,
        )
    store.mark_customer_message(ticket.id, moment)
    if not ticket.summary and body:
        store.update_ticket(
            ticket.id,
            TicketUpdate(summary=body[:240]),
            actor=Actor.SYSTEM.value,
            kind=EventKind.STATE.value,
            body="summary from the first customer message",
            at=moment,
        )


def burst_event(
    store: TicketStore, ticket_id: int, message_ids: list[str]
) -> TicketEvent | None:
    """The ticket event a burst's later rows join, or None."""
    wanted = set(message_ids)
    for event in reversed(store.events(ticket_id)):
        if event.kind == EventKind.CUSTOMER_MESSAGE.value and wanted & set(
            event.message_ids
        ):
            return event
    return None


def is_thanks_only(body: str) -> bool:
    """Whether a message is only a thank-you or a lone appreciative emoji."""
    text = body.strip()
    if not text:
        return False
    if text in _THANKS_EMOJI or _EMOJI_ONLY.fullmatch(text):
        return True
    normalized = _NON_WORD.sub("", text.casefold())
    return normalized in _THANKS


def finalize_delivery(
    store: TicketStore,
    ticket_id: int,
    update: TicketUpdate,
    *,
    delivered: bool,
    reply: str = "",
    at: int | None = None,
) -> Ticket | None:
    """Apply a staged update only after a confirmed delivery."""
    if not delivered:
        return store.ticket(ticket_id)
    resolved = completed_update(update)
    store.update_ticket(
        ticket_id,
        resolved,
        actor=Actor.ASSISTANT.value,
        kind=EventKind.REPLY.value,
        body=reply,
        at=at,
    )
    return store.ticket(ticket_id)


def completed_update(update: TicketUpdate) -> TicketUpdate:
    """Fill the default outcome when the model staged no state."""
    if update.state is not None:
        return update
    if update.missing:
        return TicketUpdate(
            state=TicketState.WAITING_CUSTOMER.value,
            summary=update.summary,
            details=update.details,
            missing=update.missing,
            done_by=update.done_by,
            urgent=update.urgent,
        )
    return TicketUpdate(
        state=TicketState.DONE.value,
        summary=update.summary,
        details=update.details,
        missing=update.missing,
        done_by=update.done_by or "assistant",
        urgent=update.urgent,
    )


def split_ticket(
    store: TicketStore, ticket: Ticket, summary: str, at: int | None = None
) -> Ticket:
    """Split an open/done ticket into a new open one.

    A ``needs_you`` or ``waiting_customer`` ticket is refused: the split
    would detach the operator's current work.
    """
    if ticket.state in (
        TicketState.NEEDS_YOU.value,
        TicketState.WAITING_CUSTOMER.value,
    ):
        raise ValueError("only an open or done ticket can be split")
    moment = store.now() if at is None else at
    if ticket.state == TicketState.OPEN.value:
        store.update_ticket(
            ticket.id,
            TicketUpdate(state=TicketState.DONE.value, done_by="assistant"),
            actor=Actor.ASSISTANT.value,
            kind=EventKind.STATE.value,
            body="split into a new ticket",
            at=moment,
        )
    contact = store.contact_by_id(ticket.contact_id)
    if contact is None:
        raise KeyError(f"contact {ticket.contact_id} not found")
    return store.create_ticket(
        contact.id,
        ticket.chat_id,
        summary=summary,
        previous_ticket_id=ticket.id,
        at=moment,
    )


def close_ticket(
    store: TicketStore, ticket: Ticket, *, actor: str = "operator", at: int | None = None
) -> Ticket:
    """Close one ticket without sending anything."""
    return store.update_ticket(
        ticket.id,
        TicketUpdate(state=TicketState.CLOSED.value),
        actor=actor,
        kind=EventKind.STATE.value,
        body="closed",
        at=at,
    )


def reopen_ticket(
    store: TicketStore, ticket: Ticket, *, actor: str = "operator", at: int | None = None
) -> Ticket:
    """Return a done/closed ticket to open, respecting one active ticket."""
    active = store.active_ticket(ticket.chat_id)
    if active is not None and active.id != ticket.id:
        raise ValueError(
            f"ticket {ticket.number} cannot reopen while {active.number} is active"
        )
    return store.update_ticket(
        ticket.id,
        TicketUpdate(state=TicketState.OPEN.value, done_by=None),
        actor=actor,
        kind=EventKind.STATE.value,
        body="reopened",
        at=at,
    )


def close_overdue(
    store: TicketStore, close_after_days: int, *, at: int | None = None
) -> list[Ticket]:
    """Close done tickets that have been silent past the window."""
    moment = store.now() if at is None else at
    closed: list[Ticket] = []
    for ticket in store.done_tickets():
        if within_close_window(ticket, moment, close_after_days):
            continue
        closed.append(
            store.update_ticket(
                ticket.id,
                TicketUpdate(state=TicketState.CLOSED.value),
                actor=Actor.SYSTEM.value,
                kind=EventKind.STATE.value,
                body="closed after the silence window",
                at=moment,
            )
        )
    return closed
