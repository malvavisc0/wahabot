"""Ticket lifecycle: opening, reopening, finalizing and closing."""

import re
import sqlite3
from dataclasses import dataclass, replace

from loguru import logger

from wahabot.core.models import WahaEvent
from wahabot.core.tickets.identity import (
    customer_details,
    normalize_lid,
    normalize_phone,
)
from wahabot.core.tickets.models import (
    Actor,
    Contact,
    DoneBy,
    EventKind,
    Ticket,
    TicketEvent,
    TicketState,
    TicketUpdate,
    text_field,
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
_EMOJI_SKIP = re.compile(r"[\s\ufe0f\u200d\U0001F3FB-\U0001F3FF]")
GLOBAL_PAUSE_FOREVER = -1


@dataclass(slots=True)
class Inbound:
    """One inbound customer message's ticket decision."""

    contact: Contact
    ticket: Ticket
    body: str
    reopened: bool = False
    operator_solved: bool = False
    previous_done_by: str | None = None


def inbound_words(events: list[WahaEvent]) -> tuple[WahaEvent, str]:
    """The event carrying text and the combined body for a ticket event."""
    bodies = [
        str(event.payload.get("body", "") or "").strip()
        for event in events
        if str(event.payload.get("body", "") or "").strip()
    ]
    content = next(
        (event for event in events if str(event.payload.get("body", "") or "").strip()),
        events[0],
    )
    return content, "\n".join(bodies)


def finalize_split(store: TicketStore, ticket: Ticket, summary: str) -> None:
    """Create the staged split ticket."""
    if not summary:
        return
    refreshed = store.ticket(ticket.id)
    if refreshed is not None:
        split_ticket(store, refreshed, summary)


def contact_language(event: WahaEvent) -> str:
    """The customer's language when the runtime already knows it."""
    return text_field(event.payload.get("_data"), "language")


def message_received_at(event: WahaEvent) -> int | None:
    """The customer's send time, when the event carries one."""
    for value in (event.payload.get("timestamp"), getattr(event, "timestamp", None)):
        if value is None:
            continue
        try:
            moment = int(value)
        except TypeError, ValueError:
            continue
        if moment > 0:
            return moment
    return None


def inbound_timestamp(
    event: WahaEvent, at: int | None, received_at: int | None, moment: int
) -> int:
    """The customer's send time, preferring explicit values."""
    if received_at is not None:
        return received_at
    if at is not None:
        return at
    return message_received_at(event) or moment


def prior_done_by(store: TicketStore, contact: Contact) -> str | None:
    """Who resolved the contact's previous ticket, when none is active."""
    latest = store.latest_ticket(contact.chat_id)
    if latest is None or store.active_ticket(contact.chat_id) is not None:
        return None
    return latest.done_by


def record_inbound(
    store: TicketStore,
    event: WahaEvent,
    *,
    message_ids: list[str],
    close_after_days: int,
    at: int | None = None,
    received_at: int | None = None,
    body: str | None = None,
    phone: str = "",
    lid: str = "",
) -> Inbound:
    """Resolve the contact, open/reopen its ticket and append the message."""
    moment = store.now() if at is None else at
    chat_id, found_phone, found_lid, name = customer_details(event)
    contact = store.upsert_contact(
        chat_id,
        phone=normalize_phone(phone) or found_phone,
        lid=normalize_lid(lid) or found_lid,
        display_name=name,
        language=contact_language(event),
        last_inbound_at=inbound_timestamp(event, at, received_at, moment),
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
        previous_done_by=prior_done_by(store, contact),
    )


def open_ticket(
    store: TicketStore,
    contact: Contact,
    moment: int,
    previous_ticket_id: int | None = None,
) -> Ticket:
    """Create a ticket, reusing the active one if a concurrent message won."""
    try:
        return store.create_ticket(
            contact.id,
            contact.chat_id,
            previous_ticket_id=previous_ticket_id,
            external_user_ref=contact.external_user_ref,
            at=moment,
        )
    except sqlite3.IntegrityError:
        active = store.active_ticket(contact.chat_id)
        if active is None:
            raise
        return active


def ticket_for_inbound(
    store: TicketStore, contact: Contact, moment: int, close_after_days: int
) -> tuple[Ticket, bool, bool]:
    """The ticket one message belongs to, reopening or creating as needed."""
    active = store.active_ticket(contact.chat_id)
    if active is not None:
        return active, False, False
    latest = store.latest_ticket(contact.chat_id)
    if latest is None:
        return open_ticket(store, contact, moment), False, False
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
        return reopened, True, latest.done_by == DoneBy.OPERATOR.value
    if latest.state == TicketState.DONE.value:
        store.update_ticket(
            latest.id,
            TicketUpdate(state=TicketState.CLOSED.value),
            actor=Actor.SYSTEM.value,
            kind=EventKind.STATE.value,
            body="closed before a new ticket opened",
            at=moment,
        )
    return open_ticket(store, contact, moment, latest.id), False, False


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
    return store.event_with_message_ids(
        ticket_id, message_ids, EventKind.CUSTOMER_MESSAGE.value
    )


def is_thanks_only(body: str) -> bool:
    """Whether a message is only a thank-you or a lone appreciative emoji."""
    text = body.strip()
    if not text:
        return False
    emoji = _EMOJI_SKIP.sub("", text)
    if emoji and all(char in _THANKS_EMOJI for char in emoji):
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
        if update.state == TicketState.DONE.value and update.done_by is None:
            return TicketUpdate(
                state=update.state,
                summary=update.summary,
                details=update.details,
                missing=update.missing,
                done_by=DoneBy.ASSISTANT.value,
                urgent=update.urgent,
            )
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
        done_by=update.done_by or DoneBy.ASSISTANT.value,
        urgent=update.urgent,
    )


def closed_mid_run(ticket: Ticket, staged: TicketUpdate) -> bool:
    """Whether an operator closed the ticket while the run was in flight."""
    return (
        ticket.state == TicketState.CLOSED.value
        and staged.state != TicketState.CLOSED.value
    )


def apply_split(
    store: TicketStore,
    ticket: Ticket,
    summary: str,
    staged: TicketUpdate,
) -> TicketUpdate:
    """Split the ticket and force the old one done, or return *staged*."""
    if not summary:
        return staged
    try:
        finalize_split(store, ticket, summary)
    except Exception:
        logger.exception("Ticket split failed for {number}", number=ticket.number)
        return staged
    return replace(
        staged,
        state=TicketState.DONE.value,
        done_by=staged.done_by or DoneBy.ASSISTANT.value,
    )


def apply_delivery(
    store: TicketStore, ticket: Ticket, staged: TicketUpdate, reply: str
) -> None:
    """Apply the staged delivery, logging instead of failing the turn."""
    try:
        finalize_delivery(store, ticket.id, staged, delivered=True, reply=reply)
    except Exception:
        logger.exception("Ticket finalize failed for {number}", number=ticket.number)


def queue_alert(store: TicketStore, ticket: Ticket, *, at: int | None = None) -> Ticket:
    """Mark a ticket needs_you when its alert is queued for retry."""
    return store.update_ticket(
        ticket.id,
        TicketUpdate(state=TicketState.NEEDS_YOU.value),
        actor=Actor.ASSISTANT.value,
        kind=EventKind.STATE.value,
        body="alert queued for retry",
        at=at,
    )


def is_paused(store: TicketStore, session: str, inbound: Inbound, moment: int) -> bool:
    """Whether a contact, any of its aliases, or the session is paused."""
    contact = inbound.contact
    if contact.paused_until is not None and contact.paused_until > moment:
        return True
    aliases = {
        contact.chat_id,
        contact.phone,
        contact.lid,
        inbound.ticket.chat_id,
    }
    for alias in aliases:
        if not alias:
            continue
        until = store.chat_paused_until(alias)
        if until is not None and until > moment:
            return True
    global_until = store.session_paused_until(session)
    return global_until is not None and (
        global_until == GLOBAL_PAUSE_FOREVER or global_until > moment
    )


def holding_due(store: TicketStore, ticket_id: int, moment: int, cooldown_s: int) -> bool:
    """Whether a needs-you ticket should send another holding reply."""
    holding = store.last_event(ticket_id, EventKind.HOLDING.value)
    if holding is None:
        return True
    alert = store.last_event(ticket_id, EventKind.ALERT.value)
    if alert is not None and (alert.at, alert.id) > (holding.at, holding.id):
        return True
    return moment - holding.at >= cooldown_s


def split_ticket(
    store: TicketStore, ticket: Ticket, summary: str, at: int | None = None
) -> Ticket:
    """Split an open/done ticket into a new open one."""
    if ticket.state not in (TicketState.OPEN.value, TicketState.DONE.value):
        raise ValueError("only an open or done ticket can be split")
    moment = store.now() if at is None else at
    if ticket.state == TicketState.OPEN.value:
        store.update_ticket(
            ticket.id,
            TicketUpdate(state=TicketState.DONE.value, done_by=DoneBy.ASSISTANT.value),
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
    moment = store.now() if at is None else at
    closed = store.update_ticket(
        ticket.id,
        TicketUpdate(state=TicketState.CLOSED.value),
        actor=actor,
        kind=EventKind.STATE.value,
        body="closed",
        at=moment,
    )
    store.cancel_notices(ticket.id, moment)
    return closed


def reopen_ticket(
    store: TicketStore, ticket: Ticket, *, actor: str = "operator", at: int | None = None
) -> Ticket:
    """Return a done/closed ticket to open, respecting one active ticket."""
    active = store.active_ticket(ticket.chat_id)
    if active is not None and active.id != ticket.id:
        raise ValueError(
            f"ticket {ticket.number} cannot reopen while {active.number} is active"
        )
    moment = store.now() if at is None else at
    reopened = store.update_ticket(
        ticket.id,
        TicketUpdate(state=TicketState.OPEN.value, done_by=None),
        actor=actor,
        kind=EventKind.STATE.value,
        body="reopened",
        at=moment,
    )
    store.cancel_notices(ticket.id, moment)
    return reopened


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


def close_stale(
    store: TicketStore, close_after_days: int, *, at: int | None = None
) -> list[Ticket]:
    """Close open/waiting tickets that nobody touched for the window."""
    moment = store.now() if at is None else at
    closed: list[Ticket] = []
    for ticket in store.tickets_in_states(
        (TicketState.OPEN.value, TicketState.WAITING_CUSTOMER.value)
    ):
        last = max(ticket.updated_at, ticket.last_customer_at or 0)
        if moment - last <= close_after_days * 86400:
            continue
        closed.append(
            store.update_ticket(
                ticket.id,
                TicketUpdate(state=TicketState.CLOSED.value),
                actor=Actor.SYSTEM.value,
                kind=EventKind.STATE.value,
                body="closed after inactivity",
                at=moment,
            )
        )
    return closed
