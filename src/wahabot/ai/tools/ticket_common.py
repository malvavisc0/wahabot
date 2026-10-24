"""Shared ticket tool helpers: context lookup and staged updates."""

from collections.abc import Callable

from wahabot.core.tickets.models import Contact, Ticket, TicketUpdate
from wahabot.core.tickets.store import TicketStore

OperatorLanguage = str | Callable[[], str]


def ticket_context(
    store: TicketStore | None, ticket_id: int, chat_id: str
) -> tuple[Ticket | None, Contact | None]:
    """The current ticket and contact for a run, falling back to the chat."""
    if store is None:
        return None, None
    ticket = store.ticket(ticket_id) if ticket_id else store.active_ticket(chat_id)
    if ticket is None:
        ticket = store.latest_ticket(chat_id)
    contact = store.contact_by_id(ticket.contact_id) if ticket is not None else None
    return ticket, contact


def merged_details(
    base: dict[str, object] | None, details: dict[str, str] | None
) -> dict[str, object] | None:
    """Merge a partial details dict into the run's staged dict."""
    if details is None:
        return base
    return {**(base or {}), **details}


def merged_missing(base: list[str] | None, missing: list[str] | None) -> list[str] | None:
    """Add missing items while allowing an explicit empty list to clear them."""
    if missing is None:
        return base
    if not missing:
        return []
    return list({item: None for item in [*(base or []), *missing]})


def merge_update(
    existing: TicketUpdate | None,
    *,
    state: str | None = None,
    summary: str | None = None,
    details: dict[str, str] | None = None,
    missing: list[str] | None = None,
    done_by: str | None = None,
) -> TicketUpdate:
    """Merge newly staged values into the run's update."""
    base = existing or TicketUpdate()
    return TicketUpdate(
        state=state if state is not None else base.state,
        summary=summary if summary is not None else base.summary,
        details=merged_details(base.details, details),
        missing=merged_missing(base.missing, missing),
        done_by=done_by if done_by is not None else base.done_by,
        urgent=base.urgent,
    )
