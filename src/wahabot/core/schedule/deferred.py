"""Deferred operator notices: queue outside hours, send when open."""

from loguru import logger

from wahabot.core.schedule.models import DeferredNotice
from wahabot.core.tickets.alerts import forward_attachments
from wahabot.core.tickets.models import Ticket
from wahabot.core.tickets.store import TicketStore
from wahabot.core.transport import Transport


def defer_notice(
    store: TicketStore,
    ticket: Ticket,
    session: str,
    chat_id: str,
    text: str,
    *,
    now: int,
    at: int | None = None,
) -> int:
    """Queue one notice for the operator until *now* at the earliest."""
    return store.defer_notice(
        ticket.id,
        session,
        chat_id,
        text,
        not_before=now,
        at=store.now() if at is None else at,
    )


def due_notices(store: TicketStore, now: int, limit: int = 10) -> list[DeferredNotice]:
    """Queued notices whose delivery window has opened."""
    return store.pending_notices(now, limit=limit)


def deliver_due_notices(
    store: TicketStore, waha: Transport, now: int, limit: int = 10
) -> int:
    """Send every due notice and mark it delivered; return the count."""
    sent = 0
    for notice in due_notices(store, now, limit=limit):
        try:
            sent_id = waha.send_text(notice.session, notice.chat_id, notice.text)
        except Exception:
            continue
        if not sent_id:
            continue
        store.mark_notice_sent(notice.id, sent_id, now)
        ticket = store.ticket(notice.ticket_id)
        if ticket is not None:
            try:
                forward_attachments(store, waha, notice.session, notice.chat_id, ticket)
            except Exception as exc:
                logger.warning(
                    "Deferred alert attachment forward failed for {id}: {exc}",
                    id=notice.ticket_id,
                    exc=exc,
                )
        sent += 1
    return sent
