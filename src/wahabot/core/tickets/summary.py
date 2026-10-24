"""The morning operator summary, counted in the session timezone."""

import contextlib
import datetime
from collections.abc import Callable, Mapping, Sequence

from wahabot.core.schedule.hours import opening_state, session_zone
from wahabot.core.schedule.models import DayHours, ScheduleWindow
from wahabot.core.tickets.models import DoneBy, Ticket
from wahabot.core.tickets.store import TicketStore
from wahabot.core.transport import Transport

SUMMARY_HEADING = "Yesterday"


def local_date(now: int, timezone: str) -> datetime.date:
    """The local calendar date at *now*."""
    return datetime.datetime.fromtimestamp(now, tz=session_zone(timezone)).date()


def day_window(now: int, timezone: str) -> tuple[int, int]:
    """Yesterday's local half-open [start, end) instants."""
    zone = session_zone(timezone)
    today = datetime.datetime.fromtimestamp(now, tz=zone).date()
    start = datetime.datetime.combine(
        today - datetime.timedelta(days=1), datetime.time.min, tzinfo=zone
    )
    end = datetime.datetime.combine(today, datetime.time.min, tzinfo=zone)
    return int(start.timestamp()), int(end.timestamp())


def needs_you_line(store: TicketStore, ticket: Ticket, now: int, timezone: str) -> str:
    """One needs-you line with the customer's age and urgency."""
    contact = store.contact_by_id(ticket.contact_id)
    person = contact.display if contact is not None else "unknown"
    since = ticket.last_customer_at or ticket.updated_at
    age = max(0, now - since) // 3600
    flag = " URGENT" if ticket.urgent else ""
    return f"- {ticket.number}: {person}, {age}h, {ticket.state}{flag}"


def build_summary(
    store: TicketStore,
    session: str,
    now: int,
    timezone: str,
    fallback_timezone: str = "UTC",
) -> str:
    """The operator summary: yesterday's counts, then needs-you work."""
    zone = session_zone(timezone, fallback_timezone)
    start, end = day_window(now, str(zone))
    yesterday = store.tickets_between(start, end)
    finished = store.tickets_done_between(start, end)
    done_assistant = sum(
        1 for ticket in finished if ticket.done_by == DoneBy.ASSISTANT.value
    )
    done_operator = sum(
        1 for ticket in finished if ticket.done_by == DoneBy.OPERATOR.value
    )
    waiting = list(needs_you_order(store))
    lines = [
        (
            f"{SUMMARY_HEADING}: {len(yesterday)} tickets. "
            f"{done_assistant} done by the assistant, {done_operator} by you, "
            f"{len(waiting)} waiting for you."
        )
    ]
    if waiting:
        lines.append("Needs you:")
        lines.extend(needs_you_line(store, ticket, now, str(zone)) for ticket in waiting)
    else:
        lines.append("No open tickets.")
    return "\n".join(lines)


def summary_due(
    store: TicketStore,
    session: str,
    now: int,
    timezone: str,
    *,
    hours: Mapping[str, object] | None = None,
    fallback_timezone: str = "UTC",
) -> bool:
    """Whether today's first opening has happened and no summary was sent."""
    zone = session_zone(timezone, fallback_timezone)
    today = str(local_date(now, str(zone)))
    if store.last_summary_date(session) == today:
        return False
    normalized = {
        day: DayHours.model_validate(value) for day, value in (hours or {}).items()
    }
    config = ScheduleWindow(timezone=str(zone), hours=normalized)
    return opening_state(config, now, fallback_timezone).open


def send_summary(
    store: TicketStore,
    waha: Transport,
    session: str,
    destination: str,
    now: int,
    timezone: str,
    *,
    hours: Mapping[str, DayHours] | None = None,
    operator_language: str = "en",
    translate: Callable[[str, str], str] | None = None,
    fallback_timezone: str = "UTC",
) -> str | None:
    """Send today's summary once and record the date and message id."""
    if not summary_due(
        store,
        session,
        now,
        timezone,
        hours=hours,
        fallback_timezone=fallback_timezone,
    ):
        return None
    zone = str(session_zone(timezone, fallback_timezone))
    text = build_summary(store, session, now, zone, fallback_timezone)
    language = (operator_language or "en").strip() or "en"
    if language != "en" and translate is not None:
        with contextlib.suppress(Exception):
            text = translate(text, language)
    sent_id = waha.send_text(session, destination, text)
    if not sent_id:
        return None
    store.set_summary_message_id(session, sent_id)
    store.set_last_summary_date(session, str(local_date(now, str(zone))))
    # The summary lists every needs-you ticket, so a queued individual alert
    # for the same ticket would page the operator twice at the same opening.
    for ticket in store.needs_you_tickets():
        store.cancel_notices(ticket.id, now)
    return sent_id


def needs_you_order(store: TicketStore) -> Sequence[Ticket]:
    """Needs-you tickets, urgent first then longest waiting first."""
    return sorted(
        store.needs_you_tickets(),
        key=lambda item: (
            not item.urgent,
            item.last_customer_at or item.updated_at,
            item.number,
        ),
    )
