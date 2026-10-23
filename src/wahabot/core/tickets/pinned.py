"""Render the one-line ticket marker for the pinned state block."""

from wahabot.core.tickets.models import Ticket, TicketState

_STATE_LABELS = {
    TicketState.OPEN.value: "open",
    TicketState.WAITING_CUSTOMER.value: "waiting for customer",
    TicketState.NEEDS_YOU.value: "needs you",
    TicketState.DONE.value: "done",
    TicketState.CLOSED.value: "closed",
}


def ticket_marker(ticket: Ticket | None) -> str:
    """One line naming the ticket and its state, or "" without a ticket."""
    if ticket is None:
        return ""
    label = _STATE_LABELS.get(ticket.state, ticket.state)
    urgent = " (urgent)" if ticket.urgent else ""
    return f"Ticket {ticket.number}: {label}{urgent}"
