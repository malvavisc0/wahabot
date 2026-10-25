"""Post-commit ticket lifecycle notifications."""

from collections.abc import Callable

from loguru import logger

from wahabot.core.tickets.models import Ticket, TicketState


class LifecycleMixin:
    """Run lifecycle listeners after the ticket transaction committed."""

    on_event: Callable[[str, Ticket], None] | None = None

    def notify(self, kind: str, ticket: Ticket) -> None:
        """Run listeners; a listener failure never rolls back or blocks a ticket."""
        if self.on_event is None:
            return
        try:
            self.on_event(kind, ticket)
        except Exception:
            logger.exception("Ticket lifecycle listener failed for {kind}", kind=kind)

    def notify_transition(self, previous: str, ticket: Ticket) -> None:
        """Map a state change to the lifecycle event skills subscribe to."""
        if ticket.state == previous:
            return
        if ticket.state == TicketState.NEEDS_YOU.value:
            self.notify("needs_you", ticket)
        elif ticket.state == TicketState.DONE.value:
            self.notify("done", ticket)
        elif ticket.state == TicketState.CLOSED.value:
            self.notify("closed", ticket)
        elif ticket.state == TicketState.OPEN.value and previous in (
            TicketState.DONE.value,
            TicketState.CLOSED.value,
        ):
            self.notify("opened", ticket)
