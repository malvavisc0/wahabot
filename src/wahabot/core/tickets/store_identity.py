"""Atomic ticket/contact external identity linking."""

from wahabot.core.db.connection import connect, transaction
from wahabot.core.tickets.models import Actor, EventKind, Ticket
from wahabot.core.tickets.store_base import (
    TICKET_COLUMNS,
    StoreBase,
    ticket_from_row,
)


class IdentityMixin(StoreBase):
    """Write a ticket and its contact's external user reference together."""

    def link_ticket_user(self, ticket_id: int, user_ref: str) -> Ticket:
        """Link a ticket and its contact to one external user reference."""
        cleaned = user_ref.strip()
        if not cleaned:
            raise ValueError("external user reference must not be empty")
        conn = connect(self.path)
        try:
            with transaction(conn):
                row = conn.execute(
                    f"SELECT {TICKET_COLUMNS} FROM tickets WHERE id = ?", (ticket_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"ticket {ticket_id} not found")
                conn.execute(
                    "UPDATE tickets SET external_user_ref = ?, updated_at = ? "
                    + "WHERE id = ?",
                    (cleaned, self.now(), ticket_id),
                )
                conn.execute(
                    "UPDATE contacts SET external_user_ref = ? WHERE id = ?",
                    (cleaned, row["contact_id"]),
                )
                conn.execute(
                    "INSERT INTO ticket_events (ticket_id, at, actor, kind, body, "
                    + "message_ids_json) VALUES (?, ?, ?, ?, ?, '[]')",
                    (
                        ticket_id,
                        self.now(),
                        Actor.OPERATOR.value,
                        EventKind.STATE.value,
                        "external user reference linked",
                    ),
                )
                updated = conn.execute(
                    f"SELECT {TICKET_COLUMNS} FROM tickets WHERE id = ?", (ticket_id,)
                ).fetchone()
        finally:
            conn.close()
        return ticket_from_row(updated)
