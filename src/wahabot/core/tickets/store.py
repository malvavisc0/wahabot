"""SQLite storage for tickets, events and attachments."""

import json
import sqlite3
from collections.abc import Iterable

from wahabot.core.db.connection import connect, transaction
from wahabot.core.tickets.models import (
    ACTIVE_STATES,
    Actor,
    Attachment,
    Contact,
    EventKind,
    Ticket,
    TicketEvent,
    TicketState,
    TicketUpdate,
    json_list,
)
from wahabot.core.tickets.store_base import (
    attachment_from_row,
    contact_from_row,
    event_from_row,
    ticket_from_row,
)
from wahabot.core.tickets.store_contacts import ContactMixin

_TICKET_COLUMNS = (
    "id, number, contact_id, chat_id, state, urgent, done_by, summary, "
    "details_json, missing_json, external_user_ref, alert_message_ids_json, "
    "previous_ticket_id, created_at, updated_at, done_at, closed_at, last_customer_at"
)
_EVENT_COLUMNS = "id, ticket_id, at, actor, kind, body, message_ids_json"
_ATTACHMENT_COLUMNS = (
    "id, ticket_id, message_id, kind, mime, bytes, sha256, path, derived_text"
)


class TicketStore(ContactMixin):
    """CRUD and transactions for the ticket tables.

    One connection per operation; every write that changes a ticket's
    state also appends a ``ticket_event`` in the same transaction.
    """

    def create_ticket(
        self,
        contact_id: int,
        chat_id: str,
        *,
        summary: str = "",
        urgent: bool = False,
        previous_ticket_id: int | None = None,
        state: str = TicketState.OPEN.value,
        external_user_ref: str = "",
        at: int | None = None,
    ) -> Ticket:
        """Insert one ticket, assigning ``number`` from its row id."""
        moment = self.now() if at is None else at
        conn = connect(self.path)
        try:
            with transaction(conn):
                cursor = conn.execute(
                    "INSERT INTO tickets (number, contact_id, chat_id, state, urgent, "
                    + "summary, external_user_ref, previous_ticket_id, created_at, "
                    + "updated_at, last_customer_at) VALUES ("
                    + "(SELECT COALESCE(MAX(id), 0) + 1 FROM tickets), "
                    + "?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        contact_id,
                        chat_id,
                        state,
                        int(urgent),
                        summary,
                        external_user_ref,
                        previous_ticket_id,
                        moment,
                        moment,
                        moment,
                    ),
                )
                ticket_id = int(cursor.lastrowid or 0)
                conn.execute(
                    "UPDATE tickets SET number = ? WHERE id = ?", (ticket_id, ticket_id)
                )
                self.insert_event(
                    conn,
                    ticket_id,
                    Actor.SYSTEM.value,
                    EventKind.OPENED.value,
                    summary,
                    (),
                    moment,
                )
                row = conn.execute(
                    f"SELECT {_TICKET_COLUMNS} FROM tickets WHERE id = ?", (ticket_id,)
                ).fetchone()
        finally:
            conn.close()
        return ticket_from_row(row)

    def ticket(self, ticket_id: int) -> Ticket | None:
        """One ticket by row id."""
        return self.query_one(
            f"SELECT {_TICKET_COLUMNS} FROM tickets WHERE id = ?",
            (ticket_id,),
            ticket_from_row,
        )

    def ticket_by_number(self, number: int) -> Ticket | None:
        """One ticket by its operator-facing number."""
        return self.query_one(
            f"SELECT {_TICKET_COLUMNS} FROM tickets WHERE number = ?",
            (number,),
            ticket_from_row,
        )

    def active_ticket(self, chat_id: str) -> Ticket | None:
        """The chat's one active ticket, when it has one."""
        marks = ",".join("?" for _ in ACTIVE_STATES)
        return self.query_one(
            f"SELECT {_TICKET_COLUMNS} FROM tickets WHERE chat_id = ? "
            + f"AND state IN ({marks}) ORDER BY number DESC LIMIT 1",
            (chat_id, *ACTIVE_STATES),
            ticket_from_row,
        )

    def latest_ticket(self, chat_id: str) -> Ticket | None:
        """The chat's newest ticket, whatever its state."""
        return self.query_one(
            f"SELECT {_TICKET_COLUMNS} FROM tickets WHERE chat_id = ? "
            + "ORDER BY number DESC LIMIT 1",
            (chat_id,),
            ticket_from_row,
        )

    def chat_tickets(self, chat_id: str, limit: int = 5) -> list[Ticket]:
        """One chat's newest tickets, whatever their states."""
        return self.query_all(
            f"SELECT {_TICKET_COLUMNS} FROM tickets WHERE chat_id = ? "
            + "ORDER BY number DESC LIMIT ?",
            (chat_id, limit),
            ticket_from_row,
        )

    def tickets_in_states(
        self, states: Iterable[str] = ACTIVE_STATES, limit: int | None = None
    ) -> list[Ticket]:
        """Tickets in *states*, newest first."""
        values = tuple(states)
        marks = ",".join("?" for _ in values)
        sql = (
            f"SELECT {_TICKET_COLUMNS} FROM tickets WHERE state IN ({marks}) "
            + "ORDER BY number DESC"
        )
        params: tuple[object, ...] = values
        if limit is not None:
            sql += " LIMIT ?"
            params = (*values, limit)
        return self.query_all(sql, params, ticket_from_row)

    def update_ticket(
        self,
        ticket_id: int,
        update: TicketUpdate,
        *,
        actor: str = Actor.ASSISTANT.value,
        kind: str = EventKind.STATE.value,
        body: str = "",
        at: int | None = None,
    ) -> Ticket:
        """Apply staged values and append the matching event atomically."""
        moment = self.now() if at is None else at
        conn = connect(self.path)
        try:
            with transaction(conn):
                current = conn.execute(
                    f"SELECT {_TICKET_COLUMNS} FROM tickets WHERE id = ?", (ticket_id,)
                ).fetchone()
                if current is None:
                    raise KeyError(f"ticket {ticket_id} not found")
                self.apply_update(conn, current, update, moment)
                self.insert_event(conn, ticket_id, actor, kind, body, (), moment)
                row = conn.execute(
                    f"SELECT {_TICKET_COLUMNS} FROM tickets WHERE id = ?", (ticket_id,)
                ).fetchone()
        finally:
            conn.close()
        return ticket_from_row(row)

    def apply_update(
        self,
        conn: sqlite3.Connection,
        current: sqlite3.Row,
        update: TicketUpdate,
        moment: int,
    ) -> None:
        """Write the non-empty parts of one staged update."""
        fields: list[str] = ["updated_at = ?"]
        params: list[object] = [moment]
        if update.state is not None:
            fields.append("state = ?")
            params.append(update.state)
            self.apply_state_times(conn, current, update, moment, fields, params)
        if update.summary is not None:
            fields.append("summary = ?")
            params.append(update.summary)
        if update.details is not None:
            fields.append("details_json = ?")
            params.append(json.dumps(update.details, sort_keys=True))
        if update.missing is not None:
            fields.append("missing_json = ?")
            params.append(json.dumps(update.missing))
        if update.urgent is not None:
            fields.append("urgent = ?")
            params.append(int(update.urgent))
        if update.done_by is not None:
            fields.append("done_by = ?")
            params.append(update.done_by)
        conn.execute(
            f"UPDATE tickets SET {', '.join(fields)} WHERE id = ?",
            (*params, current["id"]),
        )

    def apply_state_times(
        self,
        conn: sqlite3.Connection,
        current: sqlite3.Row,
        update: TicketUpdate,
        moment: int,
        fields: list[str],
        params: list[object],
    ) -> None:
        """Set the done/closed timestamps that belong to the new state."""
        if update.state == TicketState.DONE.value:
            fields.extend(["done_at = ?", "closed_at = NULL"])
            params.append(moment)
        elif update.state == TicketState.CLOSED.value:
            fields.extend(["closed_at = ?", "done_at = COALESCE(done_at, ?)"])
            params.extend([moment, moment])
        elif update.state in ACTIVE_STATES and current["state"] in ACTIVE_STATES:
            return
        elif update.state in ACTIVE_STATES:
            fields.extend(["done_at = NULL", "closed_at = NULL"])
            if update.done_by is None:
                fields.append("done_by = NULL")

    def add_event(
        self,
        ticket_id: int,
        actor: str,
        kind: str,
        body: str = "",
        message_ids: Iterable[str] = (),
        *,
        at: int | None = None,
    ) -> TicketEvent:
        """Append one timeline event."""
        moment = self.now() if at is None else at
        conn = connect(self.path)
        try:
            with transaction(conn):
                event_id = self.insert_event(
                    conn, ticket_id, actor, kind, body, tuple(message_ids), moment
                )
                row = conn.execute(
                    f"SELECT {_EVENT_COLUMNS} FROM ticket_events WHERE id = ?",
                    (event_id,),
                ).fetchone()
        finally:
            conn.close()
        return event_from_row(row)

    def insert_event(
        self,
        conn: sqlite3.Connection,
        ticket_id: int,
        actor: str,
        kind: str,
        body: str,
        message_ids: Iterable[str],
        moment: int,
    ) -> int:
        """Insert an event inside an existing transaction."""
        cursor = conn.execute(
            "INSERT INTO ticket_events (ticket_id, at, actor, kind, body, "
            + "message_ids_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                ticket_id,
                moment,
                actor,
                kind,
                body,
                json.dumps(list(message_ids)),
            ),
        )
        return int(cursor.lastrowid or 0)

    def extend_event_message_ids(self, event_id: int, message_ids: Iterable[str]) -> None:
        """Add message ids to an existing event, keeping one event per burst."""
        conn = connect(self.path)
        try:
            with transaction(conn):
                row = conn.execute(
                    "SELECT message_ids_json FROM ticket_events WHERE id = ?",
                    (event_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"ticket event {event_id} not found")
                stored = json_list(str(row["message_ids_json"]))
                for message_id in message_ids:
                    if message_id and message_id not in stored:
                        stored.append(message_id)
                conn.execute(
                    "UPDATE ticket_events SET message_ids_json = ? WHERE id = ?",
                    (json.dumps(stored), event_id),
                )
        finally:
            conn.close()

    def events(self, ticket_id: int) -> list[TicketEvent]:
        """One ticket's timeline, oldest first."""
        return self.query_all(
            f"SELECT {_EVENT_COLUMNS} FROM ticket_events WHERE ticket_id = ? "
            + "ORDER BY at, id",
            (ticket_id,),
            event_from_row,
        )

    def last_event(self, ticket_id: int, kind: str) -> TicketEvent | None:
        """The newest event of *kind* on a ticket."""
        return self.query_one(
            f"SELECT {_EVENT_COLUMNS} FROM ticket_events WHERE ticket_id = ? "
            + "AND kind = ? ORDER BY at DESC, id DESC LIMIT 1",
            (ticket_id, kind),
            event_from_row,
        )

    def event_with_message_ids(
        self, ticket_id: int, message_ids: Iterable[str], kind: str
    ) -> TicketEvent | None:
        """The newest event of *kind* carrying any of *message_ids*."""
        wanted = [item for item in message_ids if item]
        if not wanted:
            return None
        placeholders = ", ".join("?" for _ in wanted)
        return self.query_one(
            "SELECT ticket_events.* FROM ticket_events, "
            + "json_each(ticket_events.message_ids_json) "
            + "WHERE ticket_events.ticket_id = ? AND ticket_events.kind = ? "
            + f"AND json_each.value IN ({placeholders}) "
            + "ORDER BY ticket_events.id DESC LIMIT 1",
            (ticket_id, kind, *wanted),
            event_from_row,
        )

    def mark_customer_message(self, ticket_id: int, at: int | None = None) -> None:
        """Move a ticket's last-customer timestamp forward."""
        moment = self.now() if at is None else at
        self.execute(
            "UPDATE tickets SET last_customer_at = ?, updated_at = ? WHERE id = ?",
            (moment, moment, ticket_id),
        )

    def append_alert_ids(self, ticket_id: int, message_ids: Iterable[str]) -> None:
        """Store every alert or forwarded id so quote replies can route."""
        conn = connect(self.path)
        try:
            with transaction(conn):
                row = conn.execute(
                    "SELECT alert_message_ids_json FROM tickets WHERE id = ?",
                    (ticket_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"ticket {ticket_id} not found")
                stored = json_list(str(row["alert_message_ids_json"]))
                for message_id in message_ids:
                    if message_id and message_id not in stored:
                        stored.append(message_id)
                conn.execute(
                    "UPDATE tickets SET alert_message_ids_json = ? WHERE id = ?",
                    (json.dumps(stored), ticket_id),
                )
        finally:
            conn.close()

    def ticket_for_alert_message(self, message_id: str) -> Ticket | None:
        """The ticket whose alert or forwarded message carries *message_id*."""
        if not message_id:
            return None
        return self.query_one(
            "SELECT tickets.* FROM tickets, json_each(tickets.alert_message_ids_json) "
            + "WHERE json_each.value = ? ORDER BY tickets.number DESC LIMIT 1",
            (message_id,),
            ticket_from_row,
        )

    def add_attachment(
        self,
        ticket_id: int,
        message_id: str,
        kind: str,
        *,
        mime: str = "",
        size: int = 0,
        sha256: str = "",
        path: str = "",
        derived_text: str = "",
    ) -> Attachment:
        """Record one stored attachment."""
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "INSERT INTO attachments (ticket_id, message_id, kind, mime, bytes, "
                    + "sha256, path, derived_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (ticket_id, message_id, kind, mime, size, sha256, path, derived_text),
                )
                row = conn.execute(
                    f"SELECT {_ATTACHMENT_COLUMNS} FROM attachments WHERE id = "
                    + "last_insert_rowid()"
                ).fetchone()
        finally:
            conn.close()
        return attachment_from_row(row)

    def attachments(self, ticket_id: int) -> list[Attachment]:
        """One ticket's stored attachments, oldest first."""
        return self.query_all(
            f"SELECT {_ATTACHMENT_COLUMNS} FROM attachments WHERE ticket_id = ? "
            + "ORDER BY id",
            (ticket_id,),
            attachment_from_row,
        )

    def done_tickets(self) -> list[Ticket]:
        """Every done ticket, oldest first."""
        return self.query_all(
            f"SELECT {_TICKET_COLUMNS} FROM tickets WHERE state = ? ORDER BY number",
            (TicketState.DONE.value,),
            ticket_from_row,
        )

    def needs_you_tickets(self) -> list[Ticket]:
        """Every ticket waiting on the operator, newest first."""
        return self.query_all(
            f"SELECT {_TICKET_COLUMNS} FROM tickets WHERE state = ? ORDER BY number DESC",
            (TicketState.NEEDS_YOU.value,),
            ticket_from_row,
        )

    def paused_contacts(self) -> list[Contact]:
        """Every contact carrying a pause deadline."""
        return self.query_all(
            "SELECT * FROM contacts WHERE paused_until IS NOT NULL",
            (),
            contact_from_row,
        )
