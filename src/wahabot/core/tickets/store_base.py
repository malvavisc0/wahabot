"""Shared connection helpers and row mapping for the ticket stores."""

import hashlib
import sqlite3
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from wahabot.core.db.connection import connect, transaction
from wahabot.core.tickets.models import (
    Attachment,
    Contact,
    Ticket,
    TicketEvent,
    json_list,
    json_object,
)


class StoreBase:
    """Connection-per-operation helpers shared by every ticket mixin."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.clock = clock

    def now(self) -> int:
        """The store's current wall-clock second."""
        return int(self.clock())

    def execute(self, sql: str, params: Iterable[object]) -> None:
        """Run one write statement."""
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(sql, tuple(params))
        finally:
            conn.close()

    def query_one[T](
        self, sql: str, params: Iterable[object], mapper: Callable[[sqlite3.Row], T]
    ) -> T | None:
        """Run a query returning at most one mapped row."""
        conn = connect(self.path)
        try:
            row = conn.execute(sql, tuple(params)).fetchone()
        finally:
            conn.close()
        return mapper(row) if row is not None else None

    def query_all[T](
        self, sql: str, params: Iterable[object], mapper: Callable[[sqlite3.Row], T]
    ) -> list[T]:
        """Run a query returning mapped rows."""
        conn = connect(self.path)
        try:
            rows = conn.execute(sql, tuple(params)).fetchall()
        finally:
            conn.close()
        return [mapper(row) for row in rows]


def contact_from_row(row: sqlite3.Row) -> Contact:
    """Map one contacts row."""
    return Contact(
        id=int(row["id"]),
        chat_id=str(row["chat_id"]),
        phone=str(row["phone"] or ""),
        lid=str(row["lid"] or ""),
        display_name=str(row["display_name"] or ""),
        language=str(row["language"] or ""),
        external_user_ref=str(row["external_user_ref"] or ""),
        last_inbound_at=(
            int(row["last_inbound_at"]) if row["last_inbound_at"] is not None else None
        ),
        paused_until=(
            int(row["paused_until"]) if row["paused_until"] is not None else None
        ),
    )


def ticket_from_row(row: sqlite3.Row) -> Ticket:
    """Map one tickets row."""
    return Ticket(
        id=int(row["id"]),
        number=int(row["number"]),
        contact_id=int(row["contact_id"]),
        chat_id=str(row["chat_id"]),
        state=str(row["state"]),
        urgent=bool(row["urgent"]),
        done_by=str(row["done_by"]) if row["done_by"] is not None else None,
        summary=str(row["summary"] or ""),
        details=json_object(str(row["details_json"])),
        missing=json_list(str(row["missing_json"])),
        external_user_ref=str(row["external_user_ref"] or ""),
        alert_message_ids=json_list(str(row["alert_message_ids_json"])),
        previous_ticket_id=(
            int(row["previous_ticket_id"])
            if row["previous_ticket_id"] is not None
            else None
        ),
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        done_at=int(row["done_at"]) if row["done_at"] is not None else None,
        closed_at=int(row["closed_at"]) if row["closed_at"] is not None else None,
        last_customer_at=(
            int(row["last_customer_at"]) if row["last_customer_at"] is not None else None
        ),
    )


def event_from_row(row: sqlite3.Row) -> TicketEvent:
    """Map one ticket_events row."""
    return TicketEvent(
        id=int(row["id"]),
        ticket_id=int(row["ticket_id"]),
        at=int(row["at"]),
        actor=str(row["actor"]),
        kind=str(row["kind"]),
        body=str(row["body"] or ""),
        message_ids=json_list(str(row["message_ids_json"])),
    )


def attachment_from_row(row: sqlite3.Row) -> Attachment:
    """Map one attachments row."""
    return Attachment(
        id=int(row["id"]),
        ticket_id=int(row["ticket_id"]),
        message_id=str(row["message_id"] or ""),
        kind=str(row["kind"]),
        mime=str(row["mime"] or ""),
        bytes=int(row["bytes"]),
        sha256=str(row["sha256"] or ""),
        path=str(row["path"] or ""),
        derived_text=str(row["derived_text"] or ""),
    )


def digest(data: bytes) -> str:
    """The sha256 hex digest of some bytes."""
    return hashlib.sha256(data).hexdigest()
