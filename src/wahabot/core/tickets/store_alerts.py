"""Durable operator-alert intents used for exactly-once handover."""

import sqlite3
from typing import Any

from wahabot.core.db.connection import connect, transaction
from wahabot.core.tickets.models import (
    ALERT_FAILED,
    ALERT_SENDING,
    ALERT_SENT,
    AlertIntent,
)
from wahabot.core.tickets.store_base import StoreBase


def alert_from_row(row: sqlite3.Row) -> AlertIntent:
    """Map one alert_intents row."""
    return AlertIntent(
        key=str(row["alert_key"]),
        ticket_id=int(row["ticket_id"]),
        session=str(row["session"]),
        chat_id=str(row["chat_id"]),
        text=str(row["text"]),
        state=str(row["state"]),
        sent_id=str(row["sent_id"] or ""),
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
    )


class AlertMixin(StoreBase):
    """Alert-intent persistence shared by the ticket store."""

    def claim_alert(
        self,
        key: str,
        ticket_id: int,
        session: str,
        chat_id: str,
        text: str,
        *,
        at: int | None = None,
    ) -> tuple[AlertIntent, bool]:
        """Persist the alert before the transport call, or load the old one.

        Returns ``(intent, created)``; a freshly created intent is safe to
        send immediately, while an existing ``sending`` one is a retry.

        A ``sent`` intent is returned untouched. A ``failed`` one is reset
        to ``sending`` with the freshly composed text, because it is known
        not to have been delivered.
        """
        moment = self.now() if at is None else at
        conn = connect(self.path)
        try:
            with transaction(conn):
                existing = conn.execute(
                    "SELECT 1 FROM alert_intents WHERE alert_key = ?", (key,)
                ).fetchone()
                conn.execute(
                    "INSERT INTO alert_intents (alert_key, ticket_id, session, "
                    + "chat_id, text, state, sent_id, created_at, updated_at) "
                    + "VALUES (?, ?, ?, ?, ?, ?, '', ?, ?) "
                    + "ON CONFLICT(alert_key) DO UPDATE SET "
                    + "text = CASE WHEN alert_intents.state = ? "
                    + "THEN excluded.text ELSE alert_intents.text END, "
                    + "state = CASE WHEN alert_intents.state = ? "
                    + "THEN ? ELSE alert_intents.state END, "
                    + "updated_at = CASE WHEN alert_intents.state = ? "
                    + "THEN excluded.updated_at ELSE alert_intents.updated_at END",
                    (
                        key,
                        ticket_id,
                        session,
                        chat_id,
                        text,
                        ALERT_SENDING,
                        moment,
                        moment,
                        ALERT_FAILED,
                        ALERT_FAILED,
                        ALERT_SENDING,
                        ALERT_FAILED,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM alert_intents WHERE alert_key = ?", (key,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"alert intent {key} not found")
        finally:
            conn.close()
        return alert_from_row(row), existing is None

    def alert_intent(self, key: str) -> AlertIntent | None:
        """One alert intent by key."""
        return self.query_one(
            "SELECT * FROM alert_intents WHERE alert_key = ?",
            (key,),
            alert_from_row,
        )

    def fail_alert(self, key: str, *, at: int | None = None) -> None:
        """Mark a definitely failed attempt so a retry may send again."""
        self.execute(
            "UPDATE alert_intents SET state = ?, updated_at = ? WHERE alert_key = ?",
            (ALERT_FAILED, self.now() if at is None else at, key),
        )

    def restart_alert(self, key: str, text: str, *, at: int | None = None) -> None:
        """Refresh a stale attempt's text/time before resending it."""
        self.execute(
            "UPDATE alert_intents SET text = ?, state = ?, updated_at = ? "
            + "WHERE alert_key = ? AND state != ?",
            (
                text,
                ALERT_SENDING,
                self.now() if at is None else at,
                key,
                ALERT_SENT,
            ),
        )

    def cancel_alert(self, key: str, *, at: int | None = None) -> None:
        """Stop retrying an alert whose ticket no longer needs it."""
        self.execute(
            "UPDATE alert_intents SET state = ?, updated_at = ? WHERE alert_key = ?",
            (ALERT_SENT, self.now() if at is None else at, key),
        )

    def claim_stale_alert(self, key: str, cutoff: int, *, at: int) -> bool:
        """Atomically take ownership of a stale alert for one retry."""
        conn = connect(self.path)
        try:
            with transaction(conn):
                cursor = conn.execute(
                    "UPDATE alert_intents SET state = ?, updated_at = ? "
                    + "WHERE alert_key = ? AND state IN (?, ?) AND updated_at <= ?",
                    (ALERT_SENDING, at, key, ALERT_SENDING, ALERT_FAILED, cutoff),
                )
                return cursor.rowcount > 0
        finally:
            conn.close()

    def stale_alerts(self, cutoff: int, limit: int = 10) -> list[AlertIntent]:
        """Alerts whose last attempt is old enough to reconcile or resend."""
        return self.query_all(
            "SELECT * FROM alert_intents WHERE state IN (?, ?) "
            + "AND updated_at <= ? ORDER BY updated_at LIMIT ?",
            (ALERT_SENDING, ALERT_FAILED, cutoff, limit),
            alert_from_row,
        )

    def alert_id_used(self, sent_id: str, exclude_key: str) -> bool:
        """Whether another alert already adopted *sent_id*."""
        if not sent_id:
            return False
        found = self.query_one(
            "SELECT 1 AS used FROM alert_intents "
            + "WHERE sent_id = ? AND alert_key != ? LIMIT 1",
            (sent_id, exclude_key),
            lambda row: int(row["used"]),
        )
        return found is not None

    def mark_alert_sent(self, conn: Any, key: str, sent_id: str, moment: int) -> None:
        """Mark the intent sent inside the caller's transaction."""
        if not key:
            return
        conn.execute(
            "UPDATE alert_intents SET state = ?, sent_id = ?, updated_at = ? "
            + "WHERE alert_key = ?",
            (ALERT_SENT, sent_id, moment, key),
        )
