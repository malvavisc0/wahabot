"""Session summary state and deferred-notice storage."""

from wahabot.core.db.connection import connect, transaction
from wahabot.core.schedule.models import DeferredNotice
from wahabot.core.tickets.models import TicketState
from wahabot.core.tickets.store_base import StoreBase


class ScheduleMixin(StoreBase):
    """Summary dates and outside-hours notices."""

    def last_summary_date(self, session: str) -> str | None:
        """The local date of the last sent morning summary, or None."""
        return self.query_one(
            "SELECT last_summary_date FROM session_state WHERE session = ?",
            (session,),
            lambda row: (
                str(row["last_summary_date"])
                if row["last_summary_date"] is not None
                else None
            ),
        )

    def set_last_summary_date(self, session: str, day: str) -> None:
        """Record the local date a morning summary was sent."""
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "INSERT INTO session_state (session, last_summary_date, updated_at) "
                    + "VALUES (?, ?, ?) ON CONFLICT(session) DO UPDATE SET "
                    + "last_summary_date = excluded.last_summary_date, "
                    + "updated_at = excluded.updated_at",
                    (session, day, self.now()),
                )
        finally:
            conn.close()

    def summary_message_id(self, session: str) -> str:
        """The id of the last summary message, for quoted-number routing."""
        found = self.query_one(
            "SELECT summary_message_id FROM session_state WHERE session = ?",
            (session,),
            lambda row: str(row["summary_message_id"] or ""),
        )
        return found or ""

    def set_summary_message_id(self, session: str, message_id: str) -> None:
        """Store the summary message id."""
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "INSERT INTO session_state (session, summary_message_id, updated_at) "
                    + "VALUES (?, ?, ?) ON CONFLICT(session) DO UPDATE SET "
                    + "summary_message_id = excluded.summary_message_id, "
                    + "updated_at = excluded.updated_at",
                    (session, message_id, self.now()),
                )
        finally:
            conn.close()

    def defer_notice(
        self,
        ticket_id: int,
        session: str,
        chat_id: str,
        text: str,
        *,
        not_before: int,
        at: int | None = None,
    ) -> int:
        """Queue one operator notice until *not_before*."""
        moment = self.now() if at is None else at
        conn = connect(self.path)
        try:
            with transaction(conn):
                existing = conn.execute(
                    "SELECT id FROM deferred_notices "
                    + "WHERE ticket_id = ? AND sent_at IS NULL "
                    + "ORDER BY id DESC LIMIT 1",
                    (ticket_id,),
                ).fetchone()
                if existing is not None:
                    notice_id = int(existing["id"])
                    conn.execute(
                        "UPDATE deferred_notices SET session = ?, chat_id = ?, "
                        + "text = ?, not_before = ?, created_at = ? WHERE id = ?",
                        (
                            session,
                            chat_id,
                            text,
                            not_before,
                            moment,
                            notice_id,
                        ),
                    )
                else:
                    cursor = conn.execute(
                        "INSERT INTO deferred_notices "
                        + "(ticket_id, session, chat_id, text, not_before, created_at) "
                        + "VALUES (?, ?, ?, ?, ?, ?)",
                        (ticket_id, session, chat_id, text, not_before, moment),
                    )
                    notice_id = int(cursor.lastrowid or 0)
        finally:
            conn.close()
        return notice_id

    def pending_notices(self, now: int, limit: int = 10) -> list[DeferredNotice]:
        """Unsent notices for still-open needs-you tickets, oldest first."""
        return self.query_all(
            "SELECT n.id, n.ticket_id, n.session, n.chat_id, n.text, "
            + "n.not_before, n.created_at FROM deferred_notices n "
            + "JOIN tickets t ON t.id = n.ticket_id "
            + "WHERE n.sent_at IS NULL AND n.not_before <= ? AND t.state = ? "
            + "ORDER BY n.not_before, n.id LIMIT ?",
            (now, TicketState.NEEDS_YOU.value, limit),
            lambda row: DeferredNotice(
                id=int(row["id"]),
                ticket_id=int(row["ticket_id"]),
                session=str(row["session"]),
                chat_id=str(row["chat_id"]),
                text=str(row["text"]),
                not_before=int(row["not_before"]),
                created_at=int(row["created_at"]),
            ),
        )

    def mark_notice_sent(self, notice_id: int, sent_id: str, at: int) -> None:
        """Mark one deferred notice as delivered."""
        self.execute(
            "UPDATE deferred_notices SET sent_at = ?, sent_id = ? WHERE id = ?",
            (at, sent_id, notice_id),
        )

    def cancel_notices(self, ticket_id: int, at: int) -> None:
        """Drop unsent notices for a ticket (it was answered or closed)."""
        self.execute(
            "UPDATE deferred_notices SET sent_at = ?, sent_id = 'cancelled' "
            + "WHERE ticket_id = ? AND sent_at IS NULL",
            (at, ticket_id),
        )

    def setup_completed(self, session: str) -> bool:
        """Whether Setup recorded a completed first customer exchange."""
        found = self.query_one(
            "SELECT setup_completed FROM session_state WHERE session = ?",
            (session,),
            lambda row: bool(row["setup_completed"]),
        )
        return bool(found)

    def set_setup_completed(self, session: str, completed: bool = True) -> None:
        """Persist the Setup completion flag for one session."""
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "INSERT INTO session_state "
                    + "(session, setup_completed, updated_at) VALUES (?, ?, ?) "
                    + "ON CONFLICT(session) DO UPDATE SET "
                    + "setup_completed = excluded.setup_completed, "
                    + "updated_at = excluded.updated_at",
                    (session, int(completed), self.now()),
                )
        finally:
            conn.close()
