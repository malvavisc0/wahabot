"""SQLite storage for durable intake rows."""

import sqlite3
import time
from pathlib import Path

from wahabot.core.db.connection import connect, transaction
from wahabot.core.intake.store import (
    DELIVERED,
    DELIVERING,
    DONE,
    FAILED,
    OPERATOR_REQUIRED,
    PENDING,
    PROCESSING,
    IntakeRow,
)

_COLUMNS = (
    "message_id, session, chat_id, sender, raw_json, record, received_at, state, "
    + "outcome, attempts, next_attempt_at, last_error, burst_key, addressed, owner, "
    + "reason, handled_at, notified_at"
)


class SqliteIntakeStore:
    """Durable intake storage in the platform database."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self, key: tuple[str, str], limit: int | None = None) -> list[IntakeRow]:
        conn = connect(self.path)
        try:
            if limit is None:
                rows = conn.execute(
                    f"SELECT {_COLUMNS} FROM intake WHERE session = ? AND chat_id = ? "
                    + "ORDER BY received_at, rowid",
                    key,
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {_COLUMNS} FROM intake WHERE session = ? AND chat_id = ? "
                    + "ORDER BY received_at DESC, rowid DESC LIMIT ?",
                    (*key, max(0, limit)),
                ).fetchall()
                rows = list(reversed(rows))
        finally:
            conn.close()
        return [to_row(row) for row in rows]

    def get(self, key: tuple[str, str], message_id: str) -> IntakeRow | None:
        conn = connect(self.path)
        try:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM intake "
                + "WHERE message_id = ? AND session = ? AND chat_id = ?",
                (message_id, *key),
            ).fetchone()
        finally:
            conn.close()
        return to_row(row) if row is not None else None

    def record(self, row: IntakeRow) -> IntakeRow:
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "INSERT INTO intake ("
                    + "message_id, session, chat_id, sender, raw_json, record, "
                    + "received_at, state, outcome, attempts, next_attempt_at, "
                    + "last_error, burst_key, addressed, owner, reason, handled_at, "
                    + "notified_at"
                    + ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    + "ON CONFLICT(message_id) DO NOTHING",
                    values(row),
                )
                stored = conn.execute(
                    f"SELECT {_COLUMNS} FROM intake WHERE message_id = ?",
                    (row.message_id,),
                ).fetchone()
        finally:
            conn.close()
        return to_row(stored) if stored is not None else row

    def claim(
        self,
        key: tuple[str, str],
        message_ids: list[str],
        owner: str,
        lease_until: float,
    ) -> list[IntakeRow]:
        if not message_ids:
            return []
        marks = ",".join("?" for _ in message_ids)
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "UPDATE intake SET state = ?, owner = ?, next_attempt_at = ? "
                    + f"WHERE message_id IN ({marks}) AND state = ?",
                    (PROCESSING, owner, lease_until, *message_ids, PENDING),
                )
                rows = conn.execute(
                    f"SELECT {_COLUMNS} FROM intake WHERE message_id IN ({marks}) "
                    + "AND state = ? AND owner = ? ORDER BY received_at, rowid",
                    (*message_ids, PROCESSING, owner),
                ).fetchall()
        finally:
            conn.close()
        return [to_row(row) for row in rows]

    def handled(
        self, key: tuple[str, str], message_ids: list[str], addressed_only: bool
    ) -> bool:
        if not message_ids:
            return False
        marks = ",".join("?" for _ in message_ids)
        filter = " AND addressed = 1" if addressed_only else ""
        conn = connect(self.path)
        try:
            rows = conn.execute(
                f"SELECT message_id, state FROM intake WHERE message_id IN ({marks})"
                + filter,
                message_ids,
            ).fetchall()
        finally:
            conn.close()
        if addressed_only:
            return bool(rows) and all(str(row["state"]) == DONE for row in rows)
        found = {str(row["message_id"]) for row in rows}
        return found == set(message_ids) and all(
            str(row["state"]) == DONE for row in rows
        )

    def release(
        self, key: tuple[str, str], message_ids: list[str], owner: str = ""
    ) -> None:
        if not message_ids:
            return
        marks = ",".join("?" for _ in message_ids)
        owner_filter = " AND owner = ?" if owner else ""
        params: tuple[object, ...] = (
            (PENDING, *message_ids, PROCESSING, owner)
            if owner
            else (PENDING, *message_ids, PROCESSING)
        )
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "UPDATE intake SET state = ?, owner = '', next_attempt_at = NULL "
                    + f"WHERE message_id IN ({marks}) AND state = ?"
                    + owner_filter,
                    params,
                )
        finally:
            conn.close()

    def hold(self, key: tuple[str, str], message_ids: list[str], reason: str) -> None:
        """Mark rows as owned by an outbound intent; do not run them again."""
        if not message_ids:
            return
        marks = ",".join("?" for _ in message_ids)
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "UPDATE intake SET state = ?, next_attempt_at = NULL, "
                    + "last_error = ? "
                    + f"WHERE message_id IN ({marks}) AND state != ?",
                    (DELIVERING, reason, *message_ids, DONE),
                )
        finally:
            conn.close()

    def is_delivering(self, key: tuple[str, str], message_ids: list[str]) -> bool:
        if not message_ids:
            return False
        marks = ",".join("?" for _ in message_ids)
        conn = connect(self.path)
        try:
            row = conn.execute(
                f"SELECT 1 FROM intake WHERE message_id IN ({marks}) AND state = ? "
                + "LIMIT 1",
                (*message_ids, DELIVERING),
            ).fetchone()
        finally:
            conn.close()
        return row is not None

    def delivering_rows(self, limit: int) -> list[IntakeRow]:
        return self.query("state = ?", (DELIVERING,), limit)

    def reset_delivering(self, message_ids: list[str]) -> None:
        if not message_ids:
            return
        marks = ",".join("?" for _ in message_ids)
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "UPDATE intake SET state = ?, owner = '', next_attempt_at = NULL "
                    + f"WHERE message_id IN ({marks}) AND state = ?",
                    (PENDING, *message_ids, DELIVERING),
                )
        finally:
            conn.close()

    def finish(
        self,
        key: tuple[str, str],
        message_ids: list[str],
        outcome: str,
        reason: str,
        handled_at: float,
    ) -> None:
        if not message_ids:
            return
        marks = ",".join("?" for _ in message_ids)
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "UPDATE intake SET state = ?, outcome = ?, reason = ?, "
                    + "handled_at = ? "
                    + f"WHERE message_id IN ({marks}) AND state != ?",
                    (DONE, outcome, reason, handled_at, *message_ids, DONE),
                )
        finally:
            conn.close()

    def finish_turn(
        self,
        key: tuple[str, str],
        turn_ids: list[str],
        claimed_ids: list[str],
        delivered: bool,
        outcome: str,
        reason: str,
        handled_at: float,
        owner: str = "",
    ) -> None:
        conn = connect(self.path)
        try:
            with transaction(conn):
                if delivered:
                    self._complete(
                        conn,
                        [*turn_ids, *claimed_ids],
                        outcome or DELIVERED,
                        reason,
                        handled_at,
                    )
                    return
                self._release_claims(conn, claimed_ids, owner)
                if outcome:
                    self._complete(conn, turn_ids, outcome, reason, handled_at)
        finally:
            conn.close()

    def _complete(
        self,
        conn: sqlite3.Connection,
        message_ids: list[str],
        outcome: str,
        reason: str,
        handled_at: float,
    ) -> None:
        if not message_ids:
            return
        marks = ",".join("?" for _ in message_ids)
        conn.execute(
            "UPDATE intake SET state = ?, outcome = ?, reason = ?, handled_at = ? "
            + f"WHERE message_id IN ({marks}) AND state != ?",
            (DONE, outcome, reason, handled_at, *message_ids, DONE),
        )

    def _release_claims(
        self,
        conn: sqlite3.Connection,
        claimed_ids: list[str],
        owner: str,
    ) -> None:
        if not claimed_ids:
            return
        marks = ",".join("?" for _ in claimed_ids)
        owner_filter = " AND owner = ?" if owner else ""
        params: tuple[object, ...] = (
            (PENDING, *claimed_ids, PROCESSING, owner)
            if owner
            else (PENDING, *claimed_ids, PROCESSING)
        )
        conn.execute(
            "UPDATE intake SET state = ?, owner = '', next_attempt_at = NULL "
            + f"WHERE message_id IN ({marks}) AND state = ?"
            + owner_filter,
            params,
        )

    def fail(
        self,
        key: tuple[str, str],
        message_ids: list[str],
        error: str,
        next_attempt_at: float | None,
    ) -> None:
        if not message_ids:
            return
        marks = ",".join("?" for _ in message_ids)
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "UPDATE intake SET state = ?, attempts = attempts + 1, "
                    + "last_error = ?, next_attempt_at = ? "
                    + f"WHERE message_id IN ({marks}) AND state != ?",
                    (FAILED, error, next_attempt_at, *message_ids, DONE),
                )
        finally:
            conn.close()

    def set_burst_key(
        self, key: tuple[str, str], message_ids: list[str], burst_key: str
    ) -> None:
        if not message_ids:
            return
        marks = ",".join("?" for _ in message_ids)
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    f"UPDATE intake SET burst_key = ? WHERE message_id IN ({marks})",
                    (burst_key, *message_ids),
                )
        finally:
            conn.close()

    def due(self, now: float, limit: int) -> list[IntakeRow]:
        return self.query(
            "state = ? OR (state = ? AND "
            + "(next_attempt_at IS NULL OR next_attempt_at <= ?))",
            (PENDING, FAILED, now),
            limit,
        )

    def stale_processing(self, before: float, limit: int) -> list[IntakeRow]:
        return self.query(
            "state = ? AND next_attempt_at IS NOT NULL AND next_attempt_at <= ?",
            (PROCESSING, before),
            limit,
        )

    def reset_processing(self) -> int:
        """Return every claim to pending after a restart, without a retry."""
        conn = connect(self.path)
        try:
            with transaction(conn):
                cursor = conn.execute(
                    "UPDATE intake SET state = ?, owner = '', next_attempt_at = NULL "
                    + "WHERE state = ?",
                    (PENDING, PROCESSING),
                )
                return int(cursor.rowcount)
        finally:
            conn.close()

    def unnotified_operator(self, limit: int) -> list[IntakeRow]:
        return self.query(
            "state = ? AND outcome = ? AND notified_at IS NULL",
            (DONE, OPERATOR_REQUIRED),
            limit,
        )

    def mark_notified(self, message_ids: list[str], notified_at: float) -> None:
        if not message_ids:
            return
        marks = ",".join("?" for _ in message_ids)
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    f"UPDATE intake SET notified_at = ? WHERE message_id IN ({marks})",
                    (notified_at, *message_ids),
                )
        finally:
            conn.close()

    def query(
        self, where: str, params: tuple[object, ...], limit: int
    ) -> list[IntakeRow]:
        """Rows matching *where*, oldest first, capped at *limit*."""
        conn = connect(self.path)
        try:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM intake WHERE {where} "
                + "ORDER BY received_at, rowid LIMIT ?",
                (*params, limit),
            ).fetchall()
        finally:
            conn.close()
        return [to_row(row) for row in rows]

    def purge(self, before: float) -> int:
        conn = connect(self.path)
        try:
            with transaction(conn):
                cursor = conn.execute(
                    "DELETE FROM intake WHERE state = ? AND received_at < ?",
                    (DONE, before),
                )
                return int(cursor.rowcount)
        finally:
            conn.close()

    def stats(self) -> dict[str, int]:
        unfinished_states = (PENDING, PROCESSING, DELIVERING, FAILED)
        conn = connect(self.path)
        try:
            unanswered = conn.execute(
                "SELECT COUNT(*) FROM intake WHERE state IN (?, ?, ?, ?)",
                unfinished_states,
            ).fetchone()[0]
            done = conn.execute(
                "SELECT COUNT(*) FROM intake WHERE state = ?", (DONE,)
            ).fetchone()[0]
            oldest = conn.execute(
                "SELECT MIN(received_at) FROM intake WHERE state IN (?, ?, ?, ?)",
                unfinished_states,
            ).fetchone()[0]
        finally:
            conn.close()
        age = max(0, int(time.time() - oldest)) if oldest is not None else 0
        return {"unanswered": int(unanswered), "done": int(done), "oldest_age_s": age}


def values(row: IntakeRow) -> tuple[object, ...]:
    """The insert values of one row."""
    return (
        row.message_id,
        row.session,
        row.chat_id,
        row.sender,
        row.raw_json,
        row.record,
        row.received_at,
        row.state,
        row.outcome,
        row.attempts,
        row.next_attempt_at,
        row.last_error,
        row.burst_key,
        int(row.addressed),
        row.owner,
        row.reason,
        row.handled_at,
        row.notified_at,
    )


def to_row(row: sqlite3.Row) -> IntakeRow:
    """Map one SQLite row to the ledger row."""
    received_at = float(row["received_at"])
    age = max(0.0, time.time() - received_at)
    return IntakeRow(
        message_id=str(row["message_id"]),
        session=str(row["session"]),
        chat_id=str(row["chat_id"]),
        record=str(row["record"]),
        at=time.monotonic() - age,
        state=str(row["state"]),
        addressed=bool(row["addressed"]),
        raw_json=str(row["raw_json"]),
        sender=str(row["sender"]),
        received_at=received_at,
        burst_key=row["burst_key"],
        attempts=int(row["attempts"]),
        next_attempt_at=(
            float(row["next_attempt_at"]) if row["next_attempt_at"] is not None else None
        ),
        last_error=str(row["last_error"] or ""),
        outcome=row["outcome"],
        owner=str(row["owner"] or ""),
        reason=str(row["reason"] or ""),
        handled_at=(float(row["handled_at"]) if row["handled_at"] is not None else None),
        notified_at=(
            float(row["notified_at"]) if row["notified_at"] is not None else None
        ),
    )
