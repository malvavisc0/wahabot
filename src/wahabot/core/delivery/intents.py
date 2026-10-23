"""Durable outbound intents recorded before every transport send."""

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wahabot.core.db.connection import connect, transaction

PREPARED = "prepared"
SENDING = "sending"
SENT = "sent"
FAILED = "failed"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class DeliveryIntent:
    """One planned send and its confirmed, failed or unknown outcome."""

    id: str
    session: str
    intake_ids: list[str]
    chat_id: str
    kind: str
    payload: dict[str, Any]
    state: str
    transport_message_id: str
    attempts: int
    next_attempt_at: float | None
    last_error: str
    operator_required: bool
    notified_at: float | None
    created_at: float
    updated_at: float
    sent_at: float | None


class IntentStore:
    """SQLite persistence for outbound intents."""

    def __init__(self, path: Path, *, max_attempts: int = 5) -> None:
        self.path = path
        self.max_attempts = max_attempts

    def prepare(
        self,
        *,
        session: str,
        chat_id: str,
        kind: str,
        payload: dict[str, Any],
        intake_ids: list[str] | tuple[str, ...] = (),
    ) -> DeliveryIntent:
        """Persist a prepared intent before any transport call."""
        now = time.time()
        intent_id = uuid.uuid4().hex
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "INSERT INTO outbound_intents ("
                    + "id, session, intake_ids_json, chat_id, kind, payload_json, "
                    + "state, transport_message_id, attempts, next_attempt_at, "
                    + "last_error, operator_required, created_at, updated_at, sent_at"
                    + ") VALUES (?, ?, ?, ?, ?, ?, ?, '', 0, NULL, '', 0, ?, ?, NULL)",
                    (
                        intent_id,
                        session,
                        json.dumps(list(intake_ids)),
                        chat_id,
                        kind,
                        json.dumps(payload),
                        PREPARED,
                        int(now),
                        int(now),
                    ),
                )
        finally:
            conn.close()
        return self.get(intent_id)

    def sending(self, intent_id: str) -> None:
        self._set(intent_id, state=SENDING)

    def sent(self, intent_id: str, transport_message_id: str = "") -> None:
        """Mark a confirmed send and finish the intake rows it answered."""
        now = int(time.time())
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "UPDATE outbound_intents SET state = ?, transport_message_id = ?, "
                    + "next_attempt_at = NULL, last_error = '', updated_at = ?, "
                    + "sent_at = ? WHERE id = ?",
                    (SENT, transport_message_id, now, now, intent_id),
                )
                row = conn.execute(
                    "SELECT intake_ids_json FROM outbound_intents WHERE id = ?",
                    (intent_id,),
                ).fetchone()
                if row is not None:
                    ids = json.loads(row[0])
                    if ids:
                        conn.execute(
                            "UPDATE intake SET state = 'done', outcome = 'delivered', "
                            + "handled_at = ? WHERE message_id IN ("
                            + ",".join("?" for _ in ids)
                            + ")",
                            (now, *ids),
                        )
                conn.execute(
                    "UPDATE outbound_intents SET operator_required = 0 WHERE id = ?",
                    (intent_id,),
                )
        finally:
            conn.close()

    def failed(
        self,
        intent_id: str,
        error: str,
        *,
        next_attempt_at: float | None,
        operator_required: bool = False,
    ) -> None:
        """Mark a definite failure, with a retry time only when retryable."""
        now = time.time()
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "UPDATE outbound_intents SET state = ?, attempts = attempts + 1, "
                    + "next_attempt_at = ?, last_error = ?, operator_required = ?, "
                    + "updated_at = ? WHERE id = ?",
                    (
                        FAILED,
                        int(next_attempt_at) if next_attempt_at is not None else None,
                        error,
                        int(operator_required),
                        int(now),
                        intent_id,
                    ),
                )
        finally:
            conn.close()

    def unknown(self, intent_id: str, error: str) -> None:
        """Mark a send whose remote outcome is unknown; never auto-retried."""
        now = int(time.time())
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "UPDATE outbound_intents SET state = ?, attempts = attempts + 1, "
                    + "next_attempt_at = NULL, last_error = ?, operator_required = 1, "
                    + "updated_at = ? WHERE id = ?",
                    (UNKNOWN, error, now, intent_id),
                )
        finally:
            conn.close()

    def get(self, intent_id: str) -> DeliveryIntent:
        conn = connect(self.path)
        try:
            row = conn.execute(
                "SELECT * FROM outbound_intents WHERE id = ?", (intent_id,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(intent_id)
        return to_intent(row)

    def recent(self, limit: int = 10) -> list[DeliveryIntent]:
        conn = connect(self.path)
        try:
            rows = conn.execute(
                "SELECT * FROM outbound_intents ORDER BY created_at DESC, rowid DESC "
                + "LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [to_intent(row) for row in rows]

    def recoverable(self, now: float) -> list[DeliveryIntent]:
        """Prepared, stale-sending and retry-due failed intents."""
        conn = connect(self.path)
        try:
            rows = conn.execute(
                "SELECT * FROM outbound_intents WHERE (state = ? "
                + "OR (state = ? AND (next_attempt_at IS NULL OR next_attempt_at <= ?)) "
                + "OR state = ?) AND NOT (kind = 'operator_notice' AND chat_id = '') "
                + "ORDER BY created_at, rowid",
                (PREPARED, FAILED, int(now), SENDING),
            ).fetchall()
        finally:
            conn.close()
        return [to_intent(row) for row in rows]

    def recover_due(self, now: float) -> list[DeliveryIntent]:
        """Due definite failures only; unknown and sending never qualify."""
        conn = connect(self.path)
        try:
            rows = conn.execute(
                "SELECT * FROM outbound_intents WHERE state = ? "
                + "AND next_attempt_at IS NOT NULL AND next_attempt_at <= ? "
                + "ORDER BY created_at, rowid",
                (FAILED, int(now)),
            ).fetchall()
        finally:
            conn.close()
        return [to_intent(row) for row in rows]

    def unnotified_operator(self) -> list[DeliveryIntent]:
        """Operator-required outcomes that still owe a notice."""
        conn = connect(self.path)
        try:
            rows = conn.execute(
                "SELECT * FROM outbound_intents WHERE operator_required = 1 "
                + "AND notified_at IS NULL AND state IN (?, ?) "
                + "ORDER BY created_at, rowid",
                (UNKNOWN, FAILED),
            ).fetchall()
        finally:
            conn.close()
        return [to_intent(row) for row in rows]

    def mark_notified(self, intent_ids: list[str]) -> None:
        if not intent_ids:
            return
        marks = ",".join("?" for _ in intent_ids)
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "UPDATE outbound_intents SET notified_at = ?, updated_at = ? "
                    + f"WHERE id IN ({marks})",
                    (int(time.time()), int(time.time()), *intent_ids),
                )
        finally:
            conn.close()

    def pending_notices(self) -> list[DeliveryIntent]:
        """Prepared operator notices still missing a destination."""
        conn = connect(self.path)
        try:
            rows = conn.execute(
                "SELECT * FROM outbound_intents WHERE kind = 'operator_notice' "
                + "AND state = ? AND chat_id = '' ORDER BY created_at, rowid",
                (PREPARED,),
            ).fetchall()
        finally:
            conn.close()
        return [to_intent(row) for row in rows]

    def set_chat(self, intent_id: str, chat_id: str) -> None:
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "UPDATE outbound_intents SET chat_id = ?, updated_at = ? "
                    + "WHERE id = ?",
                    (chat_id, int(time.time()), intent_id),
                )
        finally:
            conn.close()

    def exhausted(self, intent: DeliveryIntent) -> bool:
        return intent.attempts >= self.max_attempts

    def _set(self, intent_id: str, *, state: str) -> None:
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "UPDATE outbound_intents SET state = ?, updated_at = ? WHERE id = ?",
                    (state, int(time.time()), intent_id),
                )
        finally:
            conn.close()


def to_intent(row: Any) -> DeliveryIntent:
    """Map one SQLite row to the intent model."""
    return DeliveryIntent(
        id=str(row["id"]),
        session=str(row["session"]),
        intake_ids=list(json.loads(row["intake_ids_json"])),
        chat_id=str(row["chat_id"]),
        kind=str(row["kind"]),
        payload=dict(json.loads(row["payload_json"])),
        state=str(row["state"]),
        transport_message_id=str(row["transport_message_id"] or ""),
        attempts=int(row["attempts"]),
        next_attempt_at=(
            float(row["next_attempt_at"]) if row["next_attempt_at"] is not None else None
        ),
        last_error=str(row["last_error"] or ""),
        operator_required=bool(row["operator_required"]),
        notified_at=(
            float(row["notified_at"]) if row["notified_at"] is not None else None
        ),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        sent_at=float(row["sent_at"]) if row["sent_at"] is not None else None,
    )
