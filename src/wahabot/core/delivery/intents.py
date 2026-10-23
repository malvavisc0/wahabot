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
CANCELLED = "cancelled"

#: Intake outcomes owned by a delivery intent.
OPERATOR_REQUIRED = "operator_required"


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
                self._insert(
                    conn,
                    intent_id=intent_id,
                    session=session,
                    chat_id=chat_id,
                    kind=kind,
                    payload=payload,
                    intake_ids=list(intake_ids),
                    now=now,
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
                    + "sent_at = ?, operator_required = 0 WHERE id = ?",
                    (SENT, transport_message_id, now, now, intent_id),
                )
                row = conn.execute(
                    "SELECT intake_ids_json, session FROM outbound_intents WHERE id = ?",
                    (intent_id,),
                ).fetchone()
                if row is not None:
                    ids = json.loads(row[0])
                    if ids:
                        self._settle(conn, ids, "delivered", "", now)
                    self._cancel_retryable(
                        conn, ids, "superseded by a confirmed send", now
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
                ids = self._intake_ids(conn, intent_id)
                if not ids:
                    return
                if operator_required or next_attempt_at is None:
                    self._settle(
                        conn, ids, OPERATOR_REQUIRED, "delivery failed", int(now)
                    )
                else:
                    self._hold(conn, ids, "delivery retrying")
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
                ids = self._intake_ids(conn, intent_id)
                if ids:
                    self._settle(conn, ids, OPERATOR_REQUIRED, "delivery unknown", now)
        finally:
            conn.close()

    def cancel(self, intent_id: str, reason: str) -> None:
        """Stop an intent that another confirmed delivery superseded."""
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "UPDATE outbound_intents SET state = ?, next_attempt_at = NULL, "
                    + "operator_required = 0, last_error = ?, updated_at = ? "
                    + "WHERE id = ?",
                    (CANCELLED, reason, int(time.time()), intent_id),
                )
        finally:
            conn.close()

    def cancel_retryable_for(self, intake_ids: list[str], reason: str) -> None:
        """Cancel retryable failures whose turn another send answered."""
        if not intake_ids:
            return
        conn = connect(self.path)
        try:
            with transaction(conn):
                self._cancel_retryable(conn, intake_ids, reason, int(time.time()))
        finally:
            conn.close()

    def reconcile_sent_intake(self) -> int:
        """Finish received rows for intents already confirmed before a crash."""
        conn = connect(self.path)
        settled = 0
        try:
            with transaction(conn):
                rows = conn.execute(
                    "SELECT intake_ids_json FROM outbound_intents WHERE state = ?",
                    (SENT,),
                ).fetchall()
                ids = sorted({mid for row in rows for mid in json.loads(row[0])})
                if ids:
                    marks = ",".join("?" for _ in ids)
                    cursor = conn.execute(
                        "UPDATE intake SET state = 'done', outcome = 'delivered', "
                        + "reason = '', handled_at = ? WHERE message_id IN ("
                        + marks
                        + ") AND state != 'done'",
                        (int(time.time()), *ids),
                    )
                    settled = int(cursor.rowcount)
        finally:
            conn.close()
        return settled

    def unsettled_intake_ids(self) -> set[str]:
        """Intake ids still owned by a prepared, sending, failed or unknown intent."""
        conn = connect(self.path)
        try:
            rows = conn.execute(
                "SELECT intake_ids_json FROM outbound_intents "
                + "WHERE state IN (?, ?, ?, ?)",
                (PREPARED, SENDING, FAILED, UNKNOWN),
            ).fetchall()
        finally:
            conn.close()
        return {message_id for row in rows for message_id in json.loads(row[0])}

    def outstanding_for(self, intake_ids: list[str]) -> list[DeliveryIntent]:
        """Unsettled intents whose turn includes any of *intake_ids*."""
        wanted = set(intake_ids)
        if not wanted:
            return []
        conn = connect(self.path)
        try:
            rows = conn.execute(
                "SELECT * FROM outbound_intents WHERE state IN (?, ?, ?, ?) "
                + "ORDER BY created_at, rowid",
                (PREPARED, SENDING, FAILED, UNKNOWN),
            ).fetchall()
        finally:
            conn.close()
        return [
            intent
            for intent in (to_intent(row) for row in rows)
            if wanted.intersection(intent.intake_ids)
        ]

    def prepend_text(self, intent_id: str, prefix: str) -> DeliveryIntent:
        """Prefix the text (or caption) a pending intent will send."""
        if not prefix:
            return self.get(intent_id)
        conn = connect(self.path)
        try:
            with transaction(conn):
                row = conn.execute(
                    "SELECT kind, payload_json FROM outbound_intents WHERE id = ?",
                    (intent_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(intent_id)
                payload = _prefix_payload(
                    str(row["kind"]), dict(json.loads(row["payload_json"])), prefix
                )
                conn.execute(
                    "UPDATE outbound_intents SET payload_json = ?, updated_at = ? "
                    + "WHERE id = ?",
                    (json.dumps(payload), int(time.time()), intent_id),
                )
        finally:
            conn.close()
        return self.get(intent_id)

    def prepare_notice(
        self,
        *,
        session: str,
        chat_id: str,
        text: str,
        source_intent_ids: list[str] | tuple[str, ...] = (),
        source_intake_ids: list[str] | tuple[str, ...] = (),
    ) -> DeliveryIntent:
        """Insert one durable notice and mark its sources as noticed atomically."""
        now = time.time()
        intent_id = uuid.uuid4().hex
        conn = connect(self.path)
        try:
            with transaction(conn):
                self._insert(
                    conn,
                    intent_id=intent_id,
                    session=session,
                    chat_id=chat_id,
                    kind="operator_notice",
                    payload={
                        "text": text,
                        "source_intent_ids": list(source_intent_ids),
                        "source_intake_ids": list(source_intake_ids),
                    },
                    intake_ids=[],
                    now=now,
                )
                if source_intent_ids:
                    marks = ",".join("?" for _ in source_intent_ids)
                    conn.execute(
                        "UPDATE outbound_intents SET notified_at = ?, updated_at = ? "
                        + f"WHERE id IN ({marks})",
                        (int(now), int(now), *source_intent_ids),
                    )
                if source_intake_ids:
                    marks = ",".join("?" for _ in source_intake_ids)
                    conn.execute(
                        "UPDATE intake SET notified_at = ? "
                        + f"WHERE message_id IN ({marks})",
                        (int(now), *source_intake_ids),
                    )
        finally:
            conn.close()
        return self.get(intent_id)

    def purge(self, before: float) -> int:
        """Prune settled intents older than *before*.

        Sent/cancelled rows and notified operator-required outcomes are
        historical; prepared, sending, retryable failed and unnoticed
        unknown rows stay.
        """
        conn = connect(self.path)
        try:
            with transaction(conn):
                cursor = conn.execute(
                    "DELETE FROM outbound_intents WHERE updated_at < ? AND ("
                    + "state IN (?, ?) OR (state IN (?, ?) "
                    + "AND operator_required = 1 AND notified_at IS NOT NULL))",
                    (before, SENT, CANCELLED, FAILED, UNKNOWN),
                )
                return int(cursor.rowcount)
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
        """Prepared, stale-sending and retry-due failed intents.

        Terminal failures (``next_attempt_at IS NULL``) and cancelled
        intents are never selected, so exhaustion does not rewrite the
        same row every tick.
        """
        conn = connect(self.path)
        try:
            rows = conn.execute(
                "SELECT * FROM outbound_intents WHERE (state = ? "
                + "OR (state = ? AND next_attempt_at IS NOT NULL "
                + "AND next_attempt_at <= ?) OR state = ?) "
                + "AND NOT (kind = 'operator_notice' AND chat_id = '') "
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
                + "AND notified_at IS NULL AND kind != 'operator_notice' "
                + "AND state IN (?, ?) ORDER BY created_at, rowid",
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

    def _insert(
        self,
        conn: Any,
        *,
        intent_id: str,
        session: str,
        chat_id: str,
        kind: str,
        payload: dict[str, Any],
        intake_ids: list[str],
        now: float,
    ) -> None:
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

    def _intake_ids(self, conn: Any, intent_id: str) -> list[str]:
        row = conn.execute(
            "SELECT intake_ids_json FROM outbound_intents WHERE id = ?",
            (intent_id,),
        ).fetchone()
        if row is None:
            return []
        return list(json.loads(row[0]))

    def _settle(
        self, conn: Any, ids: list[str], outcome: str, reason: str, now: int
    ) -> None:
        if not ids:
            return
        marks = ",".join("?" for _ in ids)
        conn.execute(
            "UPDATE intake SET state = 'done', outcome = ?, reason = ?, handled_at = ? "
            + f"WHERE message_id IN ({marks}) AND state != 'done'",
            (outcome, reason, now, *ids),
        )

    def _hold(self, conn: Any, ids: list[str], reason: str) -> None:
        if not ids:
            return
        marks = ",".join("?" for _ in ids)
        conn.execute(
            "UPDATE intake SET state = 'delivering', next_attempt_at = NULL, "
            + "last_error = ? "
            + f"WHERE message_id IN ({marks}) AND state != 'done'",
            (reason, *ids),
        )

    def _cancel_retryable(
        self, conn: Any, ids: list[str], reason: str, now: int | float
    ) -> None:
        wanted = set(ids)
        if not wanted:
            return
        rows = conn.execute(
            "SELECT id, intake_ids_json FROM outbound_intents "
            + "WHERE (state = ? AND operator_required = 0) OR state = ?",
            (FAILED, PREPARED),
        ).fetchall()
        for row in rows:
            if wanted.intersection(json.loads(row[1])):
                conn.execute(
                    "UPDATE outbound_intents SET state = ?, next_attempt_at = NULL, "
                    + "last_error = ?, updated_at = ? WHERE id = ?",
                    (CANCELLED, reason, int(now), row[0]),
                )


def _prefix_payload(kind: str, payload: dict[str, Any], prefix: str) -> dict[str, Any]:
    if kind in ("text", "operator_notice"):
        text = str(payload.get("text") or "")
        if text and not text.startswith(prefix):
            payload["text"] = prefix + text
    elif kind in ("image", "file", "video"):
        caption = payload.get("caption")
        if not str(caption or "").startswith(prefix):
            payload["caption"] = prefix + str(caption or "")
    return payload


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
