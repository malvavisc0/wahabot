"""Durable outbound intents recorded before every transport send."""

import json
import time
import uuid
from pathlib import Path
from typing import Any

from wahabot.core.db.connection import connect, transaction
from wahabot.core.delivery.model import (
    CANCELLED,
    FAILED,
    OPERATOR_REQUIRED,
    PREPARED,
    SENDING,
    SENT,
    UNKNOWN,
    DeliveryIntent,
    to_intent,
)


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
                self.insert_intent(
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
        self.set_state(intent_id, state=SENDING)

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
                    "SELECT intake_ids_json, session, chat_id, kind, payload_json "
                    + "FROM outbound_intents WHERE id = ?",
                    (intent_id,),
                ).fetchone()
                if row is not None:
                    ids = json.loads(row["intake_ids_json"])
                    if ids:
                        self.settle_rows(conn, ids, "delivered", "", now)
                    self.cancel_retryable(
                        conn,
                        ids,
                        "superseded by a confirmed send",
                        now,
                        draft=(
                            str(row["session"]),
                            str(row["chat_id"]),
                            str(row["kind"]),
                            str(row["payload_json"]),
                        ),
                        exclude=intent_id,
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
                ids = self.intake_ids_of(conn, intent_id)
                if not ids:
                    return
                if operator_required or next_attempt_at is None:
                    self.settle_rows(
                        conn, ids, OPERATOR_REQUIRED, "delivery failed", int(now)
                    )
                else:
                    self.hold_rows(conn, ids, "delivery retrying")
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
                ids = self.intake_ids_of(conn, intent_id)
                if ids:
                    self.settle_rows(
                        conn, ids, OPERATOR_REQUIRED, "delivery unknown", now
                    )
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

    def cancel_retryable_for(
        self, intake_ids: list[str], reason: str, *, chat_id: str = ""
    ) -> None:
        """Cancel retryable failures whose turn another same-chat send answered.

        A send to another chat (an escalation, an operator notice) is not an
        answer to the turn: its failure must keep its own retry.
        """
        if not intake_ids:
            return
        conn = connect(self.path)
        try:
            with transaction(conn):
                self.cancel_retryable(
                    conn, intake_ids, reason, int(time.time()), chat_id=chat_id
                )
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
                self.insert_intent(
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
        unknown rows stay. A terminal operator notice is historical
        too: ``unnotified_operator`` excludes the notice kind, so its
        ``notified_at`` can never be set — without this clause a
        definitively failed notice would be retained forever.
        """
        conn = connect(self.path)
        try:
            with transaction(conn):
                cursor = conn.execute(
                    "DELETE FROM outbound_intents WHERE updated_at < ? AND ("
                    + "state IN (?, ?) OR (state IN (?, ?) "
                    + "AND operator_required = 1 AND notified_at IS NOT NULL) "
                    + "OR (kind = 'operator_notice' AND state IN (?, ?) "
                    + "AND next_attempt_at IS NULL))",
                    (before, SENT, CANCELLED, FAILED, UNKNOWN, FAILED, UNKNOWN),
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

    def find_sent_text(self, chat_id: str, text: str, since: float) -> str:
        """The transport id of a sent text in *chat_id* since *since*."""
        conn = connect(self.path)
        try:
            rows = conn.execute(
                "SELECT * FROM outbound_intents WHERE chat_id = ? "
                + "AND kind IN ('text', 'operator_notice') AND state = ? "
                + "AND created_at >= ? ORDER BY created_at DESC LIMIT 50",
                (chat_id, SENT, since),
            ).fetchall()
        finally:
            conn.close()
        for row in rows:
            intent = to_intent(row)
            if (
                str(intent.payload.get("text") or "") == text
                and intent.transport_message_id
            ):
                return intent.transport_message_id
        return ""

    def recoverable(self, now: float, limit: int = 50) -> list[DeliveryIntent]:
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
                + "ORDER BY created_at, rowid LIMIT ?",
                (PREPARED, FAILED, int(now), SENDING, limit),
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

    def set_state(self, intent_id: str, *, state: str) -> None:
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "UPDATE outbound_intents SET state = ?, updated_at = ? WHERE id = ?",
                    (state, int(time.time()), intent_id),
                )
        finally:
            conn.close()

    def insert_intent(
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

    def intake_ids_of(self, conn: Any, intent_id: str) -> list[str]:
        row = conn.execute(
            "SELECT intake_ids_json FROM outbound_intents WHERE id = ?",
            (intent_id,),
        ).fetchone()
        if row is None:
            return []
        return list(json.loads(row[0]))

    def settle_rows(
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

    def hold_rows(self, conn: Any, ids: list[str], reason: str) -> None:
        if not ids:
            return
        marks = ",".join("?" for _ in ids)
        conn.execute(
            "UPDATE intake SET state = 'delivering', next_attempt_at = NULL, "
            + "last_error = ? "
            + f"WHERE message_id IN ({marks}) AND state != 'done'",
            (reason, *ids),
        )

    def cancel_retryable(
        self,
        conn: Any,
        ids: list[str],
        reason: str,
        now: int | float,
        *,
        chat_id: str = "",
        draft: tuple[str, str, str, str] | None = None,
        exclude: str = "",
    ) -> None:
        wanted = set(ids)
        if not wanted and draft is None:
            return
        rows = conn.execute(
            "SELECT id, intake_ids_json, session, chat_id, kind, payload_json "
            + "FROM outbound_intents "
            + "WHERE (state = ? AND operator_required = 0) OR state = ?",
            (FAILED, PREPARED),
        ).fetchall()
        for row in rows:
            if exclude and str(row["id"]) == exclude:
                continue
            if matches_retryable(row, wanted, chat_id, draft):
                conn.execute(
                    "UPDATE outbound_intents SET state = ?, next_attempt_at = NULL, "
                    + "last_error = ?, updated_at = ? WHERE id = ?",
                    (CANCELLED, reason, int(now), row["id"]),
                )


def matches_retryable(
    row: Any,
    wanted: set[str],
    chat_id: str,
    draft: tuple[str, str, str, str] | None,
) -> bool:
    """Whether a stored retryable draft is superseded by the current send."""
    same_turn = bool(wanted.intersection(json.loads(row["intake_ids_json"])))
    if same_turn and chat_id and str(row["chat_id"]) != chat_id:
        same_turn = False
    if same_turn:
        return True
    return (
        draft is not None
        and (
            str(row["session"]),
            str(row["chat_id"]),
            str(row["kind"]),
            str(row["payload_json"]),
        )
        == draft
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
