"""The durable outbound intent model and its stored-row mapping."""

import json
from dataclasses import dataclass
from typing import Any

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
