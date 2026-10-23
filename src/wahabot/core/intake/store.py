"""Durable intake rows and the storage contract behind the ledger."""

import json
import time
from dataclasses import dataclass, field
from typing import Protocol

from wahabot.core.models import WahaEvent

PENDING = "pending"
PROCESSING = "processing"
DONE = "done"
FAILED = "failed"

DELIVERED = "delivered"
SILENT = "silent"
OPERATOR_OWNED = "operator_owned"
OPERATOR_REQUIRED = "operator_required"


@dataclass(slots=True)
class IntakeRow:
    """One accepted message, its turn state and its stored event."""

    message_id: str
    session: str
    chat_id: str
    record: str
    at: float
    state: str = PENDING
    addressed: bool = True
    raw_json: str = ""
    sender: str = ""
    received_at: float = field(default_factory=time.time)
    burst_key: str | None = None
    attempts: int = 0
    next_attempt_at: float | None = None
    last_error: str = ""
    outcome: str | None = None
    owner: str = ""
    reason: str = ""
    handled_at: float | None = None

    @property
    def event(self) -> WahaEvent:
        """The accepted event, reconstructed from the stored raw payload."""
        return WahaEvent.model_validate(json.loads(self.raw_json))


class IntakeStore(Protocol):
    """Storage primitives the ledger reads and updates.

    Every mutating call is one transaction, so a claim, release,
    finish or failure cannot interleave with another run.
    """

    def read(self, key: tuple[str, str]) -> list[IntakeRow]: ...

    def record(self, row: IntakeRow) -> IntakeRow: ...

    def claim(
        self,
        key: tuple[str, str],
        message_ids: list[str],
        owner: str,
        lease_until: float,
    ) -> list[IntakeRow]: ...

    def handled(
        self, key: tuple[str, str], message_ids: list[str], addressed_only: bool
    ) -> bool: ...

    def release(self, key: tuple[str, str], message_ids: list[str]) -> None: ...

    def finish(
        self,
        key: tuple[str, str],
        message_ids: list[str],
        outcome: str,
        reason: str,
        handled_at: float,
    ) -> None: ...

    def finish_turn(
        self,
        key: tuple[str, str],
        turn_ids: list[str],
        claimed_ids: list[str],
        delivered: bool,
        outcome: str,
        reason: str,
        handled_at: float,
    ) -> None: ...

    def fail(
        self,
        key: tuple[str, str],
        message_ids: list[str],
        error: str,
        next_attempt_at: float | None,
    ) -> None: ...

    def set_burst_key(
        self, key: tuple[str, str], message_ids: list[str], burst_key: str
    ) -> None: ...

    def due(self, now: float, limit: int) -> list[IntakeRow]: ...

    def stale_processing(self, before: float, limit: int) -> list[IntakeRow]: ...

    def stats(self) -> dict[str, int]: ...

    def purge(self, before: float) -> int: ...
