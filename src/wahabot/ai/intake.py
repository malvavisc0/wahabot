"""Turn-level authority over accepted inbound messages.

The seen cache is a fast duplicate check; this ledger survives the gap
between accepting a message and its run finishing. A run claims messages
that arrive after it started, so a stale reply cannot overtake them, and
a queued run waits for an active claim instead of racing it. Storage is
injectable: the default is process memory, and the durable store keeps
the same read/write surface in SQLite.
"""

import threading
import time
from collections.abc import Collection, Sequence

from wahabot.core.intake.memory import InMemoryIntakeStore
from wahabot.core.intake.store import (
    DELIVERED,
    PENDING,
    PROCESSING,
    IntakeRow,
    IntakeStore,
)

IntakeEntry = IntakeRow

#: One chat's intake is small: messages older than this are no longer
#: relevant to revalidation, and the cap bounds memory under a flood.
#: It must outlive ``Settings.run_timeout``: a run can hold a claim for
#: its whole wall-clock budget, and an expiring claim would let the
#: queued run answer the same turn again.
MAX_AGE_S = 1800
MAX_MESSAGES = 1000

#: How long a claimed row stays a live run's before recovery may reset it.
CLAIM_LEASE_S = 1800


def max_age_for_run_timeout(run_timeout: float) -> float:
    """The intake window that keeps a claim alive for one full run.

    Claims are stamped at arrival, not at claim time, so a run that
    started before the message can claim it near its own deadline. The
    extra margin keeps the row alive slightly past that deadline.
    """
    return max(MAX_AGE_S, run_timeout + 300)


class IntakeLedger:
    """Records, claims, releases and finishes accepted inbound messages."""

    def __init__(
        self,
        store: IntakeStore | None = None,
        *,
        max_age_s: float = MAX_AGE_S,
        max_messages: int = MAX_MESSAGES,
        claim_lease_s: float = CLAIM_LEASE_S,
    ) -> None:
        self.store = store or InMemoryIntakeStore()
        self.max_age_s = max_age_s
        self.max_messages = max_messages
        self.claim_lease_s = claim_lease_s
        # Delivery tools run on worker threads while the loop records
        # new messages, so reads and claims must not interleave.
        self._lock = threading.RLock()

    def record(
        self,
        session: str,
        chat_id: str,
        message_id: str,
        record: str,
        *,
        at: float | None = None,
        addressed: bool = True,
        raw_json: str = "",
        sender: str = "",
        burst_key: str | None = None,
        received_at: float | None = None,
    ) -> IntakeRow:
        """Add one accepted message as pending, unless it is already known."""
        row = IntakeRow(
            message_id=message_id,
            session=session,
            chat_id=chat_id,
            record=record,
            at=time.monotonic() if at is None else at,
            addressed=addressed,
            raw_json=raw_json,
            sender=sender,
            received_at=time.time() if received_at is None else received_at,
            burst_key=burst_key,
        )
        with self._lock:
            return self.store.record(row)

    def read(self, session: str, chat_id: str) -> list[IntakeRow]:
        """Every stored row of one chat, oldest first."""
        with self._lock:
            return self._all((session, chat_id))

    def rows(self, session: str, chat_id: str) -> list[IntakeRow]:
        """Every stored row of one chat, oldest first."""
        return self.read(session, chat_id)

    def row(self, session: str, chat_id: str, message_id: str) -> IntakeRow | None:
        """One stored row, or None when it was never accepted."""
        with self._lock:
            return self.store.get((session, chat_id), message_id)

    def pending(
        self,
        session: str,
        chat_id: str,
        *,
        after: float | None = None,
        exclude: Collection[str] = (),
        addressed_only: bool = False,
    ) -> list[IntakeRow]:
        """Pending messages, optionally only those newer than ``after``."""
        skipped = set(exclude)
        with self._lock:
            return [
                entry
                for entry in self._live((session, chat_id))
                if entry.state == PENDING
                and entry.message_id not in skipped
                and (after is None or entry.at > after)
                and (not addressed_only or entry.addressed)
            ]

    def newer(
        self,
        session: str,
        chat_id: str,
        *,
        after: float | None = None,
        exclude: Collection[str] = (),
        addressed_only: bool = False,
    ) -> list[IntakeRow]:
        """Pending rows newer than a turn, for send revalidation."""
        skipped = set(exclude)
        with self._lock:
            return [
                entry
                for entry in self._live((session, chat_id))
                if entry.state == PENDING
                and entry.message_id not in skipped
                and (after is None or entry.at > after)
                and (not addressed_only or entry.addressed)
            ]

    def claim(
        self,
        session: str,
        chat_id: str,
        message_ids: Sequence[str],
        *,
        owner: str = "",
    ) -> list[IntakeRow]:
        """Claim pending rows for the run that is about to answer them."""
        wanted = set(message_ids)
        with self._lock:
            pending = [
                entry.message_id
                for entry in self._all((session, chat_id))
                if entry.message_id in wanted and entry.state == PENDING
            ]
            return self.store.claim(
                (session, chat_id),
                pending,
                owner,
                time.time() + self.claim_lease_s,
            )

    def claimed(self, session: str, chat_id: str, message_ids: Sequence[str]) -> bool:
        """Whether any requested id is currently claimed by an active run."""
        wanted = set(message_ids)
        if not wanted:
            return False
        with self._lock:
            return any(
                entry.message_id in wanted and entry.state == PROCESSING
                for entry in self._all((session, chat_id))
            )

    def handled(
        self,
        session: str,
        chat_id: str,
        message_ids: Sequence[str],
        *,
        addressed_only: bool = False,
    ) -> bool:
        """Whether the requested ids are known and already handled.

        With ``addressed_only`` only rows that would wake a run count:
        an album's unaddressed image rows never own an answer, so they
        must not keep an already-answered album run alive.
        """
        with self._lock:
            return self.store.handled(
                (session, chat_id), list(message_ids), addressed_only
            )

    def release(
        self,
        session: str,
        chat_id: str,
        message_ids: Sequence[str],
        *,
        owner: str = "",
    ) -> None:
        """Return rows this run claimed to pending so their own run can answer.

        Only claims: a row another run settled by delivering stays
        handled — ``DONE`` means an answer reached the chat, never a
        suppression left to undo. A non-empty owner releases only that
        run's claims.
        """
        with self._lock:
            self.store.release((session, chat_id), list(message_ids), owner)

    def finish(
        self,
        session: str,
        chat_id: str,
        message_ids: Sequence[str],
        *,
        delivered: bool,
        outcome: str | None = None,
        reason: str = "",
    ) -> None:
        """Settle rows: done after a delivery, released after silence."""
        if not delivered and outcome is None:
            self.release(session, chat_id, message_ids)
            return
        with self._lock:
            self.store.finish(
                (session, chat_id),
                list(message_ids),
                outcome or DELIVERED,
                reason,
                time.time(),
            )

    def finish_turn(
        self,
        session: str,
        chat_id: str,
        turn_ids: Sequence[str],
        claimed_ids: Sequence[str],
        *,
        delivered: bool,
        outcome: str | None = None,
        reason: str = "",
        owner: str = "",
    ) -> None:
        """Settle a run's own ids and every id it claimed while running."""
        with self._lock:
            self.store.finish_turn(
                (session, chat_id),
                list(turn_ids),
                list(claimed_ids),
                delivered,
                outcome or "",
                reason,
                time.time(),
                owner,
            )

    def hold(
        self, session: str, chat_id: str, message_ids: Sequence[str], reason: str
    ) -> None:
        """Hand rows to an outbound intent so recovery does not rerun them."""
        with self._lock:
            self.store.hold((session, chat_id), list(message_ids), reason)

    def is_delivering(
        self, session: str, chat_id: str, message_ids: Sequence[str]
    ) -> bool:
        """True when an outbound intent owns at least one of these rows."""
        with self._lock:
            return self.store.is_delivering((session, chat_id), list(message_ids))

    def delivering_rows(self, limit: int = 50) -> list[IntakeRow]:
        """Rows waiting for an outbound intent."""
        with self._lock:
            return self.store.delivering_rows(limit)

    def reset_delivering(self, message_ids: Sequence[str]) -> None:
        """Requeue rows whose intent was cancelled or already settled."""
        with self._lock:
            self.store.reset_delivering(list(message_ids))

    def fail(
        self,
        session: str,
        chat_id: str,
        message_ids: Sequence[str],
        *,
        error: str,
        next_attempt_at: float | None,
    ) -> None:
        """Record a definite failure and its next retry time."""
        with self._lock:
            self.store.fail(
                (session, chat_id),
                list(message_ids),
                error,
                next_attempt_at,
            )

    def set_burst_key(
        self, session: str, chat_id: str, message_ids: Sequence[str], burst_key: str
    ) -> None:
        """Stamp the window's key on rows that share one flush."""
        with self._lock:
            self.store.set_burst_key((session, chat_id), list(message_ids), burst_key)

    def due(self, now: float | None = None, limit: int = 50) -> list[IntakeRow]:
        """Pending and retry-due failed rows across chats, oldest first."""
        with self._lock:
            return self.store.due(time.time() if now is None else now, limit)

    def stale_processing(
        self, now: float | None = None, limit: int = 50
    ) -> list[IntakeRow]:
        """Processing rows whose claim lease has lapsed."""
        with self._lock:
            return self.store.stale_processing(time.time() if now is None else now, limit)

    def reset_processing(self) -> int:
        """Return every claim to pending at startup without a retry increment."""
        with self._lock:
            return self.store.reset_processing()

    def unnotified_operator(self, limit: int = 50) -> list[IntakeRow]:
        """Operator-required rows whose notice has not been queued yet."""
        with self._lock:
            return self.store.unnotified_operator(limit)

    def mark_notified(self, message_ids: Sequence[str]) -> None:
        """Record that a durable operator notice covers these rows."""
        with self._lock:
            self.store.mark_notified(list(message_ids), time.time())

    def stats(self) -> dict[str, int]:
        """Backlog age and unanswered count for the heartbeat."""
        with self._lock:
            return self.store.stats()

    def purge(self, retention_days: int) -> int:
        """Drop long-finished rows; unfinished work is never deleted."""
        before = time.time() - retention_days * 86400
        with self._lock:
            return self.store.purge(before)

    def _all(self, key: tuple[str, str]) -> list[IntakeRow]:
        return self.store.read(key, self.max_messages)

    def _live(self, key: tuple[str, str], now: float | None = None) -> list[IntakeRow]:
        moment = time.monotonic() if now is None else now
        return [entry for entry in self._all(key) if moment - entry.at <= self.max_age_s]
