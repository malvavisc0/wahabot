"""Turn-level authority over accepted inbound messages.

The seen cache is a fast duplicate check; this ledger survives the gap
between accepting a message and its run finishing. A run claims messages
that arrive after it started, so a stale reply cannot overtake them, and
a queued run waits for an active claim instead of racing it. Storage is
injectable: the default is process memory, and a durable store can
implement the same read/write surface later.
"""

import threading
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Protocol

PENDING = "pending"
CLAIMED = "claimed"
HANDLED = "handled"

#: One chat's intake is small: messages older than this are no longer
#: relevant to revalidation, and the cap bounds memory under a flood.
MAX_AGE_S = 900
MAX_MESSAGES = 1000


@dataclass(slots=True)
class IntakeEntry:
    """One accepted message with its rendered record and turn state."""

    message_id: str
    session: str
    chat_id: str
    record: str
    at: float
    state: str = PENDING
    #: Whether the message would wake a run of its own. Records that
    #: never wake a run (unaddressed group chatter) never supersede a
    #: reply either — they age out unclaimed.
    addressed: bool = True


class IntakeStore(Protocol):
    """Per-chat persistence for intake entries."""

    def read(self, key: tuple[str, str]) -> list[IntakeEntry]: ...

    def write(self, key: tuple[str, str], entries: Sequence[IntakeEntry]) -> None: ...


class InMemoryIntakeStore:
    """Process-memory storage, one list per (session, chat)."""

    def __init__(self) -> None:
        self._chats: dict[tuple[str, str], list[IntakeEntry]] = {}

    def read(self, key: tuple[str, str]) -> list[IntakeEntry]:
        return self._chats.get(key, [])

    def write(self, key: tuple[str, str], entries: Sequence[IntakeEntry]) -> None:
        self._chats[key] = list(entries)


class IntakeLedger:
    """Records, claims, releases and finishes accepted inbound messages."""

    def __init__(
        self,
        store: IntakeStore | None = None,
        *,
        max_age_s: float = MAX_AGE_S,
        max_messages: int = MAX_MESSAGES,
    ) -> None:
        self.store = store or InMemoryIntakeStore()
        self.max_age_s = max_age_s
        self.max_messages = max_messages
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
    ) -> None:
        """Add one accepted message as pending, unless it is already known."""
        if not message_id:
            return
        with self._lock:
            key = (session, chat_id)
            entries = self._live(key)
            if any(entry.message_id == message_id for entry in entries):
                return
            entries.append(
                IntakeEntry(
                    message_id=message_id,
                    session=session,
                    chat_id=chat_id,
                    record=record,
                    at=time.monotonic() if at is None else at,
                    addressed=addressed,
                )
            )
            self.store.write(key, entries[-self.max_messages :])

    def pending(
        self,
        session: str,
        chat_id: str,
        *,
        after: float | None = None,
        exclude: Collection[str] = (),
        addressed_only: bool = False,
    ) -> list[IntakeEntry]:
        """Pending messages, optionally only those newer than ``after``."""
        skipped = set(exclude)
        with self._lock:
            now = time.monotonic()
            return [
                entry
                for entry in self._live((session, chat_id), now)
                if entry.state == PENDING
                and entry.message_id not in skipped
                and (after is None or entry.at > after)
                and (not addressed_only or entry.addressed)
            ]

    def claim(
        self, session: str, chat_id: str, message_ids: Sequence[str]
    ) -> list[IntakeEntry]:
        """Claim pending rows for the run that is about to answer them."""
        key = (session, chat_id)
        wanted = set(message_ids)
        with self._lock:
            entries = self._live(key)
            claimed: list[IntakeEntry] = []
            for entry in entries:
                if entry.message_id in wanted and entry.state == PENDING:
                    entry.state = CLAIMED
                    claimed.append(entry)
            if claimed:
                self.store.write(key, entries)
            return claimed

    def claimed(self, session: str, chat_id: str, message_ids: Sequence[str]) -> bool:
        """Whether any requested id is currently claimed by an active run.

        A claim never outlives its run — the run's finally-block settles
        or releases it — so ``True`` means "wait for that run's decision",
        never "this turn is stranded".
        """
        wanted = set(message_ids)
        if not wanted:
            return False
        with self._lock:
            return any(
                entry.message_id in wanted and entry.state == CLAIMED
                for entry in self._live((session, chat_id))
            )

    def handled(self, session: str, chat_id: str, message_ids: Sequence[str]) -> bool:
        """Whether every requested id is known and already handled."""
        wanted = set(message_ids)
        if not wanted:
            return False
        with self._lock:
            found = [
                entry
                for entry in self._live((session, chat_id))
                if entry.message_id in wanted
            ]
            return {entry.message_id for entry in found} == wanted and all(
                entry.state == HANDLED for entry in found
            )

    def release(self, session: str, chat_id: str, message_ids: Sequence[str]) -> None:
        """Return rows this run claimed to pending so their own run can answer.

        Only claims: a row another run settled by delivering stays
        handled — ``HANDLED`` means an answer reached the chat, never a
        suppression left to undo.
        """
        with self._lock:
            self._update(
                (session, chat_id),
                set(message_ids),
                lambda entry: entry.state == CLAIMED,
                PENDING,
            )

    def finish(
        self,
        session: str,
        chat_id: str,
        message_ids: Sequence[str],
        *,
        delivered: bool,
    ) -> None:
        """Settle rows: handled after a delivery, released after silence."""
        if delivered:
            with self._lock:
                self._update(
                    (session, chat_id), set(message_ids), lambda entry: True, HANDLED
                )
        else:
            self.release(session, chat_id, message_ids)

    def finish_turn(
        self,
        session: str,
        chat_id: str,
        turn_ids: Sequence[str],
        claimed_ids: Sequence[str],
        *,
        delivered: bool,
    ) -> None:
        """Settle a run's own ids and every id it claimed while running."""
        if delivered:
            self.finish(session, chat_id, [*turn_ids, *claimed_ids], delivered=True)
        else:
            # Only the claimed ids were this run's to lose; its own turn
            # may already be handled by an earlier run.
            self.release(session, chat_id, claimed_ids)

    def _live(self, key: tuple[str, str], now: float | None = None) -> list[IntakeEntry]:
        moment = time.monotonic() if now is None else now
        entries = [
            entry for entry in self.store.read(key) if moment - entry.at <= self.max_age_s
        ][-self.max_messages :]
        self.store.write(key, entries)
        return entries

    def _update(
        self,
        key: tuple[str, str],
        wanted: set[str],
        matches: Callable[[IntakeEntry], bool],
        state: str,
    ) -> None:
        if not wanted:
            return
        entries = self._live(key)
        changed = False
        for entry in entries:
            if entry.message_id in wanted and matches(entry):
                entry.state = state
                changed = True
        if changed:
            self.store.write(key, entries)
