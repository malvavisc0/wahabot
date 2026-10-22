"""Turn-level authority over accepted inbound messages.

The seen cache is a fast duplicate check; this ledger survives the gap
between accepting a message and its run finishing. A run claims messages
that arrive after it started, so a stale reply cannot overtake them.
Storage is injectable: the default is process memory, and a durable
store can implement the same read/write surface later.
"""

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


class IntakeStore(Protocol):
    """Per-chat persistence for intake entries."""

    def read(self, key: tuple[str, str]) -> list[IntakeEntry]: ...

    def write(self, key: tuple[str, str], entries: Sequence[IntakeEntry]) -> None: ...

    def chat_keys(self) -> list[tuple[str, str]]: ...


class InMemoryIntakeStore:
    """Process-memory storage, one list per (session, chat)."""

    def __init__(self) -> None:
        self._chats: dict[tuple[str, str], list[IntakeEntry]] = {}

    def read(self, key: tuple[str, str]) -> list[IntakeEntry]:
        return self._chats.get(key, [])

    def write(self, key: tuple[str, str], entries: Sequence[IntakeEntry]) -> None:
        self._chats[key] = list(entries)

    def chat_keys(self) -> list[tuple[str, str]]:
        return list(self._chats)


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

    def record(
        self,
        session: str,
        chat_id: str,
        message_id: str,
        record: str,
        *,
        at: float | None = None,
    ) -> None:
        """Add one accepted message as pending, unless it is already known."""
        if not message_id:
            return
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
    ) -> list[IntakeEntry]:
        """Pending messages, optionally only those newer than ``after``."""
        skipped = set(exclude)
        now = time.monotonic()
        return [
            entry
            for entry in self._live((session, chat_id), now)
            if entry.state == PENDING
            and entry.message_id not in skipped
            and (after is None or entry.at > after)
        ]

    def claim(
        self, session: str, chat_id: str, message_ids: Sequence[str]
    ) -> list[IntakeEntry]:
        """Claim pending rows for the run that is about to answer them."""
        key = (session, chat_id)
        wanted = set(message_ids)
        entries = self._live(key)
        claimed: list[IntakeEntry] = []
        for entry in entries:
            if entry.message_id in wanted and entry.state == PENDING:
                entry.state = CLAIMED
                claimed.append(entry)
        if claimed:
            self.store.write(key, entries)
        return claimed

    def handled(self, message_ids: Sequence[str]) -> bool:
        """Whether every requested id is known and already handled."""
        wanted = set(message_ids)
        if not wanted:
            return False
        found = self._find(wanted)
        return {entry.message_id for entry in found} == wanted and all(
            entry.state == HANDLED for entry in found
        )

    def release(self, message_ids: Sequence[str]) -> None:
        """Return claimed rows to pending so their own run can answer them."""
        self._update(set(message_ids), lambda entry: entry.state == CLAIMED, PENDING)

    def finish(self, message_ids: Sequence[str], delivered: bool) -> None:
        """Settle rows: handled after a delivery, released after silence."""
        if delivered:
            self._update(set(message_ids), lambda entry: True, HANDLED)
        else:
            self.release(message_ids)

    def finish_turn(
        self,
        turn_ids: Sequence[str],
        claimed_ids: Sequence[str],
        *,
        delivered: bool,
    ) -> None:
        """Settle a run's own ids and every id it claimed while running."""
        self.finish([*turn_ids, *claimed_ids], delivered)

    def _live(self, key: tuple[str, str], now: float | None = None) -> list[IntakeEntry]:
        moment = time.monotonic() if now is None else now
        entries = [
            entry for entry in self.store.read(key) if moment - entry.at <= self.max_age_s
        ][-self.max_messages :]
        self.store.write(key, entries)
        return entries

    def _find(self, wanted: set[str]) -> list[IntakeEntry]:
        found: list[IntakeEntry] = []
        for key in self.store.chat_keys():
            entries = self.store.read(key)
            matched = [entry for entry in entries if entry.message_id in wanted]
            if matched:
                found.extend(matched)
        return found

    def _update(
        self, wanted: set[str], matches: Callable[[IntakeEntry], bool], state: str
    ) -> None:
        if not wanted:
            return
        for key in self.store.chat_keys():
            entries = self.store.read(key)
            changed = False
            for entry in entries:
                if entry.message_id in wanted and matches(entry):
                    entry.state = state
                    changed = True
            if changed:
                self.store.write(key, entries)
