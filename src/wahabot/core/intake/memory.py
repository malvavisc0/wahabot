"""Process-memory intake storage for tests and single-process callers."""

from wahabot.core.intake.store import (
    CLAIMABLE_STATES,
    DELIVERED,
    DELIVERING,
    DONE,
    FAILED,
    OPERATOR_REQUIRED,
    PENDING,
    PROCESSING,
    IntakeRow,
)


class InMemoryIntakeStore:
    """Rows kept per (session, chat), insertion ordered."""

    def __init__(self) -> None:
        self._chats: dict[tuple[str, str], dict[str, IntakeRow]] = {}

    def read(self, key: tuple[str, str], limit: int | None = None) -> list[IntakeRow]:
        rows = list(self._chats.get(key, {}).values())
        if limit is None:
            return rows
        if limit <= 0:
            return []
        return rows[-limit:]

    def get(self, key: tuple[str, str], message_id: str) -> IntakeRow | None:
        return self._chats.get((key[0], key[1]), {}).get(message_id)

    def record(self, row: IntakeRow) -> IntakeRow:
        rows = self._chats.setdefault((row.session, row.chat_id), {})
        existing = rows.get(row.message_id)
        if existing is not None:
            return existing
        rows[row.message_id] = row
        return row

    def select_rows(self, key: tuple[str, str], ids: list[str]) -> list[IntakeRow]:
        rows = self._chats.get((key[0], key[1]), {})
        wanted = set(ids)
        return [row for message_id, row in rows.items() if message_id in wanted]

    def claim(
        self,
        key: tuple[str, str],
        message_ids: list[str],
        owner: str,
        lease_until: float,
    ) -> list[IntakeRow]:
        claimed: list[IntakeRow] = []
        for row in self.select_rows(key, message_ids):
            if row.state in CLAIMABLE_STATES:
                row.state = PROCESSING
                row.owner = owner
                row.next_attempt_at = lease_until
                claimed.append(row)
        return claimed

    def handled(
        self, key: tuple[str, str], message_ids: list[str], addressed_only: bool
    ) -> bool:
        wanted = set(message_ids)
        if not wanted:
            return False
        found = [
            row
            for row in self.select_rows(key, message_ids)
            if not addressed_only or row.addressed
        ]
        if addressed_only:
            return bool(found) and self.all_done(found)
        return {row.message_id for row in found} == wanted and self.all_done(found)

    def all_done(self, rows: list[IntakeRow]) -> bool:
        return all(row.state == DONE for row in rows)

    def release(
        self, key: tuple[str, str], message_ids: list[str], owner: str = ""
    ) -> None:
        for row in self.select_rows(key, message_ids):
            if row.state == PROCESSING and (not owner or row.owner == owner):
                row.state = PENDING
                row.owner = ""
                row.next_attempt_at = None

    def hold(self, key: tuple[str, str], message_ids: list[str], reason: str) -> None:
        for row in self.select_rows(key, message_ids):
            if row.state != DONE:
                row.state = DELIVERING
                row.next_attempt_at = None
                row.last_error = reason

    def is_delivering(self, key: tuple[str, str], message_ids: list[str]) -> bool:
        return any(row.state == DELIVERING for row in self.select_rows(key, message_ids))

    def delivering_rows(self, limit: int) -> list[IntakeRow]:
        rows = [
            row
            for chat in self._chats.values()
            for row in chat.values()
            if row.state == DELIVERING
        ]
        rows.sort(key=lambda row: row.received_at)
        return rows[:limit]

    def reset_delivering(self, message_ids: list[str]) -> None:
        wanted = set(message_ids)
        for chat in self._chats.values():
            for message_id, row in chat.items():
                if message_id in wanted and row.state == DELIVERING:
                    row.state = PENDING
                    row.owner = ""
                    row.next_attempt_at = None

    def finish(
        self,
        key: tuple[str, str],
        message_ids: list[str],
        outcome: str,
        reason: str,
        handled_at: float,
    ) -> None:
        for row in self.select_rows(key, message_ids):
            if row.state == DONE:
                continue
            row.state = DONE
            row.outcome = outcome
            row.reason = reason
            row.handled_at = handled_at

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
        if delivered:
            self.finish(
                key,
                [*turn_ids, *claimed_ids],
                outcome or DELIVERED,
                reason,
                handled_at,
            )
            return
        self.release(key, claimed_ids, owner)
        if outcome:
            self.finish(key, turn_ids, outcome, reason, handled_at)

    def fail(
        self,
        key: tuple[str, str],
        message_ids: list[str],
        error: str,
        next_attempt_at: float | None,
    ) -> None:
        for row in self.select_rows(key, message_ids):
            if row.state == DONE:
                continue
            row.state = FAILED
            row.attempts += 1
            row.last_error = error
            row.next_attempt_at = next_attempt_at

    def set_burst_key(
        self, key: tuple[str, str], message_ids: list[str], burst_key: str
    ) -> None:
        for row in self.select_rows(key, message_ids):
            row.burst_key = burst_key

    def due(self, now: float, limit: int) -> list[IntakeRow]:
        rows = [
            row
            for chat in self._chats.values()
            for row in chat.values()
            if row.state == PENDING
            or (
                row.state == FAILED
                and (row.next_attempt_at is None or row.next_attempt_at <= now)
            )
        ]
        rows.sort(key=lambda row: row.received_at)
        return rows[:limit]

    def stale_processing(self, before: float, limit: int) -> list[IntakeRow]:
        rows = [
            row
            for chat in self._chats.values()
            for row in chat.values()
            if row.state == PROCESSING
            and row.next_attempt_at is not None
            and row.next_attempt_at <= before
        ]
        rows.sort(key=lambda row: row.received_at)
        return rows[:limit]

    def reset_processing(self) -> int:
        reset = 0
        for chat in self._chats.values():
            for row in chat.values():
                if row.state == PROCESSING:
                    row.state = PENDING
                    row.owner = ""
                    row.next_attempt_at = None
                    reset += 1
        return reset

    def unnotified_operator(self, limit: int) -> list[IntakeRow]:
        rows = [
            row
            for chat in self._chats.values()
            for row in chat.values()
            if row.state == DONE
            and row.outcome == OPERATOR_REQUIRED
            and row.notified_at is None
        ]
        rows.sort(key=lambda row: row.received_at)
        return rows[:limit]

    def purge(self, before: float) -> int:
        removed = 0
        for key, rows in list(self._chats.items()):
            for message_id, row in list(rows.items()):
                if row.state == DONE and row.received_at < before:
                    del rows[message_id]
                    removed += 1
            if not rows:
                self._chats.pop(key, None)
        return removed
