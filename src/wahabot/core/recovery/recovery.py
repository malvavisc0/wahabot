"""Startup and periodic recovery across durable intake and outbound rows."""

import asyncio
import time
from collections.abc import Callable
from typing import Protocol, cast

from loguru import logger

from wahabot.core.delivery.intents import (
    FAILED as INTENT_FAILED,
)
from wahabot.core.delivery.intents import (
    UNKNOWN as INTENT_UNKNOWN,
)
from wahabot.core.delivery.intents import (
    DeliveryIntent,
    IntentStore,
)
from wahabot.core.intake.store import (
    DONE,
    OPERATOR_REQUIRED,
    PENDING,
    IntakeRow,
)
from wahabot.core.intake.store import (
    FAILED as ROW_FAILED,
)
from wahabot.core.recovery.context import APOLOGY
from wahabot.core.runs import chat_lock
from wahabot.core.transport.protocol import Transport
from wahabot.settings import Settings
from wahabot.status.state import state

#: One notice can describe at most this many required rows; the rest stay
#: unnotified and are covered by the next tick.
_REQUIRED_NOTICE_LIMIT = 50


class RecoveryTransport(Transport, Protocol):
    """The transport surface recovery drives, including intent replay."""

    intents: IntentStore | None

    def recover_intent(
        self,
        intent: DeliveryIntent,
        now: float | None = None,
        *,
        recover_sending: bool = False,
    ) -> str | None: ...


class RecoveryRunner(Protocol):
    """The handler operations recovery drives."""

    async def recover_rows(self, rows: list[IntakeRow], *, apology: bool) -> str: ...

    def reply_window_days(self) -> int: ...


class RecoveryLedger(Protocol):
    """The durable intake operations recovery uses."""

    def due(self, now: float, limit: int) -> list[IntakeRow]: ...

    def stale_processing(self, now: float, limit: int) -> list[IntakeRow]: ...

    def delivering_rows(self, limit: int) -> list[IntakeRow]: ...

    def reset_delivering(self, message_ids: list[str]) -> None: ...

    def release(
        self,
        session: str,
        chat_id: str,
        message_ids: list[str],
        *,
        owner: str = "",
    ) -> None: ...

    def reset_processing(self) -> int: ...

    def fail(
        self,
        session: str,
        chat_id: str,
        message_ids: list[str],
        *,
        error: str,
        next_attempt_at: float | None,
    ) -> None: ...

    def finish(
        self,
        session: str,
        chat_id: str,
        message_ids: list[str],
        *,
        delivered: bool,
        outcome: str,
        reason: str = "",
    ) -> None: ...

    def unnotified_operator(self, limit: int) -> list[IntakeRow]: ...

    def row(self, session: str, chat_id: str, message_id: str) -> IntakeRow | None: ...

    def purge(self, retention_days: int) -> int: ...

    def stats(self) -> dict[str, int]: ...


class Recovery:
    """Recover accepted work after a restart and on a slow periodic tick."""

    def __init__(
        self,
        settings: Settings,
        ledger: RecoveryLedger,
        transport: RecoveryTransport,
        runner: RecoveryRunner,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings
        self.ledger = ledger
        self.transport = transport
        self.runner = runner
        self.clock = clock
        if transport.intents is None:
            raise ValueError("recovery requires a transport with intents")
        self.intents: IntentStore = transport.intents

    async def startup(self) -> None:
        """Recover after a restart, before the first periodic tick."""
        self.ledger.purge(self.settings.intake_retention_days)
        # A confirmed send settles its rows in the same transaction, but a
        # crash or an older row can still leave a sent intent with pending
        # intake rows; finish them before anything can re-answer the turn.
        self.intents.reconcile_sent_intake()
        # No live run survived the restart: every processing row is stale.
        # Return it to pending without spending a retry attempt.
        self.ledger.reset_processing()
        self._reconcile_delivering()
        await self.tick(startup=True)

    async def tick(self, *, startup: bool = False, now: float | None = None) -> None:
        """Run one bounded recovery pass; callable directly by tests."""
        moment = self.clock() if now is None else now
        self._reconcile_delivering()
        self._flush_operator_notices()
        await self._recover_intents(moment, startup=startup)
        self._reset_stale(moment)
        due = self.ledger.due(moment, limit=20)
        delayed = 0
        for group in self._groups(due):
            outcome = await self._recover_group(group, moment)
            if outcome == "delayed":
                delayed += len(group)
        await self._notify_operator(delayed, due, moment)

    async def _recover_intents(self, moment: float, *, startup: bool) -> None:
        """Replay due intents under the same chat locks as live runs."""
        for intent in self.intents.recoverable(moment):
            prepared = self._prepare_intent(intent, moment)
            if prepared is None:
                continue
            session, chat_id = self._lock_key(prepared)
            async with chat_lock(session, chat_id):
                await asyncio.to_thread(
                    self.transport.recover_intent,
                    prepared,
                    moment,
                    recover_sending=startup,
                )

    def _prepare_intent(
        self, intent: DeliveryIntent, moment: float
    ) -> DeliveryIntent | None:
        """Apply the reply-window and apology policy before replaying."""
        rows = self._intent_rows(intent)
        if rows:
            if all(row.state == DONE for row in rows):
                # Another confirmed send (or a recovery settlement) already
                # answered the turn. Never replay an old draft over it.
                self.intents.cancel(intent.id, "intake already settled")
                return None
            age = moment - min(row.received_at for row in rows)
            if age > self.runner.reply_window_days() * 86400:
                self._require_operator(rows, "outside the reply window")
                self.intents.cancel(intent.id, "outside the reply window")
                return None
            if age > self.settings.intake_recovery_minutes * 60:
                intent = self.intents.prepend_text(intent.id, APOLOGY)
        return intent

    def _intent_rows(self, intent: DeliveryIntent) -> list[IntakeRow]:
        rows: list[IntakeRow] = []
        if not intent.chat_id:
            return rows
        for message_id in intent.intake_ids:
            row = self.ledger.row(intent.session, intent.chat_id, message_id)
            if row is not None:
                rows.append(row)
        return rows

    def _lock_key(self, intent: DeliveryIntent) -> tuple[str, str]:
        rows = self._intent_rows(intent)
        if rows:
            return (rows[0].session, rows[0].chat_id)
        return (intent.session, intent.chat_id or "operator")

    def _reset_stale(self, now: float) -> None:
        for row in self.ledger.stale_processing(now, limit=50):
            self.ledger.release(
                row.session,
                row.chat_id,
                [row.message_id],
                owner=row.owner,
            )

    def _reconcile_delivering(self) -> None:
        """Requeue rows whose intent was settled without settling the row."""
        owned = self.intents.unsettled_intake_ids()
        stranded = [
            row.message_id
            for row in self.ledger.delivering_rows(limit=100)
            if row.message_id not in owned
        ]
        if stranded:
            self.ledger.reset_delivering(stranded)

    def _flush_operator_notices(self) -> None:
        destination = state.operator_jid
        if not destination:
            return
        for intent in self.intents.pending_notices():
            self.intents.set_chat(intent.id, destination)

    async def _recover_group(self, rows: list[IntakeRow], now: float) -> str:
        fresh = [row for row in rows if row.state in (PENDING, ROW_FAILED)]
        if not fresh:
            return "done"
        refusal = self._refuse_group(fresh, now)
        if refusal is not None:
            return refusal
        apology = now - min(row.received_at for row in fresh) > (
            self.settings.intake_recovery_minutes * 60
        )
        outcome = await self._run_group(fresh, apology, now)
        if outcome == OPERATOR_REQUIRED:
            return OPERATOR_REQUIRED
        return "delayed" if apology else outcome

    def _refuse_group(self, rows: list[IntakeRow], now: float) -> str | None:
        if max(row.attempts for row in rows) >= self.settings.intake_max_attempts:
            self._require_operator(rows, "retry budget exhausted")
            return OPERATOR_REQUIRED
        age = now - min(row.received_at for row in rows)
        if age > self.runner.reply_window_days() * 86400:
            self._require_operator(rows, "outside the reply window")
            return OPERATOR_REQUIRED
        return None

    async def _run_group(self, rows: list[IntakeRow], apology: bool, now: float) -> str:
        try:
            return await self.runner.recover_rows(rows, apology=apology)
        except Exception:
            logger.exception(
                "Recovery failed for {chat_id}",
                chat_id=rows[0].chat_id,
            )
            self.ledger.fail(
                rows[0].session,
                rows[0].chat_id,
                [row.message_id for row in rows],
                error="recovery run failed",
                next_attempt_at=now + 60,
            )
            return "failed"

    def _require_operator(self, rows: list[IntakeRow], reason: str) -> None:
        self.ledger.finish(
            rows[0].session,
            rows[0].chat_id,
            [row.message_id for row in rows],
            delivered=False,
            outcome=OPERATOR_REQUIRED,
            reason=reason,
        )

    def _groups(self, rows: list[IntakeRow]) -> list[list[IntakeRow]]:
        grouped: dict[tuple[str, str, str], list[IntakeRow]] = {}
        for row in rows:
            key = (row.session, row.chat_id, row.burst_key or row.message_id)
            grouped.setdefault(key, []).append(row)
        groups = list(grouped.values())
        groups.sort(key=lambda group: group[0].received_at)
        return groups

    async def _notify_operator(
        self,
        delayed: int,
        rows: list[IntakeRow],
        moment: float,
    ) -> None:
        """Queue and deliver at most one durable notice for this pass."""
        required_rows = self.ledger.unnotified_operator(_REQUIRED_NOTICE_LIMIT)
        pending = self.intents.unnotified_operator()
        if not delayed and not required_rows and not pending:
            return
        parts: list[str] = []
        if delayed:
            parts.append(f"{delayed} message(s) delayed by the outage")
        if required_rows:
            parts.append(f"{len(required_rows)} message(s) need operator attention")
        unknown = [intent for intent in pending if intent.state == INTENT_UNKNOWN]
        failed = [intent for intent in pending if intent.state == INTENT_FAILED]
        if unknown:
            drafts = "; ".join(self._draft(intent) for intent in unknown[:3])
            parts.append(
                f"{len(unknown)} send(s) had an unknown outcome and were "
                + f"not retried: {drafts}"
            )
        if failed:
            drafts = "; ".join(self._draft(intent) for intent in failed[:3])
            parts.append(f"{len(failed)} send(s) could not be delivered: {drafts}")
        text = "wahabot: " + "; ".join(parts) + "."
        destination = self._notice_destination(required_rows or rows)
        notice = self.intents.prepare_notice(
            session=self.settings.session,
            chat_id=destination,
            text=text,
            source_intent_ids=[intent.id for intent in pending],
            source_intake_ids=[row.message_id for row in required_rows],
        )
        if not destination:
            # The notice is durable and waits for the operator JID.
            return
        async with chat_lock(self.settings.session, destination):
            await asyncio.to_thread(
                self.transport.recover_intent,
                notice,
                moment,
                recover_sending=False,
            )

    def _draft(self, intent: DeliveryIntent) -> str:
        payload = intent.payload
        if "text" in payload:
            return str(payload["text"])[:80]
        file = payload.get("file")
        if isinstance(file, dict):
            reference = cast(dict[str, object], file)
            return str(reference.get("url") or reference.get("id") or "attachment")
        return intent.kind

    def _notice_destination(self, rows: list[IntakeRow]) -> str:
        if state.operator_jid:
            return state.operator_jid
        for row in rows:
            me: dict[str, object] = dict(row.event.me or {})
            for key in ("id", "lid"):
                value = me.get(key)
                if isinstance(value, str) and value:
                    return value
        return ""

    def unanswered(self) -> int:
        """The queue depth the heartbeat exposes."""
        stats = self.ledger.stats()
        return int(stats.get("unanswered", 0))
