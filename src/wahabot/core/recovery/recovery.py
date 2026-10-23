"""Startup and periodic recovery across durable intake and outbound rows."""

import asyncio
import time
from collections.abc import Callable
from typing import Protocol, cast

from loguru import logger

from wahabot.core.delivery.intents import DeliveryIntent, IntentStore
from wahabot.core.intake.store import (
    FAILED,
    OPERATOR_REQUIRED,
    PENDING,
    IntakeRow,
)
from wahabot.core.transport.waha import WahaTransport
from wahabot.settings import Settings
from wahabot.status.state import state


class RecoveryRunner(Protocol):
    """The handler operations recovery drives."""

    async def recover_rows(self, rows: list[IntakeRow], *, apology: bool) -> str: ...

    def reply_window_days(self) -> int: ...


class RecoveryLedger(Protocol):
    """The durable intake operations recovery uses."""

    def due(self, now: float, limit: int) -> list[IntakeRow]: ...

    def stale_processing(self, now: float, limit: int) -> list[IntakeRow]: ...

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

    def purge(self, retention_days: int) -> int: ...

    def stats(self) -> dict[str, int]: ...


class Recovery:
    """Recover accepted work after a restart and on a slow periodic tick."""

    def __init__(
        self,
        settings: Settings,
        ledger: RecoveryLedger,
        transport: WahaTransport,
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
        """Recover after a restart, before the first new event."""
        self.ledger.purge(self.settings.intake_retention_days)
        await self.tick(startup=True)

    async def tick(self, *, startup: bool = False, now: float | None = None) -> None:
        """Run one bounded recovery pass; callable directly by tests."""
        moment = self.clock() if now is None else now
        await asyncio.to_thread(
            self.transport.recover_intents, moment, recover_sending=startup
        )
        self._reset_stale(moment)
        self._flush_operator_notices()
        await self._notify_unrecorded_sends()
        due = self.ledger.due(moment, limit=20)
        if not due:
            return
        delayed = 0
        required = 0
        for group in self._groups(due):
            outcome = await self._recover_group(group, moment)
            if outcome == "delayed":
                delayed += len(group)
            elif outcome == "operator_required":
                required += len(group)
        if delayed or required:
            await self._notify_operator(delayed, required, due)

    def _reset_stale(self, now: float) -> None:
        for row in self.ledger.stale_processing(now, limit=50):
            self.ledger.fail(
                row.session,
                row.chat_id,
                [row.message_id],
                error="stale processing claim",
                next_attempt_at=now,
            )
            if row.attempts + 1 >= self.settings.intake_max_attempts:
                self.ledger.finish(
                    row.session,
                    row.chat_id,
                    [row.message_id],
                    delivered=False,
                    outcome=OPERATOR_REQUIRED,
                    reason="retry budget exhausted",
                )

    def _flush_operator_notices(self) -> None:
        destination = state.operator_jid
        if not destination:
            return
        for intent in self.intents.pending_notices():
            self.intents.set_chat(intent.id, destination)

    async def _recover_group(self, rows: list[IntakeRow], now: float) -> str:
        fresh = [row for row in rows if row.state in (PENDING, FAILED)]
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

    async def _notify_unrecorded_sends(self) -> None:
        pending = self.intents.unnotified_operator()
        if not pending:
            return
        destination = state.operator_jid
        drafts = "; ".join(self._draft(intent) for intent in pending[:3])
        text = (
            f"wahabot: {len(pending)} send(s) had an unknown outcome and were "
            f"not retried: {drafts}"
        )
        if not destination:
            self.intents.prepare(
                session=self.settings.session,
                chat_id="",
                kind="operator_notice",
                payload={"text": text},
            )
            return
        try:
            await asyncio.to_thread(
                self.transport.send_text, self.settings.session, destination, text
            )
        except Exception:
            logger.exception("Operator notice for uncertain sends failed")
            return
        self.intents.mark_notified([intent.id for intent in pending])

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

    async def _notify_operator(
        self, delayed: int, required: int, rows: list[IntakeRow]
    ) -> None:
        parts: list[str] = []
        if delayed:
            parts.append(f"{delayed} message(s) delayed by the outage")
        if required:
            parts.append(f"{required} message(s) need operator attention")
        text = "wahabot: " + "; ".join(parts) + "."
        destination = self._notice_destination(rows)
        if not destination:
            self.intents.prepare(
                session=self.settings.session,
                chat_id="",
                kind="operator_notice",
                payload={"text": text},
            )
            return
        try:
            await asyncio.to_thread(
                self.transport.send_text, self.settings.session, destination, text
            )
        except Exception:
            logger.exception("Operator recovery notice failed")
            self.intents.prepare(
                session=self.settings.session,
                chat_id=destination,
                kind="operator_notice",
                payload={"text": text},
            )

    def unanswered(self) -> int:
        """The queue depth the heartbeat exposes."""
        stats = self.ledger.stats()
        return int(stats.get("unanswered", 0))
