"""State and method contract shared by the handler mixins."""

# pyright: reportUninitializedInstanceVariable=false

import asyncio
import time
import uuid
from collections.abc import Sequence
from typing import Any

from loguru import logger

from wahabot.ai import albums, bursts
from wahabot.ai.intake import IntakeLedger
from wahabot.ai.tools.runcontext import delivery_path
from wahabot.ai.tools.whatsapp import EscalationChannel, RunTarget
from wahabot.ai.triage import TriageGate
from wahabot.ai.workflow import FunctionCallingAgentWorkflow
from wahabot.core.access import SessionConfig, SessionConfigReloader
from wahabot.core.answers.service import AnswerService
from wahabot.core.audit import save_run_outcome
from wahabot.core.delivery.intents import IntentStore
from wahabot.core.delivery.model import FAILED, PREPARED, SENDING, UNKNOWN
from wahabot.core.models import WahaEvent
from wahabot.core.scheduler import TicketScheduler
from wahabot.core.skills.service import SkillService
from wahabot.core.tickets.router import Route
from wahabot.core.tickets.store import TicketStore
from wahabot.core.tickets.translate import TicketTranslator
from wahabot.core.transport import Transport
from wahabot.settings import Settings

#: How often a queued run re-checks another run's claim on its turn.
_CLAIM_POLL_S = 0.1


def run_delivered(target: RunTarget | None, outcome: str) -> bool:
    """Whether a run has a confirmed delivery worth finalizing."""
    if target is None:
        return False
    return (
        outcome in ("reply", "delivered_via_tool")
        or bool(target.sent)
        or bool(target.reacted and target.ticket_update is not None)
    )


class HandlerBase:
    """Per-registration state shared by every handler mixin."""

    settings: Settings
    waha: Transport
    config: SessionConfig
    config_reloader: SessionConfigReloader
    escalation_channel: EscalationChannel
    triage: TriageGate | None
    intake: IntakeLedger
    intents: IntentStore
    ticket_store: TicketStore | None
    ticket_translator: TicketTranslator | None
    scheduler: TicketScheduler | None
    answer_service: AnswerService | None
    skill_service: SkillService | None
    bursts: bursts.Bursts
    albums: albums.Albums
    started_at: float
    agent: FunctionCallingAgentWorkflow

    async def transcribe_message(self, *args: Any, **kwargs: Any) -> Any: ...
    async def download_captioned_image(self, *args: Any, **kwargs: Any) -> Any: ...
    async def prepare_message_videos(self, *args: Any, **kwargs: Any) -> Any: ...
    async def accept_message(self, *args: Any, **kwargs: Any) -> Any: ...
    async def handle_self_command(self, *args: Any, **kwargs: Any) -> Any: ...
    async def prepare_chat_message(self, *args: Any, **kwargs: Any) -> Any: ...
    async def skip_or_buffer_message(self, *args: Any, **kwargs: Any) -> Any: ...
    async def reply_with_agent(self, *args: Any, **kwargs: Any) -> Any: ...
    async def run_burst(self, *args: Any, **kwargs: Any) -> Any: ...
    async def run_album(self, *args: Any, **kwargs: Any) -> Any: ...
    async def ticket_gate_events(self, events: list[WahaEvent]) -> bool: ...
    async def ticket_skill_route(self, event: WahaEvent, route: Route) -> bool: ...
    async def ticket_send_operator(self, event: WahaEvent, text: str) -> None: ...
    def ticket_active_id(self, chat_id: str) -> int: ...
    def finalize_ticket(
        self, target: RunTarget | None, outcome: str, reply: str = ""
    ) -> None: ...
    def pause_after_human_message(
        self, event: WahaEvent, config: SessionConfig
    ) -> None: ...
    async def handle_ticket_route(
        self, event: WahaEvent, route: Route, config: SessionConfig
    ) -> bool: ...
    def group_message_addressed(self, *args: Any, **kwargs: Any) -> Any: ...
    def settle_intake(self, event: WahaEvent, outcome: str, reason: str = "") -> None: ...

    def load_events(
        self, session: str, chat_id: str, message_ids: Sequence[str]
    ) -> list[WahaEvent]:
        """Reconstruct the not-yet-settled events of one intake window."""
        events: list[WahaEvent] = []
        for message_id in message_ids:
            row = self.intake.row(session, chat_id, message_id)
            if row is None or row.state == "done" or not row.raw_json:
                continue
            events.append(row.event)
        return events

    async def turn_suppressed(
        self, session: str, chat_id: str, turn_ids: Sequence[str]
    ) -> bool:
        """Whether another run already claimed or handled this turn.

        A claim belongs to a run that starts by claiming its own turn
        rows (``claim_turn``) or a later message it revalidated into
        (``refuse_newer``); the claim settles or releases without the
        chat lock, so a bounded wait replaces the race where both runs
        answer the newest turn once each. A claim never outlives its
        run, so the wait is bounded by the run timeout; past it the
        turn proceeds as unclaimed.
        """
        deadline = time.monotonic() + self.settings.run_timeout
        while self.intake.claimed(session, chat_id, turn_ids):
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(_CLAIM_POLL_S)
        if self.intake.is_delivering(session, chat_id, turn_ids):
            return True
        return self.intake.handled(session, chat_id, turn_ids, addressed_only=True)

    def claim_turn(self, session: str, chat_id: str, turn_ids: Sequence[str]) -> str:
        """Claim the turn's intake rows under a fresh owner.

        Called by every run path (single, burst, album, recovery
        re-run) right after ``turn_suppressed`` clears, still under the
        chat lock: the live run owns its rows until it settles, so the
        60-second recovery tick's due work (pending and failed rows
        only) can never re-answer a turn a live run is answering. Pass
        the returned owner back into ``handle_message`` (a durable run
        without it raises) so the target, the claims and the settle
        agree on the owner by construction.
        """
        owner = uuid.uuid4().hex
        self.intake.claim(session, chat_id, list(turn_ids), owner=owner)
        return owner

    def finish_run(
        self,
        session: str,
        chat_id: str,
        turn_ids: Sequence[str],
        target: RunTarget | None,
        outcome: str,
        started: float,
        failure_class: str = "",
    ) -> None:
        """Settle intake and write the one outcome row every run owes."""
        claimed = list(target.claimed_ids) if target is not None else []
        owner = target.owner if target is not None else ""
        sent = bool(target and (target.sent or target.reacted))
        delivered = outcome in ("reply", "delivered_via_tool", "reaction") or sent
        # A suppressed run never claimed these rows: another run or an
        # outbound intent owns them, so settling here would clobber its
        # claim. It still owes its audit outcome row.
        if outcome != "superseded" or target is not None:
            try:
                self.settle_run(
                    session, chat_id, turn_ids, claimed, delivered, outcome, owner
                )
            except Exception:
                logger.exception("Intake settle failed for {chat_id}", chat_id=chat_id)
        save_run_outcome(
            self.settings.data_dir,
            session,
            chat_id,
            outcome=outcome,
            duration_ms=int((time.monotonic() - started) * 1000),
            message_ids=[*turn_ids, *claimed],
            outbound_id=target.sent_id if target is not None else "",
            delivered_via=delivery_path(outcome, target),
            failure_class=failure_class,
        )

    def settle_run(
        self,
        session: str,
        chat_id: str,
        turn_ids: Sequence[str],
        claimed: list[str],
        delivered: bool,
        outcome: str,
        owner: str,
    ) -> None:
        ids = [*turn_ids, *claimed]
        outstanding = self.settle_sending(ids)
        if self.has_unknown(outstanding):
            self.finish_operator_required(session, chat_id, ids)
            return
        if self.retryable(outstanding):
            self.settle_retryable(
                session,
                chat_id,
                turn_ids,
                claimed,
                delivered,
                outcome,
                owner,
                ids,
            )
            return
        self.settle_plain(
            session, chat_id, turn_ids, claimed, delivered, outcome, owner, ids
        )

    def settle_sending(self, ids: list[str]) -> list[Any]:
        outstanding = self.intents.outstanding_for(ids)
        sending = [intent for intent in outstanding if intent.state == SENDING]
        for intent in sending:
            # The remote call returned or the process survived, but the
            # outcome was not durably recorded: treat it as unknown rather
            # than retrying it blindly.
            self.intents.unknown(intent.id, "send outcome was not recorded")
        return self.intents.outstanding_for(ids) if sending else outstanding

    def has_unknown(self, outstanding: list[Any]) -> bool:
        return any(
            intent.state == UNKNOWN
            or (intent.state == FAILED and intent.operator_required)
            for intent in outstanding
        )

    def retryable(self, outstanding: list[Any]) -> list[Any]:
        return [
            intent
            for intent in outstanding
            if intent.state in (PREPARED, SENDING)
            or (intent.state == FAILED and not intent.operator_required)
        ]

    def settle_retryable(
        self,
        session: str,
        chat_id: str,
        turn_ids: Sequence[str],
        claimed: list[str],
        delivered: bool,
        outcome: str,
        owner: str,
        ids: list[str],
    ) -> None:
        if delivered:
            self.intents.cancel_retryable_for(
                ids, "superseded by a confirmed send", chat_id=chat_id
            )
            self.finish_delivered(session, chat_id, turn_ids, claimed, owner)
            return
        if outcome in ("silence", "empty", "superseded"):
            self.intents.cancel_retryable_for(ids, outcome, chat_id=chat_id)
            self.finish_silent(session, chat_id, turn_ids, claimed, outcome, owner)
        # A failed run with a retryable intent leaves the rows in
        # ``delivering``; the intent owns the next attempt.

    def settle_plain(
        self,
        session: str,
        chat_id: str,
        turn_ids: Sequence[str],
        claimed: list[str],
        delivered: bool,
        outcome: str,
        owner: str,
        ids: list[str],
    ) -> None:
        if delivered:
            self.finish_delivered(session, chat_id, turn_ids, claimed, owner)
            return
        if outcome in ("silence", "empty", "superseded"):
            self.finish_silent(session, chat_id, turn_ids, claimed, outcome, owner)
            return
        if outcome == "operator_owned":
            self.finish_operator_owned(session, chat_id, turn_ids, claimed, owner)
            return
        self.intake.fail(
            session,
            chat_id,
            ids,
            error=outcome,
            next_attempt_at=time.time() + 60,
        )

    def finish_delivered(
        self,
        session: str,
        chat_id: str,
        turn_ids: Sequence[str],
        claimed: list[str],
        owner: str,
    ) -> None:
        self.intake.finish_turn(
            session,
            chat_id,
            turn_ids,
            claimed,
            delivered=True,
            owner=owner,
        )

    def finish_silent(
        self,
        session: str,
        chat_id: str,
        turn_ids: Sequence[str],
        claimed: list[str],
        outcome: str,
        owner: str,
    ) -> None:
        self.intake.finish_turn(
            session,
            chat_id,
            turn_ids,
            claimed,
            delivered=False,
            outcome="silent",
            reason=outcome,
            owner=owner,
        )

    def finish_operator_owned(
        self,
        session: str,
        chat_id: str,
        turn_ids: Sequence[str],
        claimed: list[str],
        owner: str,
    ) -> None:
        self.intake.finish_turn(
            session,
            chat_id,
            turn_ids,
            claimed,
            delivered=False,
            outcome="operator_owned",
            reason="held for the operator",
            owner=owner,
        )

    def finish_operator_required(
        self, session: str, chat_id: str, ids: list[str]
    ) -> None:
        self.intake.finish(
            session,
            chat_id,
            ids,
            delivered=False,
            outcome="operator_required",
            reason="send outcome unknown",
        )
