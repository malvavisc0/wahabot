"""State and method contract shared by the handler mixins."""

# pyright: reportUninitializedInstanceVariable=false

import asyncio
import time
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
from wahabot.core.audit import save_run_outcome
from wahabot.core.delivery.intents import IntentStore
from wahabot.core.models import WahaEvent
from wahabot.core.transport import Transport
from wahabot.settings import Settings

#: How often a queued run re-checks another run's claim on its turn.
_CLAIM_POLL_S = 0.1


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

        A claim belongs to a run that already released the chat lock and
        settles or releases it without the lock, so a bounded wait
        replaces the race where both runs answer the newest turn once
        each. A claim never outlives its run, so the wait is bounded by
        the run timeout; past it the turn proceeds as unclaimed.
        """
        deadline = time.monotonic() + self.settings.run_timeout
        while self.intake.claimed(session, chat_id, turn_ids):
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(_CLAIM_POLL_S)
        return self.intake.handled(session, chat_id, turn_ids, addressed_only=True)

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
        sent = bool(target and (target.sent or target.reacted))
        delivered = outcome in ("reply", "delivered_via_tool", "reaction") or sent
        try:
            if delivered:
                self.intake.finish_turn(
                    session, chat_id, turn_ids, claimed, delivered=True
                )
            elif outcome in ("silence", "empty", "superseded"):
                self.intake.finish_turn(
                    session,
                    chat_id,
                    turn_ids,
                    claimed,
                    delivered=False,
                    outcome="silent",
                    reason=outcome,
                )
            else:
                self.intake.fail(
                    session,
                    chat_id,
                    [*turn_ids, *claimed],
                    error=outcome,
                    next_attempt_at=time.time() + 60,
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
