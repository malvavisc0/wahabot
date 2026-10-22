"""State and method contract shared by the handler mixins."""

# pyright: reportUninitializedInstanceVariable=false

import time
from collections.abc import Sequence
from typing import Any

from loguru import logger

from wahabot.ai.intake import IntakeLedger
from wahabot.ai.tools.whatsapp import EscalationChannel, RunTarget
from wahabot.ai.triage import TriageGate
from wahabot.ai.workflow import FunctionCallingAgentWorkflow
from wahabot.core.access import SessionConfig, SessionConfigReloader
from wahabot.core.audit import save_run_outcome
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings


def _delivered_via(outcome: str, target: RunTarget | None) -> str:
    """The audit delivery path for an outcome."""
    if outcome == "reply":
        return "text"
    if outcome == "delivered_via_tool" or (
        target is not None and (target.sent or target.reacted)
    ):
        return (
            "reaction"
            if target is not None and target.reacted and not target.sent_id
            else "tool"
        )
    if outcome in ("silence", "empty"):
        return "silence"
    return ""


class HandlerBase:
    """Per-registration state shared by every handler mixin."""

    settings: Settings
    waha: WahaClient
    config: SessionConfig
    config_reloader: SessionConfigReloader
    escalation_channel: EscalationChannel
    triage: TriageGate | None
    intake: IntakeLedger
    started_at: float
    agent: FunctionCallingAgentWorkflow

    async def transcribe_message(self, *args: Any, **kwargs: Any) -> Any: ...
    async def download_captioned_image(self, *args: Any, **kwargs: Any) -> Any: ...
    async def prepare_message_videos(self, *args: Any, **kwargs: Any) -> Any: ...
    async def accept_message(self, *args: Any, **kwargs: Any) -> Any: ...
    async def handle_self_command(self, *args: Any, **kwargs: Any) -> Any: ...
    async def prepare_chat_message(self, *args: Any, **kwargs: Any) -> Any: ...
    async def skip_or_buffer_message(self, *args: Any, **kwargs: Any) -> Any: ...
    def group_message_addressed(self, *args: Any, **kwargs: Any) -> Any: ...

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
        delivered = outcome in ("reply", "delivered_via_tool") or sent
        try:
            self.intake.finish_turn(turn_ids, claimed, delivered=delivered)
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
            delivered_via=_delivered_via(outcome, target),
            failure_class=failure_class,
        )
