import asyncio
import time
from typing import override

import httpx
from loguru import logger

from wahabot.ai import bursts
from wahabot.ai.messages import (
    extract_text,
    is_group_addressed,
    is_operator_event,
    is_replyable,
    self_command_instruction,
)
from wahabot.ai.serials import serial_store
from wahabot.ai.turn_records import event_record
from wahabot.core.access import SessionConfig
from wahabot.core.echoes import is_self_echo
from wahabot.core.filters import chat_allowed, jid_alias_lookup
from wahabot.core.inbound import (
    buffer_album_message,
    self_chat_needs_transcript,
)
from wahabot.core.models import WahaEvent
from wahabot.core.presence import mark_seen
from wahabot.handlers.base import HandlerBase
from wahabot.handlers.dedup import forget_seen, is_stale, seen_recently
from wahabot.handlers.memory import (
    fold_operator_message,
    learn_identity,
    operator_wake,
)
from wahabot.handlers.replies import journal_send_failure, send_self_reply
from wahabot.status import (
    llm_call_timed_out,
    probe_session_recovery,
    session_healthy,
)


class IngestMixin(HandlerBase):
    @override
    async def accept_message(self, event: WahaEvent) -> SessionConfig | None:
        """Check health, dedup and freshness before learning the sender's identity."""
        message_id = str(event.payload.get("id", ""))
        if not message_id:
            logger.debug(
                "Skipping message without an id from {chat_id}",
                chat_id=event.payload.get("from"),
            )
            return
        if not session_healthy():
            # Before seen_recently: a muted message must not be marked
            # seen, so WAHA's redelivery retries it after recovery.
            # The delivery itself is evidence the session may be back
            # (a hung WAHA emits nothing) — one cheap probe against
            # GET /api/sessions decides; the poller arms only if the
            # status event never arrives (register_session_status_handler).
            logger.info(
                "Muting message {id} while WAHA session is not WORKING", id=message_id
            )
            await probe_session_recovery(self.waha, event.session)
            return
        # Hot-path config: picks up whitelist/prompt/mode edits per event.
        config = self.config_reloader.current_config()
        if seen_recently(message_id):
            logger.debug("Skipping duplicate event for message {id}", id=message_id)
            return
        if is_stale(event, self.started_at):
            logger.info(
                "Skipping stale message {id} (ts={ts})",
                id=message_id,
                ts=event.payload.get("timestamp"),
            )
            return
        if not is_replyable(event):
            logger.debug(
                "Ignoring non-replyable message from {sender}",
                sender=event.payload.get("from"),
            )
            return
        # Off the event loop: a rate-limited identity flush writes disk.
        await asyncio.to_thread(learn_identity, event)
        return config

    @override
    async def handle_self_command(
        self, event: WahaEvent, config: SessionConfig
    ) -> tuple[bool, str | None]:
        """Run a self-chat command before chat gates, preserving retry and echo rules."""
        message_id = str(event.payload.get("id", ""))
        body = extract_text(event)
        if self_chat_needs_transcript(event, body, self.settings):
            # The self-chat is the operator console: a voice note there
            # can only come from the operator's own devices, so it is
            # always worth transcribing early — it may be a spoken
            # command ("kai do x"). Everywhere else transcription stays
            # behind the chat gates. The transcript becomes the body;
            # the mention check below runs on it like typed text.
            body = await self.transcribe_message(event, body)
        instruction = (
            None
            if is_self_echo(message_id)
            else self_command_instruction(
                event,
                bot_name=config.bot_name,
                bot_mention_regex=config.bot_mention_regex,
            )
        )
        if not instruction:
            return False, body
        # Commands share the operator context and bypass chat gates;
        # send_self_reply remembers the reply id to suppress its echo.
        from wahabot.commands import build_command_event, run_command

        command = WahaEvent.model_validate(
            build_command_event(event.session, instruction)
        )
        command.payload["reply_chat_id"] = str(event.payload.get("from", ""))
        command.payload["reply_to"] = message_id
        try:
            reply = await run_command(command, self.agent, self.settings, self.waha)
        except Exception as exc:
            # Retry failed runs, even if a tool delivered before the crash;
            # the delivery latch is per-run, not a durable command receipt.
            # A permanent ContextBudgetError never reaches here: the
            # command path reports it as the reply (journaled, delivered
            # back to this self-chat) instead of raising.
            forget_seen(message_id)
            if llm_call_timed_out(exc):
                detail = (
                    f"LLM call timed out after {self.settings.llm_timeout}s on"
                    f" self-chat command {message_id} ({self.settings.llm_model} via"
                    f" {self.settings.llm_api_base}); dropped for redelivery"
                )
                logger.warning("{detail}", detail=detail)
            else:
                logger.exception("Failed to handle self-chat command {id}", id=message_id)
            return True, body
        try:
            await send_self_reply(self.waha, command, reply)
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            # The command completed: keep its seen marker even when only
            # the final reply failed, or redelivery duplicates its actions.
            journal_send_failure(
                self.settings,
                event.session,
                str(command.payload.get("reply_chat_id", "")),
                "send_text",
                message_id,
                exc,
            )
        except Exception:
            logger.exception(
                "Failed to deliver self-chat reply for command {id}", id=message_id
            )
        return True, body

    def record_intake(
        self,
        event: WahaEvent,
        config: SessionConfig,
        *,
        operator_wakes: bool | None,
    ) -> None:
        """Ledger the accepted message, fail-soft.

        Only a message that would wake a run may supersede one:
        unaddressed group chatter must not stall an answer.
        """
        try:
            addressed = bool(operator_wakes) or is_group_addressed(
                event,
                bot_name=config.bot_name,
                bot_mention_regex=config.bot_mention_regex,
                participation=config.group_participation,
            )
            self.intake.record(
                event.session,
                str(event.payload.get("from", "")),
                str(event.payload.get("id", "")),
                event_record(
                    serial_store(str(self.settings.data_dir)),
                    event,
                    self.settings.timezone,
                ),
                at=time.monotonic(),
                addressed=addressed,
            )
        except Exception:
            # The ledger is turn-level memory, not audit: an unexpected
            # payload must not cost the chat its reply.
            logger.exception("Intake record failed for {id}", id=event.payload.get("id"))

    @override
    async def prepare_chat_message(
        self, event: WahaEvent, config: SessionConfig, body: str | None
    ) -> bool | None:
        """Apply access gates and fold or buffer turns that need no immediate run."""
        operator_wakes = operator_wake(event, config)
        if operator_wakes is None:
            return
        if not chat_allowed(
            event,
            config.whitelist,
            config.blacklist,
            jid_aliases=jid_alias_lookup(event),
        ):
            return
        self.record_intake(event, config, operator_wakes=operator_wakes)
        if (
            self.settings.send_seen
            and not operator_wakes
            and not is_operator_event(event)
        ):
            # The read receipt answers *their* message, not the bot's
            # reply. It goes out before any participation decision and
            # before burst/album buffering, so holding or ignoring a
            # message never withholds it.
            await asyncio.to_thread(
                mark_seen, self.waha, event.session, str(event.payload["from"])
            )
        if is_operator_event(event) and not operator_wakes:
            await fold_operator_message(event, self.agent, self.settings, body)
            return
        if buffer_album_message(event, operator_wakes):
            return
        return operator_wakes

    @override
    async def skip_or_buffer_message(
        self, event: WahaEvent, config: SessionConfig, operator_wakes: bool
    ) -> bool:
        """Apply group addressing and queue eligible bursts."""
        if not operator_wakes and not is_group_addressed(
            event,
            bot_name=config.bot_name,
            bot_mention_regex=config.bot_mention_regex,
            participation=config.group_participation,
        ):
            logger.debug(
                "Ignoring unaddressed group message {id}", id=event.payload.get("id")
            )
            return True
        if (
            self.settings.burst_enabled
            and not operator_wakes
            and bursts.is_bufferable(event)
        ):
            # Hold for the sender's window; only messages that passed
            # every gate reach here, and albums were handled above.
            bursts.add_message(event)
            return True
        return False
