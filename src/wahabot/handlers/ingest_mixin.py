import asyncio
import time
from typing import override

import httpx
from loguru import logger

from wahabot.ai import bursts
from wahabot.ai.albums import AlbumBuffer, is_album_container
from wahabot.ai.bursts import BurstBuffer
from wahabot.ai.messages import (
    extract_text,
    is_group_addressed,
    is_operator_event,
    is_replyable,
    self_command_instruction,
)
from wahabot.ai.serials import serial_store
from wahabot.ai.turn_records import assign_event_serial, event_record
from wahabot.core.access import SessionConfig
from wahabot.core.echoes import is_self_echo
from wahabot.core.filters import chat_allowed, jid_alias_lookup
from wahabot.core.inbound import (
    self_chat_needs_transcript,
)
from wahabot.core.intake.store import IntakeRow
from wahabot.core.models import WahaEvent
from wahabot.core.presence import mark_seen
from wahabot.core.teams.groups import group_participation
from wahabot.core.tickets.router import RouteKind, is_self_chat_event, route_message
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


def routed_event(event: WahaEvent, instruction: str) -> WahaEvent:
    """A shallow copy whose body is the operator instruction for routing."""
    if instruction == str(event.payload.get("body", "")):
        return event
    routed = event.model_copy(deep=True)
    routed.payload["body"] = instruction
    return routed


class IngestMixin(HandlerBase):
    @override
    def settle_intake(self, event: WahaEvent, outcome: str, reason: str = "") -> None:
        """Settle this message's intake row without a run."""
        message_id = str(event.payload.get("id", ""))
        chat_id = str(event.payload.get("from", ""))
        if not message_id or not chat_id:
            return
        try:
            self.intake.finish(
                event.session,
                chat_id,
                [message_id],
                delivered=False,
                outcome=outcome,
                reason=reason,
            )
        except Exception:
            logger.exception("Intake settle failed for {id}", id=message_id)

    def intake_row(self, event: WahaEvent):
        """This event's row, if the webhook accepted it."""
        message_id = str(event.payload.get("id", ""))
        chat_id = str(event.payload.get("from", ""))
        if not message_id or not chat_id:
            return None
        return self.intake.row(event.session, chat_id, message_id)

    @override
    async def accept_message(self, event: WahaEvent) -> SessionConfig | None:
        """Check health, dedup and freshness before learning the sender's identity."""
        from wahabot.core.recovery.context import recovering

        message_id = str(event.payload.get("id", ""))
        if not message_id:
            logger.debug(
                "Skipping message without an id from {chat_id}",
                chat_id=event.payload.get("from"),
            )
            return
        chat_id = str(event.payload.get("from", ""))
        if self.duplicate_or_owned(event, chat_id, message_id):
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
        # Recovery owns a durable row: the live dedup and startup
        # freshness gates would otherwise drop exactly the messages the
        # outage left behind. Provenance gates below still apply.
        if not recovering.get() and self.stale_or_duplicate(event, message_id):
            return
        if not is_replyable(event):
            logger.debug(
                "Ignoring non-replyable message from {sender}",
                sender=event.payload.get("from"),
            )
            self.settle_intake(event, "silent", "not replyable")
            return
        # Off the event loop: a rate-limited identity flush writes disk.
        await asyncio.to_thread(learn_identity, event)
        return config

    def duplicate_or_owned(self, event: WahaEvent, chat_id: str, message_id: str) -> bool:
        """True when another run or outbound intent already owns this message."""
        if self.intake.handled(event.session, chat_id, [message_id]):
            logger.debug("Skipping already handled message {id}", id=message_id)
            return True
        if self.intake.claimed(event.session, chat_id, [message_id]):
            logger.debug("Skipping message {id} owned by another run", id=message_id)
            return True
        if self.intake.is_delivering(event.session, chat_id, [message_id]):
            logger.debug(
                "Skipping message {id} owned by an outbound intent", id=message_id
            )
            return True
        return False

    def stale_or_duplicate(self, event: WahaEvent, message_id: str) -> bool:
        """True when the live dedup/freshness gate must skip this event."""
        if seen_recently(message_id):
            logger.debug("Skipping duplicate event for message {id}", id=message_id)
            return True
        if not is_stale(event, self.started_at):
            return False
        logger.info(
            "Skipping stale message {id} (ts={ts})",
            id=message_id,
            ts=event.payload.get("timestamp"),
        )
        # A durable row belongs to recovery: settling it here would
        # discard exactly the backlog the outage left behind.
        if self.intake_row(event) is None:
            self.settle_intake(event, "silent", "stale")
        return True

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
        if await self.handle_self_ticket_route(event, body, config, message_id):
            return True, None
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
            reply = await run_command(
                command, self.agent, self.settings, self.waha, self.ticket_store
            )
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
        arrived_at: float | None = None,
    ) -> None:
        """Ledger an accepted message when the webhook fallback is absent."""
        try:
            self.intake.record(
                event.session,
                str(event.payload.get("from", "")),
                str(event.payload.get("id", "")),
                event_record(
                    serial_store(str(self.settings.data_dir)),
                    event,
                    self.settings.timezone,
                ),
                at=time.monotonic() if arrived_at is None else arrived_at,
                addressed=self.intake_addressed(event, config, operator_wakes),
                raw_json=event.model_dump_json(),
                sender=str(
                    event.payload.get("participant") or event.payload.get("from") or ""
                ),
                burst_key=(
                    str(event.payload.get("id", ""))
                    if is_album_container(event)
                    else None
                ),
            )
        except Exception:
            logger.exception("Intake record failed for {id}", id=event.payload.get("id"))

    def reply_window_days(self) -> int:
        """The chat's configured reply window in days."""
        return self.config_reloader.current_config().reply_window_days

    async def recover_rows(self, rows: list[IntakeRow], *, apology: bool) -> str:
        """Re-run durable rows through the normal gates and delivery paths."""
        from wahabot.core.recovery.context import APOLOGY, recovering
        from wahabot.core.recovery.context import apology as apology_var

        recovering_token = recovering.set(True)
        apology_token = apology_var.set(APOLOGY if apology else "")
        try:
            await self.recover_shape(rows)
        finally:
            apology_var.reset(apology_token)
            recovering.reset(recovering_token)
        return self.recovery_outcome(rows)

    async def recover_shape(self, rows: list[IntakeRow]) -> None:
        """Reassemble one album, burst or single message and run it."""
        album_row = next((row for row in rows if is_album_container(row.event)), None)
        if album_row is not None and len(rows) > 1:
            await self.run_album(self.recovered_album(rows, album_row.message_id))
            return
        if len(rows) > 1 or any(row.burst_key for row in rows):
            await self.run_burst(self.recovered_burst(rows))
            return
        await self.reply_with_agent(rows[0].event)

    def recovered_album(self, rows: list[IntakeRow], container_id: str) -> AlbumBuffer:
        """Rebuild an album buffer from its durable rows."""
        return AlbumBuffer(
            container_id=container_id,
            session=rows[0].session,
            chat_id=rows[0].chat_id,
            expected=len(rows) - 1,
            image_ids=[row.message_id for row in rows if row.message_id != container_id],
        )

    def recovered_burst(self, rows: list[IntakeRow]) -> BurstBuffer:
        """Rebuild a burst buffer from its durable rows."""
        sender = rows[0].sender or rows[0].chat_id
        return BurstBuffer(
            key=(rows[0].session, rows[0].chat_id, sender),
            sender=sender,
            message_ids=[row.message_id for row in rows],
        )

    def reloaded_rows(self, rows: list[IntakeRow]) -> list[IntakeRow]:
        """The same rows as they are stored now, dropping vanished ids."""
        return [
            row
            for row in (
                self.intake.row(row.session, row.chat_id, row.message_id) for row in rows
            )
            if row is not None
        ]

    def recovery_outcome(self, rows: list[IntakeRow]) -> str:
        """Classify the rows after their recovery run settled them."""
        after = self.reloaded_rows(rows)
        if any(row.outcome == "operator_required" for row in after):
            return "operator_required"
        if not after or not all(row.state == "done" for row in after):
            return "skipped"
        if all(row.outcome == "delivered" for row in after):
            return "delivered"
        return "answered"

    def intake_addressed(
        self,
        event: WahaEvent,
        config: SessionConfig,
        operator_wakes: bool | None,
    ) -> bool:
        """Whether this message would wake a run, for delivery guard ordering."""
        chat_id = str(event.payload.get("from", ""))
        return bool(operator_wakes) or is_group_addressed(
            event,
            bot_name=config.bot_name,
            bot_mention_regex=config.bot_mention_regex,
            participation=group_participation(config, chat_id),
        )

    async def record_webhook_intake(self, event: WahaEvent) -> None:
        """Persist the accepted message before dispatch; a failure is a 500."""
        config = self.config_reloader.current_config()
        operator_wakes = operator_wake(event, config)
        message_id = str(event.payload.get("id", ""))
        chat_id = str(event.payload.get("from", ""))
        if not message_id or not chat_id:
            return
        store = serial_store(str(self.settings.data_dir))
        try:
            await asyncio.to_thread(assign_event_serial, store, event)
        except Exception:
            logger.exception("Serial assignment failed for {id}", id=message_id)
        self.intake.record(
            event.session,
            chat_id,
            message_id,
            event_record(store, event, self.settings.timezone),
            at=time.monotonic(),
            addressed=self.intake_addressed(event, config, operator_wakes),
            raw_json=event.model_dump_json(),
            sender=str(
                event.payload.get("participant") or event.payload.get("from") or ""
            ),
            burst_key=message_id if is_album_container(event) else None,
        )

    @override
    async def prepare_chat_message(
        self,
        event: WahaEvent,
        config: SessionConfig,
        body: str | None,
        *,
        arrived_at: float | None = None,
    ) -> bool | None:
        """Apply access gates and fold or buffer turns that need no immediate run."""
        operator_wakes = operator_wake(event, config)
        if operator_wakes is None:
            self.settle_intake(event, "silent", "not a waking operator event")
            return None
        if not self.chat_access_allowed(event, config):
            self.settle_intake(event, "silent", "chat not allowed")
            return None
        self.pause_after_human_message(event, config)
        if await self.fold_operator_event(event, config, body, operator_wakes):
            return None
        if self.intake_row(event) is None:
            self.record_intake(
                event, config, operator_wakes=operator_wakes, arrived_at=arrived_at
            )
        await self.mark_seen_for(event, operator_wakes)
        self.buffer_album_window(event, operator_wakes)
        return operator_wakes

    async def handle_self_ticket_route(
        self, event: WahaEvent, body: str | None, config: SessionConfig, message_id: str
    ) -> bool:
        """Route a ticket command typed in the operator console."""
        if (
            self.ticket_store is None
            or not is_self_chat_event(event)
            or is_self_echo(message_id)
        ):
            return False
        instruction = (
            self_command_instruction(
                event,
                bot_name=config.bot_name,
                bot_mention_regex=config.bot_mention_regex,
            )
            or body
            or ""
        )
        route = route_message(
            self.ticket_store,
            routed_event(event, instruction),
            self.skill_service.store if self.skill_service is not None else None,
        )
        if route.kind == RouteKind.NONE:
            return False
        await self.handle_ticket_route(event, route, config)
        return True

    def chat_access_allowed(self, event: WahaEvent, config: SessionConfig) -> bool:
        """Whether the chat whitelist/blacklist lets this event through."""
        return chat_allowed(
            event,
            config.whitelist,
            config.blacklist,
            jid_aliases=jid_alias_lookup(event),
        )

    async def fold_operator_event(
        self,
        event: WahaEvent,
        config: SessionConfig,
        body: str | None,
        operator_wakes: bool,
    ) -> bool:
        """Fold a non-waking operator message; True when it was consumed."""
        if not is_operator_event(event) or operator_wakes:
            return False
        await fold_operator_message(event, self.agent, self.settings, body)
        self.settle_intake(event, "operator_owned", "operator typed")
        return True

    async def mark_seen_for(self, event: WahaEvent, operator_wakes: bool) -> None:
        """Send the read receipt for an inbound non-operator message."""
        if (
            self.settings.send_seen
            and not operator_wakes
            and not is_operator_event(event)
        ):
            await asyncio.to_thread(
                mark_seen, self.waha, event.session, str(event.payload["from"])
            )

    def buffer_album_window(self, event: WahaEvent, operator_wakes: bool) -> None:
        """Hold an album container or image until its flush completes."""
        if operator_wakes:
            return
        album_key = self.albums.buffer(event)
        if not album_key:
            return
        self.intake.set_burst_key(
            event.session,
            str(event.payload.get("from", "")),
            [str(event.payload.get("id", ""))],
            album_key,
        )

    @override
    async def skip_or_buffer_message(
        self, event: WahaEvent, config: SessionConfig, operator_wakes: bool
    ) -> bool:
        """Apply group addressing and queue eligible bursts."""
        if self.albums.consumed(event):
            return True
        chat_id = str(event.payload.get("from", ""))
        if not operator_wakes and not is_group_addressed(
            event,
            bot_name=config.bot_name,
            bot_mention_regex=config.bot_mention_regex,
            participation=group_participation(config, chat_id),
        ):
            logger.debug(
                "Ignoring unaddressed group message {id}", id=event.payload.get("id")
            )
            self.settle_intake(event, "silent", "unaddressed")
            return True
        if (
            self.settings.burst_enabled
            and not operator_wakes
            and bursts.is_bufferable(event)
        ):
            # Hold for the sender's window; only messages that passed
            # every gate reach here, and albums were handled above.
            message_id = str(event.payload.get("id", ""))
            if self.bursts.add(message_id, event):
                key = bursts.burst_key(event)
                window_ids = self.bursts.ids(key)
                self.intake.set_burst_key(
                    event.session, key[1], window_ids, window_ids[0]
                )
                return True
        return False
