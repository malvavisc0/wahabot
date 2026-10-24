import asyncio
import time
from typing import Any, override

from loguru import logger

from wahabot.ai.context import (
    handle_message,
)
from wahabot.ai.history import ContextBudgetError
from wahabot.ai.messages import (
    image_media,
    jid_string,
    replies_to_bot,
)
from wahabot.ai.observability import chat_trace_attributes
from wahabot.ai.serials import serial_store
from wahabot.ai.tools.whatsapp import RunTarget
from wahabot.ai.turn_records import (
    assign_event_serial,
    serial_tokens_for,
)
from wahabot.core.access import SessionConfig
from wahabot.core.documents.limits import DOCUMENT_MARKER_KEY
from wahabot.core.models import WahaEvent
from wahabot.core.runs import (
    chat_lock,
    context_for,
    persist_memory,
)
from wahabot.core.teams.groups import surface_for
from wahabot.handlers.base import HandlerBase
from wahabot.handlers.dedup import forget_seen
from wahabot.handlers.media import prepare_document_event, video_frames
from wahabot.handlers.replies import finish_agent_reply_with_intake
from wahabot.status import (
    classify_llm_failure,
    mark_llm_recovered,
)


class SingleMixin(HandlerBase):
    @override
    def group_message_addressed(self, event: WahaEvent, current: SessionConfig) -> bool:
        """True for DMs and for group messages that mention/reply to the bot."""
        if not str(event.payload.get("from", "")).endswith("@g.us"):
            return True
        return replies_to_bot(event, current.bot_name, current.bot_mention_regex)

    async def deliver_single_reply(
        self,
        event: WahaEvent,
        config: SessionConfig,
        image: dict[str, Any] | None,
        video: dict[str, Any] | None,
        url_video: dict[str, Any] | None,
    ) -> None:
        """Run and persist one turn under its chat lock, then deliver its reply."""
        message_id = str(event.payload.get("id", ""))
        # jid_string, not the raw payload value: a `from` carrying a
        # linked-device suffix must key the same memory (and lock) as
        # the operator fold's device-stripped chat id, or one DM splits
        # in two.
        chat_id = jid_string(event.payload["from"])
        addressed = self.group_message_addressed(event, config)
        gate_first = not addressed and config.group_participation == "judicious"
        turn_ids = [message_id]
        ticket_id = self.ticket_active_id(str(event.payload.get("from", "") or chat_id))
        started = time.monotonic()
        target: RunTarget | None = None
        outcome = "failed"
        failure_class = ""
        try:
            image = await self.download_captioned_image(image, message_id)
            async with chat_lock(event.session, chat_id):
                if await self.turn_suppressed(event.session, chat_id, turn_ids):
                    outcome = "superseded"
                    return
                # Own the turn before it is answered: while this run is
                # live, the row is a claim (not pending), so the
                # recovery tick never pulls it back as due work.
                owner = self.claim_turn(event.session, chat_id, turn_ids)
                # Runs in the same chat serialize on its memory and
                # timeline; runs in different chats proceed in parallel
                # (each binds its own run target inside the workflow).
                ctx = await context_for(event.session, chat_id, self.agent, self.settings)
                with chat_trace_attributes(
                    chat_id,
                    message_ids=turn_ids,
                    serial_tokens=serial_tokens_for(
                        serial_store(str(self.settings.data_dir)),
                        event.session,
                        chat_id,
                        turn_ids,
                    ),
                    reply_to=message_id,
                ):
                    reply, target = await handle_message(
                        event,
                        self.agent,
                        ctx=ctx,
                        image=image,
                        images=video_frames(video, url_video),
                        settings=self.settings,
                        waha=self.waha,
                        bot_name=config.bot_name,
                        bot_mention_regex=config.bot_mention_regex,
                        addressed=addressed,
                        gate_first=gate_first,
                        intake=self.intake,
                        turn_ids=turn_ids,
                        turn_started_at=started,
                        owner=owner,
                        ticket_store=self.ticket_store,
                        ticket_id=ticket_id,
                        surface=surface_for(config, str(event.payload.get("from", ""))),
                    )
                await persist_memory(self.settings, event.session, chat_id, ctx)
                delivered = bool(target.sent or target.reacted)
            # The provider answered: flip the LLM health flag back up
            # (once-per-transition notify) outside the chat lock, so
            # the notify send never serializes this chat's next run.
            await mark_llm_recovered(self.waha, event.session)
            outcome = await finish_agent_reply_with_intake(
                turn_ids,
                self.waha,
                event.session,
                chat_id,
                message_id,
                reply,
                delivered,
                self.settings,
                prefix=target.prefix,
                emoji_reaction=True,
                target=target,
                message_ids=turn_ids,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            self.finalize_ticket(target, outcome, reply)
            # Post-reply memory jobs outside the chat lock: the recap
            # merge and the embedding drain must never hold this
            # chat's next run.
            await self.agent.merge_learned()
        except Exception as exc:
            # Timeout/outage: notify and drop the seen marker so WAHA
            # retries. Budget: permanent, keep the marker. Anything else
            # is a bug and is redelivered.
            if isinstance(exc, ContextBudgetError):
                logger.warning(
                    "Message {id} in {chat_id} exceeds the budget ({limit} tokens)",
                    id=message_id,
                    chat_id=chat_id,
                    limit=self.settings.memory_token_limit,
                )
                failure_class = "budget"
                return
            failure_class = await classify_llm_failure(
                self.waha,
                event.session,
                exc,
                self.settings.llm_timeout,
                self.settings.run_timeout,
            )
            forget_seen(message_id)
            if failure_class != "bug":
                return
            logger.exception(
                "Failed to handle message {id} in {chat_id} {detail}",
                id=message_id,
                chat_id=chat_id,
                detail="(seen marker dropped; a WAHA redelivery will retry)",
            )
            return
        finally:
            self.finish_run(
                event.session,
                chat_id,
                turn_ids,
                target,
                outcome,
                started,
                failure_class,
            )

    @override
    async def reply_with_agent(self, event: WahaEvent) -> None:
        """Run the agent over an incoming message and send its reply via WAHA."""
        # Arrival order, not processing order: identity flushes and media
        # work can reorder two messages that reached the webhook together.
        arrived_at = time.monotonic()
        config = await self.accept_message(event)
        if config is None:
            return
        # Serial at ingest, before every gate: a message that never wakes a
        # run still gets its #m<n> so a later quote/reaction can anchor to it.
        store = serial_store(str(self.settings.data_dir))
        await asyncio.to_thread(assign_event_serial, store, event)
        handled, body = await self.handle_self_command(event, config)
        if handled:
            return
        operator_wakes = await self.prepare_chat_message(
            event, config, body, arrived_at=arrived_at
        )
        if operator_wakes is None:
            return
        image = image_media(event) if self.settings.vision else None
        if await self.skip_or_buffer_message(event, config, operator_wakes):
            return
        body = await self.transcribe_message(event, body)
        if str(event.payload.get("from", "")).endswith("@g.us"):
            await prepare_document_event(event, self.waha, self.settings)
        body, video, url_video = await self.prepare_message_videos(
            event, body, stamp_body=False
        )
        if await self.ticket_or_content_gate(event, body, image, video, url_video):
            return
        await self.deliver_single_reply(event, config, image, video, url_video)

    async def ticket_or_content_gate(
        self,
        event: WahaEvent,
        body: str | None,
        image: dict[str, Any] | None,
        video: dict[str, Any] | None,
        url_video: dict[str, Any] | None,
    ) -> bool:
        """Settle empty turns and ticket-handled turns before a run."""
        if body is None and image is None and video is None and url_video is None:
            logger.debug("Skipping media/album message {id}", id=event.payload.get("id"))
            # Nothing to answer, but the row must leave the durable queue or
            # recovery would retry it every tick forever — and the customer's
            # media-only turn still belongs on their ticket.
            if await self.ticket_gate_events([event]):
                return True
            if event.payload.get(DOCUMENT_MARKER_KEY):
                # A document with no caption is still a readable turn: the
                # marker/text were stamped by attachment persistence.
                return False
            self.settle_intake(event, "silent", "no readable content")
            return True
        return await self.ticket_gate_events([event])
