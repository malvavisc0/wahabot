import asyncio
import time
from typing import Any, override

from loguru import logger

from wahabot.ai import bursts
from wahabot.ai.context import (
    handle_message,
)
from wahabot.ai.history import ContextBudgetError
from wahabot.ai.messages import (
    extract_text,
    image_media,
    message_kind,
)
from wahabot.ai.observability import chat_trace_attributes
from wahabot.ai.serials import serial_store
from wahabot.ai.tools.url_images import fetch_url_images, image_urls
from wahabot.ai.tools.whatsapp import RunTarget
from wahabot.ai.turn_records import (
    LINK_VIDEO_MARKER_KEY,
    TRANSCRIPT_KEY,
    VIDEO_MARKER_KEY,
    RecordEntry,
    serial_tokens_for,
)
from wahabot.ai.video import join_anchor
from wahabot.ai.vision import caption_images
from wahabot.core.access import SessionConfig
from wahabot.core.inbound import (
    can_prepare_video,
    url_video_enabled,
)
from wahabot.core.models import WahaEvent
from wahabot.core.runs import (
    chat_lock,
    context_for,
    persist_memory,
)
from wahabot.core.transcribe import transcribe_voice_note
from wahabot.handlers.base import HandlerBase
from wahabot.handlers.dedup import forget_seen
from wahabot.handlers.media import (
    download_image,
    prepare_url_video,
    prepare_video,
)
from wahabot.handlers.replies import finish_agent_reply_with_intake
from wahabot.status import (
    classify_llm_failure,
    mark_llm_recovered,
)


class BurstMixin(HandlerBase):
    @override
    async def run_burst(self, buffer: bursts.BurstBuffer) -> None:
        """Run the agent once over a completed burst, all messages joined.

        Fire-and-forget from the burst buffer, so failures must be
        caught here or they would die silently in an unretrieved task:
        the exception is logged with traceback and every buffered
        message's seen marker is dropped so WAHA's redelivery
        reprocesses them through the normal path — a burst never
        disappears, mirroring the single-message and album paths.
        """
        events = self.load_events(buffer.key[0], buffer.key[1], buffer.ids())
        turn_ids = [str(event.payload.get("id", "")) for event in events]
        started = time.monotonic()
        target: RunTarget | None = None
        outcome = "failed"
        failure_class = ""
        try:
            if not events:
                return
            outcome, target = await self.deliver_burst_reply(buffer, events, started)
        except Exception as exc:
            # Same failure policy as a single turn: a permanent budget
            # error keeps the seen markers, everything else drops them so
            # WAHA redelivers the whole burst.
            if isinstance(exc, ContextBudgetError):
                logger.warning(
                    "Burst in {chat_id} exceeds the context budget ({limit} tokens)",
                    chat_id=buffer.key[1],
                    limit=self.settings.memory_token_limit,
                )
                failure_class = "budget"
                return
            for burst_id in buffer.ids():
                forget_seen(burst_id)
            failure_class = await classify_llm_failure(
                self.waha,
                buffer.key[0],
                exc,
                self.settings.llm_timeout,
                self.settings.run_timeout,
            )
            if failure_class == "bug":
                logger.exception(
                    "Failed to handle burst in {chat_id}",
                    chat_id=buffer.key[1],
                )
        finally:
            self.finish_run(
                buffer.key[0],
                buffer.key[1],
                turn_ids,
                target,
                outcome,
                started,
                failure_class,
            )

    @override
    async def transcribe_message(self, event: WahaEvent, body: str | None) -> str | None:
        """Replace an audio turn's body with its available voice transcript."""
        if message_kind(event) == "audio" and self.settings.transcribe_url:
            transcript = await transcribe_voice_note(event, self.waha, self.settings)
            if transcript:
                event.payload[TRANSCRIPT_KEY] = transcript
                body = f"[voice note] {transcript}"
                event.payload["body"] = body
        return body

    @override
    async def download_captioned_image(
        self, media: dict[str, Any] | None, message_id: str
    ) -> dict[str, Any] | None:
        """Download and caption an image before taking the chat's run lock."""
        if media is None:
            return None
        image = await asyncio.to_thread(
            download_image, self.waha, media, message_id, self.settings.max_image_bytes
        )
        if image is not None:
            captions = await caption_images(
                self.agent.llm, [image], semaphore=self.agent.llm_semaphore
            )
            image["caption"] = captions[0]
        return image

    async def prepare_attached_video(
        self, event: WahaEvent, body: str | None, *, stamp_body: bool
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Prepare an attached video and stamp its marker when prose is used."""
        if not can_prepare_video(event, self.settings):
            return body, None
        video = await prepare_video(
            event, self.waha, self.settings, self.agent.llm, self.agent.llm_semaphore
        )
        if video is None:
            return body, None
        event.payload[VIDEO_MARKER_KEY] = video["marker"]
        if stamp_body:
            body = join_anchor(body or "", video["marker"])
            event.payload["body"] = body
        return body, video

    async def prepare_link_video(
        self,
        event: WahaEvent,
        body: str | None,
        *,
        excluded: bool,
        stamp_body: bool,
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Prepare a video linked in the body, unless an attached one won."""
        if excluded or not url_video_enabled(self.settings) or not body:
            return body, None
        url_video = await prepare_url_video(
            self.settings, self.agent.llm, self.agent.llm_semaphore, body
        )
        if url_video is None:
            return body, None
        event.payload[LINK_VIDEO_MARKER_KEY] = url_video["marker"]
        if stamp_body:
            body = join_anchor(body, url_video["marker"])
            event.payload["body"] = body
        return body, url_video

    @override
    async def prepare_message_videos(
        self,
        event: WahaEvent,
        body: str | None,
        *,
        burst: bool = False,
        stamp_body: bool = True,
    ) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None]:
        """Prepare attached or linked video and record its analysis."""
        body, video = await self.prepare_attached_video(
            event, body, stamp_body=stamp_body
        )
        # Bursts exclude attached videos even after a failed download;
        # single turns try body URLs whenever no attached video resolved.
        excluded = message_kind(event) in ("video", "ptv") if burst else video is not None
        body, url_video = await self.prepare_link_video(
            event, body, excluded=excluded, stamp_body=stamp_body
        )
        return body, video, url_video

    async def prepare_burst_image(
        self,
        burst_event: WahaEvent,
        images: list[dict[str, Any]],
        entry: RecordEntry,
    ) -> None:
        """Download and bind one buffered image, or mark it cap-omitted."""
        if (
            message_kind(burst_event) not in ("image", "sticker")
            or not self.settings.vision
        ):
            return
        # The vision cap applies at collection time: once the shared
        # list holds ``vision_max_images`` entries, a further image can
        # never ride the prompt, so downloading and captioning it would
        # be pure waste.
        if (
            self.settings.vision_max_images > 0
            and len(images) >= self.settings.vision_max_images
        ):
            entry.omitted += 1
            return
        image = await self.download_captioned_image(
            image_media(burst_event), str(burst_event.payload.get("id", ""))
        )
        if image is not None:
            images.append(image)
            entry.images.append(image)

    async def prepare_burst_url_images(
        self,
        body: str | None,
        images: list[dict[str, Any]],
        entry: RecordEntry,
    ) -> None:
        """Fetch image URLs sniffed from one buffered body, if vision is on.

        The vision cap applies here too: the shared list already holds
        ``vision_max_images`` entries once the direct images of earlier
        burst parts filled it, so fetching more would be pure waste.
        """
        if not self.settings.vision or not body:
            return
        if (
            self.settings.vision_max_images > 0
            and len(images) >= self.settings.vision_max_images
        ):
            return
        fetched = await asyncio.to_thread(
            fetch_url_images,
            self.settings,
            image_urls(body, self.settings.max_url_images),
        )
        if fetched:
            entry.images.extend(fetched)
            images.extend(fetched)

    async def prepare_burst_media(
        self,
        burst_event: WahaEvent,
        images: list[dict[str, Any]],
    ) -> tuple[str | None, RecordEntry]:
        """Prepare one buffered turn's text and its record entry."""
        await self.transcribe_message(burst_event, None)
        body = extract_text(burst_event)
        entry = RecordEntry(event=burst_event, body=body or "")
        await self.prepare_burst_image(burst_event, images, entry)
        prepared, video, url_video = await self.prepare_message_videos(
            burst_event, body, burst=True, stamp_body=False
        )
        entry.video = video
        if url_video is not None:
            entry.extra_payloads.append(str(url_video["marker"]))
        for source in (video, url_video):
            if source is not None:
                images.extend(source["frames"])
        await self.prepare_burst_url_images(prepared, images, entry)
        return prepared, entry

    async def merge_burst_parts(
        self,
        messages: list[WahaEvent],
    ) -> tuple[list[dict[str, Any]], list[RecordEntry]]:
        """Prepare every buffered message into images and record entries."""
        images: list[dict[str, Any]] = []
        entries: list[RecordEntry] = []
        for burst_event in messages:
            _, entry = await self.prepare_burst_media(burst_event, images)
            entries.append(entry)
        return images, self.usable_entries(entries)

    def usable_entries(self, entries: list[RecordEntry]) -> list[RecordEntry]:
        """Drop empty entries with no text, media, omission or transcript."""
        return [
            entry
            for entry in entries
            if entry.body
            or entry.images
            or entry.video
            or entry.omitted
            or str(entry.event.payload.get(TRANSCRIPT_KEY) or "")
        ]

    def burst_anchor(self, last: WahaEvent) -> WahaEvent:
        """The run's anchor event; a burst must not inherit one member's quote."""
        merged = last.model_copy(deep=True)
        merged.payload["id"] = str(last.payload.get("id", ""))
        merged.payload["body"] = ""
        merged.payload["hasMedia"] = False
        merged.payload.pop("replyTo", None)
        merged.payload.get("_data", {}).pop("quotedMsg", None)
        return merged

    async def merge_burst(
        self,
        buffer: bursts.BurstBuffer,
        messages: list[WahaEvent],
    ) -> tuple[WahaEvent, list[RecordEntry]] | None:
        """Merge a burst into one turn; None when it has no usable content.

        The turn is the ordered list of record entries, one per buffered
        message; the merged event is only the run's anchor, with no welded
        text and no inherited quote.
        """
        if not messages:
            return None
        _, entries = await self.merge_burst_parts(messages)
        if not entries:
            logger.debug(
                "Burst in {chat_id} yielded no usable content",
                chat_id=buffer.key[1],
            )
            return None
        return self.burst_anchor(messages[-1]), entries

    def mark_addressed_entry(
        self, entries: list[RecordEntry], current: SessionConfig
    ) -> bool:
        """Flag exactly the record that caused the turn's addressing verdict.

        A DM addresses every message, but the grammar carries one verdict
        per turn: the newest record wins. A group burst can carry more
        than one mention; the newest addressed record is the reason the
        turn woke the bot.
        """
        verdicts = [
            self.group_message_addressed(entry.event, current) for entry in entries
        ]
        if not any(verdicts):
            return False
        winner = len(verdicts) - 1 - verdicts[::-1].index(True)
        for index, entry in enumerate(entries):
            entry.addressed = index == winner
        return True

    async def run_burst_turn(
        self,
        buffer: bursts.BurstBuffer,
        merged: WahaEvent,
        entries: list[RecordEntry],
        *,
        turn_ids: list[str],
        started: float,
    ) -> tuple[str, RunTarget | None]:
        """Run one merged burst turn under the chat lock."""
        chat_id = buffer.key[1]
        async with chat_lock(merged.session, chat_id):
            if await self.turn_suppressed(merged.session, chat_id, turn_ids):
                return "", None
            ctx = await context_for(merged.session, chat_id, self.agent, self.settings)
            with chat_trace_attributes(
                chat_id,
                message_ids=turn_ids,
                serial_tokens=serial_tokens_for(
                    serial_store(str(self.settings.data_dir)),
                    merged.session,
                    chat_id,
                    turn_ids,
                ),
                reply_to=str(merged.payload.get("id", "")),
            ):
                current = self.config_reloader.current_config()
                addressed = self.mark_addressed_entry(entries, current)
                gate_first = not addressed and current.group_participation == "judicious"
                reply, target = await handle_message(
                    merged,
                    self.agent,
                    ctx=ctx,
                    settings=self.settings,
                    waha=self.waha,
                    bot_name=current.bot_name,
                    bot_mention_regex=current.bot_mention_regex,
                    addressed=addressed,
                    gate_first=gate_first,
                    record_entries=entries,
                    intake=self.intake,
                    turn_ids=turn_ids,
                    turn_started_at=started,
                )
            await persist_memory(self.settings, merged.session, chat_id, ctx)
        return reply, target

    async def deliver_burst_reply(
        self, buffer: bursts.BurstBuffer, messages: list[WahaEvent], started: float
    ) -> tuple[str, RunTarget | None]:
        """Prepare each buffered message's media, run the agent once.

        One ``handle_message`` call over a merged turn whose text joins
        every message's body. Media rides as image blocks; quote/reply
        targets the LAST buffered message (the natural reply anchor).
        """
        merged_burst = await self.merge_burst(buffer, messages)
        if merged_burst is None:
            return "empty", None
        merged, entries = merged_burst
        chat_id = buffer.key[1]
        message_id = str(merged.payload.get("id", ""))
        turn_ids = [str(event.payload.get("id", "")) for event in messages]
        reply, target = await self.run_burst_turn(
            buffer, merged, entries, turn_ids=turn_ids, started=started
        )
        if target is None:
            return "superseded", None
        # The provider answered: flip the LLM health flag back up
        # (once-per-transition notify) outside the chat lock — the
        # burst path shares the single-message path's contract.
        await mark_llm_recovered(self.waha, merged.session)
        outcome = await finish_agent_reply_with_intake(
            turn_ids,
            self.waha,
            merged.session,
            chat_id,
            message_id,
            reply,
            bool(target.sent or target.reacted),
            self.settings,
            prefix=target.prefix,
            emoji_reaction=True,
            target=target,
            message_ids=turn_ids,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        # Post-reply memory jobs outside the chat lock: the recap merge
        # and the embedding drain must never hold the chat's next run.
        await self.agent.merge_learned()
        return outcome, target
