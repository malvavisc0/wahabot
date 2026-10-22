import asyncio
import time
from typing import Any

from loguru import logger

from wahabot.ai.albums import (
    AlbumBuffer,
)
from wahabot.ai.context import (
    handle_message,
)
from wahabot.ai.history import ContextBudgetError
from wahabot.ai.messages import (
    image_media,
)
from wahabot.ai.observability import chat_trace_attributes
from wahabot.ai.serials import serial_store
from wahabot.ai.tools.whatsapp import RunTarget
from wahabot.ai.turn_records import (
    RecordEntry,
    serial_tokens_for,
)
from wahabot.ai.vision import caption_images
from wahabot.core.runs import (
    chat_lock,
    context_for,
    persist_memory,
)
from wahabot.handlers.base import HandlerBase
from wahabot.handlers.dedup import forget_seen
from wahabot.handlers.media import album_message_id, download_image
from wahabot.handlers.replies import finish_agent_reply
from wahabot.status import llm_failure_class


class AlbumMixin(HandlerBase):
    async def run_album(self, buffer: AlbumBuffer) -> None:
        """Run the agent once over a completed album, all images attached.

        The container event drives the turn (sender tag, gating already
        done at arrival); each buffered image contributes its bytes.
        Runs under the same per-chat lock as single messages so the
        chat's memory and timeline stay consistent.

        Fire-and-forget from the album buffer, so failures must be
        caught here or they would die silently in an unretrieved task:
        the exception is logged with traceback and every buffered
        message's seen marker is dropped so WAHA's redelivery can
        retry the album, mirroring the single-message path.
        """
        chat_id = str(buffer.container.payload.get("from", ""))
        turn_ids = [
            album_message_id(buffer.container),
            *(str(image.payload.get("id", "")) for image in buffer.images),
        ]
        started = time.monotonic()
        target: RunTarget | None = None
        outcome = "failed"
        failure_class = ""
        try:
            outcome, target = await self.deliver_album_reply(buffer, started)
        except ContextBudgetError:
            # The album's images can never fit the budget; keep every
            # seen marker or redelivery retries the album forever.
            logger.warning(
                "Album in {chat_id} exceeds the context budget ({limit} tokens)",
                chat_id=chat_id,
                limit=self.settings.memory_token_limit,
            )
            failure_class = "budget"
        except Exception as exc:
            for image_event in buffer.images:
                forget_seen(str(image_event.payload.get("id", "")))
            forget_seen(str(buffer.container.payload.get("id", "")))
            failure_class = llm_failure_class(exc)
            logger.exception("Failed to handle album in {chat_id}", chat_id=chat_id)
        finally:
            self.finish_run(
                buffer.container.session,
                chat_id,
                turn_ids,
                target,
                outcome,
                started,
                failure_class,
            )

    async def download_album_images(self, buffer: AlbumBuffer) -> list[dict[str, Any]]:
        """An album's images, downloaded and capped for the vision model.

        The vision cap applies at collection time: images past
        ``vision_max_images`` can never ride the prompt (handle_message
        trims to the same cap), so downloading them would be pure
        waste.
        """
        downloaded: list[dict[str, Any]] = []
        for image_event in buffer.images:
            if (
                self.settings.vision_max_images > 0
                and len(downloaded) >= self.settings.vision_max_images
            ):
                break
            media = image_media(image_event)
            if media is None:
                continue
            image = await asyncio.to_thread(
                download_image,
                self.waha,
                media,
                str(image_event.payload.get("id", "")),
                self.settings.max_image_bytes,
            )
            if image is not None:
                downloaded.append(image)
        return downloaded

    async def deliver_album_reply(
        self, buffer: AlbumBuffer, started: float
    ) -> tuple[str, RunTarget | None]:
        """Download an album's images, run the agent once, send its reply."""
        event = buffer.container
        chat_id = str(event.payload.get("from", ""))
        turn_ids = [
            album_message_id(event),
            *(str(image.payload.get("id", "")) for image in buffer.images),
        ]
        downloaded = await self.download_album_images(buffer)
        if not downloaded:
            logger.debug("Album in {chat_id} yielded no usable images", chat_id=chat_id)
            return "empty", None
        # Caption before the lock: the vision call must not extend the
        # serialized agent-run section.
        captions = await caption_images(
            self.agent.llm, downloaded, semaphore=self.agent.llm_semaphore
        )
        for image, caption in zip(downloaded, captions, strict=True):
            image["caption"] = caption
        current = self.config_reloader.current_config()
        addressed = any(
            self.group_message_addressed(member, current)
            for member in (buffer.container, *buffer.images)
        )
        gate_first = not addressed and current.group_participation == "judicious"
        entry = RecordEntry(
            event=event,
            body=str(event.payload.get("body", "")).strip(),
            images=downloaded,
            omitted=max(0, len(buffer.images) - len(downloaded)),
            album=True,
            addressed=addressed,
        )
        async with chat_lock(event.session, chat_id):
            if self.intake.handled(turn_ids):
                return "superseded", None
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
                reply_to=album_message_id(event),
            ):
                reply, target = await handle_message(
                    event,
                    self.agent,
                    ctx=ctx,
                    images=downloaded,
                    settings=self.settings,
                    waha=self.waha,
                    bot_name=current.bot_name,
                    bot_mention_regex=current.bot_mention_regex,
                    addressed=addressed,
                    gate_first=gate_first,
                    record_entries=[entry],
                    intake=self.intake,
                    turn_ids=turn_ids,
                    turn_started_at=started,
                )
            await persist_memory(self.settings, event.session, chat_id, ctx)
        delivered = bool(target.sent or target.reacted)
        outcome = await finish_agent_reply(
            self.waha,
            event.session,
            chat_id,
            album_message_id(event),
            reply,
            delivered,
            self.settings,
            prefix=target.prefix,
            emoji_reaction=not delivered,
            target=target,
            message_ids=turn_ids,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        # Post-reply memory jobs outside the chat lock: the recap merge
        # and the embedding drain must never hold the chat's next run.
        await self.agent.merge_learned()
        return outcome, target
