"""Message handler connecting the webhook to the LlamaIndex agent."""

import asyncio
import io
import time
from collections.abc import Callable
from typing import Any, cast

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Context
from loguru import logger
from PIL import Image

from wahabot.ai.albums import (
    AlbumBuffer,
    add_album_image,
    is_album_container,
    set_completion_handler,
    start_album,
)
from wahabot.ai.context import handle_message, render_system_prompt
from wahabot.ai.messages import (
    extract_text,
    image_media,
    is_group_addressed,
    is_replyable,
    message_kind,
)
from wahabot.ai.observability import chat_trace_attributes, enable_langfuse
from wahabot.ai.tools import build_default_tools
from wahabot.ai.workflow import FunctionCallingAgentWorkflow, build_agent
from wahabot.core.access import SessionConfigReloader, load_session_config
from wahabot.core.filters import chat_allowed, jid_alias_lookup
from wahabot.core.models import WahaEvent
from wahabot.core.transcribe import transcribe_voice_note
from wahabot.core.waha import MediaTooLargeError, WahaClient
from wahabot.settings import Settings
from wahabot.webhook import on_message

_seen_ids: dict[str, float] = {}
#: Messages older than this many seconds are stale backlog, not live turns.
_MAX_MESSAGE_AGE_S = 300
#: Dedup window for redelivered message ids. Must meet or exceed
#: ``_MAX_MESSAGE_AGE_S`` so the two guards overlap seamlessly: a
#: redelivery inside this window is dropped by the seen cache, past it
#: by the staleness check — between a shorter TTL and the age bound
#: (120 < age < 300) a redelivered message would slip through both
#: and be answered twice.
_SEE_TTL_S = _MAX_MESSAGE_AGE_S
#: Keep at most this many per-chat agent contexts; least recently used
#: chats are evicted (their conversation memory is dropped).
_MAX_CONTEXTS = 1000
contexts: dict[tuple[str, str], Context] = {}
_agent_lock = asyncio.Lock()


#: Seen-id cache hard cap; past it the oldest entries are evicted.
_MAX_SEEN_IDS = 10_000


def seen_recently(message_id: str) -> bool:
    """Return True if this message id was already handled in the last TTL.

    The id is marked seen on first sight, so a duplicate redelivery
    racing the in-flight run is also deduplicated. A run that later
    fails must call :func:`forget_seen` to allow a retry.

    The cache never wipes wholesale: at the cap the oldest entries are
    evicted one by one (the insertion order of ``dict`` is oldest
    first), so a redelivery stays deduplicated even while the cache
    turns over.
    """
    if not message_id:
        return False
    now = time.monotonic()
    if message_id in _seen_ids:
        if now - _seen_ids[message_id] < _SEE_TTL_S:
            return True
        del _seen_ids[message_id]
    while len(_seen_ids) >= _MAX_SEEN_IDS:
        del _seen_ids[next(iter(_seen_ids))]
    _seen_ids[message_id] = now
    return False


def forget_seen(message_id: str) -> None:
    """Drop a message id's seen marker so a redelivery is reprocessed."""
    _seen_ids.pop(message_id, None)


def context_for(
    session: str, chat_id: str, agent: FunctionCallingAgentWorkflow
) -> Context:
    """The per-chat agent context, evicting stale chats past the cap.

    Every touch moves the chat to the end (most recently used);
    inserts past the cap drop the oldest entry.
    """
    key = (session, chat_id)
    ctx = contexts.pop(key, None)
    if ctx is None:
        ctx = Context(agent)
    contexts[key] = ctx
    while len(contexts) > _MAX_CONTEXTS:
        oldest = next(iter(contexts))
        del contexts[oldest]
    return ctx


async def remember_own_message(
    event: WahaEvent,
    agent: FunctionCallingAgentWorkflow,
    body: str | None,
) -> None:
    """Fold a ``fromMe`` message into the chat's memory as an assistant turn.

    Messages sent from the bot account by its human operator (typing in
    the WhatsApp app) are the bot's own voice as far as the chat is
    concerned — storing them as assistant messages keeps the model's
    self-history coherent (it "said" them). Memory-only: no agent run,
    so the bot can never wake on its own output and loop on itself.
    """
    if not body:
        return
    chat_id = str(event.payload.get("from", ""))
    if not chat_id:
        return
    ctx = context_for(event.session, chat_id, agent)
    memory = await ctx.store.get("memory", default=None)
    if memory is None:
        from_defaults = cast(
            Callable[..., ChatMemoryBuffer], ChatMemoryBuffer.from_defaults
        )
        memory = from_defaults(token_limit=agent.memory_token_limit, llm=agent.llm)
    await memory.aput(ChatMessage(role=MessageRole.ASSISTANT, content=body))
    await ctx.store.set("memory", memory)
    logger.debug("Remembered own outbound message {id}", id=event.payload.get("id"))


def is_stale(event: WahaEvent, started_at: float) -> bool:
    """True when the message is replayed backlog rather than a live turn.

    WhatsApp redelivers undelivered messages when the WAHA session or the
    phone reconnects, and WAHA forwards them as fresh ``message`` events.
    Two guards: anything sent before this process started is definitionally
    backlog, and anything older than ``_MAX_MESSAGE_AGE_S`` is stale even
    mid-run (phone reconnect flush). Unknown timestamps pass — better one
    late reply than silence.
    """
    ts = event.payload.get("timestamp")
    if not isinstance(ts, (int, float)):
        return False
    return ts < started_at or time.time() - ts > _MAX_MESSAGE_AGE_S


def download_image(
    waha: WahaClient, media: dict[str, Any], message_id: str, max_bytes: int
) -> dict[str, Any] | None:
    """Download an image's bytes; None keeps the turn text-only on failure."""
    url = str(media.get("url", ""))
    if not url:
        return None
    try:
        data = waha.download_media(url, max_bytes=max_bytes)
    except MediaTooLargeError:
        logger.info(
            "Skipping image over {max_bytes} B in message {id}",
            max_bytes=max_bytes,
            id=message_id,
        )
        return None
    except Exception as exc:
        logger.warning(
            "Image download failed for message {id}: {exc}", id=message_id, exc=exc
        )
        return None
    mimetype = str(media.get("mimetype") or "image/jpeg")
    logger.info("Downloaded image ({mime}, {size} B)", mime=mimetype, size=len(data))
    if mimetype == "image/webp":
        try:
            data = first_frame_png(data)
            mimetype = "image/png"
        except Exception as exc:
            logger.warning(
                "WebP first-frame conversion failed for {id}: {exc}",
                id=message_id,
                exc=exc,
            )
            return None
    return {"data": data, "mimetype": mimetype}


def first_frame_png(data: bytes) -> bytes:
    """The first frame of an animated image (webp/gif) as PNG bytes.

    Vision models accept stills, not animations — a 23-frame sticker
    webp would be rejected or misread. Callers decide by mimetype
    whether to convert; static webp converts losslessly too.
    """
    with Image.open(io.BytesIO(data)) as img:
        img.seek(0)
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()


def register_agent_handler(settings: Settings, waha: WahaClient) -> None:
    """Build the agent and register the reply handler."""
    send_tool_holder: dict[str, str] = {}
    config = load_session_config(settings.access_config)
    config_reloader = SessionConfigReloader(settings.access_config)
    enable_langfuse(settings)

    async def run_album(buffer: AlbumBuffer) -> None:
        """Run the agent once over a completed album, all images attached.

        The container event drives the turn (sender tag, gating already
        done at arrival); each buffered image contributes its bytes.
        Runs under the same agent lock as single messages so the shared
        send holder and per-chat context stay consistent.

        Fire-and-forget from the album buffer, so failures must be
        caught here or they would die silently in an unretrieved task:
        the exception is logged with traceback and every buffered
        message's seen marker is dropped so WAHA's redelivery can
        retry the album, mirroring the single-message path.
        """
        try:
            await deliver_album_reply(buffer)
        except Exception:
            for image_event in buffer.images:
                forget_seen(str(image_event.payload.get("id", "")))
            forget_seen(str(buffer.container.payload.get("id", "")))
            logger.exception(
                "Failed to handle album in {chat_id}",
                chat_id=str(buffer.container.payload.get("from", "")),
            )

    async def deliver_album_reply(buffer: AlbumBuffer) -> None:
        """Download an album's images, run the agent once, send its reply."""
        event = buffer.container
        chat_id = str(event.payload.get("from", ""))
        downloaded: list[dict[str, Any]] = []
        for image_event in buffer.images:
            media = image_media(image_event)
            if media is None:
                continue
            image = await asyncio.to_thread(
                download_image,
                waha,
                media,
                str(image_event.payload.get("id", "")),
                settings.max_image_bytes,
            )
            if image is not None:
                downloaded.append(image)
        if not downloaded:
            logger.debug("Album in {chat_id} yielded no usable images", chat_id=chat_id)
            return
        async with _agent_lock:
            send_tool_holder["session"] = event.session
            send_tool_holder["chat_id"] = chat_id
            send_tool_holder["sent"] = ""
            send_tool_holder["reacted"] = ""
            ctx = context_for(event.session, chat_id, agent)
            with chat_trace_attributes(chat_id):
                reply = await handle_message(
                    event, agent, ctx=ctx, images=downloaded, settings=settings, waha=waha
                )
            if send_tool_holder["sent"] or send_tool_holder["reacted"]:
                return
        if reply and reply.strip():
            logger.info("Replying to album in {chat_id}", chat_id=chat_id)
            await asyncio.to_thread(
                waha.send_text,
                event.session,
                chat_id,
                reply,
                str(event.payload.get("id", "")),
            )

    set_completion_handler(run_album)

    def render_prompt() -> str:
        """Re-render ``{{date}}``/``{{time}}`` and pick up config edits.

        The current config's prompt always wins over the startup
        snapshot, so prompt changes apply without a restart too.
        """
        current = config_reloader.current_config()
        return render_system_prompt(
            current.system_prompt, settings.timezone, current.bot_name, current.goal
        )

    agent = build_agent(
        settings,
        tools=build_default_tools(waha, send_tool_holder),
        system_prompt=config.system_prompt,
        prompt_renderer=render_prompt,
    )
    # The workflow reads the holder to know a delivery tool already
    # fired this run (post-delivery final texts are dropped, not sent).
    agent.send_holder = send_tool_holder
    logger.info(
        "Agent ready with {count} tools: {tools}",
        count=len(agent.tools),
        tools=", ".join(sorted(tool.metadata.get_name() for tool in agent.tools)),
    )
    started_at = time.time()

    @on_message
    async def reply_with_agent(event: WahaEvent) -> None:
        """Run the agent over an incoming message and send its reply via WAHA."""
        message_id = str(event.payload.get("id", ""))
        if not message_id:
            logger.debug(
                "Skipping message without an id from {chat_id}",
                chat_id=event.payload.get("from"),
            )
            return
        # Hot-path config: picks up whitelist/prompt/mode edits per event.
        config = config_reloader.current_config()
        if seen_recently(message_id):
            logger.debug("Skipping duplicate event for message {id}", id=message_id)
            return
        if is_stale(event, started_at):
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
        if not chat_allowed(
            event,
            config.whitelist,
            config.blacklist,
            jid_aliases=jid_alias_lookup(event),
        ):
            return
        body = extract_text(event)
        if event.payload.get("fromMe"):
            async with _agent_lock:
                await remember_own_message(event, agent, body)
            return
        if is_album_container(event):
            start_album(event)
            return
        if message_kind(event) in ("image", "sticker") and add_album_image(event):
            return
        image = image_media(event) if settings.vision else None
        if not is_group_addressed(
            event,
            bot_name=config.bot_name,
            bot_mention_regex=config.bot_mention_regex,
            participation=config.group_participation,
        ):
            logger.debug(
                "Ignoring unaddressed group message {id}", id=event.payload.get("id")
            )
            return
        if message_kind(event) == "audio" and settings.transcribe_url:
            transcript = await transcribe_voice_note(event, waha, settings)
            if transcript:
                event.payload["body"] = f"[voice note] {transcript}"
                body = event.payload["body"]
        if body is None and image is None:
            logger.debug("Skipping media/album message {id}", id=event.payload.get("id"))
            return
        chat_id = str(event.payload["from"])
        try:
            if image is not None:
                image = await asyncio.to_thread(
                    download_image, waha, image, message_id, settings.max_image_bytes
                )
            async with _agent_lock:
                # Holder writes live inside the lock, and _agent_lock
                # serializes all agent runs, so a concurrent webhook post
                # can no longer overwrite this run's chat target.
                send_tool_holder["session"] = event.session
                send_tool_holder["chat_id"] = chat_id
                send_tool_holder["sent"] = ""
                send_tool_holder["reacted"] = ""
                ctx = context_for(event.session, chat_id, agent)
                with chat_trace_attributes(chat_id):
                    reply = await handle_message(
                        event, agent, ctx=ctx, image=image, settings=settings, waha=waha
                    )
                if send_tool_holder["sent"] or send_tool_holder["reacted"]:
                    logger.debug(
                        "Agent already delivered its reply in {chat_id}",
                        chat_id=chat_id,
                    )
                    return
            if not reply or not reply.strip():
                logger.debug("Agent chose to stay silent in {chat_id}", chat_id=chat_id)
                return
            logger.info("Replying to {chat_id}", chat_id=chat_id)
            await asyncio.to_thread(
                waha.send_text, event.session, chat_id, reply, message_id
            )
        except Exception:
            # Allow WAHA's redelivery of this message to be reprocessed.
            forget_seen(message_id)
            logger.exception(
                "Failed to handle message {id} in {chat_id} {detail}",
                id=message_id,
                chat_id=chat_id,
                detail="(seen marker dropped; a WAHA redelivery will retry)",
            )
            return
