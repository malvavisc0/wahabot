"""Message handler connecting the webhook to the LlamaIndex agent."""

import asyncio
import time
from typing import Any

from llama_index.core.workflow import Context
from loguru import logger

from whabot.ai.context import handle_message, render_system_prompt
from whabot.ai.messages import (
    extract_text,
    image_media,
    is_group_addressed,
    is_replyable,
)
from whabot.ai.observability import chat_trace_attributes, enable_langfuse
from whabot.ai.tools import build_default_tools
from whabot.ai.workflow import build_agent
from whabot.core.access import load_session_config
from whabot.core.filters import chat_allowed
from whabot.core.models import WahaEvent
from whabot.core.waha import MediaTooLargeError, WahaClient
from whabot.settings import Settings
from whabot.webhook import on_message

_seen_ids: dict[str, float] = {}
_SEE_TTL_S = 120
contexts: dict[tuple[str, str], Context] = {}
_agent_lock = asyncio.Lock()

#: Messages older than this many seconds are stale backlog, not live turns.
_MAX_MESSAGE_AGE_S = 300


def seen_recently(message_id: str) -> bool:
    """Return True if this message id was already handled in the last TTL."""
    now = time.monotonic()
    if message_id in _seen_ids:
        if now - _seen_ids[message_id] < _SEE_TTL_S:
            return True
        del _seen_ids[message_id]
    if len(_seen_ids) >= 10_000:
        _seen_ids.clear()
    _seen_ids[message_id] = now
    return False


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
    return {"data": data, "mimetype": mimetype}


def register_agent_handler(settings: Settings, waha: WahaClient) -> None:
    """Build the agent and register the reply handler."""
    send_tool_holder: dict[str, str] = {}
    config = load_session_config(settings.access_config)
    enable_langfuse(settings)
    agent = build_agent(
        settings,
        tools=build_default_tools(waha, send_tool_holder),
        system_prompt=render_system_prompt(
            config.system_prompt, settings.timezone, config.bot_name
        ),
    )
    access = config
    started_at = time.time()

    @on_message
    async def reply_with_agent(event: WahaEvent) -> None:
        """Run the agent over an incoming message and send its reply via WAHA."""
        message_id = str(event.payload.get("id", ""))
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
        if event.payload.get("fromMe"):
            logger.debug("Ignoring own outbound message {id}", id=message_id)
            return
        if not is_replyable(event):
            logger.debug(
                "Ignoring non-replyable message from {sender}",
                sender=event.payload.get("from"),
            )
            return
        if not chat_allowed(event, access.whitelist, access.blacklist):
            return
        body = extract_text(event)
        image = image_media(event) if settings.vision else None
        if body is None and image is None:
            logger.debug("Skipping media/album message {id}", id=event.payload.get("id"))
            return
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
        if image is not None:
            image = download_image(waha, image, message_id, settings.max_image_bytes)
        chat_id = str(event.payload["from"])
        send_tool_holder["session"] = event.session
        send_tool_holder["chat_id"] = chat_id
        send_tool_holder["sent"] = ""
        ctx = contexts.setdefault((event.session, chat_id), Context(agent))
        async with _agent_lock:
            with chat_trace_attributes(chat_id):
                reply = await handle_message(
                    event, agent, ctx=ctx, image=image, settings=settings
                )
        if send_tool_holder["sent"]:
            logger.debug("Agent already sent its reply in {chat_id}", chat_id=chat_id)
            return
        if not reply or not reply.strip():
            logger.debug("Agent chose to stay silent in {chat_id}", chat_id=chat_id)
            return
        logger.info("Replying to {chat_id}", chat_id=chat_id)
        waha.send_text(event.session, chat_id, reply)
