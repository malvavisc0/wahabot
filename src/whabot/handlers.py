"""Message handler connecting the webhook to the LlamaIndex agent."""

import asyncio
import time

from llama_index.core.workflow import Context
from loguru import logger

from whabot.ai.context import handle_message, render_system_prompt
from whabot.ai.messages import extract_text, is_group_addressed, is_replyable
from whabot.ai.tools import build_default_tools
from whabot.ai.workflow import build_agent
from whabot.core.access import load_session_config
from whabot.core.filters import chat_allowed
from whabot.core.models import WahaEvent
from whabot.core.waha import WahaClient
from whabot.settings import Settings
from whabot.webhook import on_message

_seen_ids: dict[str, float] = {}
_SEE_TTL_S = 120
contexts: dict[tuple[str, str], Context] = {}
_agent_lock = asyncio.Lock()


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


def register_agent_handler(settings: Settings, waha: WahaClient) -> None:
    """Build the agent and register the reply handler."""
    send_tool_holder: dict[str, str] = {}
    config = load_session_config(settings.access_config)
    agent = build_agent(
        settings,
        tools=build_default_tools(waha, send_tool_holder),
        system_prompt=render_system_prompt(config.system_prompt, settings.timezone),
    )
    access = config

    @on_message
    async def reply_with_agent(event: WahaEvent) -> None:
        """Run the agent over an incoming message and send its reply via WAHA."""
        message_id = str(event.payload.get("id", ""))
        if seen_recently(message_id):
            logger.debug("Skipping duplicate event for message {id}", id=message_id)
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
        if body is None:
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
        chat_id = str(event.payload["from"])
        send_tool_holder["session"] = event.session
        send_tool_holder["chat_id"] = chat_id
        send_tool_holder["sent"] = ""
        ctx = contexts.setdefault((event.session, chat_id), Context(agent))
        async with _agent_lock:
            reply = await handle_message(event, agent, ctx=ctx)
        if send_tool_holder["sent"]:
            logger.debug("Agent already sent its reply in {chat_id}", chat_id=chat_id)
            return
        if not reply or not reply.strip():
            logger.debug("Agent chose to stay silent in {chat_id}", chat_id=chat_id)
            return
        logger.info("Replying to {chat_id}", chat_id=chat_id)
        waha.send_text(event.session, chat_id, reply)
