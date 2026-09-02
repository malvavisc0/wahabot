"""Message handler connecting the webhook to the LlamaIndex agent."""

import time

from llama_index.core.workflow import Context
from loguru import logger

from whabot.ai.agent import (
    build_agent,
    build_default_tools,
    extract_text,
    handle_message,
    is_group_addressed,
    is_replyable,
)
from whabot.ai.context import render_system_prompt
from whabot.core.access import load_session_config
from whabot.core.filters import chat_allowed
from whabot.core.models import WahaEvent
from whabot.core.waha import WahaClient
from whabot.settings import Settings
from whabot.webhook import on_message

_seen_ids: dict[str, float] = {}
_SEE_TTL_S = 120
contexts: dict[tuple[str, str], Context] = {}


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


def register_agent_handler(settings: Settings, waha: WahaClient | None = None) -> None:
    """Build the agent and WAHA client, registering the reply handler."""
    waha = waha or WahaClient(base_url=settings.waha_url, api_key=settings.waha_api_key)
    send_tool_holder: dict[str, str] = {}
    config = load_session_config(settings.access_config)
    system_prompt = config.system_prompt or settings.agent_system_prompt
    agent = build_agent(
        settings,
        tools=build_default_tools(waha, send_tool_holder),
        system_prompt=render_system_prompt(system_prompt, settings.timezone),
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
        ctx = contexts.setdefault((event.session, chat_id), Context(agent))
        reply = await handle_message(event, agent, ctx=ctx)
        if not reply or not reply.strip():
            logger.debug("Agent chose to stay silent in {chat_id}", chat_id=chat_id)
            return
        logger.info("Replying to {chat_id}", chat_id=chat_id)
        waha.send_text(event.session, chat_id, reply)
