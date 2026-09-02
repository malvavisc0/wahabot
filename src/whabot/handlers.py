"""Message handler connecting the webhook to the LlamaIndex agent."""

from loguru import logger

from whabot.ai.agent import (
    build_agent,
    extract_text,
    handle_message,
    is_group_addressed,
    is_replyable,
)
from whabot.core.access import load_access_lists
from whabot.core.filters import chat_allowed
from whabot.core.models import WahaEvent
from whabot.core.waha import WahaClient
from whabot.settings import Settings
from whabot.webhook import on_message


def register_agent_handler(settings: Settings) -> None:
    """Build the agent and WAHA client, registering the reply handler."""
    agent = build_agent(settings)
    waha = WahaClient(base_url=settings.waha_url, api_key=settings.waha_api_key)
    access = load_access_lists(settings.access_config)

    @on_message
    async def reply_with_agent(event: WahaEvent) -> None:
        """Run the agent over an incoming message and send its reply via WAHA."""
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
        if not is_group_addressed(event):
            logger.debug(
                "Ignoring unaddressed group message {id}", id=event.payload.get("id")
            )
            return
        reply = await handle_message(event, agent)
        chat_id = str(event.payload["from"])
        logger.info("Replying to {chat_id}", chat_id=chat_id)
        waha.send_text(event.session, chat_id, reply)
