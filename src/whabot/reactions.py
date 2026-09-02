"""Reaction handling: log reactions to messages the bot sent."""

from typing import Any

import httpx
from loguru import logger

from whabot.core.messages import chat_id_from_message_id, message_preview
from whabot.core.models import WahaEvent
from whabot.core.waha import WahaClient
from whabot.webhook import on_reaction


def register_reaction_handler(waha: WahaClient) -> None:
    """Log reactions the recipients give to messages the bot sent."""

    @on_reaction
    async def log_reaction(event: WahaEvent) -> None:
        reaction: dict[str, Any] = event.payload.get("reaction", {})
        target_id = reaction.get("messageId", "")
        if not target_id:
            logger.debug(
                "Reaction without a target id from {sender}",
                sender=event.payload.get("from"),
            )
            return
        message = fetch_target(waha, event.session, target_id)
        if message is None or not message.get("fromMe"):
            return
        emoji = str(reaction.get("text", "")).strip() or "(removed)"
        logger.info(
            'Reaction {emoji} to "{preview}" from {sender}',
            emoji=emoji,
            preview=message_preview(message),
            sender=event.payload.get("from"),
        )


def fetch_target(waha: WahaClient, session: str, target_id: str) -> dict[str, Any] | None:
    """Fetch the reacted-to message, or None if it cannot be retrieved."""
    try:
        return waha.get_message(session, chat_id_from_message_id(target_id), target_id)
    except httpx.HTTPError as exc:
        logger.warning(
            "Could not fetch message reacted to {target}: {exc}",
            target=target_id,
            exc=exc,
        )
        return None
