"""Reaction handling: log reactions to messages the bot sent."""

import asyncio
from typing import Any

from loguru import logger

from wahabot.core.models import WahaEvent
from wahabot.core.waha import WahaClient
from wahabot.webhook import on_reaction


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
        message = await asyncio.to_thread(fetch_target, waha, event.session, target_id)
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
    """Fetch the reacted-to message, or None if it cannot be retrieved.

    Fail-soft on every error: reaction logging is best-effort, and an
    exception here would 500 the webhook (WAHA would then redeliver
    the reaction event forever).
    """
    try:
        return waha.get_message(session, chat_id_from_message_id(target_id), target_id)
    except Exception as exc:
        logger.warning(
            "Could not fetch message reacted to {target}: {exc}",
            target=target_id,
            exc=exc,
        )
        return None


def chat_id_from_message_id(message_id: str) -> str:
    """Return the chat JID embedded in a serialized message id.

    Serialized ids have the form ``{fromMe}_{chat}_{message_id}[_{participant}]``
    and chat JIDs never contain underscores, so the chat is the second segment.
    """
    parts = message_id.split("_")
    return parts[1] if len(parts) > 1 else message_id


def message_preview(payload: dict[str, Any]) -> str:
    """Return a short human-readable preview of a message, for logs."""
    body = str(payload.get("body", "")).strip()
    if body:
        return body if len(body) <= 80 else body[:77] + "..."
    data = payload.get("_data", {})
    kind = data.get("type")
    return f"[{kind}]" if kind else "[media]"
