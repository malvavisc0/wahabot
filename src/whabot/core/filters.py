"""Access filtering: whitelist and blacklist for chats."""

from loguru import logger

from whabot.core.models import WahaEvent


def chat_allowed(event: WahaEvent, whitelist: set[str], blacklist: set[str]) -> bool:
    """Check the message's chat against the blacklist and whitelist.

    A chat is the `from` JID (user for 1:1, group for @g.us). The
    blacklist always wins; when the whitelist is non-empty only
    listed chats are answered. Every decision is logged.
    """
    chat_id = str(event.payload.get("from", ""))
    participant = str(event.payload.get("participant", ""))

    senders = [chat_id]
    if participant and participant != chat_id:
        senders.append(participant)

    if any(sender in blacklist for sender in senders):
        logger.info(
            "Dropping message from blacklisted {chat_id} (participant {participant})",
            chat_id=chat_id,
            participant=participant,
        )
        return False

    if whitelist and not any(sender in whitelist for sender in senders):
        logger.info(
            "Dropping message from non-whitelisted {chat_id} (participant {participant})",
            chat_id=chat_id,
            participant=participant,
        )
        return False

    return True
