"""Access filtering: whitelist and blacklist for chats."""

from loguru import logger

from whabot.core.models import WahaEvent


def chat_allowed(event: WahaEvent, whitelist: set[str], blacklist: set[str]) -> bool:
    """Check the message's chat against the blacklist and whitelist.

    A chat is the `from` JID (user for 1:1, group for @g.us). The
    blacklist always wins; when the whitelist is non-empty only
    listed chats are answered. Every decision is logged.
    """
    senders = {
        str(event.payload.get("from", "")),
        str(event.payload.get("participant", "")),
    }
    senders.discard("")
    if is_listed(senders, blacklist):
        return drop_message(event, "blacklisted")
    if whitelist and not is_listed(senders, whitelist):
        return drop_message(event, "non-whitelisted")
    return True


def is_listed(senders: set[str], listed: set[str]) -> bool:
    """Whether any sender appears in the given access list."""
    return any(sender in listed for sender in senders)


def drop_message(event: WahaEvent, reason: str) -> bool:
    """Log a dropped message and tell the caller to drop it."""
    logger.info(
        "Dropping message from {reason} {chat_id} (participant {participant})",
        reason=reason,
        chat_id=event.payload.get("from"),
        participant=event.payload.get("participant"),
    )
    return False
