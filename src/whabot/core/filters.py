"""Access filtering: whitelist and blacklist for chats."""

from collections.abc import Callable

from loguru import logger

from whabot.core.models import WahaEvent


def chat_allowed(
    event: WahaEvent,
    whitelist: set[str],
    blacklist: set[str],
    jid_aliases: Callable[[str], set[str]] | None = None,
) -> bool:
    """Check the message's chat against the blacklist and whitelist.

    A chat is the `from` JID (user for 1:1, group for @g.us). The
    blacklist always wins; when the whitelist is non-empty only
    listed chats are answered. ``jid_aliases`` maps one JID to its
    alternate identities (e.g. ``@c.us`` ↔ ``@lid``), so an operator
    may list either form. Every decision is logged.
    """
    senders = {
        str(event.payload.get("from", "")),
        str(event.payload.get("participant", "")),
    }
    senders.discard("")
    if jid_aliases is not None:
        senders |= {alias for jid in senders for alias in jid_aliases(jid)}
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


def jid_alias_lookup(event: WahaEvent) -> Callable[[str], set[str]]:
    """Resolve a JID to its alternate identity, based on the event's ``me``.

    WhatsApp accounts have two stable identifiers: the phone-number JID
    (``@c.us``) and the linked-device LID (``@lid``). Events carry
    whichever the chat uses, and ``me`` carries both of ours, so a JID
    equal to one of our identities resolves to the other. JIDs of
    other people cannot be resolved locally and stay as they are.
    """

    me = event.me or {}
    phone_id = str(me.get("id") or "")
    lid = str(me.get("lid") or "")
    pairs = {phone_id: lid, lid: phone_id}
    pairs.pop("", None)

    def lookup(jid: str) -> set[str]:
        return {alias for alias in (pairs.get(jid),) if alias}

    return lookup
