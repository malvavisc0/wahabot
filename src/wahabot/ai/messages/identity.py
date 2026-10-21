from typing import Any

from wahabot.core.models import WahaEvent


def jid_string(value: Any) -> str:
    """A JID field as a plain ``user@server`` string.

    WAHA carries JIDs in two shapes: plain strings (``@lid``/``@c.us``
    entries in ``mentionedJidList`` on some engines) or objects with a
    ``_serialized`` field (LID groups, ``replyTo.participant``).
    Stringifying the object blindly (``str(dict)``) never matches
    anything, so mentions and quotes from LID groups were invisible.
    A dict without ``_serialized`` falls back to joining its ``user``
    and ``server`` fields; anything else unknown yields "".
    """
    if isinstance(value, dict):
        entry: dict[str, Any] = value
        if entry.get("_serialized"):
            return strip_device(str(entry["_serialized"]))
        user, server = entry.get("user"), entry.get("server")
        return f"{user}@{server}" if user and server else ""
    return strip_device(str(value or ""))


def strip_device(jid: str) -> str:
    """*jid* without a linked-device suffix (``123:41@lid`` → ``123@lid``).

    WhatsApp tags a message or reaction typed on a secondary device of
    an account with that device's index. The suffix names a device, not
    a person: the operator reacting from WhatsApp Web arrives as
    ``<own-lid-user>:41@lid`` and must still compare equal to the
    account's own LID, or the bot mistakes its operator for a stranger.
    """
    user, at, server = jid.partition("@")
    if not at or ":" not in user:
        return jid
    return f"{user.split(':', 1)[0]}@{server}"


def conversation_jid(event: WahaEvent) -> str:
    """The chat a message belongs to, whoever sent it.

    WAHA puts the *sender* in ``from``: an incoming message's ``from``
    is the chat (group or person), but a ``fromMe`` message's ``from``
    is the account itself and the chat is in ``to``. Using ``from`` for
    both silently filed every operator-typed message under the
    account's own JID, where the whitelist dropped it.
    """
    payload = event.payload
    if payload.get("fromMe"):
        return jid_string(payload.get("to")) or jid_string(payload.get("from"))
    return jid_string(payload.get("from"))


def sent_by_bot_api(event: WahaEvent) -> bool:
    """True for the echo of a message the bot itself sent through WAHA.

    WAHA's ``source`` field (webhook events only, ``fromMe`` messages
    only) is ``api`` for a send through the API,
    i.e. the bot, and ``app`` for the human typing on a phone or
    WhatsApp Web. The bot's own output is already in memory, so its
    echo must never be folded again or treated as an operator message.
    """
    return bool(event.payload.get("fromMe")) and event.payload.get("source") == "api"


def typed_by_operator(event: WahaEvent) -> bool:
    """True for a message the human operator typed on the shared account.

    Requires WAHA's explicit ``source == "app"``: without the field the
    message could be the bot's own echo, and treating it as the operator
    could make the bot answer itself.
    """
    return bool(event.payload.get("fromMe")) and event.payload.get("source") == "app"


def bot_jids(event: WahaEvent) -> set[str]:
    """The bot's own JIDs — its phone id and, when known, its LID.

    Read from the event's own ``me`` block only: command and mention
    gates must authenticate per event, not inherit trust from
    process-global state — an event without ``me`` yields no identity
    and simply fails those gates.
    """
    me = event.me or {}
    return {jid_string(me[key]) for key in ("id", "lid") if me.get(key)}
