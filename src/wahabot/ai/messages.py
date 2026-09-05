"""Message classification and extraction helpers for WAHA events."""

import re
from typing import Any

from wahabot.core.models import WahaEvent

NON_REPLYABLE_SUFFIXES = ("@broadcast", "@newsletter")


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
            return str(entry["_serialized"])
        user, server = entry.get("user"), entry.get("server")
        return f"{user}@{server}" if user and server else ""
    return str(value or "")


def is_replyable(event: WahaEvent) -> bool:
    """Check whether the message comes from a chat the bot can answer.

    Status updates (`status@broadcast`) and newsletters arrive as
    `message` events but are not replyable conversations.
    """
    sender = str(event.payload.get("from", ""))
    return not sender.endswith(NON_REPLYABLE_SUFFIXES)


def message_kind(event: WahaEvent) -> str:
    """Classify an incoming message: text, image, video, audio, sticker, ..."""
    kind = event.payload.get("_data", {}).get("type")
    if kind == "ptt":
        return "audio"
    if kind in ("image", "video", "ptv", "audio", "sticker", "document"):
        return kind
    if kind == "album":
        return "album"
    return "text"


def extract_text(event: WahaEvent) -> str | None:
    """Return replyable text, or None when there is nothing to answer.

    Album containers carry no text of their own. Every other message —
    text, voice transcription, image/video caption — stores its text in
    the top-level `body`, so return it directly instead of skipping
    messages simply because they carry media.
    """
    if message_kind(event) == "album":
        return None
    body = str(event.payload.get("body", "")).strip()
    return body or None


def image_media(event: WahaEvent) -> dict[str, Any] | None:
    """The media dict of an image message, or None.

    ``image`` messages and ``sticker`` messages qualify — a sticker is
    a (possibly animated) webp image, and the vision model can comment
    on its first frame. Videos and documents are excluded. The dict
    holds ``url``, ``mimetype`` and optionally ``filename``.
    """
    if message_kind(event) not in ("image", "sticker"):
        return None
    media = event.payload.get("media") or _data_media(event.payload)
    if not isinstance(media, dict):
        return None
    media_dict: dict[str, Any] = media
    if not media_dict.get("url"):
        return None
    return media_dict


def _data_media(payload: dict[str, Any]) -> Any:
    """The ``_data.media`` blob of a payload, when present."""
    data = payload.get("_data")
    if isinstance(data, dict):
        data_dict: dict[str, Any] = data
        return data_dict.get("media")
    return None


def message_replies_to(event: WahaEvent) -> dict[str, Any] | None:
    """Return the quoted message when this message is a reply.

    WAHA injects a `replyTo` snippet (WAHA >= 5) or the older
    `_data.quotedMsg` dict; both carry the id, participant and body of
    the quoted message. Returns None for a plain message.
    """
    reply = event.payload.get("replyTo")
    if isinstance(reply, dict):
        return reply
    quoted = event.payload.get("_data", {}).get("quotedMsg")
    return quoted if isinstance(quoted, dict) else None


def bot_mention_pattern(
    bot_name: str | None,
    bot_mention_regex: str | None,
) -> re.Pattern[str]:
    """Build the regex used to detect when the bot is mentioned.

    Prefers the configured ``bot_mention_regex`` (the user controls the
    variants, e.g. `@?[kĸ]a[iy]` for kai/kay/ĸay). Falls back to a
    case-insensitive whole-word match of ``bot_name`` with an optional
    ``@`` prefix. Without a name or regex there is nothing to match, so
    the pattern never matches (an empty ``bot_name`` would otherwise
    compile to a zero-width regex matching almost any text).
    """
    if bot_mention_regex:
        return re.compile(bot_mention_regex)
    if not bot_name:
        return re.compile(r"(?!x)x")
    quoted = re.escape(bot_name)
    return re.compile(rf"(?iu)(?<![^\W_])@?{quoted}(?!\w)")


def bot_mentioned(
    event: WahaEvent,
    bot_name: str | None = None,
    bot_mention_regex: str | None = None,
) -> bool:
    """Whether the message mentions the bot by name/regex or via our JID.

    In groups this matches the configured regex (e.g. ``@kai``,
    ``@kay``, ``ĸay``, ``kai``) or the bot's own JID(s) in
    ``mentionedJidList``. Both identities are checked: WhatsApp
    carries mentions of the account's phone JID (``@c.us``) or its
    linked-device LID (``@lid``) depending on the group type.
    """
    payload = event.payload
    me_ids = bot_jids(event)
    mentioned = payload.get("_data", {}).get("mentionedJidList", [])
    if me_ids & {jid_string(jid) for jid in mentioned}:
        return True
    pattern = bot_mention_pattern(bot_name, bot_mention_regex)
    body = str(payload.get("body", ""))
    return bool(pattern.search(body))


def bot_jids(event: WahaEvent) -> set[str]:
    """The bot's own JIDs — its phone id and, when known, its LID."""
    me = event.me or {}
    return {str(me[key]) for key in ("id", "lid") if me.get(key)}


def is_group_addressed(
    event: WahaEvent,
    bot_name: str | None = None,
    bot_mention_regex: str | None = None,
    participation: str = "mentioned",
) -> bool:
    """Check whether a group message should wake the agent.

    In 1:1 chats every message is for us. In groups:

    - ``never`` — never reply in groups.
    - ``mentioned`` (default) — wake only when the bot is mentioned by
      name/regex (``bot_mentioned``) or when a message replies to a bot
      message.
    - ``judicious`` — wake on mention/reply as well, but also wake for
      any text so the agent can decide whether to speak.
    """
    payload = event.payload
    if not str(payload.get("from", "")).endswith("@g.us"):
        return True
    if payload.get("fromMe"):
        return False
    if participation == "never":
        return False
    if participation == "judicious":
        # Let the agent judge every text message; it may stay silent.
        return True
    return replies_to_bot(event, bot_name, bot_mention_regex)


def replies_to_bot(
    event: WahaEvent,
    bot_name: str | None = None,
    bot_mention_regex: str | None = None,
) -> bool:
    """Whether a group message mentions the bot or quotes one of its messages.

    The quoted message's sender is matched against both of the bot's
    identities: quotes of the bot's messages carry its phone JID or its
    LID depending on the group type.
    """
    if bot_mentioned(event, bot_name, bot_mention_regex):
        return True
    quoted = message_replies_to(event)
    if not quoted:
        return False
    reply: dict[str, Any] = quoted
    return bool(jid_string(reply.get("participant")) in bot_jids(event))
