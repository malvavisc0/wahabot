"""Message classification and extraction helpers for WAHA events."""

import re
from typing import Any

from whabot.core.models import WahaEvent

NON_REPLYABLE_SUFFIXES = ("@broadcast", "@newsletter")


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
    ``@`` prefix.
    """
    if bot_mention_regex:
        return re.compile(bot_mention_regex)
    quoted = re.escape(bot_name or "")
    return re.compile(rf"(?iu)(?<![^\W_])@?{quoted}(?!\w)")


def bot_mentioned(
    event: WahaEvent,
    bot_name: str | None = None,
    bot_mention_regex: str | None = None,
) -> bool:
    """Whether the message mentions the bot by name/regex or via our JID.

    In groups this matches the configured regex (e.g. ``@kai``,
    ``@kay``, ``ĸay``, ``kai``) or the bot's JID in ``mentionedJidList``.
    """
    payload = event.payload
    me = (event.me or {}).get("id")
    mentioned = payload.get("_data", {}).get("mentionedJidList", [])
    if me and me in mentioned:
        return True
    pattern = bot_mention_pattern(bot_name, bot_mention_regex)
    body = str(payload.get("body", ""))
    return bool(pattern.search(body))


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
    # "mentioned" mode
    if bot_mentioned(event, bot_name, bot_mention_regex):
        return True
    quoted = message_replies_to(event)
    if quoted:
        me = (event.me or {}).get("id")
        quoted_by = quoted.get("participant")
        if me and quoted_by == me:
            return True
    return False
