"""Message classification and extraction helpers for WAHA events."""

import re
from typing import Any

from wahabot.core.identity import authors
from wahabot.core.models import WahaEvent
from wahabot.core.waha import media_with_url

NON_REPLYABLE_SUFFIXES = ("@broadcast", "@newsletter")

#: Kwarg key tagging a folded reaction note with its target message id.
#: ``reactions`` stamps it so the superseded-note removal survives the
#: history merge; ``history`` treats a tagged turn as a fold rather
#: than a run-scoped inbound turn.
REACTION_TARGET_KWARG = "reaction_target_id"

#: Kwarg key stamped on an inbound user turn once its agent run has
#: finished. ``prepare_chat_history`` appends the turn unstamped; the
#: workflow stamps it at run end. ``history``'s trailing-user drop then
#: only removes an *unstamped* trailing turn — the scaffolding of a
#: crashed/replaced run — never a message a completed run already read
#: (in ``judicious`` group mode that is most of the conversation: a
#: silent run's message must stay in context for the next run).
TURN_HANDLED_KWARG = "turn_handled"

#: Kwarg key tagging the model's post-delivery wrap-up note — the
#: one-sentence self-record stored after a delivery fired. The note
#: was never sent to the chat (the one-delivery latch holds), so
#: anything reading history must be able to tell it apart from a
#: message the chat saw.
WRAP_UP_NOTE_KWARG = "wrap_up_note"

#: Kwarg tagging a folded operator-typed message: a no-run user turn,
#: handled by construction (never the trailing scaffolding of a run).
OPERATOR_FOLD_KWARG = "operator_fold"


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
    only — docs/openapi.json) is ``api`` for a send through the API,
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
    return media_with_url(event.payload)


def video_media(event: WahaEvent) -> dict[str, Any] | None:
    """The media dict of a video message (kinds ``video``/``ptv``), or None.

    Like :func:`image_media` but for moving pictures; only entries with
    a URL qualify — the download path checks the mimetype/size caps.
    """
    if message_kind(event) not in ("video", "ptv"):
        return None
    return media_with_url(event.payload)


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
    return account_tagged(event) or name_mentioned(event, bot_name, bot_mention_regex)


def self_command_instruction(
    event: WahaEvent,
    bot_name: str | None = None,
    bot_mention_regex: str | None = None,
) -> str | None:
    """Extract an operator instruction addressed to the bot in self-chat.

    WAHA marks messages sent from the account as ``fromMe``. A message
    qualifies when the account is on both ends of it — ``from`` is one
    of the account's own JIDs and ``to`` (when present) is too — so a
    manually typed message in another chat keeps its memory-only
    behavior. The ``to`` check matters on LID-linked accounts: WAHA
    reports the self-chat as ``from=<phone>@c.us, to=<lid>@lid`` —
    two different strings for one conversation. The mention must
    *begin* the message: a trigger word appearing mid-text (the
    operator quoting or discussing something) is not a command.
    """
    payload = event.payload
    if not payload.get("fromMe"):
        return None
    own = bot_jids(event)
    if jid_string(payload.get("from")) not in own:
        return None
    to = jid_string(payload.get("to"))
    if to and to not in own:
        return None
    body = str(payload.get("body", "")).strip()
    # A transcribed voice note arrives wrapped as "[voice note] …";
    # the mention follows the modality marker.
    body = body.removeprefix("[voice note]").strip()
    match = bot_mention_pattern(bot_name, bot_mention_regex).match(body)
    if match is None:
        return None
    instruction = body[match.end() :].lstrip(" \t:,-")
    return instruction or None


def bot_jids(event: WahaEvent) -> set[str]:
    """The bot's own JIDs — its phone id and, when known, its LID.

    Read from the event's own ``me`` block only: command and mention
    gates must authenticate per event, not inherit trust from
    process-global state — an event without ``me`` yields no identity
    and simply fails those gates.
    """
    me = event.me or {}
    return {jid_string(me[key]) for key in ("id", "lid") if me.get(key)}


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


#: Payload flag set on an operator-typed ``fromMe`` message once the
#: handler normalized it into an ordinary group/DM message shape.
OPERATOR_PAYLOAD_FLAG = "_wahabot_operator"

#: Prefix of every turn the human operator typed on the shared account.
OPERATOR_NOTE_PREFIX = "[operator message] "


def is_operator_event(event: WahaEvent) -> bool:
    """True for an operator-typed message the handler normalized."""
    return bool(event.payload.get(OPERATOR_PAYLOAD_FLAG))


def name_mentioned(
    event: WahaEvent,
    bot_name: str | None = None,
    bot_mention_regex: str | None = None,
) -> bool:
    """Whether the body names the bot (regex / bot-name match)."""
    pattern = bot_mention_pattern(bot_name, bot_mention_regex)
    return bool(pattern.search(str(event.payload.get("body", ""))))


def account_tagged(event: WahaEvent) -> bool:
    """Whether ``mentionedJidList`` tags the shared account's own JID."""
    mentioned = event.payload.get("_data", {}).get("mentionedJidList", []) or []
    return bool(bot_jids(event) & {jid_string(jid) for jid in mentioned})


def quoted_account_author(event: WahaEvent) -> str | None:
    """Who wrote the quoted message, when it came from the shared account.

    ``None`` when the message quotes nothing or someone else; otherwise
    ``"bot"``, ``"operator"`` or ``""`` (an account message from before
    authors were recorded — it could be either teammate).
    """
    quoted = message_replies_to(event)
    if not quoted:
        return None
    reply: dict[str, Any] = quoted
    if jid_string(reply.get("participant")) not in bot_jids(event):
        return None
    return authors.author(str(reply.get("id") or ""))


def addressed_note(
    event: WahaEvent,
    bot_name: str | None = None,
    bot_mention_regex: str | None = None,
) -> str:
    """The turn-level statement of *how* this message reached the bot.

    The wake gate decides mechanically, but the model never sees that
    decision — in ``judicious`` groups every message wakes it, so being
    awake proves nothing. This note hands the mechanical facts over as
    turn context, precisely, because the account is shared with the
    human operator and "addressed" has three different meanings:

    - the text names the bot (regex / bot name): it is for the bot;
    - the message tags the account's JID: that is the shared account,
      which may mean the bot *or* the human operator;
    - the message quote-replies to an account message: the note says
      whether the bot or the operator wrote it, when that is recorded.

    An operator-typed message that names the bot carries its own
    variant: the teammate is asking the bot directly.

    DM turns carry no note (every DM is for the bot).
    """
    payload = event.payload
    if is_operator_event(event):
        if name_mentioned(event, bot_name, bot_mention_regex):
            return (
                "\n[you were addressed: your operator, the human sharing this"
                " account, named you — this is for you]"
            )
        return ""
    if not str(payload.get("from", "")).endswith("@g.us") or payload.get("fromMe"):
        return ""
    if name_mentioned(event, bot_name, bot_mention_regex):
        return "\n[you were addressed: this message names you — it is for you]"
    if account_tagged(event):
        return (
            "\n[you were addressed: this message tags the shared account, which"
            " may mean you or your operator]"
        )
    author = quoted_account_author(event)
    if author == "bot":
        return "\n[you were addressed: this message replies to your message]"
    if author == "operator":
        return (
            "\n[you were addressed: this message replies to a message your"
            " operator typed — it may be meant for him]"
        )
    if author == "":
        return (
            "\n[you were addressed: this message replies to a message from the"
            " shared account, written by you or your operator]"
        )
    return ""
