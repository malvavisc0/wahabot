import re
from typing import Any

from wahabot.ai.messages.constants import (
    BLAME_RE,
    GENERIC_ADDRESSED_NOTE,
    OPERATOR_PAYLOAD_FLAG,
    UNADDRESSED_NOTE,
)
from wahabot.ai.messages.content import is_reaction_only
from wahabot.ai.messages.identity import bot_jids, jid_string
from wahabot.ai.scrub import strip_spoofed_markers
from wahabot.core.identity import AUTHOR_BOT, AUTHOR_OPERATOR, authors
from wahabot.core.jid import is_own_message_id
from wahabot.core.models import WahaEvent


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


def reply_description(message_reply: dict[str, Any]) -> str:
    """Describe the quoted message body/media, truncated for the prompt.

    The quoted body is member text riding inside a code-rendered
    ``[quoting]`` marker — scrubbed so a member cannot smuggle a
    spoofed marker in through the quote they send. The record grammar
    reuses the same description, clamped further at render time.
    """
    media = message_reply.get("hasMedia") and message_reply.get("media")
    if media:
        mimetype = media.get("mimetype", "media")
        filename = media.get("filename") or ""
        description = f"{mimetype} {filename}".strip()
    else:
        description = strip_spoofed_markers(
            str(message_reply.get("body", "") or "").strip()
        )
    return description[:400]


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
    *begin* the message in other chats (a trigger word appearing
    mid-text is usually a quote or discussion). The self-chat is the
    console: there the operator is always addressing the bot, so a
    mention anywhere is an instruction and the whole body is used.
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
    if match is not None:
        instruction = body[match.end() :].lstrip(" \t:,-")
        return instruction or None
    return body if name_mentioned(event, bot_name, bot_mention_regex) else None


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
      any text so the agent can decide whether to speak. A bare emoji or
      punctuation-only body is a reaction, not an opening, and does not
      wake it on its own.
    """
    payload = event.payload
    if not str(payload.get("from", "")).endswith("@g.us"):
        return True
    if payload.get("fromMe"):
        return False
    if participation == "never":
        return False
    if replies_to_bot(event, bot_name, bot_mention_regex):
        return True
    if participation == "judicious":
        # Let the agent judge every message; it may stay silent. A
        # reaction-only body is skipped so an emoji cannot summon a run.
        return not is_reaction_only(event)
    return False


def replies_to_bot(
    event: WahaEvent,
    bot_name: str | None = None,
    bot_mention_regex: str | None = None,
) -> bool:
    """Whether a group message mentions the bot or quotes one of its messages.

    The quoted message's sender is matched against both of the bot's
    identities: quotes of the bot's messages carry its phone JID or its
    LID depending on the group type. The author book decides bot vs
    operator when it knows the quoted id; a quote of the operator's
    output is not a reply to the bot. An unrecorded account author keeps
    the old account-JID fallback.
    """
    if bot_mentioned(event, bot_name, bot_mention_regex):
        return True
    quoted = message_replies_to(event)
    if not quoted:
        return False
    author = quoted_account_author(event)
    if author == AUTHOR_BOT:
        return True
    if author == AUTHOR_OPERATOR:
        return False
    # An account message without an author record: the classifier's
    # id/participant fallback already proved it belongs to the shared
    # account, so keep the old addressed behaviour.
    return author == ""


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
    authors were recorded — it could be either teammate). The author
    book is authoritative: an id-only quote or a phone/LID mismatch must
    not erase a recorded bot/operator author.
    """
    quoted = message_replies_to(event)
    if not quoted:
        return None
    reply: dict[str, Any] = quoted
    quoted_id = str(reply.get("id") or "")
    author = authors.author(quoted_id)
    if author in (AUTHOR_BOT, AUTHOR_OPERATOR):
        return author
    participant = jid_string(
        reply.get("participant") or reply.get("_data", {}).get("author")
    )
    if (participant and participant in bot_jids(event)) or is_own_message_id(quoted_id):
        return ""
    return None


def newest_quote_flags(event: WahaEvent) -> tuple[bool, bool, bool]:
    """Classify the newest message's quote as member, bot or operator.

    Returns ``(quoted_other, quoted_bot, quoted_operator)``. A quote of
    another member is a side-thread signal; a quote of the bot is
    addressing; a quote of the operator is the human teammate, not the
    bot. An unrecorded account author is ambiguous, not a member quote,
    so it neither grants a source signal nor silences the turn.
    """
    author = quoted_account_author(event)
    if author == AUTHOR_BOT:
        return False, True, False
    if author == AUTHOR_OPERATOR:
        return False, False, True
    if author is None:
        return bool(message_replies_to(event)), False, False
    return False, False, False


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


def is_group_turn(event: WahaEvent) -> bool:
    """True for a non-``fromMe`` group message (the only not-addressed case)."""
    payload = event.payload
    return (
        not is_operator_event(event)
        and not payload.get("fromMe")
        and str(payload.get("from", "")).endswith("@g.us")
    )


def turn_addressed_note(
    event: WahaEvent,
    addressed: bool,
    bot_name: str | None = None,
    bot_mention_regex: str | None = None,
) -> str:
    """The turn note for *event*, trusting the wake gate's *addressed*.

    ``addressed_note`` re-derives "addressed" from the event alone, which
    a merged burst turn loses (its quote/mention rides an earlier member,
    stripped before the model sees it). The gate already decided, so its
    verdict wins: an addressed turn always carries a positive note, and an
    unaddressed group turn now carries an explicit negative one instead of
    the old silent blank.
    """
    if not is_group_turn(event):
        return addressed_note(event, bot_name, bot_mention_regex)
    if addressed:
        return addressed_note(event, bot_name, bot_mention_regex) or (
            GENERIC_ADDRESSED_NOTE
        )
    return UNADDRESSED_NOTE


def is_blame_text(text: str) -> bool:
    """True when *text* tells the recipient they are wrong.

    A correction turn belongs to the bot's own mistake. The workflow
    rewrites such a final reply; delivery tools refuse it so a model
    that routes the blame through ``send_message``/``send_media`` cannot
    bypass the guard.
    """
    return bool(text and BLAME_RE.search(text))
