from typing import Any

from wahabot.ai.messages import (
    bot_jids,
    is_operator_event,
    jid_string,
    reply_description,
)
from wahabot.core.identity import AUTHOR_BOT, AUTHOR_OPERATOR, authors, display_name
from wahabot.core.models import WahaEvent


def sender_tag(event: WahaEvent, names: dict[str, str] | None = None) -> str:
    """The sender's display identity for the agent prompt, e.g. ``[Ana]``.

    Prefers the WhatsApp display name (``_data.notifyName``, then a
    top-level ``notifyName`` for engines that hoist it); falls back to
    the participant/author id so a group turn is never anonymous.
    Returns an empty tag when nothing is known (should not happen).

    Group turns render ``[Name <jid>]`` when the sender's JID is
    known: one string that is both the display identity and the
    mention handle the model copies into ``@<user-part>`` tokens /
    ``mentions`` — no suffix-parsing inference, no extra tool call.
    The JID comes from ``payload.participant``/``_data.author``
    (normalized through :func:`jid_string` — LID groups report
    participants as JID objects) and the name from *names*, the shared
    roster. Fallbacks: unknown name → ``[<full-jid>]`; known name, no
    JID → today's ``[Name]``. DM turns keep ``[Name]`` — the chat
    partner needs no mentioning, and the JID buys nothing there.
    """
    data = event.payload.get("_data", {})
    name = str(data.get("notifyName") or event.payload.get("notifyName") or "").strip()
    participant = jid_string(event.payload.get("participant") or data.get("author"))
    if not participant:
        return f"[{name}]" if name else ""
    in_group = str(event.payload.get("from", "")).endswith("@g.us")
    # The operator's DM turns keep the account JID too: in a direct
    # chat the tag is what tells the operator's lines from the partner's.
    if in_group or is_operator_event(event):
        return group_sender_tag(name, participant, names or {})
    return f"[{name}]" if name else f"[{participant}]"


def group_sender_tag(name: str, participant: str, names: dict[str, str]) -> str:
    """A group turn's ``[Name <jid>]`` tag with its two fallbacks."""
    resolved = name or display_name(participant, names)
    if resolved:
        return f"[{resolved} <{participant}>]"
    return f"[{participant}]"


def message_id_note(event: WahaEvent) -> str:
    """The incoming message's serialized id as a prompt note.

    The id is what ``send_message(reply_to=…)`` and ``react_to_message``
    need to quote or react to *this* message — without it in the turn
    text the model can only quote ids it fetched itself.
    """
    data = event.payload.get("_data", {})
    message_id = str(data.get("id", {}).get("_serialized", "")) if data else ""
    if not message_id:
        message_id = str(event.payload.get("id", ""))
    return f"\n[message id: {message_id}]" if message_id else ""


def reply_context(
    message_reply: dict[str, Any] | None,
    participant_names: dict[str, str] | None = None,
    own_jids: set[str] | None = None,
) -> str:
    """Render the quoted message as context for the agent.

    Empty/None input yields nothing; the caller decides whether to
    include a ``Reply context`` block in the prompt. *participant_names*
    maps chat JIDs to display names so the quoted sender renders as a
    name, not a raw ``@lid`` JID. *own_jids* (the account's own ids)
    lets a quote of an account message say which teammate wrote it.
    """
    if not message_reply:
        return ""
    sender = quoted_participant(message_reply, participant_names, own_jids)
    description = reply_description(message_reply)
    if sender and description:
        return f'{sender}: "{description}"'
    return sender or description


def quoted_participant(
    message_reply: dict[str, Any],
    participant_names: dict[str, str] | None = None,
    own_jids: set[str] | None = None,
) -> str:
    """The quoted message's sender display name; empty when unknown.

    Prefers the quoted message's own ``_data.notifyName``, then the
    chat's participant roster (*participant_names*), then the raw
    participant/author id. With a resolved name the render is
    ``Name <jid>`` — the same mention-handle shape the sender tags
    carry, so the model can copy the id without a tool call; without
    one it falls back to the bare user part (a bare number reads like
    an id, a full JID reads like noise). The participant field may be
    a JID object (LID groups), so it is normalized via
    :func:`jid_string` before use.
    """
    data = message_reply.get("_data", {})
    jid = jid_string(message_reply.get("participant") or data.get("author"))
    if own_jids and jid in own_jids:
        return own_account_label(str(message_reply.get("id") or ""), jid)
    name = str(data.get("notifyName") or "").strip()
    if not jid:
        return name
    name = name or display_name(jid, participant_names)
    return f"{name} <{jid}>" if name else jid.split("@", 1)[0]


def own_account_label(message_id: str, jid: str) -> str:
    """Who wrote a message from the shared account, as a quote sender label.

    The bot and the human operator share one JID, so the JID alone
    cannot say which teammate wrote it; the author book recorded it from
    WAHA's ``source`` field when the message was sent.
    """
    author = authors.author(message_id)
    if author == AUTHOR_BOT:
        return f"you <{jid}>"
    if author == AUTHOR_OPERATOR:
        return f"your operator <{jid}>"
    return f"this account, you or your operator <{jid}>"


def mentions_note(
    event: WahaEvent, participant_names: dict[str, str] | None = None
) -> str:
    """Who each ``@<digits>`` in the body is, from ``mentionedJidList``.

    WhatsApp writes a mention into the text as bare digits
    (``@111222333444555``) and lists the tagged JIDs separately; the
    model sees only the digits, so it cannot tell who was tagged — or
    that the tag is the shared account itself. Names come from the
    roster and the name book; unknown members stay bare JIDs. Only the
    first ten tags render — a mass tagging must not flood the turn.
    """
    mentioned = event.payload.get("_data", {}).get("mentionedJidList") or []
    jids = list(dict.fromkeys(j for j in map(jid_string, mentioned) if j))
    if not jids:
        return ""
    own = bot_jids(event)
    parts = [mention_label(jid, own, participant_names) for jid in jids[:10]]
    return "\n[mentions: " + "; ".join(parts) + "]"


def mention_label(jid: str, own: set[str], names: dict[str, str] | None) -> str:
    """``@<digits> is <who>`` for one tagged JID."""
    user = jid.split("@", 1)[0]
    if jid in own:
        return f"@{user} is the shared account (you and your operator)"
    name = display_name(jid, names)
    return f"@{user} is {name} <{jid}>" if name else f"@{user} is <{jid}>"


def reply_context_section(
    message_reply: dict[str, Any] | None,
    participant_names: dict[str, str] | None = None,
    own_jids: set[str] | None = None,
) -> str:
    """Message text with the quoted/replied-to message attached as context."""
    if not message_reply:
        return ""
    line = reply_context(message_reply, participant_names, own_jids)
    return f"\n[quoting] {line}" if line else ""
