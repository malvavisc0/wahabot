"""WhatsApp identity: serialized message ids and JID equivalence.

One parser for the serialized id shape (``{fromMe}_{chat}_{msgid}``)
and one normalization core for JIDs, over which the two equivalence
predicates sit:

- :func:`same_chat` — the security fence. Person JIDs compare on user
  id across the interchangeable domains (``c.us``/``s.whatsapp.net``/
  ``lid``); everything else compares exactly, and unknown shapes fail
  closed (a false refusal is the safe direction for the fence).
- :func:`jid_aliases` — the access-list convenience. Resolves only the
  bot's own two identities (the ``me`` id/lid pair an event carries);
  other people's JIDs cannot be resolved locally and stay as they are.

The two predicates are intentionally separate: the fence fails closed
while the access list resolves only self — merging them into one
equivalence would either weaken the fence or break the access list.
"""

from typing import Any

__all__ = [
    "chat_from_message_id",
    "is_own_message_id",
    "jid_aliases",
    "jid_user",
    "parse_message_id",
    "roster_entries",
    "same_chat",
]

#: Server domains WhatsApp uses interchangeably for one person's JID
#: (the phone-number identity). Group (``@g.us``) and broadcast JIDs
#: are never aliased, so they compare by exact string.
_PERSON_JID_DOMAINS = ("c.us", "s.whatsapp.net", "lid")


def parse_message_id(message_id: str) -> tuple[bool, str]:
    """Split a serialized id into ``(from_me, chat_jid)``.

    Serialized ids have the form ``{fromMe}_{chat}_{msgid}[_{participant}]``
    and chat JIDs never contain underscores (user ids are digits, group
    ids digits-dash-digits), so the chat is the second segment. A value
    with no recognizable shape returns ``(False, "")``.
    """
    parts = message_id.split("_")
    if len(parts) < 3 or parts[0] not in ("true", "false"):
        return False, ""
    chat = parts[1]
    return parts[0] == "true", chat if "@" in chat else ""


def chat_from_message_id(message_id: str) -> str:
    """The chat JID embedded in a serialized id, else ""."""
    return parse_message_id(message_id)[1]


def is_own_message_id(message_id: str) -> bool:
    """True when the serialized id marks the message as sent by us."""
    return message_id.startswith("true_")


def jid_user(jid: str) -> tuple[str, str]:
    """The ``(user, domain)`` halves of a JID (empty strings when absent)."""
    user, _, domain = jid.partition("@")
    return user, domain


def same_chat(a: str, b: str) -> bool:
    """True when two JIDs name the same chat.

    WAHA reports a person's chat as ``<phone>@c.us`` in some payloads
    and ``<phone>@lid`` (linked-device id) or the classic
    ``<phone>@s.whatsapp.net`` in others; comparing raw strings would
    fence the current chat against itself. Person JIDs compare on
    user id when both domains are interchangeable; every other shape
    (groups, broadcasts, unknown domains) falls back to exact
    equality. Failing closed — a false refusal — is the safe
    direction for the fence.
    """
    if a == b:
        return True
    a_user, a_domain = jid_user(a)
    b_user, b_domain = jid_user(b)
    if not a_user or not b_user:
        return False
    return (
        a_user == b_user
        and a_domain in _PERSON_JID_DOMAINS
        and b_domain in _PERSON_JID_DOMAINS
    )


def jid_aliases(me: dict[str, Any]) -> dict[str, str]:
    """The bot's own id ↔ lid alias map from an event's ``me`` block.

    WhatsApp accounts have two stable identifiers: the phone-number JID
    (``@c.us``) and the linked-device LID (``@lid``). Events carry
    whichever the chat uses, and ``me`` carries both of ours, so a JID
    equal to one of our identities resolves to the other. JIDs of
    other people cannot be resolved locally and stay as they are.
    """
    phone_id = str(me.get("id") or "")
    lid = str(me.get("lid") or "")
    pairs = {phone_id: lid, lid: phone_id}
    pairs.pop("", None)
    return pairs


def roster_entries(overview: dict[str, Any]) -> list[Any]:
    """The participant list, wherever WAHA put it.

    Engines differ: top level, under ``_chat``, or (LID groups) nested
    inside ``_chat.groupMetadata.participants``; entries are plain JID
    strings or ``{"id": {...}}`` objects.
    """
    blob = overview.get("_chat")
    chat_blob: dict[str, Any] = blob if isinstance(blob, dict) else {}
    for candidates in (
        overview.get("participants"),
        chat_blob.get("participants"),
        chat_blob.get("groupMetadata", {}).get("participants"),
    ):
        if isinstance(candidates, list):
            return candidates
    return []
