import re
import unicodedata
from typing import Any

from loguru import logger

from wahabot.ai.messages import jid_string
from wahabot.ai.tools.envelope import error, ok
from wahabot.core.identity import display_name
from wahabot.core.identity import names as name_book
from wahabot.core.jid import roster_entries
from wahabot.core.waha import WahaClient
from wahabot.status import state as _status_state


def summarize_chat(
    chat_id: str,
    overview: dict[str, Any],
    names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a compact, model-friendly summary of a chat overview dict.

    Extracts the stable scalar fields and the participant roster,
    normally skipping nested blobs. If only an id or no stable fields
    remain, ``_raw`` is a possibly truncated 1000-character string
    fallback, not structured or complete metadata. Recent display names
    enrich roster entries but do not verify identity.
    """
    scalar = chat_scalars(overview)
    add_participant_summary(scalar, overview, names or {})
    if not scalar or all(key == "id" for key in scalar):
        scalar["chat_id"] = chat_id
        scalar["_raw"] = str(overview)[:1000]
    return scalar


def sender_names(
    waha: WahaClient, session: str, chat_id: str, limit: int = 100
) -> dict[str, str]:
    """JID → recent display label, from at most *limit* messages (default 100).

    The group roster carries only JIDs and admin flags; display names
    ride the messages themselves (``_data.notifyName`` per sender).
    Fails soft — an unreadable chat yields no names and the summary
    falls back to bare JIDs.
    """
    try:
        messages = waha.fetch_chat_messages(session, chat_id, limit=limit)
    except Exception as exc:
        logger.debug(
            "sender names unavailable for {chat_id}: {exc}", chat_id=chat_id, exc=exc
        )
        return {}
    return names_from_messages(messages)


def names_from_messages(messages: list[dict[str, Any]]) -> dict[str, str]:
    """JID → display name from chat messages' ``notifyName`` fields.

    Shared by the roster enrichment (``participant_names``) and the
    chat-summary path (``sender_names``): the first name a sender's
    message carries wins, JID objects are normalized via
    :func:`jid_string` (LID groups report participants as objects), and
    entries without a name are dropped.
    """
    names: dict[str, str] = {}
    for message in messages:
        jid = jid_string(message.get("participant"))
        name = str(message.get("_data", {}).get("notifyName") or "").strip()
        if jid and name and jid not in names:
            names[jid] = name
            name_book.learn(jid, name)
    return names


def chat_scalars(overview: dict[str, Any]) -> dict[str, Any]:
    """Available scalar chat fields, preserving known false/zero values."""
    return {
        key: overview[key]
        for key in (
            "id",
            "name",
            "isReadOnly",
            "isGroup",
            "muted",
            "archived",
            "pinned",
            "unreadCount",
        )
        if overview.get(key) is not None and overview.get(key) != ""
    }


def add_participant_summary(
    scalar: dict[str, Any],
    overview: dict[str, Any],
    names: dict[str, str],
) -> None:
    """Add a nonempty roster count and up to 20 extractable id/name pairs.

    Names come from *names* (recent senders); roster entries without a
    known name keep the bare JID. An absent/empty roster omits both
    fields; omission does not establish that the chat has no members.
    """
    participants = roster_entries(overview)
    if not participants:
        return
    scalar["participants"] = len(participants)
    jids = [jid for jid in map(participant_jid, participants) if jid]
    pairs = [roster_pair(jid, names) for jid in jids]
    if 0 < len(pairs) <= 20:
        scalar["participant_list"] = pairs


def roster_pair(jid: str, names: dict[str, str]) -> dict[str, str]:
    """One ``participant_list`` entry: the JID, plus its name when known."""
    name = display_name(jid, names)
    return {"id": jid, "name": name} if name else {"id": jid}


def participant_jid(participant: Any) -> str:
    """The JID of a string or roster entry wrapping a string/JID object in id."""
    if isinstance(participant, dict):
        entry: dict[str, Any] = participant
        return jid_string(entry.get("id"))
    return jid_string(participant)


_MENTION_RE = re.compile(r"@([\dA-Za-z.-]+@[\da-z.-]+|\d{6,})")


def mention_tokens(text: str) -> list[str]:
    """The ``@``-tokens of *text* that can name a chat member.

    Each is a bare user part (``@111222333444555``) or a full JID
    (``@111222333444555@lid``); ``@`` followed by anything else is
    ordinary prose and never a mention.
    """
    return _MENTION_RE.findall(text)


def resolve_mentions(text: str, roster: list[str]) -> list[str]:
    """Roster JIDs named by *text*'s ``@``-tokens, in token order.

    A token matches a roster JID when their user parts are equal: the
    chat writes mentions as bare LID/phone user parts, so
    ``@111222333444555`` resolves against ``111222333444555@lid``.
    Unresolved tokens are left out — inventing a mention for a
    non-member would tag nobody (or worse, the wrong person in another
    chat's namespace).
    """
    by_user = {jid.split("@", 1)[0]: jid for jid in roster}
    resolved: list[str] = []
    for token in dict.fromkeys(mention_tokens(text)):
        jid = by_user.get(token.split("@", 1)[0])
        if jid is not None and jid not in resolved:
            resolved.append(jid)
    return resolved


def dangling_mentions(text: str, roster: list[str]) -> list[str]:
    """*text*'s ``@``-tokens naming nobody on *roster*, in token order.

    The counterpart of :func:`resolve_mentions`: the tokens it drops.
    Only numeric or full-JID-shaped tokens are recognized, including a
    stylized ``@L@s``. Bare ``@Lorenzo`` is not a token. This checks
    auto-resolution against the roster, not explicit mention JIDs or
    notification delivery.
    """
    by_user = {jid.split("@", 1)[0] for jid in roster}
    return [
        token
        for token in dict.fromkeys(mention_tokens(text))
        if token.split("@", 1)[0] not in by_user
    ]


def chat_roster(waha: WahaClient, session: str, chat_id: str) -> list[str]:
    """Best-effort participant JIDs; may be empty for DMs or unreadable chats.

    Two sources, both fail-soft — the roster only enriches mentions, so
    an unavailable one must not break the send itself. The overview's
    ``groupMetadata.participants`` carries phone JIDs, but LID groups
    speak ``@lid`` in message ``participant`` fields (verified against
    WAHA: a 34-member LID group's overview roster shows zero ``@lid``
    entries, while its messages carry only ``@lid`` participants), so
    recent senders are merged in to bridge the two namespaces.
    """
    roster: list[str] = []
    try:
        overview = waha.get_chat_overview(session, chat_id)
    except Exception:
        overview = None
    if overview:
        roster.extend(
            jid for jid in map(participant_jid, roster_entries(overview)) if jid
        )
    try:
        messages = waha.fetch_chat_messages(session, chat_id, limit=50)
    except Exception:
        messages = []
    for message in messages:
        jid = jid_string(message.get("participant"))
        if jid:
            roster.append(jid)
    return list(dict.fromkeys(roster))


_FULL_JID_TOKEN_RE = re.compile(r"@(\d{6,})@(?:lid|c\.us|s\.whatsapp\.net)\b")

_NAME_TOKEN_RE = re.compile(r"(?<![\w@/])@([^\W\d_]{2,}(?:[ \u00a0][^\W\d_]{2,})?)")


def fold_name(name: str) -> str:
    """Case- and accent-insensitive comparison key for a display name."""
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def roster_name_index(roster: list[str], own: set[str]) -> dict[str, set[str]]:
    """Folded full name / first name → roster JIDs, other members only.

    The shared account's own JIDs are excluded: tagging "@kAI" or the
    operator would tag the account itself, which notifies nobody.
    """
    index: dict[str, set[str]] = {}
    for jid in roster:
        if jid in own:
            continue
        name = name_book.name(jid)
        if not name:
            continue
        folded = fold_name(name)
        index.setdefault(folded, set()).add(jid)
        first = folded.split()[0] if folded.split() else ""
        if first and first != folded:
            index.setdefault(first, set()).add(jid)
    return index


def prepare_mentions(
    waha: WahaClient, session: str, text: str, roster: list[str]
) -> tuple[str, list[str]]:
    """Rewrite *text*'s mention tokens into the shape WhatsApp tags.

    Three repairs, all conservative:

    1. ``@123@lid`` → ``@123``: the JID tail would stay visible text.
    2. ``@Ana`` → ``@<digits>`` when exactly one other roster member is
       known by that full or first name; ambiguous or unknown names are
       left untouched (they stay plain text, as before).
    3. A numeric token missing from the roster is checked against
       WAHA's LID↔phone map, so a member who has not spoken recently
       (LID groups list members by phone number, messages by LID) is
       still taggable.

    Returns the rewritten text and the roster extended with any JID
    the LID map confirmed.
    """
    text = _FULL_JID_TOKEN_RE.sub(lambda m: f"@{m.group(1)}", text)
    own = {jid for jid in (_status_state.operator_jid, _status_state.operator_lid) if jid}
    index = roster_name_index(roster, own)
    if index:

        def by_name(match: re.Match[str]) -> str:
            words = match.group(1)
            for candidate in (words, words.split()[0]):
                jids = index.get(fold_name(candidate))
                if jids and len(jids) == 1:
                    (jid,) = jids
                    rest = words[len(candidate) :]
                    return f"@{jid.split('@', 1)[0]}{rest}"
            return match.group(0)

        text = _NAME_TOKEN_RE.sub(by_name, text)
    extended = list(roster)
    phones = {jid.split("@", 1)[0] for jid in roster if jid.endswith("@c.us")}
    for token in dangling_mentions(text, roster):
        user = token.split("@", 1)[0]
        if not user.isdigit():
            continue
        phone = waha.lid_phone(session, f"{user}@lid")
        if phone and phone.split("@", 1)[0] in phones:
            extended.append(f"{user}@lid")
    return text, extended


def ordered_merge(explicit: list[str], resolved: list[str]) -> list[str]:
    """Explicit mention JIDs first, then newly resolved ones; no dupes.

    The model's own ``mentions`` list wins on ordering so the envelope
    reports what it asked for first; resolver additions follow.
    """
    return list(dict.fromkeys([*explicit, *resolved]))


def resolve_explicit_mentions(
    waha: WahaClient,
    session: str,
    chat_id: str,
    mentions: list[str],
    roster: list[str],
) -> tuple[list[str], list[str]]:
    """Resolve explicit ``mentions`` names to JIDs; return the unresolved.

    A JID or numeric id passes through. A name resolves only when exactly
    one named member matches — two members sharing a name tag neither, and
    the name rides back as a warning so the model can quote the ``#m<n>``
    record instead. Roster reads fail soft: entries stay unresolved.
    """
    if not mentions:
        return [], []
    try:
        names = sender_names(waha, session, chat_id)
    except Exception:
        return [], [mention for mention in mentions if "@" not in mention]
    members = [{"id": jid, "name": display_name(jid, names)} for jid in roster]
    resolved: list[str] = []
    unresolved: list[str] = []
    for mention in mentions:
        jid, missing = resolve_one_mention(mention, members)
        if jid:
            resolved.append(jid)
        elif missing:
            unresolved.append(missing)
    return resolved, unresolved


def resolve_one_mention(mention: str, members: list[dict[str, str]]) -> tuple[str, str]:
    """One explicit mention as ``(jid, unresolved_name)``; one side is empty."""
    value = mention.strip()
    if not value:
        return "", ""
    if "@" in value or value.isdigit():
        return value, ""
    needle = value.casefold()
    hits = [m for m in members if m["name"] and m["name"].casefold() == needle]
    return (hits[0]["id"], "") if len(hits) == 1 else ("", value)


def resolve_in_current_chat(
    waha: WahaClient, session: str, chat_id: str, name: str
) -> str:
    """Match a name — or digits/JID — against the current chat's members.

    Members come from the overview roster plus recent senders (LID
    groups list the roster by phone JID but speak by LID), named by
    the overview, recent messages and the webhook-learned name book.
    Digits or a JID match the member's id directly, so the model can
    ask who a bare ``@<digits>`` is. Each match carries the ``tag`` to write in a
    reply. A miss can mean an unnamed member, not absence.
    """
    try:
        roster = chat_roster(waha, session, chat_id)
        names = sender_names(waha, session, chat_id)
    except Exception as exc:
        return error(f"could not read the chat's participants: {exc}")
    members = [{"id": jid, "name": display_name(jid, names)} for jid in roster]
    query = name.strip().lstrip("@")
    digits = query.split("@", 1)[0]
    hits = (
        members_by_id(members, digits)
        if digits.isdigit()
        else [m for m in search_matches(members, query) if m["name"]]
    )
    if not hits:
        return error(
            f"no match for {name!r} among this chat's named members"
            + " (resolving other chats is reserved to the operator)"
        )
    return ok(name=name, matches=[tagged(m) for m in hits[:_RESOLVE_CHAT_CANDIDATES]])


def members_by_id(members: list[dict[str, str]], digits: str) -> list[dict[str, str]]:
    """Members whose JID user part is *digits*, else a name-book hit."""
    hits = [m for m in members if m["id"].split("@", 1)[0] == digits]
    if hits:
        return hits
    jid = f"{digits}@lid"
    learned = name_book.name(jid)
    return [{"id": jid, "name": learned}] if learned else []


def tagged(member: dict[str, str]) -> dict[str, str]:
    """A resolve match with the ``@<digits>`` tag to write in a reply."""
    return {**member, "tag": "@" + member["id"].split("@", 1)[0]}


RECENT_CHATS_CAP = 30

_RESOLVE_CHAT_CANDIDATES = 5


def search_matches(entries: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """Chat/contact entries whose name matches *name*, best first.

    Exact (case-insensitive) name matches come first, then substring
    matches; each keeps only its ``id`` and ``name``. The list is capped
    at ``_RESOLVE_CHAT_CANDIDATES``.
    """
    needle = name.casefold()
    pairs = [
        {"id": jid_string(e.get("id", "")), "name": str(e.get("name", ""))}
        for e in entries
        if e.get("id")
    ]
    exact = [p for p in pairs if p["name"].casefold() == needle]
    partial = [p for p in pairs if needle in p["name"].casefold() and p not in exact]
    return (exact + partial)[:_RESOLVE_CHAT_CANDIDATES]
