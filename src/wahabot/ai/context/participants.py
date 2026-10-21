from typing import Any

from loguru import logger

from wahabot.ai.tools.whatsapp import (
    sender_names,
)
from wahabot.core.cache import TtlCache
from wahabot.core.jid import roster_entries
from wahabot.core.notebook import (
    KIND_RULE,
    Notebook,
    notebook_operation,
)
from wahabot.core.waha import WahaClient

ROSTER_TTL_S = 3600

_ROSTER_CAP = 1000

roster_cache: TtlCache[tuple[str, str], dict[str, str]] = TtlCache(
    ROSTER_TTL_S, _ROSTER_CAP
)


def participant_names(
    waha: WahaClient | None, session: str, chat_id: str
) -> dict[str, str]:
    """JID → display name for a chat's participants, cached per chat.

    Two sources, both fail-soft: the overview roster first, then — for
    every roster JID still missing a name (LID groups carry bare JIDs
    only) — a backfill from the chat's recent messages via
    :func:`sender_names`, whose ``notifyName`` walk is the only place
    WAHA surfaces display names there. Any error yields fewer names;
    quoted senders fall back to their bare id. Non-group chats skip the
    lookup — a DM partner's name already rides the sender tag.
    """
    if waha is None or not chat_id.endswith("@g.us"):
        return {}
    cached = roster_cache.get((session, chat_id))
    if cached is None:
        cached = fetch_participant_names(waha, session, chat_id)
        roster_cache.put((session, chat_id), cached)
    return cached


def fetch_participant_names(
    waha: WahaClient, session: str, chat_id: str
) -> dict[str, str]:
    """The roster's names plus names on recent messages (uncached)."""
    try:
        names = roster_names(waha.get_chat_overview(session, chat_id))
    except Exception as exc:
        logger.debug(
            "Participant roster fetch failed for {chat}: {exc}", chat=chat_id, exc=exc
        )
        names = {}
    # Roster names win; the message walk only backfills JIDs the roster
    # left bare — a recent sender's notifyName is better than a raw
    # number, never worse than a roster entry.
    for jid, name in sender_names(waha, session, chat_id).items():
        names.setdefault(jid, name)
    return names


def roster_names(overview: dict[str, Any]) -> dict[str, str]:
    """Extract JID → name from a chat overview's participant list."""
    pairs = (roster_entry(entry) for entry in roster_entries(overview))
    return {jid: name for jid, name in pairs if jid and name}


def roster_entry(entry: Any) -> tuple[str, str]:
    """One participant's ``(jid, display_name)``; empty strings when absent."""
    if not isinstance(entry, dict):
        return "", ""
    item: dict[str, Any] = entry
    jid = str(item.get("id") or "")
    name = str(
        item.get("name") or item.get("pushname") or item.get("notifyName") or ""
    ).strip()
    return jid, name


async def chat_rules(
    notebook: Notebook | None, session: str, chat_id: str
) -> dict[str, str]:
    """The chat's `rule` notes keyed by key; fail-soft to ``{}``.

    Rules are the deterministic seams the code applies — the text
    prefix and the hard mute. An unreadable store yields no rules,
    never a blocked run.
    """
    if notebook is None:
        return {}
    try:
        notes = await notebook_operation(notebook.read, session, chat_id, kind=KIND_RULE)
    except Exception:
        logger.warning("Chat rules unreadable; continuing without them")
        return {}
    return {note.key: note.text for note in notes}
