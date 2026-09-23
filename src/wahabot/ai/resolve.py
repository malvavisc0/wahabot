"""Deterministic chat resolution for operator commands naming a chat.

The misdelivered-reply incident (docs/bug-report-wrong-message-voice-note.md):
asked to "reply to the last message in <chat>", the model answered the
most recent turn of its shared operator history — a different chat's
message — because a flat, channel-agnostic memory has no per-chat
"last message" to find. This module removes the guesswork *before* the
model starts reasoning: when a command names a chat, code resolves the
name against the operator's chat list (contacts as fallback — the same
precedence :func:`~wahabot.ai.tools.whatsapp.resolve_chat` uses) and
fetches that chat's real last message, so the run starts from the
correct target instead of inferring it from mixed history.

Resolution matches by *evidence, not phrasing*: any chat/contact name
appearing in the instruction text is a candidate, the longest name wins
(a chat literally named in the command outranks a one-word substring),
and "último/last message" keywords never gate anything — a command that
names a chat gets that chat pinned, whatever wording it used. Empty
outcome when nothing matches: the command then runs exactly as before.
"""

from dataclasses import dataclass
from typing import Any

from loguru import logger

from wahabot.ai.messages import jid_string
from wahabot.ai.tools.whatsapp import slim_message
from wahabot.core.waha import WahaClient

__all__ = ["chat_context_note", "resolve_last_message"]


@dataclass(frozen=True)
class ResolvedChat:
    """What a command's named chat resolved to, with its last message.

    ``candidates`` carries the full match list when the name was
    ambiguous (several chats share it); the note then presents all of
    them and leaves the *choice* to the model — with the JIDs in hand,
    so it no longer has to guess.
    """

    chat_id: str
    chat_name: str
    last_message: dict[str, Any] | None = None
    candidates: tuple[dict[str, Any], ...] = ()


def resolve_last_message(
    waha: WahaClient, session: str, instruction: str
) -> ResolvedChat | None:
    """The chat an operator command names, with its real last message.

    Searches the operator's chat list first, then the contact book —
    mirroring the operator ``resolve_chat`` path, since only operator
    commands reach this code. The longest name found wins so a
    two-word chat name beats a coincidental one-word substring of
    another chat's name. On a hit, the chat's recent messages are
    fetched and the newest slimmed message kept; the fetch fails soft
    (an unreadable chat still resolves, just without a pinned message).
    ``None`` when the instruction names no known chat.
    """
    named = named_chats(waha, session, instruction)
    if not named:
        return None
    best = named[0]
    chat_id, chat_name = jid_string(best["id"]), str(best["name"])
    outcome = ResolvedChat(
        chat_id=chat_id,
        chat_name=chat_name,
        candidates=tuple(named) if len(named) > 1 else (),
    )
    try:
        messages = waha.fetch_chat_messages(session, chat_id, limit=5)
    except Exception as exc:
        logger.warning(
            "Could not fetch last message of {chat} for the pinned note: {exc}",
            chat=chat_id,
            exc=exc,
        )
        return outcome
    if messages:
        outcome = ResolvedChat(
            chat_id=chat_id,
            chat_name=chat_name,
            last_message=slim_message(messages[0]),
            candidates=outcome.candidates,
        )
    return outcome


def named_chats(waha: WahaClient, session: str, instruction: str) -> list[dict[str, Any]]:
    """Chat/contact entries whose name literally appears in *instruction*.

    The operator's whole chat list is the search space (contacts as
    fallback — including when the chat list is unreachable, mirroring
    the operator ``resolve_chat`` path), filtered to names the
    instruction actually contains — case-insensitive, longest name
    first, so the most specific mention wins and ambiguous matches
    surface instead of silently resolving.
    """
    text = instruction.casefold()
    entries: list[dict[str, Any]] = []
    try:
        entries = waha.list_chats(session)
    except Exception as exc:
        logger.warning("Chat list fetch failed for command pinning: {exc}", exc=exc)
    matches = _entries_named(entries, text)
    if not matches:
        try:
            matches = _entries_named(waha.list_contacts(session), text)
        except Exception as exc:
            logger.warning("Contact list fetch failed for pinning: {exc}", exc=exc)
    matches.sort(key=lambda m: len(m["name"]), reverse=True)
    return matches[:5]


def _entries_named(entries: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    """The subset of *entries* whose name appears in the lowercased *text*."""
    return [
        {"id": jid_string(e.get("id", "")), "name": str(e.get("name", ""))}
        for e in entries
        if jid_string(e.get("id")) and str(e.get("name", "")).casefold() in text
    ]


def chat_context_note(outcome: ResolvedChat) -> str:
    """The pinned turn note for a resolved chat, rendered for the prompt.

    One bracketed block (the prompt's marker convention: guidance the
    model reads but never repeats verbatim) carrying the resolved chat,
    its real last message — id included, so ``reply_to`` can quote it —
    and, when several chats matched, the candidate list so the model
    can pick and say which. The block is the command's ground truth for
    "this chat": the model no longer infers the target from its mixed
    history.
    """
    header = (
        f"[chat context] The command names the chat: {outcome.chat_name!r} "
        f"(`{outcome.chat_id}`)."
    )
    lines = [header]
    if outcome.candidates:
        others = ", ".join(f"{m['name']} (`{m['id']}`)" for m in outcome.candidates[1:])
        lines.append(f"Other chats also match that name: {others}.")
    if outcome.last_message is not None:
        message = outcome.last_message
        sender = str(message.get("from", "")) or str(message.get("participant", ""))
        body = str(message.get("body", "")).strip()
        described = body or ("media message" if message.get("hasMedia") else "(no text)")
        last_line = (
            f"Its last message is id `{message.get('id', '')}` "
            f"from `{sender}`: {described}"
        )
        lines.append(last_line)
        reply_rule = (
            "Reply to that message (pass its id as `reply_to`) unless the "
            "instruction says otherwise — do not infer the target from "
            "conversation history."
        )
        lines.append(reply_rule)
    else:
        fallback = (
            "Its last message could not be fetched — read the chat with "
            "`fetch_chat_messages` before replying there."
        )
        lines.append(fallback)
    return "\n" + "\n".join(lines)
