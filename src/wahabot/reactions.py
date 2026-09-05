"""Reaction handling: feed reactions to the bot's messages into memory.

Serialized message ids answer the "is this ours?" question for free:
``true_<chat>_<msgid>`` marks our own messages, ``false_…`` everyone
else's (see :func:`chat_id_from_message_id`, which parses the same
shape). Checking the prefix skips the WAHA fetch for the majority of
reactions in busy groups; the fetched ``fromMe`` field stays the
authoritative check for the rare ambiguous id.

Reactions to the bot's own messages become a lightweight memory note in
that chat's context — not an agent run (a 👍 must not wake the LLM):
the next real turn sees the reaction in its history. One note per
(chat, target message), latest wins, so ten 👍 on one message cannot
dilute the memory buffer. Reactions to other people's messages stay
ignored.
"""

import asyncio
from typing import Any

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from loguru import logger

from wahabot.core.models import WahaEvent
from wahabot.core.waha import WahaClient
from wahabot.webhook import on_reaction

#: Latest reaction note per target message, so a second reaction to the
#: same target replaces the first note instead of stacking near-duplicates.
_last_reaction_notes: dict[tuple[str, str, str], str] = {}


def is_own_message_id(message_id: str) -> bool:
    """True when the serialized id marks the message as sent by us."""
    return message_id.startswith("true_")


def register_reaction_handler(
    waha: WahaClient,
    remember: Any,
) -> None:
    """Log reactions to the bot's messages and fold them into memory.

    *remember* is ``handlers.append_to_memory`` — passed in to avoid a
    handlers→reactions import cycle.
    """

    @on_reaction
    async def log_reaction(event: WahaEvent) -> None:
        reaction: dict[str, Any] = event.payload.get("reaction", {})
        target_id = reaction.get("messageId", "")
        if not target_id:
            logger.debug(
                "Reaction without a target id from {sender}",
                sender=event.payload.get("from"),
            )
            return
        if not is_own_message_id(str(target_id)):
            logger.debug(
                "Ignoring reaction to someone else's message {target}",
                target=target_id,
            )
            return
        message = await asyncio.to_thread(fetch_target, waha, event.session, target_id)
        if message is None or not message.get("fromMe"):
            return
        emoji = str(reaction.get("text", "")).strip() or "(removed)"
        preview = message_preview(message)
        # In groups `from` is the group JID and `participant` the actual
        # reactor — the note must name the person, not the room.
        sender = str(
            event.payload.get("participant") or event.payload.get("from") or "someone"
        )
        chat_id = chat_id_from_message_id(str(target_id))
        note = f"[reaction {emoji} from {sender} to your message: {preview}]"
        await remember_reaction_note(event.session, chat_id, target_id, note, remember)
        logger.info(
            'Reaction {emoji} to "{preview}" from {sender}',
            emoji=emoji,
            preview=preview,
            sender=sender,
        )


async def remember_reaction_note(
    session: str,
    chat_id: str,
    target_id: str,
    note: str,
    remember: Any,
) -> None:
    """Fold one reaction note into the chat's memory, latest per target.

    A prior note for the same target is dropped from the buffer first so
    ten 👍 on one message stay one note; the emoji of the latest
    reaction wins.
    """
    key = (session, chat_id, target_id)
    previous = _last_reaction_notes.get(key)
    if previous:
        await forget_memory_message(session, chat_id, previous)
    _last_reaction_notes[key] = note
    await remember(
        session,
        chat_id,
        agent=None,
        message=ChatMessage(role=MessageRole.USER, content=note),
    )


async def forget_memory_message(session: str, chat_id: str, content: str) -> None:
    """Remove the superseded reaction note from the chat's memory."""
    from wahabot.handlers import contexts

    ctx = contexts.get((session, chat_id))
    if ctx is None:
        return
    memory = await ctx.store.get("memory", default=None)
    if memory is None:
        return
    messages = await memory.aget_all()
    kept = [m for m in messages if str(m.content) != content]
    if len(kept) != len(messages):
        await memory.aset(kept)


def fetch_target(waha: WahaClient, session: str, target_id: str) -> dict[str, Any] | None:
    """Fetch the reacted-to message, or None if it cannot be retrieved.

    Fail-soft on every error: reaction logging is best-effort, and an
    exception here would 500 the webhook (WAHA would then redeliver
    the reaction event forever).
    """
    try:
        return waha.get_message(session, chat_id_from_message_id(target_id), target_id)
    except Exception as exc:
        logger.warning(
            "Could not fetch message reacted to {target}: {exc}",
            target=target_id,
            exc=exc,
        )
        return None


def chat_id_from_message_id(message_id: str) -> str:
    """Return the chat JID embedded in a serialized message id.

    Serialized ids have the form ``{fromMe}_{chat}_{message_id}[_{participant}]``
    and chat JIDs never contain underscores, so the chat is the second segment.
    """
    parts = message_id.split("_")
    return parts[1] if len(parts) > 1 else message_id


def message_preview(payload: dict[str, Any]) -> str:
    """Return a short human-readable preview of a message, for logs."""
    body = str(payload.get("body", "")).strip()
    if body:
        return body if len(body) <= 80 else body[:77] + "..."
    data = payload.get("_data", {})
    kind = data.get("type")
    return f"[{kind}]" if kind else "[media]"
