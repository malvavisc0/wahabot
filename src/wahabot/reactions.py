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

The memory fold runs under the chat's run lock (the WAHA fetch stays outside
it) and is persisted like any other fold, so a reaction to a bot
message in an LRU-evicted chat reloads from disk instead of being
silently dropped.
"""

import asyncio
from typing import Any

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from loguru import logger

from wahabot.ai.workflow import FunctionCallingAgentWorkflow
from wahabot.core.models import WahaEvent
from wahabot.core.waha import WahaClient
from wahabot.handlers import append_to_memory, chat_lock, context_for, persist_memory
from wahabot.settings import Settings
from wahabot.webhook import on_reaction

#: Latest reaction note per target message, so a second reaction to the
#: same target replaces the first note instead of stacking near-duplicates.
_last_reaction_notes: dict[tuple[str, str, str], str] = {}


def is_own_message_id(message_id: str) -> bool:
    """True when the serialized id marks the message as sent by us."""
    return message_id.startswith("true_")


def register_reaction_handler(
    waha: WahaClient,
    agent: FunctionCallingAgentWorkflow,
    settings: Settings,
) -> None:
    """Log reactions to the bot's messages and fold them into memory.

    The WAHA fetch of the reacted-to message runs outside the chat's
    run lock (a network call must never extend the serialized section);
    only the memory fold locks (against that chat's runs), and the fold
    is persisted like any other.
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
        async with chat_lock(event.session, chat_id):
            await remember_reaction_note(
                event.session, chat_id, target_id, note, agent, settings
            )
        logger.info(
            'Reaction {emoji} to "{preview}" from {sender}',
            emoji=emoji,
            preview=preview,
            sender=sender,
        )


#: Key carrying the reaction target id on a folded note message; the
#: superseded-note removal filters on it, so a note survives merging
#: with a neighboring turn (content equality would miss it there).
_REACTION_TARGET_KWARG = "reaction_target_id"


async def remember_reaction_note(
    session: str,
    chat_id: str,
    target_id: str,
    note: str,
    agent: FunctionCallingAgentWorkflow,
    settings: Settings,
) -> None:
    """Fold one reaction note into the chat's memory, latest per target.

    A prior note for the same target is dropped from the buffer first so
    ten 👍 on one message stay one note; the emoji of the latest
    reaction wins. The fold is persisted like any other memory change.
    """
    key = (session, chat_id, target_id)
    previous = _last_reaction_notes.get(key)
    if previous:
        await forget_reaction_notes(session, chat_id, target_id, agent, settings)
    _last_reaction_notes[key] = note
    ctx = await append_to_memory(
        session,
        chat_id,
        agent,
        settings,
        ChatMessage(
            role=MessageRole.USER,
            content=note,
            additional_kwargs={_REACTION_TARGET_KWARG: target_id},
        ),
    )
    await persist_memory(settings, session, chat_id, ctx)


async def forget_reaction_notes(
    session: str,
    chat_id: str,
    target_id: str,
    agent: FunctionCallingAgentWorkflow,
    settings: Settings,
) -> None:
    """Remove every note reacting to *target_id* from the chat's memory.

    The buffer may hold one note for the target or — once a run's
    sanitize pass has merged it into a neighboring user turn — a
    message whose content merely contains the note text. Filtering on
    the tagged kwarg instead of content equality removes both shapes
    without ever touching words around them.
    """
    ctx = await context_for(session, chat_id, agent, settings)
    memory = await ctx.store.get("memory", default=None)
    if memory is None:
        return
    messages = await memory.aget_all()

    def targets(msg: ChatMessage) -> Any:
        return msg.additional_kwargs.get(_REACTION_TARGET_KWARG)

    kept = [m for m in messages if targets(m) != target_id]
    if len(kept) != len(messages):
        if not kept or kept[0].role != MessageRole.USER:
            # Dropping the note left a history that no longer starts
            # with a user turn; sanitize would drop the leading
            # messages on the next run anyway — better to keep the
            # superseded note than to orphan everything after it.
            return
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
