import asyncio
from collections.abc import Callable
from typing import Any, cast

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Context
from loguru import logger

from wahabot.ai.messages import (
    OPERATOR_FOLD_KWARG,
    OPERATOR_PAYLOAD_FLAG,
    conversation_jid,
    jid_string,
    name_mentioned,
    sent_by_bot_api,
    typed_by_operator,
)
from wahabot.ai.records import ledger_turn
from wahabot.ai.serials import serial_store
from wahabot.ai.turn_records import (
    RecordEntry,
    render_entries,
)
from wahabot.ai.workflow import FunctionCallingAgentWorkflow
from wahabot.core import identity
from wahabot.core.access import SessionConfig
from wahabot.core.identity import AUTHOR_BOT, AUTHOR_OPERATOR
from wahabot.core.models import WahaEvent
from wahabot.core.persistence import memory_file
from wahabot.core.runs import (
    chat_lock,
    context_for,
    contexts,
    persist_memory,
)
from wahabot.settings import Settings


async def append_to_memory(
    session: str,
    chat_id: str,
    agent: FunctionCallingAgentWorkflow,
    settings: Settings,
    message: ChatMessage,
) -> Context:
    """Append one message to a chat's memory buffer, no agent run.

    Shared fold-in path for out-of-band turns (operator-typed ``fromMe``
    messages, reaction notes): the model's self-history should reflect
    them, but a fold must never wake the LLM. Returns the chat's context
    so the caller can persist the fold at its save point.
    """
    ctx = await context_for(session, chat_id, agent, settings)
    memory = await ctx.store.get("memory", default=None)
    if memory is None:
        from_defaults = cast(
            Callable[..., ChatMemoryBuffer], ChatMemoryBuffer.from_defaults
        )
        memory = from_defaults(token_limit=agent.memory_token_limit, llm=agent.llm)
    await memory.aput(message)
    await ctx.store.set("memory", memory)
    return ctx


async def remember_own_message(
    event: WahaEvent,
    agent: FunctionCallingAgentWorkflow,
    settings: Settings,
    body: str | None,
) -> Context | None:
    """Fold an operator-typed message into the chat's memory, no run.

    The human operator writes from the bot's own account. Those words
    are a teammate's, not the bot's: they are stored as a *user* turn
    tagged ``[Name <jid>] [operator message] …`` — the same shape a
    waking operator turn renders — so the model always knows who said
    what: it must never take the operator's lines for its own, nor
    answer them as if a stranger wrote them. A mention of the bot
    wakes a real run instead (``reply_with_agent``).

    *event* is already normalized (``from`` is the chat). Chats with no
    conversation yet — neither in memory nor on disk — fold nothing.
    Returns the chat's context so the caller can persist the fold.
    """
    if not body:
        logger.debug(
            "Skipping operator message {id}: no text to fold",
            id=event.payload.get("id"),
        )
        return None
    chat_id = str(event.payload.get("from", ""))
    if not chat_id:
        return None
    if (event.session, chat_id) not in contexts and not memory_file(
        settings.data_dir, event.session, chat_id
    ).exists():
        logger.info(
            "Skipping operator message {id} in {chat_id}: no conversation yet",
            id=event.payload.get("id"),
            chat_id=chat_id,
        )
        return None
    # One shape for operator lines whether they wake the bot or not: a
    # record with the <operator> speaker token, never a JID tag.
    registry = serial_store(str(settings.data_dir)).for_chat(event.session, chat_id)
    content, _, _ = await asyncio.to_thread(
        render_entries,
        [RecordEntry(event=event, body=body)],
        registry=registry,
        names={},
        tz_name=settings.timezone,
    )
    ctx = await append_to_memory(
        event.session,
        chat_id,
        agent,
        settings,
        ChatMessage(
            role=MessageRole.USER,
            content=ledger_turn(content),
            additional_kwargs={OPERATOR_FOLD_KWARG: True},
        ),
    )
    logger.info(
        "Remembered operator message {id} in {chat}",
        id=event.payload.get("id"),
        chat=chat_id,
    )
    return ctx


def operator_wake(event: WahaEvent, config: SessionConfig) -> bool | None:
    """Whether an operator message wakes the bot; None means skip it.

    A ``fromMe`` message with no explicit source cannot be attributed
    safely: folding the bot's own echo as the operator's would make it
    answer itself. A known operator message is normalized to the chat
    it belongs to, then only a mention of the bot wakes the agent.
    """
    if not event.payload.get("fromMe"):
        return False
    if not typed_by_operator(event):
        logger.debug(
            "Skipping own message {id} (source={source})",
            id=str(event.payload.get("id", "")),
            source=event.payload.get("source"),
        )
        return None
    normalize_operator_event(event)
    return name_mentioned(event, config.bot_name, config.bot_mention_regex)


async def fold_operator_message(
    event: WahaEvent,
    agent: FunctionCallingAgentWorkflow,
    settings: Settings,
    body: str | None,
) -> None:
    """Store an operator-typed message in the chat's memory, no run."""
    chat = str(event.payload.get("from", ""))
    async with chat_lock(event.session, chat):
        ctx = await remember_own_message(event, agent, settings, body)
        if ctx is not None:
            await persist_memory(settings, event.session, chat, ctx)


def normalize_operator_event(event: WahaEvent) -> None:
    """Reshape an operator-typed ``fromMe`` message like any member's.

    WAHA reports a ``fromMe`` message with ``from`` = the account and
    ``to`` = the chat. Everything downstream (gates, memory, sender
    tags, runs) keys on ``from`` as the chat, so the chat moves into
    ``from`` and the account's own JID becomes the ``participant`` —
    exactly the shape a member's group message has. The flag marks the
    turn as the operator's for rendering.
    """
    payload = event.payload
    me = event.me or {}
    chat = conversation_jid(event)
    own = jid_string(me.get("lid")) or jid_string(me.get("id"))
    payload["from"] = chat
    if own:
        payload["participant"] = own
    payload[OPERATOR_PAYLOAD_FLAG] = True


def learn_identity(event: WahaEvent) -> None:
    """Record who wrote *event* and its sender's display name.

    The display name (``notifyName``) and WAHA's ``source`` exist only
    in the live webhook — history fetches carry neither — so both are
    captured here, for quotes, reactions and ``read_chat`` later.
    """
    payload = event.payload
    data = cast(dict[str, Any], payload.get("_data") or {})
    message_id = str(payload.get("id", ""))
    if payload.get("fromMe"):
        if sent_by_bot_api(event):
            identity.authors.record(message_id, AUTHOR_BOT)
        elif typed_by_operator(event):
            identity.authors.record(message_id, AUTHOR_OPERATOR)
        return
    sender = jid_string(payload.get("participant") or data.get("author"))
    if not sender and not str(payload.get("from", "")).endswith("@g.us"):
        sender = jid_string(payload.get("from"))
    name = str(data.get("notifyName") or payload.get("notifyName") or "")
    identity.names.learn(sender, name)
