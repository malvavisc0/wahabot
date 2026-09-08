"""Per-chat run serialization and the shared agent-context table.

Chat runs and operator commands both need the same three services:
acquiring a chat's rolling conversation context (LRU-bounded, lazily
restored from disk), serializing runs that share one memory buffer,
and persisting a buffer at run end. They live here — not in
``handlers`` — so both the message path and the command path import
them without a module cycle (handlers lazily imports commands for the
self-chat command path).
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from llama_index.core.workflow import Context
from loguru import logger

from wahabot.ai.workflow import FunctionCallingAgentWorkflow
from wahabot.core.persistence import load_memory, save_memory
from wahabot.settings import Settings

__all__ = ["chat_lock", "context_for", "persist_memory"]

#: Keep at most this many per-chat agent contexts; least recently used
#: chats are evicted (their conversation memory is dropped).
_MAX_CONTEXTS = 1000

#: The shared context table. Keyed by ``(session, chat_id)`` — the
#: operator's shared history lives under the ``operator`` chat id.
contexts: dict[tuple[str, str], Context] = {}

#: Per-chat run serialization: runs in DIFFERENT chats proceed in
#: parallel (a slow group turn never delays a DM), while two runs in
#: the SAME chat queue — they share one memory buffer and one
#: conversation timeline, so interleaving them would corrupt both.
#: Keyed like ``contexts``; operator commands lock their own
#: ``operator`` key, never a chat's.
_chat_locks: dict[tuple[str, str], asyncio.Lock] = {}

#: Acquisitions in flight per key (queued or holding). ``Lock.locked``
#: is already False in the window between ``release()`` and the queued
#: waiter actually resuming, so a lock with a waiter would look
#: evictable there — the pending count is the only witness.
_chat_lock_pending: dict[tuple[str, str], int] = {}
_MAX_CHAT_LOCKS = 1000


@asynccontextmanager
async def chat_lock(session: str, chat_id: str) -> AsyncIterator[None]:
    """Serialize agent runs for one chat (``async with chat_lock(…)``).

    Eviction keeps the table bounded, idle entries going oldest-first;
    a held or waited-on lock is never evicted (the waiter-window
    invariant, see docs/agent-workflow.md). When every entry is busy
    the table may temporarily exceed its cap rather than spin — the
    cap is a memory bound, not an invariant.
    """
    key = (session, chat_id)
    lock = _chat_locks.pop(key, None)
    if lock is None:
        lock = asyncio.Lock()
    _chat_locks[key] = lock
    _chat_lock_pending[key] = _chat_lock_pending.get(key, 0) + 1
    for oldest in list(_chat_locks):
        if len(_chat_locks) <= _MAX_CHAT_LOCKS:
            break
        if _chat_locks[oldest].locked() or _chat_lock_pending.get(oldest, 0):
            continue
        del _chat_locks[oldest]
    try:
        async with lock:
            yield
    finally:
        pending = _chat_lock_pending.get(key, 0) - 1
        if pending > 0:
            _chat_lock_pending[key] = pending
        else:
            _chat_lock_pending.pop(key, None)


async def context_for(
    session: str,
    chat_id: str,
    agent: FunctionCallingAgentWorkflow,
    settings: Settings,
) -> Context:
    """The per-chat agent context, evicting stale chats past the cap.

    Every touch moves the chat to the end (most recently used);
    inserts past the cap drop the oldest entry. On a miss the chat's
    memory is lazily restored from disk (when persistence is on), so an
    LRU-evicted chat reloads its history instead of starting blank.
    """
    key = (session, chat_id)
    ctx = contexts.pop(key, None)
    if ctx is None:
        ctx = Context(agent)
        if settings.memory_persist:
            memory = load_memory(settings.data_dir, session, chat_id)
            if memory is not None:
                await ctx.store.set("memory", memory)
                logger.info(
                    "Restored memory for {chat_id}: {count} messages",
                    chat_id=chat_id,
                    count=len(memory.get_all()),
                )
    contexts[key] = ctx
    while len(contexts) > _MAX_CONTEXTS:
        oldest = next(iter(contexts))
        del contexts[oldest]
    return ctx


async def persist_memory(
    settings: Settings, session: str, chat_id: str, ctx: Context
) -> None:
    """Write the chat's run-end buffer to disk; a no-op when disabled.

    The buffer is the coherent snapshot of a run (sanitized, trimmed,
    delivery groups collapsed). Write failures are logged and swallowed
    inside ``save_memory`` so a disk problem never breaks a reply.
    """
    if not settings.memory_persist:
        return
    memory = await ctx.store.get("memory", default=None)
    if memory is None:
        return
    save_memory(settings.data_dir, session, chat_id, memory)
