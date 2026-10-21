import asyncio
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.vector_stores import SimpleVectorStore
from loguru import logger

from wahabot.core.persistence import persistable
from wahabot.core.thread_index.constants import (
    DRAIN_ROUNDS,
    EMBED_BATCH,
    EMBED_RETRY_S,
    EMBED_TEXT_CHARS,
    EMBED_TEXT_CHARS_RETRY,
    MAX_CACHED_STORES,
    OPERATOR_CHAT_ID,
    THREAD_QUERY_CHAR_LIMIT,
    THREAD_TOP_K,
    THREAD_WINDOW,
)
from wahabot.core.thread_index.models import (
    EmbeddingModel,
    ThreadTurn,
    turn_from_message,
    vector_file,
)
from wahabot.core.thread_index.recall import reciprocal_rank_fusion, slices
from wahabot.core.thread_index.storage import (
    bound_pending,
    dense_ids,
    fts_ids,
    insert_turns,
    read_store,
    store_vectors,
    unembedded_turns,
)


class ThreadIndexer:
    """Records trimmed turns; indexes and recalls them per chat.

    ``schedule`` is pure state: no I/O, no task, no embedding. ``drain``
    is awaited at the persist points (handlers, after the reply) and is
    single-flight per chat: one flush at a time, later schedules replayed
    by the next drain. ``recall`` reads only the requested chat's files
    and returns ready-to-render slices.
    """

    def __init__(
        self,
        data_dir: Path,
        embed_model: EmbeddingModel | None = None,
        *,
        top_k: int = THREAD_TOP_K,
        window: int = THREAD_WINDOW,
        query_timeout: float = 2.0,
    ) -> None:
        self.data_dir = data_dir
        self.embed_model = embed_model
        self.top_k = top_k
        self.window = window
        self.query_timeout = query_timeout
        self._pending: dict[tuple[str, str], list[ThreadTurn]] = {}
        self._unembedded: set[tuple[str, str]] = set()
        self._down_until: dict[tuple[str, str], float] = {}
        self._stores: dict[tuple[str, str], SimpleVectorStore] = {}
        self._store_mtimes: dict[tuple[str, str], int] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._warned: set[tuple[str, str, str]] = set()

    def schedule(
        self, session: str, chat_id: str, messages: Sequence[ChatMessage]
    ) -> None:
        """Record *messages* for a later index; empty/no-op when filtered."""
        if not persistable(chat_id) or chat_id == OPERATOR_CHAT_ID:
            return
        turns = [turn for message in messages if (turn := turn_from_message(message))]
        if not turns:
            return
        key = (session, chat_id)
        pending = self._pending.setdefault(key, [])
        known = {turn.node_id for turn in pending}
        for turn in turns:
            if turn.node_id not in known:
                pending.append(turn)
                known.add(turn.node_id)
        bound_pending(pending)

    def forget(self, session: str, chat_id: str) -> None:
        """Drop in-memory work for a chat whose index files were wiped."""
        key = (session, chat_id)
        self._pending.pop(key, None)
        self._unembedded.discard(key)
        self._down_until.pop(key, None)
        self._stores.pop(key, None)
        self._store_mtimes.pop(key, None)
        self._warned = {marker for marker in self._warned if marker[:2] != key}

    async def drain(self) -> None:
        """Flush every pending chat; a failed write is retried next time."""
        processed: set[tuple[str, str]] = set()
        for _ in range(DRAIN_ROUNDS):
            if not self._pending:
                break
            items = list(self._pending.items())
            self._pending.clear()
            failed = False
            for key, turns in items:
                processed.add(key)
                if not await self.flush(key, turns):
                    self._pending.setdefault(key, []).extend(turns)
                    bound_pending(self._pending[key])
                    failed = True
            if failed:
                break
        for key in list(self._unembedded):
            if key in processed:
                continue
            await self.flush(key, [])
        if self._pending:
            logger.warning("Thread recall drain left work pending; retrying next time")

    async def recall(
        self, session: str, chat_id: str, query: str, limit: int | None = None
    ) -> list[str]:
        """Hybrid FTS5 + dense recall for one chat; [] on any failure."""
        query = query.strip()[:THREAD_QUERY_CHAR_LIMIT]
        if not query or not persistable(chat_id) or chat_id == OPERATOR_CHAT_ID:
            return []
        key = (session, chat_id)
        count = limit or self.top_k
        fts_ids = self.safe_fts(key, session, chat_id, query, count * 3)
        dense_ids = await self.safe_dense(key, session, chat_id, query, count * 3)
        fused = reciprocal_rank_fusion([fts_ids, dense_ids])[:count]
        if not fused:
            return []
        try:
            return slices(self.data_dir, session, chat_id, fused, self.window)
        except Exception as exc:
            self.warn_once(
                key,
                "slices",
                "Thread recall slice read failed for {chat_id}: {exc}",
                chat_id=chat_id,
                exc=exc,
            )
            return []

    async def flush(self, key: tuple[str, str], turns: list[ThreadTurn]) -> bool:
        """Insert one chat's turns, then embed a bounded batch of them.

        Returns False only when the turn write failed; the caller
        requeues that batch so a transient lock or disk error does not
        silently lose the trim. Embedding failures are fail-soft and do
        not requeue the already-written turns.
        """
        session, chat_id = key
        async with self.lock(key):
            try:
                if turns:
                    insert_turns(self.data_dir, session, chat_id, turns)
            except Exception as exc:
                self.warn_once(
                    key,
                    "index",
                    "Thread index write failed for {chat_id}: {exc}",
                    chat_id=chat_id,
                    exc=exc,
                )
                return False
            if self.embed_model is not None:
                await self.embed_pending(key)
            return True

    async def embed_pending(self, key: tuple[str, str]) -> None:
        """Embed up to one batch; leave the rest for the next persist point."""
        session, chat_id = key
        if self.down(key):
            return
        try:
            rows = unembedded_turns(self.data_dir, session, chat_id, EMBED_BATCH)
            if not rows:
                self._unembedded.discard(key)
                return
            self._unembedded.add(key)
            assert self.embed_model is not None
            texts = [f"[{role}] {text[:EMBED_TEXT_CHARS]}" for _, role, text in rows]
            try:
                embeddings = await self.embed_model.aget_text_embedding_batch(texts)
            except Exception:
                # A single long or dense input can exceed the LAN model's
                # 512-token physical batch; retry the whole batch shorter
                # before giving up and leaving FTS-only rows.
                texts = [
                    f"[{role}] {text[:EMBED_TEXT_CHARS_RETRY]}" for _, role, text in rows
                ]
                embeddings = await self.embed_model.aget_text_embedding_batch(texts)
            path = vector_file(self.data_dir, session, chat_id)
            store = self.load_store(key, path)
            store_vectors(self.data_dir, session, chat_id, rows, embeddings, store)
            self._store_mtimes[key] = path.stat().st_mtime_ns
            if unembedded_turns(self.data_dir, session, chat_id, 1):
                self._unembedded.add(key)
            else:
                self._unembedded.discard(key)
            self._down_until.pop(key, None)
            self._warned.discard((*key, "embed"))
        except Exception as exc:
            self._unembedded.add(key)
            self._down_until[key] = time.monotonic() + EMBED_RETRY_S
            self.warn_once(
                key,
                "embed",
                "Thread index embedding failed for {chat_id}; FTS recall remains: {exc}",
                chat_id=chat_id,
                exc=exc,
            )

    def down(self, key: tuple[str, str]) -> bool:
        """Whether the embeddings service is in its post-failure cooldown."""
        return time.monotonic() < self._down_until.get(key, 0.0)

    def lock(self, key: tuple[str, str]) -> asyncio.Lock:
        """The per-chat single-flight lock, created lazily."""
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def safe_fts(
        self,
        key: tuple[str, str],
        session: str,
        chat_id: str,
        query: str,
        limit: int,
    ) -> list[str]:
        """FTS ids for one chat; [] on any failure."""
        try:
            return fts_ids(self.data_dir, session, chat_id, query, limit)
        except Exception as exc:
            self.warn_once(
                key,
                "fts",
                "Thread recall FTS search failed for {chat_id}: {exc}",
                chat_id=chat_id,
                exc=exc,
            )
            return []

    async def safe_dense(
        self,
        key: tuple[str, str],
        session: str,
        chat_id: str,
        query: str,
        limit: int,
    ) -> list[str]:
        """Dense ids for one chat; [] when unavailable or failed."""
        if self.embed_model is None:
            return []
        if self.down(key):
            return []
        path = vector_file(self.data_dir, session, chat_id)
        if not path.exists():
            return []
        try:
            store = self.load_store(key, path)
            return await asyncio.wait_for(
                dense_ids(store, query, limit, self.embed_model),
                timeout=self.query_timeout,
            )
        except Exception as exc:
            self._down_until[key] = time.monotonic() + EMBED_RETRY_S
            self.warn_once(
                key,
                "dense",
                "Thread recall dense search failed for {chat_id}: {exc}",
                chat_id=chat_id,
                exc=exc,
            )
            return []

    def load_store(self, key: tuple[str, str], path: Path) -> SimpleVectorStore:
        """A chat's vector store, cached until its file changes on disk.

        ``SimpleVectorStore.from_persist_path`` validates every node and
        takes seconds for a thousand embeddings, so loading it on every
        run would put that latency in front of the model. The mtime check
        keeps the cache honest if another writer replaces the file.
        """
        if not path.exists():
            self._stores.pop(key, None)
            self._store_mtimes.pop(key, None)
            return SimpleVectorStore()
        mtime = path.stat().st_mtime_ns
        if self._store_mtimes.get(key) == mtime:
            cached = self._stores.get(key)
            if cached is not None:
                self._stores.pop(key)
                self._stores[key] = cached
                return cached
        store = read_store(path)
        self._stores.pop(key, None)
        self._stores[key] = store
        self._store_mtimes[key] = mtime
        while len(self._stores) > MAX_CACHED_STORES:
            oldest = next(iter(self._stores))
            self._stores.pop(oldest, None)
            self._store_mtimes.pop(oldest, None)
        return store

    def warn_once(
        self,
        key: tuple[str, str],
        kind: str,
        message: str,
        **fields: Any,
    ) -> None:
        """Log one failure per chat and kind."""
        marker = (*key, kind)
        if marker in self._warned:
            return
        self._warned.add(marker)
        logger.warning(message, **fields)
