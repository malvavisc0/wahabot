"""Per-chat thread recall: hybrid FTS5 + dense over trimmed-away turns.

Thread recall keeps old transcript slices recoverable after the FIFO
trims them. This module owns the per-chat index: one SQLite file
(turns and an FTS5/BM25 table) and one :class:`SimpleVectorStore` JSON
file per chat, both under ``data/memory/<session>/``. Retrieval never
crosses chats — the chat id is part of every path and every query.

Indexing rides the same trim hook as the Tier 1 recap. ``schedule`` only
records the dropped user/assistant turns in memory; ``drain`` runs at
the persist points, after the reply is out, and outside the chat lock —
never inside the agent run, and never holding the chat's run lock
(embedding must not extend the serialized section).
Content-hash node ids make redeliveries, ``fromMe`` folds and repeated
batches idempotent. Tool results, narration payloads and operator
commands are never indexed. Every failure is logged once and swallowed:
recall is a lead, never a dependency.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import time
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
from llama_index.core.vector_stores.simple import SimpleVectorStoreData
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from loguru import logger

from wahabot.ai.history import INBOUND_ID_RE, narration_kind
from wahabot.ai.messages import WRAP_UP_NOTE_KWARG
from wahabot.core.persistence import persistable

#: The internal operator chat is commands, not conversation: never indexed.
OPERATOR_CHAT_ID = "operator"

#: Query embedding cost stays flat regardless of message size.
THREAD_QUERY_CHAR_LIMIT = 200

#: Hits per recall; each hit expands to a small cache-locality window.
THREAD_TOP_K = 3
THREAD_WINDOW = 2

#: One drain embeds at most this many unembedded turns; the rest waits for
#: the next persist point (bounded work, never an unbounded burst).
EMBED_BATCH = 64

#: The LAN embedding model's physical batch is 512 tokens, so one input
#: must stay well under it. 1000 characters is ~370 tokens for Latin
#: text; the 500-character retry covers denser scripts and punctuation.
#: The full turn stays in SQLite for FTS/slices; only the vector is cut.
EMBED_TEXT_CHARS = 1000
EMBED_TEXT_CHARS_RETRY = 500

#: A failed embeddings service leaves turns unembedded; retry them at the
#: next persist point, not in a tight loop inside one drain.
_DRAIN_ROUNDS = 5

#: After an embeddings failure, skip the service for this long so a down
#: endpoint never adds its timeout to every run.
_EMBED_RETRY_S = 60.0

#: Pending text is bounded like the recap batch: the oldest drops first.
_PENDING_TURN_LIMIT = 500
_PENDING_CHAR_LIMIT = 100_000

#: Bound the in-memory vector-store cache. Per-chat indexes are small,
#: but a bot can see many chats; reloading is cheap enough to evict.
_MAX_CACHED_STORES = 32

_RRF_K = 60
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_TURNS_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS turns (id TEXT PRIMARY KEY, "
    + "ordinal INTEGER NOT NULL UNIQUE, role TEXT NOT NULL, "
    + "text TEXT NOT NULL, embedded INTEGER NOT NULL DEFAULT 0)"
)
_FTS_SCHEMA = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5("
    + "id UNINDEXED, text, tokenize='unicode61')"
)


class EmbeddingModel(Protocol):
    """The two embedding calls the indexer needs, sync-free."""

    async def aget_query_embedding(self, query: str) -> list[float]:
        """Embed one retrieval query."""
        ...

    async def aget_text_embedding_batch(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float]]:
        """Embed a batch of turn texts."""
        ...


@dataclass(frozen=True, slots=True)
class ThreadTurn:
    """One indexable turn: a user or assistant message, never a tool."""

    role: str
    text: str

    @property
    def node_id(self) -> str:
        """Stable content hash: the same turn is always one node."""
        return thread_node_id(self.role, self.text)


def thread_node_id(role: str, text: str) -> str:
    """SHA-256 of ``role + NUL + text`` — the idempotent node key."""
    digest = hashlib.sha256()
    digest.update(role.encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def index_file(data_dir: Path, session: str, chat_id: str) -> Path:
    """Path of a chat's FTS/turn index, next to its memory file."""
    return data_dir / "memory" / session / f"{chat_id}.index.sqlite3"


def vector_file(data_dir: Path, session: str, chat_id: str) -> Path:
    """Path of a chat's dense vector store, next to its memory file."""
    return data_dir / "memory" / session / f"{chat_id}.vectors.json"


def turn_from_message(message: ChatMessage) -> ThreadTurn | None:
    """The indexable turn a message carries, or ``None``.

    Only user/assistant text is indexed. Tool results are raw JSON,
    narration (silence/emoji/error self-reports) is not conversation, and
    a private wrap-up note was never sent to the chat.
    """
    if message.role not in (MessageRole.USER, MessageRole.ASSISTANT):
        return None
    text = INBOUND_ID_RE.sub("", str(message.content or "")).strip()
    if not text or message.additional_kwargs.get(WRAP_UP_NOTE_KWARG):
        return None
    if message.role == MessageRole.ASSISTANT and narration_kind(text):
        return None
    return ThreadTurn(message.role.value, text)


def recall_query(messages: Sequence[ChatMessage]) -> str:
    """The latest user text, capped for embedding/FTS cost."""
    for message in reversed(messages):
        if message.role == MessageRole.USER:
            text = INBOUND_ID_RE.sub("", str(message.content or "")).strip()
            return text[:THREAD_QUERY_CHAR_LIMIT]
    return ""


def forget_thread_index(data_dir: Path, session: str, chat_id: str) -> bool:
    """Delete a chat's thread index files; True when one was removed."""
    if not persistable(chat_id):
        logger.warning("Not forgetting thread index of bad chat id {id!r}", id=chat_id)
        return False
    removed = False
    for path in (
        index_file(data_dir, session, chat_id),
        vector_file(data_dir, session, chat_id),
    ):
        removed = unlink(path) or removed
        removed = unlink(path.with_name(path.name + ".bad")) or removed
        removed = unlink(path.with_name(path.name + ".tmp")) or removed
    return removed


def unlink(path: Path) -> bool:
    """Delete one file; False when it is already absent."""
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning(
            "Failed to delete thread index file {path}: {exc}", path=path, exc=exc
        )
        return False
    return True


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
        for _ in range(_DRAIN_ROUNDS):
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
            self._down_until[key] = time.monotonic() + _EMBED_RETRY_S
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
            self._down_until[key] = time.monotonic() + _EMBED_RETRY_S
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
        while len(self._stores) > _MAX_CACHED_STORES:
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


def bound_pending(pending: list[ThreadTurn]) -> None:
    """Drop the oldest pending turns past the batch caps."""
    total = sum(len(turn.text) for turn in pending)
    while pending and (len(pending) > _PENDING_TURN_LIMIT or total > _PENDING_CHAR_LIMIT):
        total -= len(pending.pop(0).text)


def insert_turns(
    data_dir: Path, session: str, chat_id: str, turns: Sequence[ThreadTurn]
) -> None:
    """Insert turns and their FTS rows; identical content is ignored."""
    path = index_file(data_dir, session, chat_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(_TURNS_SCHEMA)
        connection.execute(_FTS_SCHEMA)
        ordinal = connection.execute(
            "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM turns"
        ).fetchone()[0]
        for turn in turns:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO turns (id, ordinal, role, text) "
                + "VALUES (?, ?, ?, ?)",
                (turn.node_id, ordinal, turn.role, turn.text),
            )
            if not cursor.rowcount:
                continue
            connection.execute(
                "INSERT INTO turns_fts (id, text) VALUES (?, ?)",
                (turn.node_id, turn.text),
            )
            ordinal += 1


def unembedded_turns(
    data_dir: Path, session: str, chat_id: str, limit: int
) -> list[tuple[str, str, str]]:
    """Unembedded ``(id, role, text)`` rows in insertion order."""
    path = index_file(data_dir, session, chat_id)
    if not path.exists():
        return []
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(
            "SELECT id, role, text FROM turns WHERE embedded = 0 "
            + "ORDER BY ordinal LIMIT ?",
            (limit,),
        ).fetchall()
    return [(str(row[0]), str(row[1]), str(row[2])) for row in rows]


def store_vectors(
    data_dir: Path,
    session: str,
    chat_id: str,
    rows: Sequence[tuple[str, str, str]],
    embeddings: Sequence[Sequence[float]],
    store: SimpleVectorStore,
) -> None:
    """Add embedded turns to the chat's vector store, then mark them done."""
    for (node_id, _role, _text), embedding in zip(rows, embeddings, strict=True):
        store.data.embedding_dict[node_id] = list(embedding)
        store.data.text_id_to_ref_doc_id[node_id] = "None"
    path = vector_file(data_dir, session, chat_id)
    persist_store(store, path)
    mark_embedded(data_dir, session, chat_id, [row[0] for row in rows])


def read_store(path: Path) -> SimpleVectorStore:
    """Read a chat's vector store; a corrupt file is moved aside.

    The direct :class:`SimpleVectorStoreData` construction skips
    ``from_persist_path``'s per-node validation, which costs seconds for
    a thousand embeddings. The file is ours and has no metadata filters,
    so the dataclass payload is enough.
    """
    if not path.exists():
        return SimpleVectorStore()
    try:
        with path.open("rb") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("vector store payload is not an object")
        payload = cast(dict[str, Any], raw)
        data = SimpleVectorStoreData(
            embedding_dict=payload.get("embedding_dict") or {},
            text_id_to_ref_doc_id=payload.get("text_id_to_ref_doc_id") or {},
            metadata_dict=payload.get("metadata_dict") or {},
        )
        return SimpleVectorStore(data)
    except Exception as exc:
        logger.warning(
            "Corrupt thread vector store {path}, renamed to .bad: {exc}",
            path=path,
            exc=exc,
        )
        try:
            os.replace(path, path.with_name(path.name + ".bad"))
        except OSError as rename_error:
            logger.warning(
                "Could not rename {path} to .bad: {error}",
                path=path,
                error=rename_error,
            )
        return SimpleVectorStore()


def persist_store(store: SimpleVectorStore, path: Path) -> None:
    """Write the vector store atomically (tmp sibling, then replace).

    ``SimpleVectorStore.persist`` re-validates the whole payload on
    every save; the plain JSON dump is the same format and fast enough
    to run after each small embedding batch.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "embedding_dict": store.data.embedding_dict,
                "text_id_to_ref_doc_id": store.data.text_id_to_ref_doc_id,
                "metadata_dict": store.data.metadata_dict,
            },
            handle,
            separators=(",", ":"),
        )
    os.replace(tmp, path)


def mark_embedded(
    data_dir: Path, session: str, chat_id: str, node_ids: Sequence[str]
) -> None:
    """Mark rows embedded after their vectors are persisted."""
    path = index_file(data_dir, session, chat_id)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.executemany(
            "UPDATE turns SET embedded = 1 WHERE id = ?",
            [(node_id,) for node_id in node_ids],
        )


def fts_ids(
    data_dir: Path, session: str, chat_id: str, query: str, limit: int
) -> list[str]:
    """BM25-ranked node ids for *query*; [] when the index is absent."""
    path = index_file(data_dir, session, chat_id)
    if not path.exists():
        return []
    terms = fts_query(query)
    if not terms:
        return []
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(
            "SELECT id FROM turns_fts WHERE turns_fts MATCH ? "
            + "ORDER BY bm25(turns_fts) LIMIT ?",
            (terms, limit),
        ).fetchall()
    return [str(row[0]) for row in rows]


def fts_query(query: str) -> str:
    """A safe FTS5 OR-query from the query's word tokens."""
    terms = _TOKEN_RE.findall(query[:THREAD_QUERY_CHAR_LIMIT])[:32]
    return " OR ".join(f'"{term}"' for term in terms)


async def dense_ids(
    store: SimpleVectorStore,
    query: str,
    limit: int,
    embed_model: EmbeddingModel,
) -> list[str]:
    """Dense-ranked node ids for *query*; [] when no vectors exist."""
    if not store.data.embedding_dict:
        return []
    embedding = await embed_model.aget_query_embedding(query)
    result = store.query(
        VectorStoreQuery(
            query_embedding=list(embedding),
            similarity_top_k=limit,
            mode=VectorStoreQueryMode.DEFAULT,
        )
    )
    return [str(node_id) for node_id in (result.ids or []) if node_id]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]]) -> list[str]:
    """Reciprocal-rank fusion of the FTS and dense rankings."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, node_id in enumerate(ranking):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores, key=lambda node_id: scores[node_id], reverse=True)


def slices(
    data_dir: Path,
    session: str,
    chat_id: str,
    node_ids: Sequence[str],
    window: int,
) -> list[str]:
    """Expand each hit to its neighbouring turns and flatten to one line."""
    path = index_file(data_dir, session, chat_id)
    if not path.exists():
        return []
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA query_only = ON")
        turn_ordinals = ordinals(connection, node_ids)
        ranges = merge_ranges(
            [(max(0, ordinal - window), ordinal + window) for ordinal in turn_ordinals]
        )
        return [
            text for low, high in ranges if (text := slice_text(connection, low, high))
        ]


def ordinals(connection: sqlite3.Connection, node_ids: Sequence[str]) -> list[int]:
    """The ordinals of the requested node ids, skipping missing rows."""
    ordinals: list[int] = []
    for node_id in node_ids:
        row = connection.execute(
            "SELECT ordinal FROM turns WHERE id = ?", (node_id,)
        ).fetchone()
        if row is not None:
            ordinals.append(int(row[0]))
    return ordinals


def merge_ranges(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent ordinal ranges."""
    merged: list[tuple[int, int]] = []
    for low, high in sorted(ranges):
        if merged and low <= merged[-1][1] + 1:
            previous_low, previous_high = merged[-1]
            merged[-1] = (previous_low, max(previous_high, high))
        else:
            merged.append((low, high))
    return merged


def slice_text(connection: sqlite3.Connection, low: int, high: int) -> str:
    """The turns in one ordinal range as a flattened slice line."""
    rows = connection.execute(
        "SELECT role, text FROM turns WHERE ordinal BETWEEN ? AND ? "
        + "ORDER BY ordinal",
        (low, high),
    ).fetchall()
    return " | ".join(f"{role}: {' '.join(str(text).split())}" for role, text in rows)
