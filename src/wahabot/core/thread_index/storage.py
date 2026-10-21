import json
import os
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any, cast

from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
from llama_index.core.vector_stores.simple import SimpleVectorStoreData
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from loguru import logger

from wahabot.core.thread_index.constants import (
    FTS_SCHEMA,
    PENDING_CHAR_LIMIT,
    PENDING_TURN_LIMIT,
    THREAD_QUERY_CHAR_LIMIT,
    TOKEN_RE,
    TURNS_SCHEMA,
)
from wahabot.core.thread_index.models import (
    EmbeddingModel,
    ThreadTurn,
    index_file,
    vector_file,
)


def bound_pending(pending: list[ThreadTurn]) -> None:
    """Drop the oldest pending turns past the batch caps."""
    total = sum(len(turn.text) for turn in pending)
    while pending and (len(pending) > PENDING_TURN_LIMIT or total > PENDING_CHAR_LIMIT):
        total -= len(pending.pop(0).text)


def insert_turns(
    data_dir: Path, session: str, chat_id: str, turns: Sequence[ThreadTurn]
) -> None:
    """Insert turns and their FTS rows; identical content is ignored."""
    path = index_file(data_dir, session, chat_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(TURNS_SCHEMA)
        connection.execute(FTS_SCHEMA)
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
    terms = TOKEN_RE.findall(query[:THREAD_QUERY_CHAR_LIMIT])[:32]
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
