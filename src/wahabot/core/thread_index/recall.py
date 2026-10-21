import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from wahabot.ai.history import INBOUND_ID_RE
from wahabot.core.thread_index.constants import (
    RRF_K,
    THREAD_QUERY_CHAR_LIMIT,
)
from wahabot.core.thread_index.models import index_file


def recall_query(messages: Sequence[ChatMessage]) -> str:
    """The latest user text, capped for embedding/FTS cost."""
    for message in reversed(messages):
        if message.role == MessageRole.USER:
            text = INBOUND_ID_RE.sub("", str(message.content or "")).strip()
            return text[:THREAD_QUERY_CHAR_LIMIT]
    return ""


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]]) -> list[str]:
    """Reciprocal-rank fusion of the FTS and dense rankings."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, node_id in enumerate(ranking):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (RRF_K + rank + 1)
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
