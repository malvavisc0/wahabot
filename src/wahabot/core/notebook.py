"""Transactional notes kept independently of rolling conversation history."""

import asyncio
import json
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from wahabot.core.tokens import text_tokens

MAX_NOTE_KEY_LENGTH = 80
MAX_NOTE_LENGTH = 2000
MAX_READ_NOTES = 20
SESSION_NOTEBOOK_CHAT = "@session"
SESSION_POLICY_TOKEN_LIMIT = 1024

#: Marker attribute set on tools whose calls must finish their SQLite
#: work before a canceled run releases its chat lock (see
#: ``notebook_operation``): cancelling the awaited call early would let
#: the next run's write interleave with this one's unfinished commit.
#: Lives here so both the tool builder and the workflow dispatcher read
#: it without an import cycle.
HOLDS_CHAT_LOCK = "holds_chat_lock"


@dataclass(frozen=True, slots=True)
class Note:
    """A named, durable fact or lesson belonging to one conversation."""

    key: str
    text: str


class Notebook:
    """Persist session/chat-scoped notes with atomic SQLite transactions.

    Connections belong to individual operations, so the same notebook can
    serve tool threads, concurrent chats, and a restarted bot safely.
    """

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "notebook.sqlite3"

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Open the database and commit or roll back one operation."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS notes (
                    session TEXT NOT NULL, chat_id TEXT NOT NULL,
                    key TEXT NOT NULL, text TEXT NOT NULL, updated_at INTEGER NOT NULL,
                    PRIMARY KEY (session, chat_id, key))"""
            )
            yield connection

    def read(
        self,
        session: str,
        chat_id: str,
        key: str = "",
        query: str = "",
        limit: int = MAX_READ_NOTES,
    ) -> list[Note]:
        """Read newest matching notes, with literal case-insensitive search."""
        if not 1 <= limit <= MAX_READ_NOTES:
            raise ValueError(f"Notebook limit must be between 1 and {MAX_READ_NOTES}")
        needle = query.strip().casefold()
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT key, text FROM notes WHERE session = ? AND chat_id = ?
                    AND (? = '' OR key = ?) ORDER BY updated_at DESC, key""",
                (session, chat_id, key.strip(), key.strip()),
            )
            notes = (Note(key=str(row[0]), text=str(row[1])) for row in rows)
            matches = (
                note for note in notes if needle in f"{note.key}\n{note.text}".casefold()
            )
            return list(islice(matches, limit))

    def write(
        self,
        session: str,
        chat_id: str,
        key: str,
        text: str,
        *,
        max_total_tokens: int | None = None,
    ) -> None:
        """Replace a note, atomically enforcing an optional whole-notebook budget."""
        if chat_id == SESSION_NOTEBOOK_CHAT:
            max_total_tokens = SESSION_POLICY_TOKEN_LIMIT
        key = valid_note_key(key)
        text = text.strip()
        if not text or len(text) > MAX_NOTE_LENGTH:
            raise ValueError(
                f"Notebook text must contain 1 to {MAX_NOTE_LENGTH} characters"
            )
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO notes (session, chat_id, key, text, updated_at)
                    VALUES (?, ?, ?, ?, ?) ON CONFLICT (session, chat_id, key)
                    DO UPDATE SET text = excluded.text,
                        updated_at = excluded.updated_at""",
                (session, chat_id, key, text, time.time_ns()),
            )
            if max_total_tokens is not None:
                rows = connection.execute(
                    """SELECT key, text FROM notes WHERE session = ? AND chat_id = ?
                        ORDER BY updated_at DESC, key""",
                    (session, chat_id),
                ).fetchall()
                contents = [{"key": str(row[0]), "text": str(row[1])} for row in rows]
                if (
                    len(contents) > MAX_READ_NOTES
                    or text_tokens(json.dumps(contents, ensure_ascii=False))
                    > max_total_tokens
                ):
                    raise ValueError(
                        "Session notebook is full; consolidate or delete policies "
                        + "before adding more"
                    )

    def scopes(self, session: str) -> list[tuple[str, int]]:
        """Every chat scope with at least one note, newest first.

        The operator-facing listing (``wahabot notebook list``): one
        row per chat_id with its note count, empty scopes omitted.
        """
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT chat_id, COUNT(*) FROM notes WHERE session = ?
                    GROUP BY chat_id ORDER BY MAX(updated_at) DESC""",
                (session,),
            )
            return [(str(row[0]), int(row[1])) for row in rows]

    def delete(self, session: str, chat_id: str, key: str) -> bool:
        """Delete one named note, returning whether it existed."""
        key = valid_note_key(key)
        with self.connection() as connection:
            result = connection.execute(
                "DELETE FROM notes WHERE session = ? AND chat_id = ? AND key = ?",
                (session, chat_id, key),
            )
            return result.rowcount > 0

    def forget(self, session: str, chat_id: str) -> None:
        """Delete every note in exactly one session/chat scope."""
        with self.connection() as connection:
            connection.execute(
                "DELETE FROM notes WHERE session = ? AND chat_id = ?",
                (session, chat_id),
            )


def valid_note_key(key: str) -> str:
    """Normalize a note key and reject empty or oversized names."""
    key = key.strip()
    if not key or len(key) > MAX_NOTE_KEY_LENGTH:
        raise ValueError(
            f"Notebook key must contain 1 to {MAX_NOTE_KEY_LENGTH} characters"
        )
    return key


async def notebook_operation[T](
    operation: Callable[..., T], *args: Any, **kwargs: Any
) -> T:
    """Finish a database operation before releasing a canceled run's chat lock."""
    task = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return task.result()
