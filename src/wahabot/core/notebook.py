"""Transactional notes kept independently of rolling conversation history."""

import asyncio
import json
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

from wahabot.core.tokens import text_tokens

MAX_NOTE_KEY_LENGTH = 80
MAX_NOTE_LENGTH = 2000
MAX_READ_NOTES = 20
SESSION_NOTEBOOK_CHAT = "@session"
SESSION_POLICY_TOKEN_LIMIT = 1024
#: A chat's `rule` notes together are capped here; a write beyond it is
#: refused, so the authoritative part of the block stays small.
CHAT_RULE_TOKEN_LIMIT = 512
#: The snapshot budget for notebook content riding a run's prompt —
#: shared by the notebook tool's reads and the pinned state block so
#: the two surfaces can never drift apart.
NOTEBOOK_SNAPSHOT_TOKEN_LIMIT = 1024

#: A note's kind. `rule` is a standing instruction for its chat;
#: `fact` is a durable chat-local fact or preference; `lesson` is an
#: operational workaround learned by doing.
KIND_RULE = "rule"
KIND_FACT = "fact"
KIND_LESSON = "lesson"
NOTE_KINDS = (KIND_RULE, KIND_FACT, KIND_LESSON)
#: Who set a note. Derived from the run target by the tool, never
#: supplied by the model.
AUTHOR_OPERATOR = "operator"
AUTHOR_MEMBER = "member"
AUTHOR_SYSTEM = "system"
AUTHOR_KINDS = (AUTHOR_OPERATOR, AUTHOR_MEMBER, AUTHOR_SYSTEM)

#: Rule keys a component reads and applies deterministically. Only
#: behaviours with a safe deterministic transform belong here; anything
#: judgement-shaped stays model-obeyed through the prompt.
RULE_MESSAGE_PREFIX = "chat-message-prefix"
RULE_REPLY_MUTE_PREFERENCES = "chat-reply-mute-preferences"
RULE_HARD_MUTE = "chat-hard-mute"
#: Rule keys a non-operator run may not write or delete.
RESERVED_RULE_KEYS = frozenset(
    (RULE_MESSAGE_PREFIX, RULE_REPLY_MUTE_PREFERENCES, RULE_HARD_MUTE)
)

#: Marker attribute set on tools whose calls must finish their SQLite
#: work before a canceled run releases its chat lock (see
#: ``notebook_operation``): cancelling the awaited call early would let
#: the next run's write interleave with this one's unfinished commit.
#: Lives here so both the tool builder and the workflow dispatcher read
#: it without an import cycle.
HOLDS_CHAT_LOCK = "holds_chat_lock"


@dataclass(frozen=True, slots=True)
class Note:
    """A named, durable note belonging to one conversation."""

    key: str
    text: str
    kind: str = field(default=KIND_FACT, compare=False)
    author_kind: str = field(default=AUTHOR_MEMBER, compare=False)
    created_at: int = field(default=0, compare=False)
    updated_at: int = field(default=0, compare=False)


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
                    key TEXT NOT NULL, text TEXT NOT NULL,
                    kind TEXT NOT NULL, author_kind TEXT NOT NULL,
                    created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
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
        kind: str = "",
    ) -> list[Note]:
        """Read newest matching notes, with literal case-insensitive search."""
        if not 1 <= limit <= MAX_READ_NOTES:
            raise ValueError(f"Notebook limit must be between 1 and {MAX_READ_NOTES}")
        if kind and kind not in NOTE_KINDS:
            raise ValueError(f"Notebook kind must be one of {NOTE_KINDS}")
        needle = query.strip().casefold()
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT key, text, kind, author_kind,
                        created_at, updated_at
                    FROM notes WHERE session = ? AND chat_id = ?
                    AND (? = '' OR key = ?) ORDER BY updated_at DESC, key""",
                (session, chat_id, key.strip(), key.strip()),
            )
            notes = (note_from_row(row) for row in rows)
            matches = (
                note
                for note in notes
                if (not kind or note.kind == kind)
                and needle in f"{note.key}\n{note.text}".casefold()
            )
            return list(islice(matches, limit))

    def write(
        self,
        session: str,
        chat_id: str,
        key: str,
        text: str,
        *,
        kind: str = KIND_FACT,
        author_kind: str = AUTHOR_MEMBER,
        max_total_tokens: int | None = None,
    ) -> None:
        """Replace a note, atomically enforcing an optional whole-notebook budget."""
        if chat_id == SESSION_NOTEBOOK_CHAT:
            # Operator policies: one bounded, kindless scope.
            kind, author_kind = KIND_RULE, AUTHOR_OPERATOR
            max_total_tokens = SESSION_POLICY_TOKEN_LIMIT
        if kind not in NOTE_KINDS:
            raise ValueError(f"Notebook kind must be one of {NOTE_KINDS}")
        if author_kind not in AUTHOR_KINDS:
            raise ValueError(f"Notebook author must be one of {AUTHOR_KINDS}")
        key = valid_note_key(key)
        text = text.strip()
        if not text or len(text) > MAX_NOTE_LENGTH:
            raise ValueError(
                f"Notebook text must contain 1 to {MAX_NOTE_LENGTH} characters"
            )
        now = time.time_ns()
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO notes
                    (session, chat_id, key, text, kind, author_kind,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (session, chat_id, key)
                    DO UPDATE SET text = excluded.text, kind = excluded.kind,
                        author_kind = excluded.author_kind,
                        updated_at = excluded.updated_at""",
                (session, chat_id, key, text, kind, author_kind, now, now),
            )
            if max_total_tokens is not None:
                enforce_scope_budget(connection, session, chat_id, max_total_tokens)
            if kind == "rule" and chat_id != SESSION_NOTEBOOK_CHAT:
                enforce_rule_budget(connection, session, chat_id)

    def scopes(self, session: str | None = None) -> list[tuple[str, str, int]]:
        """Every scope holding notes — (session, chat_id, count), newest first.

        The operator-facing listing (``wahabot notebook``): chats are
        scoped by session — one server can run several — so the
        session rides every row. Pass *session* to filter to one bot
        instance; None lists across all of them.
        """
        with self.connection() as connection:
            if session is None:
                rows = connection.execute(
                    """SELECT session, chat_id, COUNT(*) FROM notes
                        GROUP BY session, chat_id
                        ORDER BY MAX(updated_at) DESC"""
                )
            else:
                rows = connection.execute(
                    """SELECT session, chat_id, COUNT(*) FROM notes WHERE session = ?
                        GROUP BY session, chat_id
                        ORDER BY MAX(updated_at) DESC""",
                    (session,),
                )
            return [(str(row[0]), str(row[1]), int(row[2])) for row in rows]

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


def note_from_row(row: tuple[Any, ...]) -> Note:
    """One notes row as a :class:`Note`."""
    return Note(
        key=str(row[0]),
        text=str(row[1]),
        kind=str(row[2]),
        author_kind=str(row[3]),
        created_at=int(row[4]),
        updated_at=int(row[5]),
    )


def enforce_scope_budget(
    connection: sqlite3.Connection, session: str, chat_id: str, max_total_tokens: int
) -> None:
    """Refuse a write that pushes a whole notebook scope past its budget."""
    rows = connection.execute(
        """SELECT key, text FROM notes WHERE session = ? AND chat_id = ?
            ORDER BY updated_at DESC, key""",
        (session, chat_id),
    ).fetchall()
    contents = [{"key": str(row[0]), "text": str(row[1])} for row in rows]
    if (
        len(contents) > MAX_READ_NOTES
        or text_tokens(json.dumps(contents, ensure_ascii=False)) > max_total_tokens
    ):
        raise ValueError(
            "Session notebook is full; consolidate or delete policies "
            + "before adding more"
        )


def enforce_rule_budget(
    connection: sqlite3.Connection, session: str, chat_id: str
) -> None:
    """Refuse a chat rule that pushes the chat's rules past their cap."""
    rows = connection.execute(
        "SELECT text FROM notes WHERE session = ? AND chat_id = ? AND kind = 'rule'",
        (session, chat_id),
    ).fetchall()
    total = sum(text_tokens(str(row[0])) for row in rows)
    if total > CHAT_RULE_TOKEN_LIMIT:
        raise ValueError(
            f"chat rules exceed {CHAT_RULE_TOKEN_LIMIT} tokens; "
            + "delete or shorten a rule before adding more"
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
