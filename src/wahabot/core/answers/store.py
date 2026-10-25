"""SQLite storage for answers, FTS retrieval, uses and candidates."""

import re
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

from wahabot.core.answers.models import Answer, AnswerCandidate, AnswerHit
from wahabot.core.db.connection import connect, transaction
from wahabot.core.tickets.store_base import StoreBase

_ANSWER_COLUMNS = "id, question, answer, enabled, created_at, updated_at, deleted_at"
_CANDIDATE_COLUMNS = "id, ticket_id, question, answer, created_at, saved_at"
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def answer_from_row(row: sqlite3.Row) -> Answer:
    """Map one answers row."""
    return Answer(
        id=int(row["id"]),
        question=str(row["question"]),
        answer=str(row["answer"]),
        enabled=bool(row["enabled"]),
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        deleted_at=int(row["deleted_at"]) if row["deleted_at"] is not None else None,
    )


def candidate_from_row(row: sqlite3.Row) -> AnswerCandidate:
    """Map one answer_candidates row."""
    return AnswerCandidate(
        id=int(row["id"]),
        ticket_id=int(row["ticket_id"]),
        question=str(row["question"]),
        answer=str(row["answer"]),
        created_at=int(row["created_at"]),
        saved_at=int(row["saved_at"]) if row["saved_at"] is not None else None,
    )


def fts_query(query: str) -> str:
    """A safe FTS5 disjunction of the query's words."""
    words = [word.casefold() for word in _WORD.findall(query) if len(word) > 1]
    return " OR ".join(f'"{word}"' for word in dict.fromkeys(words))


def like_query(query: str) -> str:
    """A literal LIKE pattern for a syntax-error fallback."""
    return f"%{query.strip()}%"


class AnswerStore(StoreBase):
    """One connection per operation; FTS triggers keep the index in sync."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        super().__init__(path, clock)

    def add(self, question: str, answer: str, *, at: int | None = None) -> Answer:
        """Insert one enabled answer."""
        moment = self.now() if at is None else at
        conn = connect(self.path)
        try:
            with transaction(conn):
                cursor = conn.execute(
                    "INSERT INTO answers (question, answer, enabled, created_at, "
                    + "updated_at) VALUES (?, ?, 1, ?, ?)",
                    (question.strip(), answer.strip(), moment, moment),
                )
                row = conn.execute(
                    f"SELECT {_ANSWER_COLUMNS} FROM answers WHERE id = ?",
                    (int(cursor.lastrowid or 0),),
                ).fetchone()
        finally:
            conn.close()
        return answer_from_row(row)

    def get(self, answer_id: int, *, include_deleted: bool = False) -> Answer | None:
        """One answer by id."""
        sql = f"SELECT {_ANSWER_COLUMNS} FROM answers WHERE id = ?"
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        return self.query_one(sql, (answer_id,), answer_from_row)

    def list(self, *, include_deleted: bool = False) -> list[Answer]:
        """Every answer, newest first, for the Teach list."""
        sql = f"SELECT {_ANSWER_COLUMNS} FROM answers"
        if not include_deleted:
            sql += " WHERE deleted_at IS NULL"
        sql += " ORDER BY id DESC"
        return self.query_all(sql, (), answer_from_row)

    def enabled(self) -> list[Answer]:
        """Enabled, non-deleted answers in stable id order."""
        return self.query_all(
            f"SELECT {_ANSWER_COLUMNS} FROM answers "
            + "WHERE enabled = 1 AND deleted_at IS NULL ORDER BY id",
            (),
            answer_from_row,
        )

    def update(
        self,
        answer_id: int,
        *,
        question: str | None = None,
        answer: str | None = None,
        enabled: bool | None = None,
        at: int | None = None,
    ) -> Answer:
        """Update one answer's mutable fields."""
        moment = self.now() if at is None else at
        fields = ["updated_at = ?"]
        params: list[object] = [moment]
        if question is not None:
            fields.append("question = ?")
            params.append(question.strip())
        if answer is not None:
            fields.append("answer = ?")
            params.append(answer.strip())
        if enabled is not None:
            fields.append("enabled = ?")
            params.append(int(enabled))
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    f"UPDATE answers SET {', '.join(fields)} WHERE id = ?",
                    (*params, answer_id),
                )
                row = conn.execute(
                    f"SELECT {_ANSWER_COLUMNS} FROM answers WHERE id = ?",
                    (answer_id,),
                ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(f"answer {answer_id} not found")
        return answer_from_row(row)

    def soft_delete(self, answer_id: int, *, at: int | None = None) -> Answer:
        """Mark one answer deleted without losing it."""
        moment = self.now() if at is None else at
        self.execute(
            "UPDATE answers SET deleted_at = ?, updated_at = ? WHERE id = ?",
            (moment, moment, answer_id),
        )
        found = self.get(answer_id, include_deleted=True)
        if found is None:
            raise KeyError(f"answer {answer_id} not found")
        return found

    def restore(self, answer_id: int, *, at: int | None = None) -> Answer:
        """Clear one answer's deletion marker."""
        moment = self.now() if at is None else at
        self.execute(
            "UPDATE answers SET deleted_at = NULL, updated_at = ? WHERE id = ?",
            (moment, answer_id),
        )
        found = self.get(answer_id)
        if found is None:
            raise KeyError(f"answer {answer_id} not found")
        return found

    def latest_deleted(self) -> Answer | None:
        """The most recently soft-deleted answer, when one exists."""
        return self.query_one(
            f"SELECT {_ANSWER_COLUMNS} FROM answers WHERE deleted_at IS NOT NULL "
            + "ORDER BY deleted_at DESC, id DESC LIMIT 1",
            (),
            answer_from_row,
        )

    def undo_delete(self, answer_id: int | None = None) -> Answer | None:
        """Restore the named deletion, or the latest one."""
        target = (
            self.get(answer_id, include_deleted=True)
            if answer_id is not None
            else self.latest_deleted()
        )
        if target is None or target.deleted_at is None:
            return None
        return self.restore(target.id)

    def search(self, query: str, *, limit: int = 5) -> list[AnswerHit]:
        """Enabled answers matching *query*, best first; LIKE on FTS errors."""
        text = fts_query(query)
        if not text:
            return []
        try:
            return self.query_all(
                "SELECT answers.id, answers.question, answers.answer, answers.enabled, "
                + "answers.created_at, answers.updated_at, answers.deleted_at, "
                + "bm25(answers_fts) AS rank FROM answers_fts "
                + "JOIN answers ON answers.id = answers_fts.rowid "
                + "WHERE answers_fts MATCH ? AND answers.enabled = 1 "
                + "AND answers.deleted_at IS NULL ORDER BY rank LIMIT ?",
                (text, limit),
                lambda row: AnswerHit(
                    answer=answer_from_row(row), rank=float(row["rank"])
                ),
            )
        except sqlite3.OperationalError:
            return self.search_like(query, limit=limit)

    def search_like(self, query: str, *, limit: int = 5) -> list[AnswerHit]:
        """Literal fallback for an FTS syntax error."""
        pattern = like_query(query)
        if pattern == "%%":
            return []
        return self.query_all(
            f"SELECT {_ANSWER_COLUMNS}, 0.0 AS rank FROM answers "
            + "WHERE enabled = 1 AND deleted_at IS NULL "
            + "AND (question LIKE ? OR answer LIKE ?) ORDER BY id LIMIT ?",
            (pattern, pattern, limit),
            lambda row: AnswerHit(answer=answer_from_row(row), rank=float(row["rank"])),
        )

    def record_use(self, answer_id: int, *, at: int | None = None) -> None:
        """Append one answer-use ledger row."""
        self.execute(
            "INSERT INTO answer_uses (answer_id, at) VALUES (?, ?)",
            (answer_id, self.now() if at is None else at),
        )

    def uses_count(self, answer_id: int) -> int:
        """How many times one answer has been returned."""
        found = self.query_one(
            "SELECT COUNT(*) AS total FROM answer_uses WHERE answer_id = ?",
            (answer_id,),
            lambda row: int(row["total"]),
        )
        return found or 0

    def add_candidate(
        self, ticket_id: int, question: str, answer: str, *, at: int | None = None
    ) -> AnswerCandidate:
        """Store one unsaved candidate from an operator reply."""
        moment = self.now() if at is None else at
        conn = connect(self.path)
        try:
            with transaction(conn):
                cursor = conn.execute(
                    "INSERT INTO answer_candidates (ticket_id, question, answer, "
                    + "created_at) VALUES (?, ?, ?, ?)",
                    (ticket_id, question.strip(), answer.strip(), moment),
                )
                row = conn.execute(
                    f"SELECT {_CANDIDATE_COLUMNS} FROM answer_candidates WHERE id = ?",
                    (int(cursor.lastrowid or 0),),
                ).fetchone()
        finally:
            conn.close()
        return candidate_from_row(row)

    def latest_candidate(self, ticket_id: int | None = None) -> AnswerCandidate | None:
        """The newest candidate that has not been saved, optionally per ticket."""
        if ticket_id is None:
            return self.query_one(
                f"SELECT {_CANDIDATE_COLUMNS} FROM answer_candidates "
                + "WHERE saved_at IS NULL ORDER BY id DESC LIMIT 1",
                (),
                candidate_from_row,
            )
        return self.query_one(
            f"SELECT {_CANDIDATE_COLUMNS} FROM answer_candidates "
            + "WHERE saved_at IS NULL AND ticket_id = ? ORDER BY id DESC LIMIT 1",
            (ticket_id,),
            candidate_from_row,
        )

    def candidate(self, candidate_id: int) -> AnswerCandidate | None:
        """One candidate by id."""
        return self.query_one(
            f"SELECT {_CANDIDATE_COLUMNS} FROM answer_candidates WHERE id = ?",
            (candidate_id,),
            candidate_from_row,
        )

    def save_candidate(
        self, candidate_id: int, question: str, answer: str, *, at: int | None = None
    ) -> Answer | None:
        """Atomically create one answer and stamp its candidate as saved."""
        moment = self.now() if at is None else at
        conn = connect(self.path)
        try:
            with transaction(conn):
                claimed = conn.execute(
                    "UPDATE answer_candidates SET saved_at = ? "
                    + "WHERE id = ? AND saved_at IS NULL",
                    (moment, candidate_id),
                )
                if claimed.rowcount != 1:
                    return None
                cursor = conn.execute(
                    "INSERT INTO answers (question, answer, enabled, created_at, "
                    + "updated_at) VALUES (?, ?, 1, ?, ?)",
                    (question, answer, moment, moment),
                )
                row = conn.execute(
                    f"SELECT {_ANSWER_COLUMNS} FROM answers WHERE id = ?",
                    (int(cursor.lastrowid or 0),),
                ).fetchone()
        finally:
            conn.close()
        return answer_from_row(row)
