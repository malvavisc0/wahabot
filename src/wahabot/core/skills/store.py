"""SQLite storage for skills, secrets, calls and staged write requests."""

import json
import sqlite3
import time
from pathlib import Path

from wahabot.core.db.connection import connect, transaction
from wahabot.core.skills.models import (
    Skill,
    SkillCall,
    SkillRequest,
    skill_from_row,
)

_SKILL_COLUMNS = (
    "name, lifecycle, enabled, base_urls_json, auth_type, secret_ref, "
    "customer_chats, writes_need_operator, source_url, instructions, skill_md, "
    "draft_stage, draft_inputs_json, test_config_hash, test_state, test_reason, "
    "last_tested_at, failure_count, created_by, created_at, updated_at"
)
_CALL_COLUMNS = (
    "id, skill, ticket_id, chat_id, started_by, method, path, status, "
    "duration_ms, outcome, error, request_json, created_at"
)
_REQUEST_COLUMNS = (
    "id, skill, ticket_id, chat_id, request_json, summary, state, "
    "operator_message_id, created_at, resolved_at"
)
PENDING_REQUEST_TTL_S = 24 * 60 * 60


class SkillStore:
    """CRUD for the four skill tables, one connection per operation."""

    def __init__(self, path: Path, clock: object = time.time) -> None:
        self.path = path
        self.clock = clock

    def now(self) -> int:
        """The store's current wall-clock second."""
        return int(self.clock())  # type: ignore[operator]

    def connect(self) -> sqlite3.Connection:
        """A fresh connection to the platform database."""
        return connect(self.path)

    def upsert(self, skill: Skill) -> Skill:
        """Insert or replace one skill row by name."""
        moment = self.now()
        skill.updated_at = moment
        if not skill.created_at:
            skill.created_at = moment
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    f"INSERT INTO skills ({_SKILL_COLUMNS}) VALUES ("
                    + ",".join("?" for _ in skill_columns())
                    + ") ON CONFLICT(name) DO UPDATE SET "
                    + ", ".join(
                        f"{column}=excluded.{column}"
                        for column in skill_columns()
                        if column not in ("name", "created_at")
                    ),
                    skill_values(skill),
                )
        finally:
            conn.close()
        return self.get(skill.name) or skill

    def get(self, name: str) -> Skill | None:
        """One skill by name."""
        conn = connect(self.path)
        try:
            row = conn.execute(
                f"SELECT {_SKILL_COLUMNS} FROM skills WHERE name = ?", (name,)
            ).fetchone()
        finally:
            conn.close()
        return skill_from_row(row) if row is not None else None

    def list(self) -> list[Skill]:
        """Every skill row, name order."""
        conn = connect(self.path)
        try:
            rows = conn.execute(
                f"SELECT {_SKILL_COLUMNS} FROM skills ORDER BY name"
            ).fetchall()
        finally:
            conn.close()
        return [skill_from_row(row) for row in rows]

    def enabled(self) -> list[Skill]:
        """Every enabled skill row, name order."""
        return [skill for skill in self.list() if skill.enabled]

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Flip one skill's enabled flag."""
        self.run_update(
            "UPDATE skills SET enabled = ?, updated_at = ? WHERE name = ?",
            (int(enabled), self.now(), name),
        )

    def delete(self, name: str) -> None:
        """Delete one skill row and its staged requests."""
        self.run_update("DELETE FROM skills WHERE name = ?", (name,))

    def save_draft_inputs(self, name: str, inputs: dict[str, object]) -> None:
        """Persist the resumable draft inputs."""
        self.run_update(
            "UPDATE skills SET draft_inputs_json = ?, updated_at = ? WHERE name = ?",
            (json.dumps(inputs, sort_keys=True), self.now(), name),
        )

    def save_draft(
        self,
        name: str,
        *,
        inputs: dict[str, object] | None = None,
        stage: str | None = None,
        instructions: str | None = None,
        skill_md: str | None = None,
    ) -> Skill:
        """Merge partial draft inputs and optionally advance the stage."""
        skill = self.get(name)
        if skill is None:
            raise KeyError(f"skill {name} not found")
        if inputs:
            skill.draft_inputs.update(inputs)
        if stage is not None:
            skill.draft_stage = stage
        if instructions is not None:
            skill.instructions = instructions
        if skill_md is not None:
            skill.skill_md = skill_md
        return self.upsert(skill)

    def set_test(
        self,
        name: str,
        *,
        state: str,
        reason: str,
        config_hash: str,
        at: int | None = None,
    ) -> None:
        """Store one test result and the configuration hash it covered."""
        self.run_update(
            "UPDATE skills SET test_state = ?, test_reason = ?, "
            + "test_config_hash = ?, last_tested_at = ?, updated_at = ? "
            + "WHERE name = ?",
            (
                state,
                reason,
                config_hash,
                self.now() if at is None else at,
                self.now(),
                name,
            ),
        )

    def record_failure(self, name: str) -> int:
        """Increment one skill's consecutive failure count and return it."""
        moment = self.now()
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "UPDATE skills SET failure_count = failure_count + 1, "
                    + "updated_at = ? WHERE name = ?",
                    (moment, name),
                )
                row = conn.execute(
                    "SELECT failure_count FROM skills WHERE name = ?", (name,)
                ).fetchone()
        finally:
            conn.close()
        return int(row["failure_count"]) if row is not None else 0

    def reset_failures(self, name: str) -> None:
        """Clear a skill's consecutive failure count after a success."""
        self.run_update("UPDATE skills SET failure_count = 0 WHERE name = ?", (name,))

    def add_call(self, call: SkillCall) -> int:
        """Append one call row and return its id."""
        conn = connect(self.path)
        try:
            with transaction(conn):
                cursor = conn.execute(
                    "INSERT INTO skill_calls (skill, ticket_id, chat_id, started_by, "
                    + "method, path, status, duration_ms, outcome, error, "
                    + "request_json, created_at) "
                    + "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        call.skill,
                        call.ticket_id,
                        call.chat_id,
                        call.started_by,
                        call.method,
                        call.path,
                        call.status,
                        call.duration_ms,
                        call.outcome,
                        call.error,
                        call.request_json,
                        call.created_at,
                    ),
                )
                return int(cursor.lastrowid or 0)
        finally:
            conn.close()

    def calls(self, skill: str, limit: int = 50) -> list[SkillCall]:
        """One skill's newest calls."""
        conn = connect(self.path)
        try:
            rows = conn.execute(
                f"SELECT {_CALL_COLUMNS} FROM skill_calls WHERE skill = ? "
                + "ORDER BY id DESC LIMIT ?",
                (skill, limit),
            ).fetchall()
        finally:
            conn.close()
        return [call_from_row(row) for row in rows]

    def calls_for_ticket(self, ticket_id: int, limit: int = 20) -> list[SkillCall]:
        """The calls linked to one ticket, newest first."""
        conn = connect(self.path)
        try:
            rows = conn.execute(
                f"SELECT {_CALL_COLUMNS} FROM skill_calls WHERE ticket_id = ? "
                + "ORDER BY id DESC LIMIT ?",
                (ticket_id, limit),
            ).fetchall()
        finally:
            conn.close()
        return [call_from_row(row) for row in rows]

    def create_request(
        self,
        *,
        skill: str,
        ticket_id: int | None,
        chat_id: str,
        request_json: str,
        summary: str,
    ) -> SkillRequest:
        """Insert one pending write request."""
        conn = connect(self.path)
        try:
            with transaction(conn):
                cursor = conn.execute(
                    "INSERT INTO skill_requests (skill, ticket_id, chat_id, "
                    + "request_json, summary, state, created_at) "
                    + "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                    (skill, ticket_id, chat_id, request_json, summary, self.now()),
                )
                row = conn.execute(
                    f"SELECT {_REQUEST_COLUMNS} FROM skill_requests WHERE id = ?",
                    (int(cursor.lastrowid or 0),),
                ).fetchone()
        finally:
            conn.close()
        return request_from_row(row)

    def set_operator_message(self, request_id: int, message_id: str) -> None:
        """Store the self-chat message id a yes/no reply can quote."""
        self.run_update(
            "UPDATE skill_requests SET operator_message_id = ? WHERE id = ?",
            (message_id, request_id),
        )

    def request(self, request_id: int) -> SkillRequest | None:
        """One staged request by id."""
        conn = connect(self.path)
        try:
            row = conn.execute(
                f"SELECT {_REQUEST_COLUMNS} FROM skill_requests WHERE id = ?",
                (request_id,),
            ).fetchone()
        finally:
            conn.close()
        return request_from_row(row) if row is not None else None

    def latest_pending(self) -> SkillRequest | None:
        """The newest request still waiting for a yes/no."""
        pending = self.pending_requests()
        return pending[0] if pending else None

    def pending_requests(self) -> list[SkillRequest]:
        """Every actionable pending request, newest first."""
        self.expire_requests()
        conn = connect(self.path)
        try:
            rows = conn.execute(
                f"SELECT {_REQUEST_COLUMNS} FROM skill_requests "
                + "WHERE state = 'pending' ORDER BY id DESC"
            ).fetchall()
        finally:
            conn.close()
        return [request_from_row(row) for row in rows]

    def expire_requests(self, *, now: int | None = None) -> None:
        """Refuse pending requests that have waited longer than the TTL."""
        moment = self.now() if now is None else now
        self.run_update(
            "UPDATE skill_requests SET state = 'refused', resolved_at = ? "
            + "WHERE state = 'pending' AND created_at < ?",
            (moment, moment - PENDING_REQUEST_TTL_S),
        )

    def pending_for_message(self, message_id: str) -> SkillRequest | None:
        """The pending request whose self-chat message was quoted."""
        if not message_id:
            return None
        self.expire_requests()
        return self.pending_message_request(
            "SELECT {columns} FROM skill_requests WHERE state = 'pending' "
            + "AND operator_message_id = ? ORDER BY id DESC LIMIT 1",
            (message_id,),
        )

    def claim_request(self, request_id: int) -> SkillRequest | None:
        """Atomically claim a pending request; None when it was already handled."""
        self.expire_requests()
        moment = self.now()
        conn = connect(self.path)
        try:
            with transaction(conn):
                cursor = conn.execute(
                    "UPDATE skill_requests SET state = 'approved', resolved_at = ? "
                    + "WHERE id = ? AND state = 'pending' AND created_at >= ?",
                    (moment, request_id, moment - PENDING_REQUEST_TTL_S),
                )
                if cursor.rowcount != 1:
                    return None
        finally:
            conn.close()
        return self.request(request_id)

    def resolve_request(self, request_id: int, state: str) -> None:
        """Mark one claimed request done, failed or refused."""
        self.run_update(
            "UPDATE skill_requests SET state = ?, resolved_at = ? WHERE id = ?",
            (state, self.now(), request_id),
        )

    def secret_exists(self, name: str) -> bool:
        """Whether a secret row exists for *name*."""
        return (
            self.query_scalar("SELECT 1 FROM secrets WHERE name = ?", (name,)) is not None
        )

    def pending_message_request(
        self, query: str, params: tuple[object, ...] = ()
    ) -> SkillRequest | None:
        """The one request selected by a caller-supplied pending query."""
        conn = connect(self.path)
        try:
            row = conn.execute(query.format(columns=_REQUEST_COLUMNS), params).fetchone()
        finally:
            conn.close()
        return request_from_row(row) if row is not None else None

    def query_scalar(self, query: str, params: tuple[object, ...]) -> object:
        """The first column of one row, or None."""
        conn = connect(self.path)
        try:
            row = conn.execute(query, params).fetchone()
        finally:
            conn.close()
        return None if row is None else row[0]

    def run_update(self, query: str, params: tuple[object, ...]) -> None:
        """Run one write in its own transaction."""
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(query, params)
        finally:
            conn.close()


def skill_columns() -> tuple[str, ...]:
    """The skill table columns in insert order."""
    return tuple(column.strip() for column in _SKILL_COLUMNS.split(","))


def skill_values(skill: Skill) -> tuple[object, ...]:
    """The insert values matching :func:`skill_columns`."""
    return (
        skill.name,
        skill.lifecycle,
        int(skill.enabled),
        json.dumps(skill.base_urls),
        skill.auth_type,
        skill.secret_ref,
        skill.customer_chats,
        int(skill.writes_need_operator),
        skill.source_url,
        skill.instructions,
        skill.skill_md,
        skill.draft_stage,
        json.dumps(skill.draft_inputs, sort_keys=True),
        skill.test_config_hash,
        skill.test_state,
        skill.test_reason,
        skill.last_tested_at,
        skill.failure_count,
        skill.created_by,
        skill.created_at,
        skill.updated_at,
    )


def call_from_row(row: sqlite3.Row) -> SkillCall:
    """Map one ``skill_calls`` row."""
    return SkillCall(
        id=int(row["id"]),
        skill=str(row["skill"]),
        ticket_id=int(row["ticket_id"]) if row["ticket_id"] is not None else None,
        chat_id=str(row["chat_id"] or ""),
        started_by=str(row["started_by"]),
        method=str(row["method"] or ""),
        path=str(row["path"] or ""),
        status=int(row["status"]) if row["status"] is not None else None,
        duration_ms=int(row["duration_ms"]),
        outcome=str(row["outcome"] or ""),
        error=str(row["error"] or ""),
        request_json=str(row["request_json"] or "{}"),
        created_at=int(row["created_at"]),
    )


def request_from_row(row: sqlite3.Row) -> SkillRequest:
    """Map one ``skill_requests`` row."""
    return SkillRequest(
        id=int(row["id"]),
        skill=str(row["skill"]),
        ticket_id=int(row["ticket_id"]) if row["ticket_id"] is not None else None,
        chat_id=str(row["chat_id"] or ""),
        request_json=str(row["request_json"]),
        summary=str(row["summary"]),
        state=str(row["state"]),
        operator_message_id=str(row["operator_message_id"] or ""),
        created_at=int(row["created_at"]),
        resolved_at=int(row["resolved_at"]) if row["resolved_at"] is not None else None,
    )
