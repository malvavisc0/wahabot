"""Read-only SQLite and Postgres execution for SQL skills."""

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from wahabot.core.skills.errors import SkillError
from wahabot.core.skills.models import SkillConfig

MAX_ROWS = 200
MAX_COLUMNS = 50
MAX_BYTES = 64 * 1024
TIMEOUT_S = 15.0

_FIRST = re.compile(r"^\s*([a-zA-Z]+)")
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|CREATE|DROP|"
    + r"ALTER|TRIGGER|REINDEX|GRANT|REVOKE|COPY|CALL|EXECUTE)\b",
    re.IGNORECASE,
)


def load_psycopg() -> Any:
    """The optional Postgres driver, with a clear install message when absent."""
    try:
        import psycopg
    except ImportError as exc:
        raise SkillError(
            "the Postgres driver is not installed; install the postgres extra"
        ) from exc
    return psycopg


def validate_read_query(query: str) -> str:
    """One SELECT or WITH...SELECT, with comments and extra statements refused."""
    cleaned = query.strip().rstrip(";").strip()
    if not cleaned:
        raise SkillError("the SQL query is empty")
    _require_single_statement(cleaned)
    _require_read_shape(cleaned)
    return cleaned


def _require_single_statement(query: str) -> None:
    if "--" in query or "/*" in query or "*/" in query:
        raise SkillError("SQL comments are refused; they can hide a second statement")
    if ";" in query:
        raise SkillError("only one statement is allowed")


def _require_read_shape(query: str) -> None:
    first = _FIRST.match(query)
    keyword = first.group(1).upper() if first else ""
    if keyword not in ("SELECT", "WITH"):
        raise SkillError("only read-only SELECT or WITH...SELECT queries are allowed")
    if keyword == "WITH" and not re.search(r"\bSELECT\b", query, re.IGNORECASE):
        raise SkillError("a WITH query must end in SELECT")
    if _FORBIDDEN.search(query):
        raise SkillError("only non-mutating read queries are allowed")


def query_sqlite(path: Path, query: str) -> dict[str, object]:
    """Run one bounded read query against a SQLite file opened read-only."""
    if not path.is_file():
        raise SkillError("the configured SQLite database does not exist")
    uri = f"file:{path}?mode=ro"
    deadline = time.monotonic() + TIMEOUT_S
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=TIMEOUT_S)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        try:
            cursor = conn.execute(query)
            columns = [str(item[0]) for item in cursor.description or ()][:MAX_COLUMNS]
            raw_rows = cursor.fetchmany(MAX_ROWS + 1)
            rows, truncated = _bounded_rows(raw_rows, columns)
            return {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "truncated": truncated,
            }
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        raise SkillError(f"the SQLite query failed: {exc}") from exc


def query_postgres(dsn: str, query: str) -> dict[str, object]:
    """Run one bounded read query inside a read-only Postgres transaction."""
    psycopg = load_psycopg()
    try:
        with (
            psycopg.connect(dsn, connect_timeout=int(TIMEOUT_S)) as conn,
            conn.cursor() as cursor,
        ):
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute(f"SET LOCAL statement_timeout = {int(TIMEOUT_S * 1000)}")
            cursor.execute(query)
            columns = [str(item[0]) for item in cursor.description or ()][:MAX_COLUMNS]
            raw_rows = cursor.fetchmany(MAX_ROWS + 1)
            rows, truncated = _bounded_rows(raw_rows, columns)
            return {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "truncated": truncated,
            }
    except Exception as exc:
        if isinstance(exc, SkillError):
            raise
        raise SkillError(f"the Postgres query failed: {type(exc).__name__}") from exc


def execute_query(config: SkillConfig, query: str, secret: str = "") -> dict[str, object]:
    """Validate and run one read query for a configured SQL skill."""
    if config.sql is None:
        raise SkillError("this skill has no SQL configuration")
    cleaned = validate_read_query(query)
    if config.sql.engine == "sqlite":
        return query_sqlite(Path(config.sql.sqlite_path), cleaned)
    if not secret:
        raise SkillError("the Postgres DSN secret is not available")
    return query_postgres(secret, cleaned)


def _bounded_rows(
    raw_rows: list[Any], columns: list[str]
) -> tuple[list[dict[str, object]], bool]:
    truncated = len(raw_rows) > MAX_ROWS
    rows: list[dict[str, object]] = []
    size = 0
    for raw in raw_rows[:MAX_ROWS]:
        row = _row_dict(raw, columns)
        size += len(json.dumps(row, default=str))
        if size > MAX_BYTES:
            truncated = True
            break
        rows.append(row)
    return rows, truncated


def _row_dict(raw: Any, columns: list[str]) -> dict[str, object]:
    values = list(raw)[: len(columns)]
    return {
        column: _json_value(value) for column, value in zip(columns, values, strict=False)
    }


def _json_value(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
