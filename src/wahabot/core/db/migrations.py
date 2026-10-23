"""Numbered SQL migrations with one recorded version per applied file."""

import re
import sqlite3
import time
from pathlib import Path

from wahabot.core.db.connection import connect

MIGRATIONS_DIR = Path(__file__).with_name("sql")

_VERSION_RE = re.compile(r"^(\d+)_")


class MigrationError(RuntimeError):
    """A migration could not be applied; the process must not serve."""


def migration_files(sql_dir: Path = MIGRATIONS_DIR) -> list[tuple[int, Path]]:
    """The numbered migration files, in version order."""
    found: list[tuple[int, Path]] = []
    for path in sql_dir.glob("*.sql"):
        match = _VERSION_RE.match(path.name)
        if match is not None:
            found.append((int(match.group(1)), path))
    return sorted(found)


def migrate(path: Path | str, sql_dir: Path = MIGRATIONS_DIR) -> None:
    """Apply every unapplied migration; a failure rolls back and raises."""
    conn = connect(path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            + "version INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL)"
        )
        applied = {
            int(row[0]) for row in conn.execute("SELECT version FROM schema_migrations")
        }
        for version, migration in migration_files(sql_dir):
            if version in applied:
                continue
            script = (
                "BEGIN IMMEDIATE;\n"
                + migration.read_text()
                + "\nINSERT INTO schema_migrations(version, applied_at) VALUES ("
                + f"{version}, {int(time.time())});\nCOMMIT;"
            )
            try:
                conn.executescript(script)
            except sqlite3.Error as exc:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise MigrationError(f"migration {migration.name} failed: {exc}") from exc
    finally:
        conn.close()
