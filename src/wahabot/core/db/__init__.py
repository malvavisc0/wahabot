"""The platform SQLite database and its migration entry point."""

from pathlib import Path

from wahabot.core.db.connection import connect
from wahabot.core.db.migrations import MIGRATIONS_DIR, migrate

__all__ = ["Database", "connect", "migrate"]


class Database:
    """A SQLite file opened per operation, with numbered migrations."""

    def __init__(self, path: Path, sql_dir: Path = MIGRATIONS_DIR) -> None:
        self.path = path
        self.sql_dir = sql_dir

    def connect(self):
        """A fresh connection for one operation."""
        return connect(self.path)

    def migrate(self) -> None:
        """Apply every unapplied migration; raise when one fails."""
        migrate(self.path, self.sql_dir)
