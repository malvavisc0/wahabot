from pathlib import Path

import pytest

from wahabot.core.db import Database
from wahabot.core.db.migrations import MigrationError


def test_database_opens_in_wal_mode_with_pragmas(database: Database) -> None:
    database.migrate()

    with database.connect() as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]

    assert str(mode).lower() == "wal"
    assert foreign_keys == 1
    assert timeout == 5000


def test_migration_applies_once_and_records_the_version(database: Database) -> None:
    database.migrate()
    database.migrate()

    with database.connect() as conn:
        versions = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

    assert [row[0] for row in versions] == [1, 2]
    assert {"intake", "outbound_intents", "schema_migrations"} <= tables


def test_migration_creates_the_listed_indexes(database: Database) -> None:
    database.migrate()

    with database.connect() as conn:
        indexes = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }

    assert {
        "intake_state_next_attempt",
        "intake_chat_received",
        "intake_burst_key",
        "outbound_state_next_attempt",
        "outbound_chat_created",
        "intake_operator_notice",
    } <= indexes


def test_broken_migration_rolls_back_and_blocks_startup(tmp_path: Path) -> None:
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    (sql_dir / "0001_broken.sql").write_text(
        "CREATE TABLE half_applied (id TEXT);\nCREATE TABLE half_applied (id TEXT);\n"
    )
    database = Database(tmp_path / "wahabot.db", sql_dir=sql_dir)

    with pytest.raises(MigrationError):
        database.migrate()

    with database.connect() as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        applied = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]

    assert "half_applied" not in tables
    assert applied == 0


def test_fresh_data_directory_starts_clean(tmp_path: Path) -> None:
    data_dir = tmp_path / "nested" / "data"
    database = Database(data_dir / "wahabot.db")

    database.migrate()

    assert (data_dir / "wahabot.db").exists()
    with database.connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
    assert count == 2


def test_intake_settings_have_the_documented_defaults() -> None:
    from wahabot.settings import Settings

    settings = Settings(
        webhook_hmac_key="k",
        waha_url="http://waha.invalid",
        waha_api_key="k",
        llm_api_base="http://llm.invalid/v1",
        llm_api_key="k",
        _env_file=None,
    )

    assert settings.intake_recovery_minutes == 30
    assert settings.intake_max_attempts == 5
    assert settings.intake_retention_days == 30


def test_reply_window_is_bounded_and_defaults_to_seven_days() -> None:
    from wahabot.core.access import SessionConfig

    config = SessionConfig(system_prompt="be helpful")

    assert config.reply_window_days == 7
    assert SessionConfig(system_prompt="x", reply_window_days=1).reply_window_days == 1
    assert SessionConfig(system_prompt="x", reply_window_days=30).reply_window_days == 30
    for invalid in (0, 31):
        with pytest.raises(ValueError):
            SessionConfig(system_prompt="x", reply_window_days=invalid)
