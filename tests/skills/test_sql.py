import sqlite3
from pathlib import Path

from tests.harness import RecordingWaha
from tests.skills.conftest import json_body, write_skill

from wahabot.core.skills.loader import import_skill
from wahabot.core.skills.models import SkillRun
from wahabot.core.skills.service import SkillService


def sql_setup(tmp_path: Path, skill_store, secret_store, *, engine: str = "sqlite"):
    database = tmp_path / "customer.sqlite3"
    if engine == "sqlite":
        conn = sqlite3.connect(database)
        conn.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, name TEXT)")
        conn.executemany(
            "INSERT INTO records (name) VALUES (?)",
            [(f"record-{index}",) for index in range(5)],
        )
        conn.commit()
        conn.close()
        sql = f"{{engine: sqlite, sqlite_path: {database}}}"
    else:
        sql = "{engine: postgres, dsn_secret: SMOKE_KEY}"
    folder = write_skill(
        tmp_path,
        "customer-sql",
        "",
        kind="sql",
        auth="{type: none}",
        customer_chats="none",
        test_check="{}",
        sql=sql,
    )
    import_skill(folder, tmp_path / "skills", skill_store)
    skill_store.set_enabled("customer-sql", True)
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )
    run = SkillRun(session="test", surface="operator", armed=True)
    return service, run


def test_a_read_query_returns_rows(tmp_path: Path, skill_store, secret_store) -> None:
    service, run = sql_setup(tmp_path, skill_store, secret_store)

    result = json_body(
        service.execute_sql("customer-sql", "SELECT id, name FROM records", run)
    )

    assert result["ok"] is True
    assert result["columns"] == ["id", "name"]
    assert len(result["rows"]) == 5


def test_a_write_is_refused(tmp_path: Path, skill_store, secret_store) -> None:
    service, run = sql_setup(tmp_path, skill_store, secret_store)

    result = json_body(
        service.execute_sql("customer-sql", "DELETE FROM records WHERE id = 1", run)
    )

    assert result["ok"] is False
    assert "read" in result["error"].lower()


def test_two_statements_are_refused(tmp_path: Path, skill_store, secret_store) -> None:
    service, run = sql_setup(tmp_path, skill_store, secret_store)

    result = json_body(service.execute_sql("customer-sql", "SELECT 1; SELECT 2", run))

    assert result["ok"] is False
    assert "one statement" in result["error"].lower()


def test_comments_that_hide_a_statement_are_refused(
    tmp_path: Path, skill_store, secret_store
) -> None:
    service, run = sql_setup(tmp_path, skill_store, secret_store)

    result = json_body(
        service.execute_sql("customer-sql", "SELECT 1 -- ; DELETE FROM records", run)
    )

    assert result["ok"] is False


def test_attach_and_pragma_writes_are_refused(
    tmp_path: Path, skill_store, secret_store
) -> None:
    service, run = sql_setup(tmp_path, skill_store, secret_store)

    attach = json_body(
        service.execute_sql("customer-sql", "ATTACH DATABASE 'x' AS y", run)
    )
    pragma = json_body(
        service.execute_sql("customer-sql", "PRAGMA journal_mode=DELETE", run)
    )

    assert attach["ok"] is False
    assert pragma["ok"] is False


def test_rows_are_capped(tmp_path: Path, skill_store, secret_store) -> None:
    service, run = sql_setup(tmp_path, skill_store, secret_store)
    import wahabot.core.skills.sql as sql_module

    original = sql_module.MAX_ROWS
    sql_module.MAX_ROWS = 2
    try:
        result = json_body(
            service.execute_sql("customer-sql", "SELECT * FROM records", run)
        )
    finally:
        sql_module.MAX_ROWS = original

    assert result["ok"] is True
    assert result["truncated"] is True
    assert len(result["rows"]) == 2


def test_a_customer_run_cannot_use_a_sql_skill(
    tmp_path: Path, skill_store, secret_store
) -> None:
    service, _run = sql_setup(tmp_path, skill_store, secret_store)
    customer = SkillRun(session="test", chat_id="491555000001@c.us", surface="customer")

    result = json_body(service.execute_sql("customer-sql", "SELECT 1", customer))

    assert result["ok"] is False
    assert "operator" in result["error"].lower()


def test_a_missing_postgres_driver_is_a_clear_error(
    tmp_path: Path, skill_store, secret_store, monkeypatch
) -> None:
    import wahabot.core.skills.sql as sql_module
    from wahabot.core.skills.errors import SkillError

    def absent():
        raise SkillError(
            "the Postgres driver is not installed; install wahabot[postgres]"
        )

    monkeypatch.setattr(sql_module, "load_psycopg", absent)
    service, run = sql_setup(tmp_path, skill_store, secret_store, engine="postgres")

    result = json_body(service.execute_sql("customer-sql", "SELECT 1", run))

    assert result["ok"] is False
    assert "driver" in result["error"].lower()
    assert "postgres" in result["error"].lower()


def test_replace_function_and_punctuation_inside_literals_are_allowed(
    tmp_path: Path, skill_store, secret_store
) -> None:
    service, _run = sql_setup(tmp_path, skill_store, secret_store)
    for query in (
        "SELECT REPLACE(name, 'record', 'row') FROM records",
        "SELECT 'hello;world'",
        "SELECT * FROM records WHERE name = 'delete this'",
    ):
        run = SkillRun(session="test", surface="operator", armed=True)
        result = json_body(service.execute_sql("customer-sql", query, run))
        assert result["ok"] is True, result
