import sqlite3
from pathlib import Path

from tests.harness import RecordingWaha
from tests.skills.conftest import SECRET_VALUE, json_body, write_skill

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
    merge = json_body(
        service.execute_sql(
            "customer-sql",
            "WITH x AS (SELECT 1) MERGE INTO records USING x ON true "
            "WHEN MATCHED THEN DELETE",
            run,
        )
    )

    assert result["ok"] is False
    assert merge["ok"] is False
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


def test_a_refused_query_does_not_count_as_a_runtime_failure(
    tmp_path: Path, skill_store, secret_store
) -> None:
    service, run = sql_setup(tmp_path, skill_store, secret_store)
    before = skill_store.get("customer-sql")

    for query in (
        "DELETE FROM records",
        "PRAGMA journal_mode=DELETE",
        "SELECT 1; SELECT 2",
    ):
        assert json_body(service.execute_sql("customer-sql", query, run))["ok"] is False

    after = skill_store.get("customer-sql")
    assert after is not None and before is not None
    assert after.failure_count == before.failure_count
    assert after.test_state == before.test_state


def test_a_failed_query_is_audited_with_its_text(
    tmp_path: Path, skill_store, secret_store
) -> None:
    service, run = sql_setup(tmp_path, skill_store, secret_store)

    result = json_body(
        service.execute_sql("customer-sql", "SELECT * FROM missing_table", run)
    )

    assert result["ok"] is False
    call = skill_store.calls("customer-sql")[0]
    assert "missing_table" in call.path
    assert "missing_table" in call.request_json


def test_column_truncation_sets_the_truncated_flag(
    tmp_path: Path, skill_store, secret_store
) -> None:
    service, run = sql_setup(tmp_path, skill_store, secret_store)
    import wahabot.core.skills.sql as sql_module

    original = sql_module.MAX_COLUMNS
    sql_module.MAX_COLUMNS = 1
    try:
        result = json_body(
            service.execute_sql("customer-sql", "SELECT id, name FROM records", run)
        )
    finally:
        sql_module.MAX_COLUMNS = original

    assert result["ok"] is True
    assert result["columns"] == ["id"]
    assert result["truncated"] is True


def test_a_successful_sql_probe_resets_the_failure_streak(
    tmp_path: Path, skill_store, secret_store
) -> None:
    service, _run = sql_setup(tmp_path, skill_store, secret_store)
    for _ in range(3):
        skill_store.record_failure("customer-sql")

    result = service.test_sql_probe("customer-sql")

    assert result.passed is True
    assert skill_store.get("customer-sql").failure_count == 0


def test_sql_results_redact_stored_secrets(
    tmp_path: Path, skill_store, secret_store
) -> None:
    service, run = sql_setup(tmp_path, skill_store, secret_store)

    result = json_body(
        service.execute_sql("customer-sql", f"SELECT '{SECRET_VALUE}' AS token", run)
    )

    assert result["ok"] is True
    assert result["rows"][0]["token"] == "***"


def test_non_finite_floats_stay_json_safe(
    tmp_path: Path, skill_store, secret_store
) -> None:
    service, run = sql_setup(tmp_path, skill_store, secret_store)

    result = json_body(service.execute_sql("customer-sql", "SELECT 1e999 AS value", run))

    assert result["rows"][0]["value"] == "inf"
