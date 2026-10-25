from pathlib import Path

from tests.harness import RecordingWaha
from tests.skills.conftest import SENDER, json_body, write_skill

from wahabot.core.skills.drafts import SkillDraftService
from wahabot.core.skills.loader import import_skill
from wahabot.core.skills.models import SkillRun
from wahabot.core.skills.service import SkillService


def draft_service(tmp_path: Path, skill_store, secret_store, fake_rest):
    folder = write_skill(tmp_path, "customer-records", fake_rest.base)
    import_skill(folder, tmp_path / "skills", skill_store)
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )
    return SkillDraftService(
        skill_store, secret_store, tmp_path / "skills", service
    ), folder


def test_enable_refuses_an_untested_skill(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    drafts, _folder = draft_service(tmp_path, skill_store, secret_store, fake_rest)

    assert drafts.enable("customer-records") is False
    assert skill_store.get("customer-records").enabled is False


def test_a_current_passing_test_allows_enable(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    drafts, _folder = draft_service(tmp_path, skill_store, secret_store, fake_rest)

    result = drafts.run_test("customer-records")

    assert result.passed is True
    assert skill_store.get("customer-records").test_state == "passed"
    assert drafts.enable("customer-records") is True
    assert skill_store.get("customer-records").enabled is True


def test_a_changed_configuration_marks_the_test_stale(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    drafts, _folder = draft_service(tmp_path, skill_store, secret_store, fake_rest)
    drafts.run_test("customer-records")
    skill_store.save_draft_inputs("customer-records", {"instructions": "changed"})

    drafts.refresh_staleness("customer-records")

    assert skill_store.get("customer-records").test_state == "stale"
    assert drafts.enable("customer-records") is False


def test_a_changed_secret_marks_the_test_stale(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    drafts, _folder = draft_service(tmp_path, skill_store, secret_store, fake_rest)
    drafts.run_test("customer-records")
    secret_store.put("SMOKE_KEY", "sk-fictional-rotated-key")

    drafts.refresh_staleness("customer-records")

    assert skill_store.get("customer-records").test_state == "stale"


def test_no_safe_check_is_explained_and_leaves_the_skill_inactive(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    folder = write_skill(
        tmp_path,
        "customer-records",
        fake_rest.base,
        test_check="{}",
    )
    import_skill(folder, tmp_path / "skills", skill_store)
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )
    drafts = SkillDraftService(skill_store, secret_store, tmp_path / "skills", service)

    result = drafts.run_test("customer-records")

    assert result.passed is False
    assert "no safe check" in result.reason.lower()
    assert drafts.enable("customer-records") is False


def test_a_failed_check_is_recorded_and_blocks_enable(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    folder = write_skill(
        tmp_path,
        "customer-records",
        fake_rest.base,
        test_check="{method: GET, path: /unhealthy}",
    )
    import_skill(folder, tmp_path / "skills", skill_store)
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )
    drafts = SkillDraftService(skill_store, secret_store, tmp_path / "skills", service)

    result = drafts.run_test("customer-records")

    assert result.passed is False
    assert "500" in result.reason
    assert skill_store.get("customer-records").test_state == "failed"
    assert drafts.enable("customer-records") is False


def test_a_write_is_never_used_as_a_test(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    folder = write_skill(
        tmp_path,
        "customer-records",
        fake_rest.base,
        test_check="{method: POST, path: /leads}",
    )
    import_skill(folder, tmp_path / "skills", skill_store)
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )
    drafts = SkillDraftService(skill_store, secret_store, tmp_path / "skills", service)

    result = drafts.run_test("customer-records")

    assert result.passed is False
    assert "safe" in result.reason.lower()
    assert fake_rest.lead_count() == 0


def test_create_draft_saves_partial_inputs_and_stage(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )
    drafts = SkillDraftService(skill_store, secret_store, tmp_path / "skills", service)

    draft = drafts.create_draft(
        "customer-records", inputs={"base_url": "https://crm.example/api"}
    )
    updated = drafts.save_draft(
        "customer-records", inputs={"customer_chats": "read"}, stage="test"
    )

    assert draft.lifecycle == "draft"
    assert updated.lifecycle == "draft"
    assert updated.draft_stage == "test"
    assert updated.draft_inputs["base_url"] == "https://crm.example/api"
    assert updated.draft_inputs["customer_chats"] == "read"
    assert drafts.enable("customer-records") is False


def test_a_sql_skill_passes_its_read_only_probe_and_can_enable(
    tmp_path: Path, skill_store, secret_store
) -> None:
    import sqlite3

    database = tmp_path / "customer.sqlite3"
    sqlite3.connect(database).execute("CREATE TABLE records (id INTEGER)")
    folder = write_skill(
        tmp_path,
        "customer-sql",
        "",
        kind="sql",
        auth="{type: none}",
        customer_chats="none",
        test_check="{}",
        sql=f"{{engine: sqlite, sqlite_path: {database}}}",
    )
    import_skill(folder, tmp_path / "skills", skill_store)
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )
    drafts = SkillDraftService(skill_store, secret_store, tmp_path / "skills", service)

    result = drafts.run_test("customer-sql")

    assert result.passed is True
    assert skill_store.get("customer-sql").test_state == "passed"
    assert drafts.enable("customer-sql") is True


def test_a_3xx_safe_check_is_not_a_pass(
    tmp_path: Path, skill_store, secret_store, fake_rest, monkeypatch
) -> None:
    from wahabot.core.skills.models import HttpResult

    drafts, _folder = draft_service(tmp_path, skill_store, secret_store, fake_rest)
    monkeypatch.setattr(
        drafts.service,
        "test_probe",
        lambda *_args: HttpResult(status=304, text="", truncated=False),
    )

    result = drafts.run_test("customer-records")

    assert result.passed is False
    assert drafts.enable("customer-records") is False


def test_editing_an_enabled_skill_blocks_calls_until_retested(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    drafts, _folder = draft_service(tmp_path, skill_store, secret_store, fake_rest)
    assert drafts.run_test("customer-records").passed is True
    assert drafts.enable("customer-records") is True
    config = tmp_path / "skills" / "customer-records" / "skill.yaml"
    config.write_text(
        config.read_text().replace("customer_chats: read", "customer_chats: none")
    )
    run = SkillRun(session="test", chat_id=SENDER, surface="customer")

    result = json_body(
        drafts.service.execute_http(
            "customer-records", "GET", "/records", None, None, run
        )
    )

    assert result["ok"] is False
    assert "configuration changed" in result["error"]
    assert skill_store.get("customer-records").test_state == "stale"
