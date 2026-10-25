from pathlib import Path

import pytest
from tests.skills.conftest import write_skill

from wahabot.core.skills.errors import SkillError
from wahabot.core.skills.loader import import_skill, read_config


def test_import_creates_a_configured_row_and_copies_the_folder(
    tmp_path: Path, skill_store
) -> None:
    source = write_skill(
        tmp_path,
        "customer-records",
        "https://crm.example/api",
        auth="{type: none}",
    )

    imported = import_skill(source, tmp_path / "skills", skill_store)

    assert imported.name == "customer-records"
    assert imported.lifecycle == "configured"
    assert imported.enabled is False
    assert (tmp_path / "skills" / "customer-records" / "SKILL.md").is_file()
    assert (tmp_path / "skills" / "customer-records" / "skill.yaml").is_file()
    assert skill_store.get("customer-records") is not None


def test_import_is_idempotent_for_a_disabled_skill(tmp_path: Path, skill_store) -> None:
    source = write_skill(
        tmp_path,
        "customer-records",
        "https://crm.example/api",
        auth="{type: none}",
    )

    first = import_skill(source, tmp_path / "skills", skill_store)
    second = import_skill(source, tmp_path / "skills", skill_store)

    assert first.name == second.name
    assert len(skill_store.list()) == 1


def test_import_refuses_to_overwrite_an_enabled_skill(
    tmp_path: Path, skill_store
) -> None:
    source = write_skill(
        tmp_path,
        "customer-records",
        "https://crm.example/api",
        auth="{type: none}",
    )
    import_skill(source, tmp_path / "skills", skill_store)
    skill_store.set_enabled("customer-records", True)

    with pytest.raises(SkillError, match="enabled"):
        import_skill(source, tmp_path / "skills", skill_store)


def test_an_unsafe_folder_name_is_refused(tmp_path: Path, skill_store) -> None:
    source = write_skill(
        tmp_path, "Bad Name", "https://crm.example/api", auth="{type: none}"
    )

    with pytest.raises(SkillError, match="name"):
        import_skill(source, tmp_path / "skills", skill_store)


def test_a_missing_secret_reference_is_refused(tmp_path: Path, skill_store) -> None:
    source = write_skill(
        tmp_path,
        "customer-records",
        "https://crm.example/api",
        auth="{type: bearer, secret: MISSING_KEY}",
    )

    with pytest.raises(SkillError, match="secret"):
        import_skill(source, tmp_path / "skills", skill_store)


def test_a_base_url_with_credentials_is_refused(tmp_path: Path, skill_store) -> None:
    source = write_skill(
        tmp_path, "customer-records", "https://user:pass@crm.example/api"
    )

    with pytest.raises(SkillError, match="base URL"):
        import_skill(source, tmp_path / "skills", skill_store)


def test_load_skill_reads_the_folder_configuration(tmp_path: Path, skill_store) -> None:
    source = write_skill(
        tmp_path,
        "customer-records",
        "https://crm.example/api",
        auth="{type: none}",
        events={"done": True},
        sign_events=True,
        signing_secret="SIGNING_KEY",
    )
    config = read_config(source)

    assert config.effective_base_urls == ["https://crm.example/api"]
    assert config.customer_chats == "read"
    assert config.writes_need_operator is True
    assert config.events["done"] is True
    assert config.sign_events is True
    assert config.signing_secret == "SIGNING_KEY"


def test_a_base_url_with_a_query_is_refused(tmp_path: Path, skill_store) -> None:
    source = write_skill(
        tmp_path, "customer-records", "https://crm.example/api?key=value"
    )

    with pytest.raises(SkillError, match="base URL"):
        import_skill(source, tmp_path / "skills", skill_store)


def test_a_sql_skill_cannot_subscribe_to_ticket_events(
    tmp_path: Path, skill_store
) -> None:
    source = write_skill(
        tmp_path,
        "customer-sql",
        "",
        kind="sql",
        auth="{type: none}",
        customer_chats="none",
        events={"done": True},
        sql=f"{{engine: sqlite, sqlite_path: {tmp_path / 'customer.sqlite3'}}}",
    )

    with pytest.raises(SkillError, match="events"):
        import_skill(source, tmp_path / "skills", skill_store)
