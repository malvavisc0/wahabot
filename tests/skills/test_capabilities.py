from pathlib import Path

from tests.harness import RecordingWaha
from tests.skills.conftest import SECRET_VALUE, write_skill

from wahabot.core.skills.loader import import_skill
from wahabot.core.skills.models import SkillRun
from wahabot.core.skills.service import SkillService


def test_large_docs_use_the_read_skill_fallback_and_cache_per_run(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    docs = "reference " * 5000
    folder = write_skill(tmp_path, "customer-records", fake_rest.base, docs=docs)
    import_skill(folder, tmp_path / "skills", skill_store)
    skill_store.set_enabled("customer-records", True)
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )
    run = SkillRun(session="test", surface="operator", armed=True)

    section = service.prompt_section(run)

    assert "customer-records" in section
    assert "read_skill" in section
    assert docs not in section
    assert run.skill_docs == {}
    first = service.read_skill("customer-records", run)
    second = service.read_skill("customer-records", run)
    assert docs.strip() in first
    assert first == second
    assert "customer-records" in run.skill_docs


def test_capabilities_are_plain_language_and_never_include_urls_or_secrets(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    folder = write_skill(
        tmp_path,
        "customer-records",
        fake_rest.base,
        docs="Look up a customer's open records by phone.",
    )
    import_skill(folder, tmp_path / "skills", skill_store)
    skill_store.set_enabled("customer-records", True)
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )

    summary = service.capability_summary()

    assert "customer-records" in summary
    assert "open records" in summary
    assert fake_rest.base not in summary
    assert SECRET_VALUE not in summary
