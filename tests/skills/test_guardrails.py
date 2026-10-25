from pathlib import Path

import pytest
from tests.harness import RecordingWaha
from tests.skills.conftest import SENDER, json_body, write_skill

from wahabot.core.skills.guardrails import (
    EndpointRefused,
    ensure_public_endpoint,
    is_write_method,
    join_skill_url,
)
from wahabot.core.skills.loader import import_skill
from wahabot.core.skills.models import SkillConfig, SkillRun
from wahabot.core.skills.service import SkillService
from wahabot.core.tickets.store import TicketStore


def config(base_url: str, **overrides) -> SkillConfig:
    data = {"name": "customer-records", "base_url": base_url, **overrides}
    return SkillConfig.model_validate(data)


def test_a_path_escape_is_refused() -> None:
    skill = config("https://crm.example/api")

    with pytest.raises(EndpointRefused, match="path"):
        join_skill_url(skill, "../admin")


def test_an_absolute_url_in_the_path_is_refused() -> None:
    skill = config("https://crm.example/api")

    with pytest.raises(EndpointRefused, match="path"):
        join_skill_url(skill, "https://evil.example/steal")


def test_a_leading_slash_stays_under_the_base_path() -> None:
    skill = config("https://crm.example/api")

    assert join_skill_url(skill, "/records") == "https://crm.example/api/records"


def test_methods_are_classified_reads_and_writes() -> None:
    assert is_write_method("GET") is False
    assert is_write_method("HEAD") is False
    assert is_write_method("OPTIONS") is False
    assert is_write_method("POST") is True
    assert is_write_method("PATCH") is True


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/health",
        "http://10.0.0.4/health",
        "http://192.168.1.4/health",
        "http://[::1]/health",
        "http://169.254.169.254/latest/meta-data",
    ],
)
def test_private_and_metadata_addresses_are_refused_for_public_skills(url: str) -> None:
    with pytest.raises(EndpointRefused):
        ensure_public_endpoint(url, allow_private=False)


def test_metadata_stays_blocked_even_for_a_private_skill() -> None:
    with pytest.raises(EndpointRefused):
        ensure_public_endpoint("http://169.254.169.254/latest", allow_private=True)


def test_a_private_skill_may_reach_its_own_private_origin() -> None:
    ensure_public_endpoint("http://127.0.0.1:8080/health", allow_private=True)


def test_a_different_host_redirect_is_refused(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    folder = write_skill(tmp_path, "customer-records", fake_rest.base)
    import_skill(folder, tmp_path / "skills", skill_store)
    skill_store.set_enabled("customer-records", True)
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )
    run = SkillRun(session="test", surface="operator", armed=True)

    result = json_body(
        service.execute_http(
            "customer-records", "GET", "/redirect-other", None, None, run
        )
    )

    assert result["ok"] is False
    assert "host" in result["error"].lower() or "origin" in result["error"].lower()


def test_a_same_host_redirect_is_followed(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    folder = write_skill(tmp_path, "customer-records", fake_rest.base)
    import_skill(folder, tmp_path / "skills", skill_store)
    skill_store.set_enabled("customer-records", True)
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )
    run = SkillRun(session="test", surface="operator", armed=True)

    result = json_body(
        service.execute_http("customer-records", "GET", "/redirect-same", None, None, run)
    )

    assert result["ok"] is True
    assert result["body"] == {"ok": True}


def test_another_persons_phone_is_refused_in_a_customer_run(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    folder = write_skill(tmp_path, "customer-records", fake_rest.base)
    import_skill(folder, tmp_path / "skills", skill_store)
    skill_store.set_enabled("customer-records", True)
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )
    tickets = TicketStore(skill_store.path)
    tickets.upsert_contact(
        "491555000002@c.us", phone="491555000002@c.us", display_name="Other"
    )
    contact = tickets.upsert_contact(SENDER, phone=SENDER, display_name="Smoke")
    ticket = tickets.create_ticket(contact.id, SENDER)
    run = SkillRun(
        session="test",
        chat_id=SENDER,
        surface="customer",
        ticket_id=ticket.id,
        ticket_store=tickets,
    )

    result = json_body(
        service.execute_http(
            "customer-records",
            "GET",
            "/records",
            {"phone": "491555000002"},
            None,
            run,
        )
    )

    assert result["ok"] is False
    assert "another" in result["error"].lower() or "identifier" in result["error"].lower()


def test_an_operator_run_may_look_up_another_person(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    folder = write_skill(tmp_path, "customer-records", fake_rest.base)
    import_skill(folder, tmp_path / "skills", skill_store)
    skill_store.set_enabled("customer-records", True)
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )
    run = SkillRun(session="test", surface="operator", armed=True)

    result = json_body(
        service.execute_http(
            "customer-records", "GET", "/records", {"phone": "491555000002"}, None, run
        )
    )

    assert result["ok"] is True


def test_a_same_host_redirect_outside_the_base_path_is_refused(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    folder = write_skill(tmp_path, "customer-records", f"{fake_rest.base}/api")
    import_skill(folder, tmp_path / "skills", skill_store)
    skill_store.set_enabled("customer-records", True)
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )
    run = SkillRun(session="test", surface="operator", armed=True)

    result = json_body(
        service.execute_http(
            "customer-records", "GET", "/redirect-outside", None, None, run
        )
    )

    assert result["ok"] is False


def test_an_order_number_is_not_mistaken_for_a_foreign_phone(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    service, run, _tickets = _customer_run(tmp_path, skill_store, secret_store, fake_rest)

    result = json_body(
        service.execute_http(
            "customer-records", "GET", "/records", {"order": "123456789"}, None, run
        )
    )

    assert result["ok"] is True


def test_an_alphanumeric_external_ref_of_another_contact_is_refused(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    service, run, tickets = _customer_run(tmp_path, skill_store, secret_store, fake_rest)
    tickets.upsert_contact(
        "491555000002@c.us", phone="491555000002@c.us", external_user_ref="user-88"
    )

    result = json_body(
        service.execute_http(
            "customer-records", "GET", "/records", {"ref": "user-88"}, None, run
        )
    )

    assert result["ok"] is False


def _customer_run(tmp_path, skill_store, secret_store, fake_rest):
    folder = write_skill(tmp_path, "customer-records", fake_rest.base)
    import_skill(folder, tmp_path / "skills", skill_store)
    skill_store.set_enabled("customer-records", True)
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )
    tickets = TicketStore(skill_store.path)
    tickets.upsert_contact(
        "491555000002@c.us", phone="491555000002@c.us", display_name="Other"
    )
    contact = tickets.upsert_contact(SENDER, phone=SENDER, display_name="Smoke")
    ticket = tickets.create_ticket(contact.id, SENDER)
    run = SkillRun(
        session="test",
        chat_id=SENDER,
        surface="customer",
        ticket_id=ticket.id,
        ticket_store=tickets,
    )
    return service, run, tickets
