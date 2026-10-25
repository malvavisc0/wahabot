from pathlib import Path
from urllib.parse import urlsplit

import pytest
from tests.harness import RecordingWaha
from tests.harness.rest import FakeRest
from tests.skills.conftest import SENDER, TEAM_GROUP, json_body, write_skill

from wahabot.core.skills.guardrails import (
    EndpointRefused,
    ensure_public_endpoint,
    is_blocked_ip,
    is_write_method,
    join_skill_url,
    same_origin,
    sender_values,
    url_under_base,
)
from wahabot.core.skills.loader import import_skill
from wahabot.core.skills.models import SkillConfig, SkillRun
from wahabot.core.skills.secrets import SecretStore
from wahabot.core.skills.service import SkillService
from wahabot.core.skills.store import SkillStore
from wahabot.core.skills.support import prepare_request, substitute_body
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


def test_another_persons_phone_is_refused_in_acustomer_run(
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
    service, run, _tickets = customer_run(tmp_path, skill_store, secret_store, fake_rest)

    result = json_body(
        service.execute_http(
            "customer-records", "GET", "/records", {"order": "123456789"}, None, run
        )
    )

    assert result["ok"] is True


def test_an_alphanumeric_external_ref_of_another_contact_is_refused(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    service, run, tickets = customer_run(tmp_path, skill_store, secret_store, fake_rest)
    tickets.upsert_contact(
        "491555000002@c.us", phone="491555000002@c.us", external_user_ref="user-88"
    )

    result = json_body(
        service.execute_http(
            "customer-records", "GET", "/records", {"ref": "user-88"}, None, run
        )
    )

    assert result["ok"] is False


def test_an_encoded_backslash_path_is_refused() -> None:
    skill = config("https://crm.example/api")

    with pytest.raises(EndpointRefused):
        join_skill_url(skill, "/records%5C..%5Cadmin")
    with pytest.raises(EndpointRefused):
        join_skill_url(skill, "/%252e%252e/admin")
    with pytest.raises(EndpointRefused):
        join_skill_url(skill, "/records#")


def test_default_ports_count_as_the_same_origin() -> None:
    assert same_origin("https://crm.example/api", "https://crm.example:443/api")
    assert not url_under_base(
        config("https://crm.example:443/api"), "https://crm.example/records"
    )


def test_shared_and_cloud_metadata_addresses_are_blocked() -> None:
    for raw in (
        "100.100.100.200",
        "168.63.129.16",
        "100.64.0.1",
        "::ffff:169.254.169.254",
    ):
        assert is_blocked_ip(raw)


def test_a_team_group_uses_the_participant_as_the_sender(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    service, _run, tickets = customer_run(tmp_path, skill_store, secret_store, fake_rest)
    run = SkillRun(
        session="test",
        chat_id=TEAM_GROUP,
        sender_id=SENDER,
        surface="team_group",
        ticket_store=tickets,
    )

    result = json_body(
        service.execute_http(
            "customer-records",
            "GET",
            "/records",
            {"phone": "{sender.phone}"},
            None,
            run,
        )
    )

    assert result["ok"] is True
    assert result["body"]["phone"] == "491555000001"


def test_a_sender_value_cannot_inject_a_query_or_escape_a_path() -> None:
    run = SkillRun(session="test", chat_id=SENDER, sender_id="a/b?c")
    prepared = prepare_request(
        config("https://crm.example/api"),
        "GET",
        "/records/{sender.chat_id}",
        {"phone": "{sender.chat_id}"},
        None,
        run,
    )

    parsed = urlsplit(prepared.url)
    assert parsed.path.endswith("/records/a%2Fb%3Fc")
    assert parsed.query == "phone=a%2Fb%3Fc"


def test_a_body_keeps_its_shape_when_a_value_contains_json_syntax() -> None:
    run = SkillRun(session="test", chat_id=SENDER, sender_id='a"b\\c')
    body = {"note": {"id": "{sender.chat_id}"}, "items": ["{sender.chat_id}"]}

    assert substitute_body(body, run) == {
        "note": {"id": 'a"b\\c'},
        "items": ['a"b\\c'],
    }


def test_an_encoded_foreign_jid_in_the_path_is_refused(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    from wahabot.core.skills.support import claim_request

    _service, run, _tickets = customer_run(tmp_path, skill_store, secret_store, fake_rest)
    prepared = prepare_request(
        config("https://crm.example/api"),
        "GET",
        "/records/491555000002%40c.us",
        None,
        None,
        run,
    )

    assert "another chat" in (claim_request("customer-records", prepared, run) or "")
    second_run = SkillRun(
        session="test",
        chat_id=SENDER,
        surface="customer",
        ticket_id=run.ticket_id,
        ticket_store=run.ticket_store,
    )
    encoded = prepare_request(
        config("https://crm.example/api"),
        "GET",
        "/records/491555000002%2540c.us",
        None,
        None,
        second_run,
    )
    assert "another chat" in (
        claim_request("customer-records", encoded, second_run) or ""
    )


def test_sender_values_use_the_participant_in_a_team_group() -> None:
    run = SkillRun(
        session="test",
        chat_id=TEAM_GROUP,
        sender_id=SENDER,
        surface="team_group",
    )

    assert sender_values(run, required={"sender.chat_id"})["sender.chat_id"] == SENDER


def test_a_lid_participant_uses_the_resolved_phone_alias() -> None:
    run = SkillRun(
        session="test",
        chat_id=TEAM_GROUP,
        sender_id="123456789@lid",
        sender_phone="491555000009@c.us",
        surface="team_group",
    )

    values = sender_values(run, required={"sender.phone"})

    assert values["sender.phone"] == "491555000009"


def test_an_unknown_team_sender_is_not_the_group() -> None:
    run = SkillRun(
        session="test",
        chat_id=TEAM_GROUP,
        sender_id=TEAM_GROUP,
        surface="team_group",
    )

    values = sender_values(run, required=set())

    assert values["sender.chat_id"] == ""
    assert values["sender.phone"] == ""


def test_sender_values_without_a_requirement_allow_an_empty_ticket() -> None:
    run = SkillRun(
        session="test",
        chat_id=TEAM_GROUP,
        sender_id=SENDER,
        surface="team_group",
    )

    assert sender_values(run)["ticket.number"] == ""


def customer_run(
    tmp_path: Path,
    skill_store: SkillStore,
    secret_store: SecretStore,
    fake_rest: FakeRest,
) -> tuple[SkillService, SkillRun, TicketStore]:
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
