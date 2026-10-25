from pathlib import Path

from tests.harness import RecordingWaha
from tests.skills.conftest import SENDER, json_body, write_skill

from wahabot.core.skills.loader import import_skill
from wahabot.core.skills.models import SkillRun
from wahabot.core.skills.service import SkillService
from wahabot.core.tickets.store import TicketStore


def install(tmp_path: Path, skill_store, secret_store, fake_rest, **overrides):
    folder = write_skill(
        tmp_path,
        "customer-records",
        fake_rest.base,
        **overrides,
    )
    import_skill(folder, tmp_path / "skills", skill_store)
    skill_store.set_enabled("customer-records", True)
    transport = RecordingWaha()
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", transport
    )
    tickets = TicketStore(skill_store.path)
    tickets.upsert_contact(
        "491555000002@c.us", phone="491555000002@c.us", display_name="Other"
    )
    contact = tickets.upsert_contact(SENDER, phone=SENDER, display_name="Smoke Sender")
    ticket = tickets.create_ticket(contact.id, SENDER, summary="smoke")
    run = SkillRun(
        session="test",
        chat_id=SENDER,
        surface="customer",
        ticket_id=ticket.id,
        ticket_store=tickets,
    )
    return service, run, transport, tickets


def test_a_read_lookup_returns_customer_data_and_substitutes_the_sender(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    service, run, _transport, _tickets = install(
        tmp_path, skill_store, secret_store, fake_rest
    )

    result = json_body(
        service.execute_http(
            "customer-records", "GET", "/records", {"phone": "{sender.phone}"}, None, run
        )
    )

    assert result["ok"] is True
    assert result["status"] == 200
    assert result["body"]["phone"] == "491555000001"


def test_the_secret_is_added_at_call_time_and_never_echoed(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    service, run, _transport, _tickets = install(
        tmp_path, skill_store, secret_store, fake_rest
    )

    result = service.execute_http(
        "customer-records", "GET", "/secret-echo", None, None, run
    )

    assert "sk-fictional-smoke-key" not in result
    assert "***" in result


def test_a_repeated_identical_call_is_refused(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    service, run, _transport, _tickets = install(
        tmp_path, skill_store, secret_store, fake_rest
    )
    service.execute_http("customer-records", "GET", "/health", None, None, run)

    result = json_body(
        service.execute_http("customer-records", "GET", "/health", None, None, run)
    )

    assert result["ok"] is False
    assert "repeat" in result["error"].lower()


def test_a_turn_is_limited_to_five_calls(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    service, run, _transport, _tickets = install(
        tmp_path, skill_store, secret_store, fake_rest
    )
    for index in range(5):
        result = json_body(
            service.execute_http(
                "customer-records", "GET", f"/records?n={index}", None, None, run
            )
        )
        assert result["ok"] is True

    refused = json_body(
        service.execute_http("customer-records", "GET", "/records?n=6", None, None, run)
    )

    assert refused["ok"] is False
    assert "limit" in refused["error"].lower()


def test_a_large_response_is_cut_with_a_note(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    service, run, _transport, _tickets = install(
        tmp_path, skill_store, secret_store, fake_rest
    )

    result = json_body(
        service.execute_http("customer-records", "GET", "/large", None, None, run)
    )

    assert result["ok"] is True
    assert result["truncated"] is True
    assert len(str(result["body"])) < 100_000


def test_an_oversized_request_body_is_refused(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    service, run, _transport, _tickets = install(
        tmp_path, skill_store, secret_store, fake_rest, customer_chats="read_write"
    )

    result = json_body(
        service.execute_http(
            "customer-records", "POST", "/leads", None, {"blob": "x" * 40_000}, run
        )
    )

    assert result["ok"] is False
    assert "body" in result["error"].lower()


def test_a_timeout_is_reported_without_crashing(
    tmp_path: Path, skill_store, secret_store, fake_rest, monkeypatch
) -> None:
    import wahabot.core.skills.http as http

    monkeypatch.setattr(http, "REQUEST_TIMEOUT_S", 0.05)
    service, run, _transport, _tickets = install(
        tmp_path, skill_store, secret_store, fake_rest
    )

    result = json_body(
        service.execute_http("customer-records", "GET", "/slow", None, None, run)
    )

    assert result["ok"] is False
    assert "timed out" in result["error"].lower()


def test_every_call_is_logged_with_its_safe_path(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    service, run, _transport, _tickets = install(
        tmp_path, skill_store, secret_store, fake_rest
    )

    service.execute_http("customer-records", "GET", "/records", None, None, run)

    calls = skill_store.calls("customer-records")
    assert len(calls) == 1
    assert calls[0].method == "GET"
    assert calls[0].path == "/records"
    assert calls[0].status == 200
    assert calls[0].outcome == "ok"
    assert "sk-fictional-smoke-key" not in str(calls[0])


def test_system_runs_are_attributed_to_system() -> None:
    from wahabot.core.skills.execution import initiator

    assert initiator(SkillRun(session="test", surface="system", armed=True)) == "system"


def test_redirect_methods_follow_http_semantics() -> None:
    from wahabot.core.skills.http import redirected_request

    assert redirected_request("POST", b"x", 302) == ("GET", None)
    assert redirected_request("POST", b"x", 303) == ("GET", None)
    assert redirected_request("HEAD", b"", 303) == ("HEAD", b"")
    assert redirected_request("POST", b"x", 307) == ("POST", b"x")
    assert redirected_request("PUT", b"x", 308) == ("PUT", b"x")
