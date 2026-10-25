from pathlib import Path

from tests.harness import ME_JID, RecordingWaha
from tests.skills.conftest import SENDER, write_skill

from wahabot.core.skills.loader import import_skill
from wahabot.core.skills.models import SkillRun
from wahabot.core.skills.service import SkillService
from wahabot.core.skills.store import PENDING_REQUEST_TTL_S, SkillStore
from wahabot.core.tickets.router import RouteKind, route_skill_confirmation
from wahabot.core.tickets.store import TicketStore
from wahabot.status import state as status_state


def test_an_unquoted_yes_is_ambiguous_when_several_requests_wait(
    skill_store: SkillStore,
) -> None:
    first = skill_store.create_request(
        skill="a", ticket_id=None, chat_id="c1", request_json="{}", summary="one"
    )
    skill_store.create_request(
        skill="b", ticket_id=None, chat_id="c2", request_json="{}", summary="two"
    )
    skill_store.set_operator_message(first.id, "message-a")

    ambiguous = route_skill_confirmation(skill_store, "yes", "")
    quoted = route_skill_confirmation(skill_store, "yes", "message-a")

    assert ambiguous is not None
    assert ambiguous.kind == RouteKind.SKILL_CONFIRM
    assert ambiguous.skill_request is None
    assert quoted is not None
    assert quoted.skill_request is not None
    assert quoted.skill_request.id == first.id


def test_old_pending_requests_expire(skill_store: SkillStore) -> None:
    skill_store.clock = lambda: 1_000
    request = skill_store.create_request(
        skill="a", ticket_id=None, chat_id="c1", request_json="{}", summary="one"
    )
    skill_store.clock = lambda: 1_000 + PENDING_REQUEST_TTL_S + 1

    assert skill_store.pending_requests() == []
    assert skill_store.request(request.id).state == "refused"


def test_a_confirmation_after_a_base_change_fails_and_is_audited(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    folder = write_skill(
        tmp_path,
        "customer-records",
        f"{fake_rest.base}/old",
        customer_chats="read_write",
    )
    import_skill(folder, tmp_path / "skills", skill_store)
    skill_store.set_enabled("customer-records", True)
    tickets = TicketStore(skill_store.path)
    contact = tickets.upsert_contact(SENDER, phone=SENDER)
    ticket = tickets.create_ticket(contact.id, SENDER)
    service = SkillService(
        skill_store,
        secret_store,
        tmp_path / "skills",
        "test",
        RecordingWaha(),
        ticket_store=tickets,
    )
    run = SkillRun(
        session="test",
        chat_id=SENDER,
        surface="customer",
        ticket_id=ticket.id,
        ticket_store=tickets,
    )
    previous = status_state.operator_jid
    status_state.operator_jid = ME_JID
    try:
        staged = service.execute_http(
            "customer-records", "POST", "/leads", None, {"name": "Smoke"}, run
        )
        assert '"staged": true' in staged
        pending = skill_store.latest_pending()
        assert pending is not None
        assert any(event.kind == "skill_call" for event in tickets.events(ticket.id))

        config = tmp_path / "skills" / "customer-records" / "skill.yaml"
        config.write_text(config.read_text().replace("/old", "/new"))

        result = service.confirm(pending.id, approve=True)

        assert "failed" in result.lower()
        assert fake_rest.lead_count() == 0
        assert skill_store.request(pending.id).state == "failed"
        assert any(
            call.method == "CONFIRM" and call.outcome == "failed"
            for call in skill_store.calls("customer-records")
        )
    finally:
        status_state.operator_jid = previous


def test_declining_a_staged_write_is_audited(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    folder = write_skill(
        tmp_path, "customer-records", fake_rest.base, customer_chats="read_write"
    )
    import_skill(folder, tmp_path / "skills", skill_store)
    skill_store.set_enabled("customer-records", True)
    tickets = TicketStore(skill_store.path)
    contact = tickets.upsert_contact(SENDER, phone=SENDER)
    ticket = tickets.create_ticket(contact.id, SENDER)
    service = SkillService(
        skill_store,
        secret_store,
        tmp_path / "skills",
        "test",
        RecordingWaha(),
        ticket_store=tickets,
    )
    run = SkillRun(
        session="test",
        chat_id=SENDER,
        surface="customer",
        ticket_id=ticket.id,
        ticket_store=tickets,
    )
    previous = status_state.operator_jid
    status_state.operator_jid = ME_JID
    try:
        service.execute_http(
            "customer-records", "POST", "/leads", None, {"name": "Smoke"}, run
        )
        pending = skill_store.latest_pending()
        assert pending is not None

        assert service.confirm(pending.id, approve=False) == "Request declined."

        assert skill_store.request(pending.id).state == "refused"
        assert any(
            call.method == "CONFIRM" and call.outcome == "refused"
            for call in skill_store.calls("customer-records")
        )
    finally:
        status_state.operator_jid = previous


def test_a_permission_change_blocks_a_staged_write(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    from wahabot.core.skills.drafts import SkillDraftService

    folder = write_skill(
        tmp_path, "customer-records", fake_rest.base, customer_chats="read_write"
    )
    import_skill(folder, tmp_path / "skills", skill_store)
    tickets = TicketStore(skill_store.path)
    contact = tickets.upsert_contact(SENDER, phone=SENDER)
    ticket = tickets.create_ticket(contact.id, SENDER)
    service = SkillService(
        skill_store,
        secret_store,
        tmp_path / "skills",
        "test",
        RecordingWaha(),
        ticket_store=tickets,
    )
    drafts = SkillDraftService(skill_store, secret_store, tmp_path / "skills", service)
    assert drafts.run_test("customer-records").passed is True
    assert drafts.enable("customer-records") is True
    run = SkillRun(
        session="test",
        chat_id=SENDER,
        surface="customer",
        ticket_id=ticket.id,
        ticket_store=tickets,
    )
    previous = status_state.operator_jid
    status_state.operator_jid = ME_JID
    try:
        service.execute_http(
            "customer-records", "POST", "/leads", None, {"name": "Smoke"}, run
        )
        pending = skill_store.latest_pending()
        assert pending is not None
        config = tmp_path / "skills" / "customer-records" / "skill.yaml"
        config.write_text(
            config.read_text().replace(
                "writes_need_operator: true", "writes_need_operator: false"
            )
        )

        result = service.confirm(pending.id, approve=True)

        assert "failed" in result.lower()
        assert fake_rest.lead_count() == 0
        assert skill_store.request(pending.id).state == "failed"
    finally:
        status_state.operator_jid = previous
