import hashlib
import hmac
import json
from pathlib import Path

from tests.harness import RecordingWaha
from tests.skills.conftest import SENDER, SIGNING_SECRET, write_skill

from wahabot.core.skills.events import EventDispatcher
from wahabot.core.skills.loader import PREDEFINED_DIR, import_skill, load_skill
from wahabot.core.skills.service import SkillService
from wahabot.core.tickets.models import TicketUpdate
from wahabot.core.tickets.store import TicketStore


def test_the_four_predefined_skills_ship_and_validate() -> None:
    names = sorted(path.name for path in PREDEFINED_DIR.iterdir() if path.is_dir())

    assert names == ["postgres-readonly", "rest", "send-to-url", "supabase"]
    for name in names:
        assert (PREDEFINED_DIR / name / "SKILL.md").is_file()
        assert load_skill(name, PREDEFINED_DIR).name == name


def test_an_event_payload_is_signed_and_carries_no_message_body(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    secret_store.put("SIGNING_KEY", SIGNING_SECRET)
    folder = write_skill(
        tmp_path,
        "send-to-url",
        fake_rest.base,
        auth="{type: none}",
        customer_chats="none",
        sign_events=True,
        signing_secret="SIGNING_KEY",
        events={"done": True},
    )
    import_skill(folder, tmp_path / "skills", skill_store)
    skill_store.set_enabled("send-to-url", True)
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )
    tickets = TicketStore(skill_store.path, on_event=EventDispatcher(service).emit)
    contact = tickets.upsert_contact(
        "491555000001@c.us", phone="491555000001@c.us", display_name="Smoke Sender"
    )
    ticket = tickets.create_ticket(
        contact.id,
        "491555000001@c.us",
        summary="refund request",
        external_user_ref="user-88",
    )

    tickets.update_ticket(
        ticket.id,
        TicketUpdate(state="done", done_by="assistant"),
        actor="assistant",
        body="resolved internally",
    )

    hook = fake_rest.hooks[-1]
    signature = str(hook["signature"]).removeprefix("sha256=")
    expected = hmac.new(SIGNING_SECRET.encode(), hook["body"], hashlib.sha256).hexdigest()
    payload = json.loads(hook["body"])
    assert hmac.compare_digest(signature, expected)
    assert payload["event"] == "done"
    assert payload["ticket"] == ticket.number
    assert payload["event_id"] == hook["event_id"]
    assert payload["contact_ref"] == "user-88"
    assert "body" not in payload
    assert "resolved internally" not in str(payload)


def test_a_disabled_event_skill_does_not_fire(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    secret_store.put("SIGNING_KEY", SIGNING_SECRET)
    folder = write_skill(
        tmp_path,
        "send-to-url",
        fake_rest.base,
        auth="{type: none}",
        customer_chats="none",
        sign_events=True,
        signing_secret="SIGNING_KEY",
        events={"done": True},
    )
    import_skill(folder, tmp_path / "skills", skill_store)
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )
    tickets = TicketStore(skill_store.path, on_event=EventDispatcher(service).emit)
    contact = tickets.upsert_contact("491555000001@c.us", phone="491555000001@c.us")
    ticket = tickets.create_ticket(contact.id, "491555000001@c.us")

    tickets.update_ticket(
        ticket.id, TicketUpdate(state="done", done_by="assistant"), actor="assistant"
    )

    assert fake_rest.hooks == []


def test_tickets_and_answers_work_with_no_skill_configured(
    tmp_path: Path, skill_store
) -> None:
    tickets = TicketStore(skill_store.path)
    contact = tickets.upsert_contact("491555000001@c.us", phone="491555000001@c.us")

    ticket = tickets.create_ticket(contact.id, "491555000001@c.us", summary="help")
    tickets.update_ticket(
        ticket.id, TicketUpdate(state="done", done_by="assistant"), actor="assistant"
    )

    assert tickets.ticket(ticket.id).state == "done"


def test_all_four_ticket_events_fire_after_the_transition(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    secret_store.put("SIGNING_KEY", SIGNING_SECRET)
    folder = write_skill(
        tmp_path,
        "send-to-url",
        fake_rest.base,
        auth="{type: none}",
        customer_chats="none",
        sign_events=True,
        signing_secret="SIGNING_KEY",
        events={"opened": True, "needs_you": True, "done": True, "closed": True},
    )
    import_skill(folder, tmp_path / "skills", skill_store)
    skill_store.set_enabled("send-to-url", True)
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )
    tickets = TicketStore(skill_store.path, on_event=EventDispatcher(service).emit)
    contact = tickets.upsert_contact(SENDER, phone=SENDER)

    ticket = tickets.create_ticket(contact.id, SENDER)
    tickets.update_ticket(ticket.id, TicketUpdate(state="needs_you"), actor="assistant")
    tickets.update_ticket(ticket.id, TicketUpdate(state="done"), actor="assistant")
    tickets.update_ticket(ticket.id, TicketUpdate(state="closed"), actor="assistant")

    events = [json.loads(hook["body"])["event"] for hook in fake_rest.hooks]
    assert events == ["opened", "needs_you", "done", "closed"]
    assert all("body" not in json.loads(hook["body"]) for hook in fake_rest.hooks)


def test_active_state_shuffles_do_not_repeat_opened(tmp_path: Path, skill_store) -> None:
    events: list[str] = []
    tickets = TicketStore(
        skill_store.path, on_event=lambda kind, _ticket: events.append(kind)
    )
    contact = tickets.upsert_contact(SENDER, phone=SENDER)
    ticket = tickets.create_ticket(contact.id, SENDER)

    tickets.update_ticket(
        ticket.id, TicketUpdate(state="waiting_customer"), actor="assistant"
    )
    tickets.update_ticket(ticket.id, TicketUpdate(state="open"), actor="assistant")
    tickets.update_ticket(ticket.id, TicketUpdate(state="done"), actor="assistant")
    tickets.update_ticket(ticket.id, TicketUpdate(state="open"), actor="assistant")
    tickets.update_ticket(ticket.id, TicketUpdate(state="closed"), actor="assistant")

    assert events == ["opened", "done", "opened", "closed"]


def test_event_http_errors_are_retried_and_counted(
    tmp_path: Path, skill_store, secret_store, fake_rest
) -> None:
    folder = write_skill(
        tmp_path,
        "send-to-url",
        fake_rest.base,
        auth="{type: none}",
        customer_chats="none",
        events={"done": True},
    )
    import_skill(folder, tmp_path / "skills", skill_store)
    skill_store.set_enabled("send-to-url", True)
    fake_rest.hook_status = 500
    service = SkillService(
        skill_store, secret_store, tmp_path / "skills", "test", RecordingWaha()
    )
    tickets = TicketStore(skill_store.path, on_event=EventDispatcher(service).emit)
    contact = tickets.upsert_contact(SENDER, phone=SENDER)
    ticket = tickets.create_ticket(contact.id, SENDER)

    tickets.update_ticket(ticket.id, TicketUpdate(state="done", done_by="assistant"))

    assert len(fake_rest.hooks) == 3
    skill = skill_store.get("send-to-url")
    assert skill.failure_count == 3
    assert skill.test_state == "failed"
