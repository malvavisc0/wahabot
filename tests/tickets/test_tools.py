import json
from collections.abc import Iterator
from contextlib import contextmanager

from tests.harness import ME_JID, RecordingWaha
from wahabot.ai.tools.runcontext import RunTarget, bind_target, reset_target
from wahabot.ai.tools.tickets import (
    hand_to_operator,
    new_ticket,
    support_state,
    update_ticket,
)
from wahabot.core.tickets.models import Contact, Ticket, TicketState, TicketUpdate
from wahabot.core.tickets.store import TicketStore
from wahabot.settings import Settings

CUSTOMER = "491555000001@c.us"
OTHER = "491555000002@c.us"


@contextmanager
def target_for(target: RunTarget) -> Iterator[RunTarget]:
    token = bind_target(target)
    try:
        yield target
    finally:
        reset_target(token)


def make_ticket(
    store: TicketStore, chat: str = CUSTOMER, name: str = "Smoke Sender"
) -> tuple[Contact, Ticket]:
    contact = store.upsert_contact(chat, display_name=name)
    ticket = store.create_ticket(contact.id, chat, summary="issue")
    return contact, ticket


def test_customer_support_state_is_fenced_to_its_own_chat(
    store: TicketStore,
) -> None:
    _, mine = make_ticket(store)
    _, other = make_ticket(store, OTHER, "Other Person")
    tool = support_state()
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        ticket_store=store,
        ticket_id=mine.id,
    )

    with target_for(target):
        payload = json.loads(tool.fn(number=other.number, name="Other Person"))

    assert payload["ticket"]["number"] == mine.number
    assert payload["contact"]["name"] == "Smoke Sender"


def test_operator_support_state_can_look_up_by_number(
    store: TicketStore,
) -> None:
    contact, ticket = make_ticket(store)
    tool = support_state()
    target = RunTarget(
        session="default",
        chat_id=ME_JID,
        ticket_store=store,
        armed=True,
    )

    with target_for(target):
        payload = json.loads(tool.fn(number=ticket.number))

    assert payload["ticket"]["number"] == ticket.number
    assert payload["contact"]["name"] == contact.display_name


def test_update_ticket_stages_and_refuses_operator_resolver(
    store: TicketStore,
) -> None:
    _, ticket = make_ticket(store)
    tool = update_ticket()
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        ticket_store=store,
        ticket_id=ticket.id,
    )

    with target_for(target):
        staged = json.loads(tool.fn(summary="new summary", missing=["address"]))
        refused = json.loads(tool.fn(done_by="operator"))

    assert staged["ok"] and target.ticket_update is not None
    assert target.ticket_update.summary == "new summary"
    assert target.ticket_update.missing == ["address"]
    assert not refused["ok"]
    assert store.ticket(ticket.id).summary == "issue"


def test_handover_sends_the_alert_and_stages_needs_you(
    store: TicketStore, settings: Settings, monkeypatch
) -> None:
    from wahabot.status.state import state as status_state

    monkeypatch.setattr(status_state, "operator_jid", ME_JID)
    _, ticket = make_ticket(store)
    store.add_event(ticket.id, "customer", "customer_message", "I need a person")
    waha = RecordingWaha()
    tool = hand_to_operator(settings, waha)
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        ticket_store=store,
        ticket_id=ticket.id,
    )

    with target_for(target):
        payload = json.loads(tool.fn(reason="needs a person"))

    assert payload["ok"]
    assert store.ticket(ticket.id).state == TicketState.NEEDS_YOU.value
    assert target.ticket_update is not None
    assert target.ticket_update.state == TicketState.NEEDS_YOU.value
    assert waha.sent[0][1] == ME_JID


def test_handover_refuses_paths_and_foreign_message_ids(
    store: TicketStore, settings: Settings, monkeypatch
) -> None:
    from wahabot.status.state import state as status_state

    monkeypatch.setattr(status_state, "operator_jid", ME_JID)
    _, ticket = make_ticket(store)
    tool = hand_to_operator(settings, RecordingWaha())
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        ticket_store=store,
        ticket_id=ticket.id,
    )

    with target_for(target):
        path = json.loads(tool.fn(reason="x", attach_message_ids=["/tmp/secret"]))
        foreign = json.loads(
            tool.fn(reason="x", attach_message_ids=[f"false_{OTHER}_FOREIGN"])
        )

    assert not path["ok"] and "path" in path["error"]
    assert not foreign["ok"] and "conversation" in foreign["error"]


def test_new_ticket_stages_a_split_and_refuses_needs_you(
    store: TicketStore,
) -> None:
    _, ticket = make_ticket(store)
    tool = new_ticket()
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        ticket_store=store,
        ticket_id=ticket.id,
    )

    with target_for(target):
        staged = json.loads(tool.fn(summary="second issue"))

    assert staged["ok"] and target.ticket_split == "second issue"

    store.update_ticket(ticket.id, TicketUpdate(state=TicketState.NEEDS_YOU.value))
    with target_for(target):
        refused = json.loads(tool.fn(summary="second issue"))

    assert not refused["ok"]
    assert "open or done" in refused["error"]

    store.update_ticket(ticket.id, TicketUpdate(state=TicketState.CLOSED.value))
    with target_for(target):
        closed = json.loads(tool.fn(summary="second issue"))

    assert not closed["ok"]
    assert "open or done" in closed["error"]


def test_escalate_refuses_inside_a_ticket_run(
    store: TicketStore, settings: Settings
) -> None:
    from wahabot.ai.tools.whatsapp.escalation import EscalationChannel, escalate

    _, ticket = make_ticket(store)
    tool = escalate(RecordingWaha(), EscalationChannel(), settings)
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        ticket_store=store,
        ticket_id=ticket.id,
    )

    with target_for(target):
        payload = json.loads(tool.fn(report="a human is needed"))

    assert not payload["ok"]
    assert "hand_to_operator" in payload["error"]


def test_handover_reads_the_operator_language_at_call_time(
    store: TicketStore, settings: Settings, monkeypatch
) -> None:
    from wahabot.status.state import state as status_state

    monkeypatch.setattr(status_state, "operator_jid", ME_JID)
    _, ticket = make_ticket(store)
    store.add_event(ticket.id, "customer", "customer_message", "I need a person")
    language = {"value": ""}
    translated: list[str] = []

    def translator(text: str, target: str) -> str:
        translated.append(target)
        return "TRADUCIDO"

    tool = hand_to_operator(
        settings,
        RecordingWaha(),
        language=lambda: language["value"],
        translator=translator,
    )
    language["value"] = "es"
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        ticket_store=store,
        ticket_id=ticket.id,
    )

    with target_for(target):
        payload = json.loads(tool.fn(reason="needs a person"))

    assert payload["ok"]
    assert translated == ["es"]


def test_handover_alert_includes_staged_values_and_urgency(
    store: TicketStore, settings: Settings, monkeypatch
) -> None:
    from wahabot.status.state import state as status_state

    monkeypatch.setattr(status_state, "operator_jid", ME_JID)
    _, ticket = make_ticket(store)
    store.add_event(ticket.id, "customer", "customer_message", "I need a person")
    waha = RecordingWaha()
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        ticket_store=store,
        ticket_id=ticket.id,
    )

    with target_for(target):
        update_ticket().fn(
            summary="late order",
            details={"order": "A-17"},
            missing=["address"],
        )
        payload = json.loads(
            hand_to_operator(settings, waha).fn(reason="needs a person", urgent=True)
        )

    text = waha.sent[0][2]
    assert payload["ok"]
    assert text.startswith("URGENT")
    assert "order: A-17" in text
    assert "Missing: address" in text
    stored = store.ticket(ticket.id)
    assert stored.urgent is True
    assert stored.details == {"order": "A-17"}
    assert stored.missing == ["address"]
    assert stored.summary == "late order"


def test_urgent_escalation_after_an_alert_is_not_silently_dropped(
    store: TicketStore, settings: Settings, monkeypatch
) -> None:
    from wahabot.status.state import state as status_state

    monkeypatch.setattr(status_state, "operator_jid", ME_JID)
    _, ticket = make_ticket(store)
    store.add_event(ticket.id, "customer", "customer_message", "I need a person")
    waha = RecordingWaha()
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        ticket_store=store,
        ticket_id=ticket.id,
    )

    with target_for(target):
        tool = hand_to_operator(settings, waha)
        first = json.loads(tool.fn(reason="needs a person"))
        urgent = json.loads(tool.fn(reason="now urgent", urgent=True))

    assert first["ok"] and urgent["ok"]
    assert len(waha.sent) == 2
    assert waha.sent[1][2].startswith("URGENT")
    assert store.ticket(ticket.id).urgent is True


def test_handover_refuses_a_ticket_closed_mid_run(
    store: TicketStore, settings: Settings, monkeypatch
) -> None:
    from wahabot.status.state import state as status_state

    monkeypatch.setattr(status_state, "operator_jid", ME_JID)
    _, ticket = make_ticket(store)
    store.update_ticket(ticket.id, TicketUpdate(state=TicketState.CLOSED.value))
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        ticket_store=store,
        ticket_id=ticket.id,
    )

    with target_for(target):
        payload = json.loads(
            hand_to_operator(settings, RecordingWaha()).fn(reason="needs a person")
        )

    assert not payload["ok"]
    assert "closed" in payload["error"]
    assert store.ticket(ticket.id).state == TicketState.CLOSED.value


def test_split_is_refused_with_a_staged_active_state(store: TicketStore) -> None:
    _, ticket = make_ticket(store)
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        ticket_store=store,
        ticket_id=ticket.id,
    )

    with target_for(target):
        update_ticket().fn(state=TicketState.WAITING_CUSTOMER.value)
        refused = json.loads(new_ticket().fn(summary="second issue"))

    assert not refused["ok"]


def test_finalize_forces_done_after_a_split_and_records_the_reply(
    store: TicketStore,
) -> None:
    from wahabot.handlers.ticket_mixin import TicketMixin

    _, ticket = make_ticket(store)
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        ticket_store=store,
        ticket_id=ticket.id,
        ticket_split="second issue",
        ticket_update=TicketUpdate(state=TicketState.WAITING_CUSTOMER.value),
    )

    TicketMixin().finalize_ticket(target, "reply", "here is your answer")

    old = store.ticket(ticket.id)
    tickets = store.chat_tickets(CUSTOMER, limit=5)
    assert old.state == TicketState.DONE.value
    assert [item.state for item in tickets].count(TicketState.OPEN.value) == 1
    replies = [item.body for item in store.events(ticket.id) if item.kind == "reply"]
    assert replies == ["here is your answer"]


def test_finalize_does_not_reopen_a_ticket_closed_mid_run(store: TicketStore) -> None:
    from wahabot.handlers.ticket_mixin import TicketMixin

    _, ticket = make_ticket(store)
    store.update_ticket(ticket.id, TicketUpdate(state=TicketState.CLOSED.value), at=1000)
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        ticket_store=store,
        ticket_id=ticket.id,
        ticket_update=TicketUpdate(state=TicketState.WAITING_CUSTOMER.value),
    )

    TicketMixin().finalize_ticket(target, "reply", "late answer")

    closed = store.ticket(ticket.id)
    assert closed.state == TicketState.CLOSED.value
    assert closed.closed_at is not None


def test_update_ticket_merges_details_and_missing_within_a_run(
    store: TicketStore,
) -> None:
    _, ticket = make_ticket(store)
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        ticket_store=store,
        ticket_id=ticket.id,
    )

    with target_for(target):
        tool = update_ticket()
        tool.fn(details={"order": "A-17"}, missing=["address"])
        tool.fn(details={"address": "Main St"}, missing=["phone"])

    assert target.ticket_update is not None
    assert target.ticket_update.details == {
        "order": "A-17",
        "address": "Main St",
    }
    assert target.ticket_update.missing == ["address", "phone"]


def test_finalize_records_a_tool_delivered_text(store: TicketStore) -> None:
    from wahabot.handlers.ticket_mixin import TicketMixin

    _, ticket = make_ticket(store)
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        ticket_store=store,
        ticket_id=ticket.id,
        sent=CUSTOMER,
        sent_text="delivered by the tool",
    )

    TicketMixin().finalize_ticket(target, "delivered_via_tool")

    replies = [item.body for item in store.events(ticket.id) if item.kind == "reply"]
    assert replies == ["delivered by the tool"]
