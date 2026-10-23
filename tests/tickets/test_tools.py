import json
from contextlib import contextmanager

from tests.harness import ME_JID, RecordingWaha
from wahabot.ai.tools.runcontext import RunTarget, bind_target, reset_target
from wahabot.ai.tools.tickets import (
    hand_to_operator,
    new_ticket,
    support_state,
    update_ticket,
)
from wahabot.core.tickets.models import TicketState, TicketUpdate
from wahabot.core.tickets.store import TicketStore
from wahabot.settings import Settings

CUSTOMER = "491555000001@c.us"
OTHER = "491555000002@c.us"


@contextmanager
def target_for(target: RunTarget):
    token = bind_target(target)
    try:
        yield target
    finally:
        reset_target(token)


def make_ticket(store: TicketStore, chat: str = CUSTOMER, name: str = "Smoke Sender"):
    contact = store.upsert_contact(chat, display_name=name)
    ticket = store.create_ticket(contact.id, chat, summary="issue")
    return contact, ticket


def test_customer_support_state_is_fenced_to_its_own_chat(
    store: TicketStore, settings: Settings
) -> None:
    _, mine = make_ticket(store)
    _, other = make_ticket(store, OTHER, "Other Person")
    tool = support_state(settings)
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
    store: TicketStore, settings: Settings
) -> None:
    contact, ticket = make_ticket(store)
    tool = support_state(settings)
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
    store: TicketStore, settings: Settings
) -> None:
    _, ticket = make_ticket(store)
    tool = update_ticket(settings)
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
    store: TicketStore, settings: Settings
) -> None:
    _, ticket = make_ticket(store)
    tool = new_ticket(settings)
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
    assert "operator" in refused["error"]


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
