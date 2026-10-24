"""Morning summary and the internal team-group surface."""

import datetime
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager

from tests.harness import ME_JID, RecordingWaha

from wahabot.ai.messages import is_group_addressed
from wahabot.ai.tools.runcontext import RunTarget, bind_target, fenced_chat, reset_target
from wahabot.ai.tools.tickets import new_ticket, update_ticket
from wahabot.core.access import SessionConfig
from wahabot.core.models import WahaEvent
from wahabot.core.schedule.deferred import defer_notice
from wahabot.core.teams.groups import (
    bot_mention_pattern,
    group_participation,
    mention_match,
    mention_pattern_ok,
    surface_for,
    team_capability_error,
)
from wahabot.core.tickets.models import DoneBy, TicketState, TicketUpdate
from wahabot.core.tickets.store import TicketStore
from wahabot.core.tickets.summary import build_summary, send_summary, summary_due

CUSTOMER = "491555000001@c.us"
TEAM_GROUP = "1234567890-1234567890@g.us"
OTHER_GROUP = "1234567890-9999999999@g.us"


@contextmanager
def target_for(target: RunTarget) -> Iterator[RunTarget]:
    token = bind_target(target)
    try:
        yield target
    finally:
        reset_target(token)


def group_event(mid: str, body: str, chat: str = TEAM_GROUP) -> WahaEvent:
    return WahaEvent.model_validate(
        {
            "event": "message",
            "session": "default",
            "payload": {
                "id": f"false_{chat}_{mid}",
                "from": chat,
                "fromMe": False,
                "participant": CUSTOMER,
                "body": body,
                "timestamp": 1_800_000_000,
                "_data": {"type": "text"},
            },
        }
    )


def test_summary_counts_and_urgent_first(store: TicketStore) -> None:
    yesterday = int(datetime.datetime(2026, 6, 1, 12, tzinfo=datetime.UTC).timestamp())
    now = int(datetime.datetime(2026, 6, 2, 8, tzinfo=datetime.UTC).timestamp())

    def ticket(index: int) -> object:
        chat = f"49155500000{index}@c.us"
        contact = store.upsert_contact(chat, display_name=f"Smoke Sender {index}")
        return store.create_ticket(contact.id, chat, summary="x", at=yesterday)

    done = ticket(1)
    store.update_ticket(
        done.id,
        TicketUpdate(state=TicketState.DONE.value, done_by=DoneBy.ASSISTANT.value),
        at=yesterday,
    )
    done_by_operator = ticket(2)
    store.update_ticket(
        done_by_operator.id,
        TicketUpdate(state=TicketState.DONE.value, done_by=DoneBy.OPERATOR.value),
        at=yesterday,
    )
    older = ticket(3)
    store.update_ticket(
        older.id, TicketUpdate(state=TicketState.NEEDS_YOU.value), at=yesterday
    )
    urgent = ticket(4)
    store.update_ticket(
        urgent.id,
        TicketUpdate(state=TicketState.NEEDS_YOU.value, urgent=True),
        at=yesterday,
    )

    text = build_summary(store, "default", now, "UTC")
    assert "4 tickets" in text
    assert "1 done by the assistant" in text
    assert "1 by you" in text
    assert text.index(str(urgent.number)) < text.index(str(older.number))
    assert "URGENT" in text


def test_summary_counts_work_done_yesterday(store: TicketStore) -> None:
    two_days = int(datetime.datetime(2026, 5, 31, 12, tzinfo=datetime.UTC).timestamp())
    yesterday = int(datetime.datetime(2026, 6, 1, 12, tzinfo=datetime.UTC).timestamp())
    now = int(datetime.datetime(2026, 6, 2, 8, tzinfo=datetime.UTC).timestamp())
    contact = store.upsert_contact("491555000099@c.us", display_name="Older")
    ticket = store.create_ticket(contact.id, contact.chat_id, summary="x", at=two_days)
    store.update_ticket(
        ticket.id,
        TicketUpdate(state=TicketState.DONE.value, done_by=DoneBy.OPERATOR.value),
        at=yesterday,
    )

    text = build_summary(store, "default", now, "UTC")

    assert "0 tickets" in text
    assert "1 by you" in text


def test_needs_you_orders_by_waiting_age_not_number(store: TicketStore) -> None:
    now = int(datetime.datetime(2026, 6, 2, 8, tzinfo=datetime.UTC).timestamp())
    first_contact = store.upsert_contact("491555000101@c.us", display_name="First")
    first = store.create_ticket(first_contact.id, first_contact.chat_id, at=now)
    second_contact = store.upsert_contact("491555000102@c.us", display_name="Second")
    second = store.create_ticket(second_contact.id, second_contact.chat_id, at=now)
    store.update_ticket(
        first.id, TicketUpdate(state=TicketState.NEEDS_YOU.value), at=now - 3600
    )
    store.update_ticket(
        second.id, TicketUpdate(state=TicketState.NEEDS_YOU.value), at=now - 36_000
    )

    text = build_summary(store, "default", now, "UTC")

    assert text.index(str(second.number)) < text.index(str(first.number))


def test_summary_cancels_deferred_alerts_it_already_lists(
    store: TicketStore,
) -> None:
    now = int(datetime.datetime(2026, 6, 2, 8, tzinfo=datetime.UTC).timestamp())
    contact = store.upsert_contact("491555000103@c.us", display_name="Queued")
    ticket = store.create_ticket(contact.id, contact.chat_id, at=now - 3600)
    store.update_ticket(
        ticket.id, TicketUpdate(state=TicketState.NEEDS_YOU.value), at=now - 3600
    )
    defer_notice(store, ticket, "default", ME_JID, "queued alert", now=now, at=now - 3600)
    assert store.pending_notices(now)

    send_summary(
        store,
        RecordingWaha(),
        "default",
        ME_JID,
        now,
        "UTC",
        hours={"tue": {"open": "08:00", "close": "17:00"}},
    )
    assert store.pending_notices(now) == []


def test_summary_once_per_day_and_restart_safe(store: TicketStore) -> None:
    now = int(datetime.datetime(2026, 6, 2, 8, tzinfo=datetime.UTC).timestamp())
    hours = {"tue": {"open": "08:00", "close": "17:00"}}
    assert summary_due(store, "default", now, "UTC", hours=hours)
    waha = RecordingWaha()
    sent_id = send_summary(
        store,
        waha,
        "default",
        ME_JID,
        now,
        "UTC",
        hours=hours,
    )
    assert sent_id is not None
    assert store.last_summary_date("default") == "2026-06-02"
    assert store.summary_message_id("default") == sent_id
    assert not summary_due(store, "default", now + 3600, "UTC", hours=hours)
    assert TicketStore(store.path).last_summary_date("default") == "2026-06-02"


def test_team_group_allowlist_override_and_surface() -> None:
    config = SessionConfig(
        system_prompt="smoke",
        bot_name="Kai",
        bot_mention_regex="@?kai",
        group_participation="never",
        team_groups=[TEAM_GROUP],
    )
    assert group_participation(config, TEAM_GROUP) == "mentioned"
    assert group_participation(config, OTHER_GROUP) == "never"


def test_team_group_session_rule_overrides_never() -> None:
    from types import SimpleNamespace

    from wahabot.handlers.config_mixin import ConfigMixin

    config = SessionConfig(
        system_prompt="smoke",
        team_groups=[TEAM_GROUP],
        group_participation="never",
    )
    handler = SimpleNamespace(
        config_reloader=SimpleNamespace(current_config=lambda: config),
        settings=SimpleNamespace(timezone="UTC"),
    )
    handler.participation_brief = lambda current, target: (  # type: ignore[attr-defined]
        ConfigMixin.participation_brief(handler, current, target)  # type: ignore[arg-type]
    )
    target = RunTarget(session="default", chat_id=TEAM_GROUP)
    token = bind_target(target)
    try:
        rules = ConfigMixin.render_session_rules(handler)  # type: ignore[arg-type]
    finally:
        reset_target(token)

    assert "never reply" not in rules
    assert "reply only when addressed" in rules
    assert "Internal team group" in rules
    assert surface_for(config, TEAM_GROUP) == "team_group"
    assert surface_for(config, OTHER_GROUP) == "customer"

    named = group_event("T1", "kai what's open?")
    unnamed = group_event("T2", "anyone seen the invoice?")
    unlisted = group_event("T3", "kai what's open?", chat=OTHER_GROUP)
    assert is_group_addressed(
        named,
        bot_name="Kai",
        bot_mention_regex="@?kai",
        participation=group_participation(config, TEAM_GROUP),
    )
    assert not is_group_addressed(
        unnamed,
        bot_name="Kai",
        bot_mention_regex="@?kai",
        participation=group_participation(config, TEAM_GROUP),
    )
    assert not is_group_addressed(
        unlisted,
        bot_name="Kai",
        bot_mention_regex="@?kai",
        participation=group_participation(config, OTHER_GROUP),
    )


def test_team_group_is_fenced_in_code(store: TicketStore) -> None:
    contact = store.upsert_contact(CUSTOMER, display_name="Smoke Sender")
    ticket = store.create_ticket(contact.id, CUSTOMER)
    target = RunTarget(
        session="default",
        chat_id=TEAM_GROUP,
        surface="team_group",
        ticket_store=store,
        ticket_id=ticket.id,
    )
    chat, refused = fenced_chat(CUSTOMER, target)
    assert chat is None and refused
    assert team_capability_error(target, "ticket_write")
    assert team_capability_error(target, "tts")
    assert team_capability_error(target, "customer_system_write")
    assert team_capability_error(target, "read") is None

    with target_for(target):
        updated = json.loads(update_ticket().fn(state="done"))
        created = json.loads(new_ticket().fn(summary="new issue"))
    assert not updated["ok"] and "team-group" in updated["error"]
    assert not created["ok"] and "team-group" in created["error"]


def test_mention_pattern_is_compiled_and_sanity_checked() -> None:
    pattern = bot_mention_pattern("Kai")
    assert mention_pattern_ok(pattern, "Kai")
    assert re.search(pattern, "Kai")
    assert re.search(pattern, "@kai")
    assert not re.search(pattern, "kayak")
    assert mention_match(pattern, "ask Kai please")
    assert not mention_match(pattern, "kayak")


def test_quoted_summary_still_routes_a_ticket_number(store: TicketStore) -> None:
    from wahabot.core.tickets.router import RouteKind, route_message

    contact = store.upsert_contact(CUSTOMER, display_name="Smoke Sender")
    ticket = store.create_ticket(contact.id, CUSTOMER, summary="late order")
    store.set_summary_message_id("default", "summary-msg")
    event = WahaEvent.model_validate(
        {
            "event": "message",
            "session": "default",
            "payload": {
                "id": f"true_{ME_JID}_SUMMARY_REPLY",
                "from": ME_JID,
                "to": ME_JID,
                "fromMe": True,
                "participant": ME_JID,
                "body": f"{ticket.number}: refunded",
                "timestamp": 1_800_000_000,
                "replyTo": {"id": "summary-msg"},
                "_data": {"type": "text"},
            },
        }
    )
    route = route_message(store, event)
    assert route.kind == RouteKind.TICKET_REPLY
    assert route.ticket is not None and route.ticket.id == ticket.id
