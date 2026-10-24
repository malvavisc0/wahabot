"""Opening hours, deferred notices, night readiness and handover paths."""

import datetime
import json
from collections.abc import Iterator
from contextlib import contextmanager
from zoneinfo import ZoneInfo

from tests.harness import ME_JID, RecordingWaha

from wahabot.ai.tools.runcontext import RunTarget, bind_target, reset_target
from wahabot.ai.tools.tickets import hand_to_operator
from wahabot.core.access import SessionConfig
from wahabot.core.schedule.deferred import defer_notice, due_notices
from wahabot.core.schedule.hours import (
    next_opening,
    opening_state,
    render_away_message,
)
from wahabot.core.schedule.models import OpeningState
from wahabot.core.tickets.lifecycle import close_ticket
from wahabot.core.tickets.models import TicketState, TicketUpdate
from wahabot.core.tickets.store import TicketStore
from wahabot.settings import Settings

CUSTOMER = "491555000001@c.us"
NIGHT = "491555000009@c.us"


@contextmanager
def target_for(target: RunTarget) -> Iterator[RunTarget]:
    token = bind_target(target)
    try:
        yield target
    finally:
        reset_target(token)


def config_with_hours(**hours: dict[str, object]) -> SessionConfig:
    return SessionConfig(
        system_prompt="smoke",
        timezone="Europe/Madrid",
        hours=hours,  # type: ignore[arg-type]
        away_message="Ticket {ticket}: we reply after {next_opening}.",
    )


def test_opening_state_overnight_and_dst() -> None:
    config = config_with_hours(
        mon={"open": "09:00", "close": "17:00"},
        tue={"open": "09:00", "close": "17:00"},
        wed={"closed": True},
        fri={"open": "22:00", "close": "02:00"},
        sat={"open": "10:00", "close": "14:00"},
    )
    madrid = ZoneInfo("Europe/Madrid")

    def at(day: int, hour: int) -> int:
        return int(datetime.datetime(2026, 6, day, hour, tzinfo=madrid).timestamp())

    assert opening_state(config, at(1, 10)).open
    assert not opening_state(config, at(1, 8)).open
    assert next_opening(config, at(1, 20)) == at(2, 9)
    assert not opening_state(config, at(3, 12)).open
    assert opening_state(config, at(6, 1)).open
    assert not opening_state(config, at(6, 3)).open

    dst = SessionConfig(
        system_prompt="smoke",
        timezone="Europe/Madrid",
        hours={"sun": {"open": "01:00", "close": "04:00"}},
    )
    moment = datetime.datetime(2026, 3, 29, 3, 30, tzinfo=madrid)
    assert opening_state(dst, int(moment.timestamp())).open

    away = render_away_message(config.away_message, 12, at(2, 9), "Europe/Madrid")
    assert "Ticket 12" in away and "Tuesday" in away

    always = SessionConfig(system_prompt="smoke")
    assert opening_state(always, at(1, 3)).open


def test_overnight_window_reports_its_own_close_time() -> None:
    config = config_with_hours(
        **{
            "fri": {"open": "22:00", "close": "02:00"},
            "sat": {"open": "10:00", "close": "14:00"},
        }
    )
    zone = ZoneInfo("Europe/Madrid")
    at = int(datetime.datetime(2026, 6, 6, 1, 0, tzinfo=zone).timestamp())
    expected = int(datetime.datetime(2026, 6, 6, 2, 0, tzinfo=zone).timestamp())

    state = opening_state(config, at)

    assert state.open
    assert state.next_opening == expected


def test_deferred_notice_and_night_readiness(store: TicketStore) -> None:
    contact = store.upsert_contact(CUSTOMER, phone=CUSTOMER)
    ticket = store.create_ticket(contact.id, CUSTOMER)
    store.update_ticket(
        ticket.id,
        TicketUpdate(state=TicketState.NEEDS_YOU.value),
        at=100,
    )
    assert not store.night_ready("")
    assert not store.night_ready(NIGHT)
    store.upsert_contact(NIGHT, phone=NIGHT)
    assert store.night_ready(NIGHT)

    defer_notice(store, ticket, "default", ME_JID, "queued", now=100, at=100)
    assert due_notices(store, 99) == []
    assert len(due_notices(store, 101)) == 1
    assert store.pending_notices(102)[0].text == "queued"

    # A retried handover updates the one pending notice instead of queuing twice.
    defer_notice(store, ticket, "default", ME_JID, "updated", now=100, at=101)
    pending = store.pending_notices(102)
    assert len(pending) == 1 and pending[0].text == "updated"

    close_ticket(store, ticket, at=102)
    assert store.pending_notices(103) == []


def test_handover_defers_non_urgent_outside_hours(
    store: TicketStore, settings: Settings, monkeypatch
) -> None:
    from wahabot.status.state import state as status_state

    monkeypatch.setattr(status_state, "operator_jid", ME_JID)
    contact = store.upsert_contact(CUSTOMER, display_name="Smoke Sender")
    ticket = store.create_ticket(contact.id, CUSTOMER, summary="late order")
    store.add_event(ticket.id, "customer", "customer_message", "where is it?")
    waha = RecordingWaha()
    state = OpeningState(
        open=False,
        next_opening=store.now() + 3600,
        timezone="UTC",
    )
    tool = hand_to_operator(
        settings,
        waha,
        schedule=lambda: state,
        away_message=lambda: "Ticket {ticket}: back at {next_opening}.",
    )
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        surface="customer",
        ticket_store=store,
        ticket_id=ticket.id,
    )
    with target_for(target):
        payload = json.loads(tool.fn(reason="needs a person"))

    assert payload["ok"] and payload["deferred"]
    assert payload["away_message"].startswith("Ticket 1:")
    assert waha.sent == []
    assert store.ticket(ticket.id).state == TicketState.NEEDS_YOU.value
    assert store.pending_notices(state.next_opening or 0)[0].ticket_id == ticket.id


def test_urgent_after_hours_uses_the_night_number(
    store: TicketStore, settings: Settings, monkeypatch
) -> None:
    from wahabot.status.state import state as status_state

    monkeypatch.setattr(status_state, "operator_jid", ME_JID)
    contact = store.upsert_contact(CUSTOMER, display_name="Smoke Sender")
    night = store.upsert_contact(NIGHT, phone=NIGHT)
    ticket = store.create_ticket(contact.id, CUSTOMER, summary="urgent leak")
    store.add_event(ticket.id, "customer", "customer_message", "water everywhere")
    waha = RecordingWaha()
    state = OpeningState(open=False, next_opening=store.now() + 3600, timezone="UTC")
    tool = hand_to_operator(
        settings,
        waha,
        schedule=lambda: state,
        night_number=lambda: NIGHT,
    )
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        surface="customer",
        ticket_store=store,
        ticket_id=ticket.id,
    )
    with target_for(target):
        payload = json.loads(tool.fn(reason="urgent leak", urgent=True))

    assert payload["ok"] and not payload.get("deferred")
    recipients = [item[1] for item in waha.sent]
    assert ME_JID in recipients and night.chat_id in recipients


def test_unready_night_number_is_never_messaged_first(
    store: TicketStore, settings: Settings, monkeypatch
) -> None:
    from wahabot.status.state import state as status_state

    monkeypatch.setattr(status_state, "operator_jid", ME_JID)
    contact = store.upsert_contact(CUSTOMER, display_name="Smoke Sender")
    ticket = store.create_ticket(contact.id, CUSTOMER, summary="urgent")
    store.add_event(ticket.id, "customer", "customer_message", "help")
    waha = RecordingWaha()
    state = OpeningState(open=False, next_opening=store.now() + 3600, timezone="UTC")
    tool = hand_to_operator(
        settings,
        waha,
        schedule=lambda: state,
        night_number=lambda: NIGHT,
    )
    target = RunTarget(
        session="default",
        chat_id=CUSTOMER,
        surface="customer",
        ticket_store=store,
        ticket_id=ticket.id,
    )
    with target_for(target):
        payload = json.loads(tool.fn(reason="urgent", urgent=True))

    assert payload["ok"]
    assert [item[1] for item in waha.sent] == [ME_JID]
