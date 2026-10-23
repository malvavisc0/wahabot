from tests.harness import RecordingWaha
from wahabot.core.access import SessionConfig
from wahabot.core.scheduler import TicketScheduler, age_hours
from wahabot.core.tickets.models import TicketState, TicketUpdate
from wahabot.core.tickets.store import TicketStore
from wahabot.settings import Settings

CUSTOMER = "491555000001@c.us"
HOUR = 3600
DAY = 86400


class Reloader:
    def __init__(self, config: SessionConfig) -> None:
        self.config = config

    def current_config(self) -> SessionConfig:
        return self.config


def scheduler_for(
    store: TicketStore, settings: Settings, config: SessionConfig
) -> tuple[TicketScheduler, RecordingWaha]:
    waha = RecordingWaha()
    return TicketScheduler(settings, store, waha, Reloader(config)), waha


def done_ticket(store: TicketStore, at: int):
    contact = store.upsert_contact(
        CUSTOMER, display_name="Smoke Sender", last_inbound_at=at
    )
    ticket = store.create_ticket(contact.id, CUSTOMER, at=at)
    store.update_ticket(
        ticket.id,
        TicketUpdate(state=TicketState.DONE.value, done_by="assistant"),
        at=at,
    )
    return contact, store.ticket(ticket.id)


def test_done_ticket_closes_after_the_window_without_sending(
    store: TicketStore, settings: Settings
) -> None:
    _, ticket = done_ticket(store, 1000)
    scheduler, waha = scheduler_for(store, settings, SessionConfig(system_prompt="x"))

    scheduler.tick(now=1000 + 4 * DAY)
    assert store.ticket(ticket.id).state == TicketState.DONE.value

    scheduler.tick(now=1000 + 5 * DAY + 1)
    closed = store.ticket(ticket.id)
    assert closed.state == TicketState.CLOSED.value
    assert closed.closed_at is not None
    assert waha.sent == []


def test_needs_you_never_closes(store: TicketStore, settings: Settings) -> None:
    contact = store.upsert_contact(
        CUSTOMER, display_name="Smoke Sender", last_inbound_at=1000
    )
    ticket = store.create_ticket(contact.id, CUSTOMER, at=1000)
    store.update_ticket(
        ticket.id, TicketUpdate(state=TicketState.NEEDS_YOU.value), at=1000
    )
    scheduler, _ = scheduler_for(store, settings, SessionConfig(system_prompt="x"))

    scheduler.tick(now=1000 + 365 * DAY)

    assert store.ticket(ticket.id).state == TicketState.NEEDS_YOU.value


def test_pause_releases_only_after_the_customer_is_quiet(
    store: TicketStore, settings: Settings
) -> None:
    store.upsert_contact(CUSTOMER, last_inbound_at=1000)
    store.set_contact_paused(CUSTOMER, 1000 + 2 * HOUR)
    scheduler, _ = scheduler_for(store, settings, SessionConfig(system_prompt="x"))

    scheduler.tick(now=1000 + 3 * HOUR)
    assert store.contact(CUSTOMER).paused_until is None

    store.set_contact_paused(CUSTOMER, 1000 + 2 * HOUR)
    store.upsert_contact(CUSTOMER, last_inbound_at=1000 + HOUR + HOUR // 2)
    scheduler.tick(now=1000 + 3 * HOUR)
    assert store.contact(CUSTOMER).paused_until is not None


def test_global_pause_survives_a_new_store_instance(store: TicketStore, tmp_path) -> None:
    store.set_session_paused("default", 123456)

    from wahabot.core.tickets.store import TicketStore

    reopened = TicketStore(store.path)
    assert reopened.session_paused_until("default") == 123456


def test_nudge_goes_out_once_per_interval(
    store: TicketStore, settings: Settings, monkeypatch
) -> None:
    from wahabot.status.state import state as status_state

    monkeypatch.setattr(status_state, "operator_jid", "491555000000@c.us")
    contact = store.upsert_contact(
        CUSTOMER, display_name="Smoke Sender", last_inbound_at=1000
    )
    ticket = store.create_ticket(contact.id, CUSTOMER, at=1000)
    store.update_ticket(
        ticket.id, TicketUpdate(state=TicketState.NEEDS_YOU.value), at=1000
    )
    store.add_event(ticket.id, "assistant", "alert", "Ticket 1", at=1000)
    store.add_event(ticket.id, "customer", "customer_message", "any update?", at=2000)
    scheduler, waha = scheduler_for(store, settings, SessionConfig(system_prompt="x"))

    scheduler.tick(now=2000 + HOUR)
    assert len(waha.sent) == 1
    assert "asked again about 1" in waha.sent[0][2]

    scheduler.tick(now=2000 + 12 * HOUR)
    assert len(waha.sent) == 1

    store.add_event(
        ticket.id, "customer", "customer_message", "still waiting", at=2000 + 13 * HOUR
    )
    scheduler.tick(now=2000 + 25 * HOUR)
    assert len(waha.sent) == 2


def test_age_hours_uses_the_configured_zone() -> None:
    assert age_hours(0, 26 * HOUR, "UTC") == 26
    assert age_hours(0, 26 * HOUR, "America/New_York") == 26
    assert age_hours(0, 26 * HOUR, "Not/AZone") == 26
