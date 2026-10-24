from tests.harness import RecordingWaha
from wahabot.core.access import SessionConfig
from wahabot.core.scheduler import TicketScheduler, age_hours
from wahabot.core.tickets.models import Ticket, TicketState, TicketUpdate
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
    store: TicketStore,
    settings: Settings,
    config: SessionConfig,
    waha: RecordingWaha | None = None,
) -> tuple[TicketScheduler, RecordingWaha]:
    sender = waha or RecordingWaha()
    return TicketScheduler(settings, store, sender, Reloader(config)), sender


def done_ticket(store: TicketStore, at: int) -> Ticket:
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
    assert "any update?" in waha.sent[0][2]

    scheduler.tick(now=2000 + 12 * HOUR)
    assert len(waha.sent) == 1

    store.add_event(
        ticket.id, "customer", "customer_message", "still waiting", at=2000 + 13 * HOUR
    )
    scheduler.tick(now=2000 + 25 * HOUR)
    assert len(waha.sent) == 2


def test_nudge_waits_for_a_message_after_the_alert(
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
    store.add_event(ticket.id, "customer", "customer_message", "where is it?", at=1000)
    store.add_event(ticket.id, "assistant", "alert", "Ticket 1", at=2000)
    scheduler, waha = scheduler_for(store, settings, SessionConfig(system_prompt="x"))

    scheduler.tick(now=2000 + HOUR)

    assert waha.sent == []


class SilentWaha(RecordingWaha):
    def send_text(self, *args, **kwargs) -> str:
        return ""


def test_unconfirmed_nudge_is_not_recorded(
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
    scheduler, _ = scheduler_for(
        store, settings, SessionConfig(system_prompt="x"), SilentWaha()
    )

    scheduler.tick(now=2000 + HOUR)

    assert store.last_event(ticket.id, "nudge") is None


def test_age_hours_uses_the_configured_zone() -> None:
    assert age_hours(0, 26 * HOUR, "UTC") == 26
    assert age_hours(0, 26 * HOUR, "America/New_York") == 26
    assert age_hours(0, 26 * HOUR, "Not/AZone") == 26


def test_chat_pause_releases_only_after_the_customer_is_quiet(
    store: TicketStore, settings: Settings
) -> None:
    store.upsert_contact(CUSTOMER, last_inbound_at=1000)
    store.set_chat_paused(CUSTOMER, 1000 + 2 * HOUR)
    scheduler, _ = scheduler_for(store, settings, SessionConfig(system_prompt="x"))

    scheduler.tick(now=1000 + 3 * HOUR)
    assert store.chat_paused_until(CUSTOMER) is None

    store.set_chat_paused(CUSTOMER, 1000 + 2 * HOUR)
    store.upsert_contact(CUSTOMER, last_inbound_at=1000 + HOUR + HOUR // 2)
    scheduler.tick(now=1000 + 3 * HOUR)
    assert store.chat_paused_until(CUSTOMER) is not None


def test_stale_open_and_waiting_tickets_close_without_touching_needs_you(
    store: TicketStore, settings: Settings
) -> None:
    contact = store.upsert_contact(CUSTOMER, last_inbound_at=1000)
    open_ticket = store.create_ticket(contact.id, CUSTOMER, at=1000)
    waiting = store.create_ticket(contact.id, "491555000002@c.us", at=1000)
    store.update_ticket(
        waiting.id,
        TicketUpdate(state=TicketState.WAITING_CUSTOMER.value),
        at=1000,
    )
    needs = store.create_ticket(contact.id, "491555000003@c.us", at=1000)
    store.update_ticket(
        needs.id, TicketUpdate(state=TicketState.NEEDS_YOU.value), at=1000
    )
    scheduler, _ = scheduler_for(store, settings, SessionConfig(system_prompt="x"))

    scheduler.tick(now=1000 + 6 * DAY)

    assert store.ticket(open_ticket.id).state == TicketState.CLOSED.value
    assert store.ticket(waiting.id).state == TicketState.CLOSED.value
    assert store.ticket(needs.id).state == TicketState.NEEDS_YOU.value


def test_scheduler_retries_a_queued_alert(store: TicketStore, settings: Settings) -> None:
    from tests.harness import ME_JID
    from wahabot.core.tickets.alerts import ALERT_RECONCILE_GRACE_S

    contact = store.upsert_contact(CUSTOMER, last_inbound_at=1000)
    ticket = store.create_ticket(contact.id, CUSTOMER, at=1000)
    store.claim_alert("queued", ticket.id, "default", ME_JID, "queued alert", at=1000)
    scheduler, waha = scheduler_for(store, settings, SessionConfig(system_prompt="x"))

    scheduler.tick(now=1000 + ALERT_RECONCILE_GRACE_S + 1)

    assert len(waha.sent) == 1
    assert waha.sent[0][2] == "queued alert"
    stored = store.ticket(ticket.id)
    assert stored is not None and stored.state == TicketState.NEEDS_YOU.value
