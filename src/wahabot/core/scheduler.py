"""The 60-second ticket tick: close, release pauses and nudge."""

import contextlib
import datetime
import threading
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from loguru import logger

from wahabot.core.access import SessionConfig, SessionConfigReloader
from wahabot.core.echoes import remember_self_echo
from wahabot.core.schedule.deferred import deliver_due_notices
from wahabot.core.schedule.hours import effective_timezone, opening_state
from wahabot.core.tickets.alerts import Translator, operator_language, retry_alerts
from wahabot.core.tickets.lifecycle import close_overdue, close_stale
from wahabot.core.tickets.models import (
    Actor,
    Contact,
    EventKind,
    Ticket,
)
from wahabot.core.tickets.store import TicketStore
from wahabot.core.tickets.summary import send_summary
from wahabot.core.transport import Transport
from wahabot.settings import Settings
from wahabot.status.state import state as status_state

NUDGE_INTERVAL_S = 12 * 3600


class TicketScheduler:
    """Applies the time-based ticket rules on every tick."""

    def __init__(
        self,
        settings: Settings,
        store: TicketStore,
        waha: Transport,
        config_reloader: SessionConfigReloader,
        translator: Translator | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.waha = waha
        self.config_reloader = config_reloader
        self.translator = translator
        self.nudge_lock = threading.Lock()

    def tick(self, now: int | None = None) -> None:
        """Run one timer pass: close tickets, release pauses, summarize, nudge."""
        moment = self.store.now() if now is None else now
        config = self.config_reloader.current_config()
        retry_alerts(self.store, self.waha, moment)
        close_overdue(self.store, config.close_after_days, at=moment)
        close_stale(self.store, config.close_after_days, at=moment)
        self.release_pauses(moment, config)
        # The summary is the opening notification: it lists every needs-you
        # ticket, cancels their queued notices and records a nudge per ticket
        # so the same opening cannot page twice.
        self.morning_summary(moment, config)
        self.nudge(moment, config)
        self.scheduled_notices(moment, config)

    def scheduled_notices(self, moment: int, config: SessionConfig) -> None:
        """Deliver deferred alerts only while the business is open."""
        state = opening_state(config, moment, self.settings.timezone)
        if not state.open:
            return
        deliver_due_notices(self.store, self.waha, moment)

    def morning_summary(self, moment: int, config: SessionConfig) -> bool:
        """Send the morning summary once per local day at the first opening."""
        if not config.hours:
            return False
        destination = status_state.operator_jid
        if not destination:
            return False
        zone = effective_timezone(config.timezone, self.settings.timezone)
        try:
            sent = send_summary(
                self.store,
                self.waha,
                self.settings.session,
                destination,
                moment,
                zone,
                hours=config.hours,
                operator_language=config.operator_language,
                translate=self.translator,
                fallback_timezone=self.settings.timezone,
            )
        except Exception as exc:
            logger.warning("Morning summary failed: {exc}", exc=exc)
            return False
        return sent is not None

    def release_pauses(self, moment: int, config: SessionConfig) -> None:
        """Clear contact and chat pauses once the customer has been quiet."""
        quiet_s = config.pause_after_human_hours * 3600
        self.release_contact_pauses(moment, quiet_s)
        self.release_chat_pauses(moment, quiet_s)

    def release_contact_pauses(self, moment: int, quiet_s: int) -> None:
        """Release expired pauses stored on a contact row."""
        for contact in self.store.paused_contacts():
            if contact.paused_until is None or contact.paused_until > moment:
                continue
            if (
                contact.last_inbound_at is not None
                and moment - contact.last_inbound_at < quiet_s
            ):
                continue
            self.store.set_contact_paused(contact.chat_id, None)

    def release_chat_pauses(self, moment: int, quiet_s: int) -> None:
        """Release expired pauses stored against a chat id."""
        for chat_id, until in self.store.paused_chats():
            if until > moment:
                continue
            contact = self.store.contact(chat_id)
            if (
                contact is not None
                and contact.last_inbound_at is not None
                and moment - contact.last_inbound_at < quiet_s
            ):
                continue
            self.store.set_chat_paused(chat_id, None)

    def nudge(self, moment: int, config: SessionConfig | None = None) -> None:
        """Tell the operator once per interval that a customer asked again."""
        current = config if config is not None else self.config_reloader.current_config()
        state = opening_state(current, moment, self.settings.timezone)
        if not state.open:
            return
        zone = effective_timezone(current.timezone, self.settings.timezone)
        language = operator_language(current.operator_language)
        with self.nudge_lock:
            self.nudge_locked(moment, zone, language)

    def nudge_locked(self, moment: int, timezone: str, language: str) -> None:
        operator_jid = status_state.operator_jid
        if not operator_jid:
            return
        queued = {notice.ticket_id for notice in self.store.pending_notices(moment)}
        for ticket in self.store.needs_you_tickets():
            if ticket.id not in queued and self.nudge_due(ticket, moment):
                self.send_nudge(ticket, operator_jid, moment, timezone, language)

    def nudge_message(
        self, ticket: Ticket, moment: int, timezone: str, language: str
    ) -> str | None:
        """The nudge text, or None when the contact is gone."""
        contact = self.store.contact_by_id(ticket.contact_id)
        if contact is None:
            return None
        alert = self.store.last_event(ticket.id, EventKind.ALERT.value)
        since = alert.at if alert is not None else ticket.updated_at
        latest = self.store.last_event(ticket.id, EventKind.CUSTOMER_MESSAGE.value)
        text = nudge_text(
            contact,
            ticket,
            moment,
            timezone,
            since=since,
            words=latest.body if latest is not None else "",
        )
        if language != "en" and self.translator is not None:
            with contextlib.suppress(Exception):
                text = self.translator(text, language)
        return text

    def send_nudge(
        self,
        ticket: Ticket,
        operator_jid: str,
        moment: int,
        timezone: str,
        language: str,
    ) -> None:
        """Send and record one nudge for a needs-you ticket."""
        text = self.nudge_message(ticket, moment, timezone, language)
        if text is None:
            return
        try:
            sent_id = self.waha.send_text(self.settings.session, operator_jid, text)
        except Exception as exc:
            logger.warning("Ticket {n} nudge failed: {exc}", n=ticket.number, exc=exc)
            return
        if not sent_id:
            logger.warning(
                "Ticket {n} nudge returned no message id; not recorded",
                n=ticket.number,
            )
            return
        remember_self_echo(sent_id)
        self.store.append_alert_ids(ticket.id, [sent_id])
        self.store.add_event(
            ticket.id,
            Actor.SYSTEM.value,
            EventKind.NUDGE.value,
            text,
            at=moment,
        )

    def nudge_due(self, ticket: Ticket, moment: int) -> bool:
        """Whether a needs-you ticket has an un-nudged customer re-ask."""
        customer = self.store.last_event(ticket.id, EventKind.CUSTOMER_MESSAGE.value)
        if customer is None:
            return False
        alert = self.store.last_event(ticket.id, EventKind.ALERT.value)
        if alert is None or (customer.at, customer.id) <= (alert.at, alert.id):
            return False
        last_nudge = self.store.last_event(ticket.id, EventKind.NUDGE.value)
        if last_nudge is not None:
            if (last_nudge.at, last_nudge.id) >= (customer.at, customer.id):
                return False
            if moment - last_nudge.at < NUDGE_INTERVAL_S:
                return False
        return True


def age_hours(start: int, now: int, tz_name: str) -> int:
    """Whole hours between two instants, measured in the session timezone."""
    try:
        zone = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError, ValueError, OSError:
        zone = datetime.UTC
    begin = datetime.datetime.fromtimestamp(start, tz=zone)
    end = datetime.datetime.fromtimestamp(now, tz=zone)
    return max(0, int((end - begin).total_seconds() // 3600))


def nudge_text(
    contact: Contact,
    ticket: Ticket,
    now: int,
    tz_name: str,
    *,
    since: int | None = None,
    words: str = "",
) -> str:
    """The operator nudge for a re-asked needs-you ticket."""
    start = since if since is not None else ticket.updated_at
    waiting = age_hours(start, now, tz_name)
    line = f"{contact.display} asked again about {ticket.number}, waiting {waiting} h."
    snippet = words.strip()
    return f'{line} "{snippet[:240]}"' if snippet else line
