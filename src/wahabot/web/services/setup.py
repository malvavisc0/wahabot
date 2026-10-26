"""Setup view models: WAHA session lifecycle, QR, pairing and the first test."""

import base64
import re
from dataclasses import dataclass

import httpx
from loguru import logger

from wahabot.core.tickets.models import Contact, EventKind, Ticket, TicketEvent
from wahabot.core.tickets.store import TicketStore
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings

EVENTS = ["message", "message.any", "message.reaction", "session.status"]


@dataclass(frozen=True, slots=True)
class Exchange:
    """The first answered customer question Setup shows as proof."""

    ticket_number: int
    contact_name: str
    question: str
    answer: str


@dataclass(frozen=True, slots=True)
class SetupState:
    """Everything one Setup render needs."""

    status: str
    qr_data: str
    qr_value: str
    exchange: Exchange | None
    completed: bool
    error: str = ""


def first_body(events: list[TicketEvent], kinds: set[str]) -> str:
    """The first non-empty body of one of *kinds*."""
    for event in events:
        if event.kind in kinds and event.body.strip():
            return event.body
    return ""


class SetupService:
    """Drive WAHA's session lifecycle for the Setup and reconnect pages."""

    def __init__(self, settings: Settings, store: TicketStore, waha: WahaClient) -> None:
        self.settings = settings
        self.store = store
        self.waha = waha

    def status(self) -> str:
        """The WAHA session status, or NOT_CREATED when it does not exist."""
        try:
            payload = self.waha.get_session(self.settings.session)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return "NOT_CREATED"
            logger.warning(
                "WAHA session probe failed: {status}", status=exc.response.status_code
            )
            return "UNKNOWN"
        except httpx.HTTPError:
            return "UNKNOWN"
        return str(payload.get("status") or "UNKNOWN")

    def ensure_session(self) -> str:
        """Create or start the session so a QR is available.

        A FAILED session is left alone: restarting it on every poll
        would log the phone out repeatedly, so the operator uses the
        explicit Restart control instead.
        """
        status = self.status()
        if status == "NOT_CREATED":
            self.waha.create_session(
                self.settings.session,
                webhook_url=self.webhook_url(),
                hmac_key=self.settings.webhook_hmac_key,
                events=EVENTS,
            )
            return "STARTING"
        if status in {"STOPPED", "UNKNOWN"}:
            self.waha.start_session(self.settings.session)
            return "STARTING"
        return status

    def restart(self) -> None:
        """Restart the session in place after six expired codes or a failure."""
        self.waha.restart_session(self.settings.session)

    def webhook_url(self) -> str:
        """The internal URL WAHA posts the bot's events to."""
        return (
            f"http://{self.settings.webhook_host}:{self.settings.port}"
            + f"/api/webhook/{self.settings.session}"
        )

    def qr(self) -> tuple[str, str]:
        """The QR as a data URL, or its raw text value."""
        try:
            payload = self.waha.get_qr(self.settings.session)
        except httpx.HTTPError as exc:
            logger.warning("WAHA QR fetch failed: {error}", error=type(exc).__name__)
            return "", ""
        if isinstance(payload, bytes):
            encoded = base64.b64encode(payload).decode()
            return f"data:image/png;base64,{encoded}", ""
        value = str(payload.get("value") or payload.get("qr") or "")
        return "", value

    def support_number(self) -> str:
        """The connected support number without its chat suffix."""
        try:
            payload = self.waha.get_me(self.settings.session)
        except httpx.HTTPError:
            return ""
        raw = str(payload.get("id") or "")
        return re.sub(r"\D", "", raw.split("@", 1)[0])

    def pairing_code(self, phone_number: str = "") -> str:
        """Request the pairing code; QR remains the fallback."""
        number = re.sub(r"\D", "", phone_number or self.support_number())
        if not number:
            raise ValueError("enter the support number for a pairing code")
        self.ensure_session()
        payload = self.waha.request_code(self.settings.session, number)
        code = payload.get("code") or payload.get("pairingCode") or payload.get("value")
        return str(code or "Pairing code requested")

    def first_exchange(self) -> Exchange | None:
        """The first customer conversation answered outside the self chat.

        Without the account's own ids the exclusion cannot be verified,
        so no exchange is reported rather than a possible self chat.
        """
        own = self.own_ids()
        if not own:
            return None
        tickets = sorted(self.store.all_tickets(), key=lambda item: item.number)
        for ticket in tickets:
            contact = self.store.contact_by_id(ticket.contact_id)
            if not self.customer_contact(contact, own):
                continue
            exchange = self.exchange_for(ticket)
            if exchange is not None:
                return exchange
        return None

    def customer_contact(self, contact: Contact | None, own: set[str]) -> bool:
        """Whether a contact is a real customer, not the support account or a group."""
        if contact is None:
            return False
        return contact.chat_id not in own and not contact.chat_id.endswith("@g.us")

    def exchange_for(self, ticket: Ticket) -> Exchange | None:
        """The first answered question on one ticket, when both sides exist."""
        events = self.store.events(ticket.id)
        question = first_body(events, {EventKind.CUSTOMER_MESSAGE.value})
        answer = first_body(events, {EventKind.REPLY.value, EventKind.SENT.value})
        if not question or not answer:
            return None
        contact = self.store.contact_by_id(ticket.contact_id)
        return Exchange(
            ticket_number=ticket.number,
            contact_name=contact.display if contact is not None else "",
            question=question,
            answer=answer,
        )

    def own_ids(self) -> set[str]:
        """Chat ids that belong to the support account itself."""
        try:
            payload = self.waha.get_me(self.settings.session)
        except httpx.HTTPError:
            return set()
        found = {str(payload.get("id") or ""), str(payload.get("lid") or "")}
        return {item for item in found if item}

    def state(self) -> SetupState:
        """Load the page state, completing Setup when its two conditions hold."""
        status = self.ensure_session()
        exchange = self.first_exchange()
        completed = self.store.setup_completed(self.settings.session)
        if status == "WORKING" and exchange is not None and not completed:
            self.store.set_setup_completed(self.settings.session)
            completed = True
        qr_data, qr_value = ("", "") if status == "WORKING" else self.qr()
        return SetupState(
            status=status,
            qr_data=qr_data,
            qr_value=qr_value,
            exchange=exchange,
            completed=completed,
        )
