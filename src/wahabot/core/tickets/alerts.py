"""Compose and send the operator alert for a ticket."""

import base64
import hashlib
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from wahabot.core.echoes import remember_self_echo
from wahabot.core.tickets.alert_history import find_delivered_alert
from wahabot.core.tickets.models import (
    ALERT_SENDING,
    ALERT_SENT,
    AlertIntent,
    Attachment,
    Contact,
    EventKind,
    Ticket,
    TicketState,
)
from wahabot.core.tickets.store import TicketStore
from wahabot.core.transport import Transport

Translator = Callable[[str, str], str]

_ALERT_FAILED = "Could not attach {items}; they stay stored on the ticket."
#: Wait before treating an in-flight alert as definitely absent.
ALERT_RECONCILE_GRACE_S = 90
#: Search back a little before the intent for WAHA clock skew.


@dataclass(slots=True)
class AlertResult:
    """What one alert send confirmed."""

    text: str
    message_ids: list[str]
    failed_attachments: int = 0
    pending: bool = False

    @property
    def delivered(self) -> bool:
        """Whether the alert text itself reached the operator."""
        return bool(self.message_ids)


def operator_language(config_language: str) -> str:
    """The alert language: configured setting, then English."""
    from wahabot.core.tickets.language import operator_language_for

    return operator_language_for(config_language)


def display_phone(phone: str) -> str:
    """A phone for operator text: digits only, never a raw chat JID."""
    return phone.split("@", 1)[0] if "@" in phone else phone


def translate_cached(
    store: TicketStore,
    text: str,
    target_language: str,
    translator: Translator | None,
) -> str:
    """Translate one text, caching by source hash and target language."""
    if not text.strip() or target_language == "en":
        return text
    key = hashlib.sha256(text.encode()).hexdigest()
    cached = store.translation(key, target_language)
    if cached is not None:
        return cached
    if translator is None:
        return text
    translated = (translator(text, target_language) or "").strip()
    if not translated:
        return text
    store.save_translation(key, target_language, translated)
    return translated


def last_customer_words(store: TicketStore, ticket: Ticket) -> str:
    """The newest customer message on a ticket."""
    event = store.last_event(ticket.id, EventKind.CUSTOMER_MESSAGE.value)
    return event.body if event is not None else ticket.summary


def attachment_label(kind: str) -> str:
    """A short operator-facing label for one attachment kind."""
    if kind == "image":
        return "photo"
    if kind == "voice":
        return "voice note"
    if kind == "video":
        return "video"
    return "file"


def attachment_lines(attachments: Iterable[Attachment]) -> list[str]:
    """Operator-facing descriptions of a ticket's attachments."""
    lines: list[str] = []
    for item in attachments:
        label = attachment_label(item.kind).capitalize()
        text = item.derived_text.removeprefix("[voice note] ").strip()
        detail = f": {text}" if text else ""
        lines.append(f"- {label}{detail}")
    return lines


def detail_lines(details: dict[str, object]) -> list[str]:
    """Flatten stored details into one line each."""
    return [f"- {key}: {value}" for key, value in details.items() if str(value).strip()]


def english_alert(
    store: TicketStore,
    ticket: Ticket,
    contact: Contact,
    attachments: list[Attachment],
) -> str:
    """The alert's source text, before any operator-language translation."""
    lines = [
        f"Ticket {ticket.number}: {contact.display}",
        f"Customer: {contact.display} ({display_phone(contact.phone) or 'no phone'})",
        f"Language: {contact.language or 'unknown'}",
        f"Request: {last_customer_words(store, ticket) or '(no text)'}",
    ]
    details = detail_lines(ticket.details)
    if details:
        lines.append("Details:")
        lines.extend(details)
    lines.append(
        "Missing: " + (", ".join(ticket.missing) if ticket.missing else "nothing known")
    )
    attachment_text = attachment_lines(attachments)
    if attachment_text:
        lines.append("Attachments:")
        lines.extend(attachment_text)
    lines.append("Nothing was promised.")
    lines.append("Reply to this message with what you did.")
    body = "\n".join(lines)
    if ticket.urgent:
        body = f"URGENT\n{body}"
    return body


def compose_alert(
    store: TicketStore,
    ticket: Ticket,
    contact: Contact,
    language: str,
    translator: Translator | None,
) -> str:
    """The alert in the operator's language, with the original retained."""
    attachments = store.attachments(ticket.id)
    source = english_alert(store, ticket, contact, attachments)
    translated = translate_text(store, source, language, translator)
    if translated == source:
        return source
    original = last_customer_words(store, ticket)
    if original:
        return f"{translated}\nOriginal: {original}"
    return translated


def translate_text(
    store: TicketStore,
    text: str,
    language: str,
    translator: Translator | None,
) -> str:
    """Translate to the operator's language when one is configured."""
    if not language or language == "en":
        return text
    return translate_cached(store, text, language, translator)


def compose_delta(store: TicketStore, ticket: Ticket, contact: Contact) -> str:
    """The already-needs-you update: only what is new."""
    words = last_customer_words(store, ticket)
    text = f"Ticket {ticket.number} update from {contact.display}: {words or '(no text)'}"
    if ticket.urgent:
        text = f"URGENT\n{text}"
    return text


def alert_intent_key(store: TicketStore, ticket: Ticket, suffix: str) -> str:
    """A stable key for one handover decision on the current customer message."""
    event = store.last_event(ticket.id, EventKind.CUSTOMER_MESSAGE.value)
    return f"{ticket.id}:{event.id if event is not None else 0}:{suffix}"


def finish_alert(
    store: TicketStore,
    waha: Transport,
    session: str,
    destination: str,
    ticket: Ticket,
    intent: AlertIntent,
    sent_id: str,
    *,
    only_message_ids: list[str] | None,
    moment: int,
) -> AlertResult:
    """Record a confirmed alert text and forward its remaining attachments."""
    remember_self_echo(sent_id)
    store.cancel_notices(ticket.id, moment)
    ids = [sent_id]
    failed = forward_attachments(
        store,
        waha,
        session,
        destination,
        ticket,
        only_message_ids=only_message_ids,
    )
    ids.extend(failed.ids)
    updated = store.record_alert(
        ticket.id,
        ids,
        intent.text,
        alert_key=intent.key,
        at=moment,
    )
    if failed.count:
        notice = _ALERT_FAILED.format(items=", ".join(failed.failed))
        try:
            notice_id = waha.send_text(session, destination, notice)
        except Exception as exc:
            logger.warning("Attachment-failure notice failed: {exc}", exc=exc)
        else:
            if notice_id:
                remember_self_echo(notice_id)
                store.append_alert_ids(ticket.id, [notice_id])
    ticket.state = updated.state
    return AlertResult(
        text=intent.text,
        message_ids=ids,
        failed_attachments=failed.count,
    )


def outgoing_alert(
    store: TicketStore,
    ticket: Ticket,
    contact: Contact,
    language: str,
    translator: Translator | None,
) -> str:
    """Compose the full or delta alert for the ticket's current state."""
    if ticket.state == TicketState.NEEDS_YOU.value:
        return translate_text(
            store, compose_delta(store, ticket, contact), language, translator
        )
    return compose_alert(store, ticket, contact, language, translator)


def resume_attempt(
    store: TicketStore,
    waha: Transport,
    session: str,
    destination: str,
    ticket: Ticket,
    intent: AlertIntent,
    text: str,
    only_message_ids: list[str] | None,
    moment: int,
) -> AlertResult | None:
    """Settle or defer a re-entered alert; None means it may be sent."""
    if intent.state == ALERT_SENT:
        return AlertResult(text=intent.text, message_ids=[intent.sent_id])
    if intent.state != ALERT_SENDING:
        return None
    confirmed = find_delivered_alert(store, waha, session, destination, intent)
    if confirmed:
        return finish_alert(
            store,
            waha,
            session,
            destination,
            ticket,
            intent,
            confirmed,
            only_message_ids=only_message_ids,
            moment=moment,
        )
    if moment - intent.updated_at < ALERT_RECONCILE_GRACE_S:
        return AlertResult(text=intent.text, message_ids=[], pending=True)
    store.restart_alert(intent.key, text, at=moment)
    return None


def send_intent(
    store: TicketStore,
    waha: Transport,
    session: str,
    destination: str,
    ticket: Ticket,
    intent: AlertIntent,
    only_message_ids: list[str] | None,
    moment: int,
) -> AlertResult:
    """Send the intent's text once and record the confirmed result."""
    try:
        sent_id = waha.send_text(session, destination, intent.text)
    except Exception as exc:
        logger.warning("Alert send failed; queued for retry: {exc}", exc=exc)
        store.fail_alert(intent.key, at=store.now())
        return AlertResult(text=intent.text, message_ids=[], pending=True)
    if not sent_id:
        store.fail_alert(intent.key, at=store.now())
        return AlertResult(text=intent.text, message_ids=[], pending=True)
    return finish_alert(
        store,
        waha,
        session,
        destination,
        ticket,
        intent,
        sent_id,
        only_message_ids=only_message_ids,
        moment=moment,
    )


def send_alert(
    store: TicketStore,
    waha: Transport,
    session: str,
    destination: str,
    ticket: Ticket,
    contact: Contact,
    *,
    language: str = "en",
    translator: Translator | None = None,
    only_message_ids: list[str] | None = None,
    alert_key: str = "",
    now: int | None = None,
) -> AlertResult:
    """Send the alert exactly once per durable intent.

    The intent is written before the transport call. A retry adopts a
    message already visible in WAHA (or in the transport's intent ledger)
    instead of sending a second page; only after the grace window, with
    no trace of the message, does it resend.
    """
    moment = store.now() if now is None else now
    key = alert_key or f"{ticket.id}:{uuid.uuid4().hex}"
    text = outgoing_alert(store, ticket, contact, language, translator)
    intent, created = store.claim_alert(
        key, ticket.id, session, destination, text, at=moment
    )
    if not created:
        early = resume_attempt(
            store,
            waha,
            session,
            destination,
            ticket,
            intent,
            text,
            only_message_ids,
            moment,
        )
        if early is not None:
            if early.pending or early.delivered:
                store.cancel_notices(ticket.id, moment)
            return early
        intent = store.alert_intent(key) or intent
    result = send_intent(
        store,
        waha,
        session,
        destination,
        ticket,
        intent,
        only_message_ids,
        moment,
    )
    if result.pending or result.delivered:
        store.cancel_notices(ticket.id, moment)
    return result


def retry_alert(
    store: TicketStore, waha: Transport, intent: AlertIntent, moment: int
) -> bool:
    """Reconcile or resend one stale alert intent from the scheduler."""
    if not store.claim_stale_alert(
        intent.key, moment - ALERT_RECONCILE_GRACE_S, at=moment
    ):
        return False
    ticket = store.ticket(intent.ticket_id)
    if ticket is None or ticket.state == TicketState.CLOSED.value:
        store.cancel_alert(intent.key, at=moment)
        return False
    confirmed = find_delivered_alert(store, waha, intent.session, intent.chat_id, intent)
    if not confirmed:
        try:
            confirmed = waha.send_text(intent.session, intent.chat_id, intent.text)
        except Exception as exc:
            logger.warning("Queued alert retry failed: {exc}", exc=exc)
            store.fail_alert(intent.key, at=store.now())
            return False
        if not confirmed:
            store.fail_alert(intent.key, at=store.now())
            return False
    finish_alert(
        store,
        waha,
        intent.session,
        intent.chat_id,
        ticket,
        intent,
        confirmed,
        only_message_ids=None,
        moment=moment,
    )
    return True


def retry_alerts(
    store: TicketStore, waha: Transport, moment: int, limit: int = 10
) -> int:
    """Retry every alert whose last attempt is older than the grace."""
    done = 0
    for intent in store.stale_alerts(moment - ALERT_RECONCILE_GRACE_S, limit=limit):
        try:
            if retry_alert(store, waha, intent, moment):
                done += 1
        except Exception:
            logger.exception("Alert retry failed for {key}", key=intent.key)
    return done


@dataclass(slots=True)
class ForwardResult:
    """The forwarded ids and the labels of attachments that failed."""

    ids: list[str]
    failed: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        """How many attachments could not be sent."""
        return len(self.failed)


def forward_attachments(
    store: TicketStore,
    waha: Transport,
    session: str,
    destination: str,
    ticket: Ticket,
    *,
    only_message_ids: list[str] | None = None,
) -> ForwardResult:
    """Forward new attachments, falling back to the stored local file."""
    ids: list[str] = []
    failed: list[str] = []
    selected = set(only_message_ids) if only_message_ids else None
    previous = set(ticket.alert_message_ids)
    for attachment in store.attachments(ticket.id):
        if selected is not None and attachment.message_id not in selected:
            continue
        if attachment.message_id in previous and (
            selected is None or attachment.message_id not in selected
        ):
            continue
        message_id = forward_one(waha, session, destination, attachment)
        if message_id:
            remember_self_echo(message_id)
            ids.extend([attachment.message_id, message_id])
            store.append_alert_ids(ticket.id, [attachment.message_id, message_id])
        else:
            failed.append(attachment_label(attachment.kind))
    return ForwardResult(ids=ids, failed=failed)


def forward_one(
    waha: Transport, session: str, destination: str, attachment: Attachment
) -> str:
    """Forward one attachment, falling back to a local media send."""
    try:
        forwarded = waha.forward_message(session, destination, attachment.message_id)
        if forwarded:
            return forwarded
        logger.info(
            "Ticket attachment forward returned no id for {id}; using local file",
            id=attachment.message_id,
        )
    except Exception as exc:
        logger.info(
            "Ticket attachment forward failed for {id}: {exc}; using local file",
            id=attachment.message_id,
            exc=exc,
        )
    try:
        return send_local(waha, session, destination, attachment)
    except Exception as exc:
        logger.warning(
            "Ticket attachment local send failed for {id}: {exc}",
            id=attachment.message_id,
            exc=exc,
        )
        return ""


def send_local(
    waha: Transport, session: str, destination: str, attachment: Attachment
) -> str:
    """Send the stored copy through the matching media endpoint."""
    data = Path(attachment.path).read_bytes()
    payload: dict[str, str] = {
        "mimetype": attachment.mime or "application/octet-stream",
        "filename": Path(attachment.path).name,
        "data": base64.b64encode(data).decode(),
    }
    if attachment.kind == "image":
        return waha.send_image(session, destination, payload)
    if attachment.kind == "video":
        return waha.send_video(session, destination, payload)
    if attachment.kind == "voice":
        return waha.send_voice(session, destination, payload)
    return waha.send_file(session, destination, payload)
