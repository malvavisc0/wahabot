"""Compose and send the operator alert for a ticket."""

import base64
import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from wahabot.core.tickets.models import (
    Actor,
    Attachment,
    Contact,
    EventKind,
    Ticket,
    TicketState,
    TicketUpdate,
)
from wahabot.core.tickets.store import TicketStore
from wahabot.core.transport import Transport

Translator = Callable[[str, str], str]

_ALERT_FAILED = "Could not attach {count} file(s); they stay stored on the ticket."


@dataclass(slots=True)
class AlertResult:
    """What one alert send confirmed."""

    text: str
    message_ids: list[str]
    failed_attachments: int = 0

    @property
    def delivered(self) -> bool:
        """Whether the alert text itself reached the operator."""
        return bool(self.message_ids)


def operator_language(config_language: str, contact_language: str = "") -> str:
    """The alert language: the operator's setting, then English."""
    return (config_language or "en").strip() or "en"


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
    for event in reversed(store.events(ticket.id)):
        if event.kind == EventKind.CUSTOMER_MESSAGE.value:
            return event.body
    return ticket.summary


def attachment_lines(attachments: Iterable[Attachment]) -> list[str]:
    """Operator-facing descriptions of a ticket's attachments."""
    lines: list[str] = []
    for item in attachments:
        if item.kind == "voice":
            label = "Voice note"
        elif item.kind == "image":
            label = "Photo"
        elif item.kind == "video":
            label = "Video"
        else:
            label = "File"
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
    if language == "en":
        return source
    translated = translate_cached(store, source, language, translator)
    if translated == source:
        return source
    original = last_customer_words(store, ticket)
    if original:
        return f"{translated}\nOriginal: {original}"
    return translated


def compose_delta(store: TicketStore, ticket: Ticket, contact: Contact) -> str:
    """The already-needs-you update: only what is new."""
    words = last_customer_words(store, ticket)
    text = f"Ticket {ticket.number} update from {contact.display}: {words or '(no text)'}"
    if ticket.urgent:
        text = f"URGENT\n{text}"
    return text


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
) -> AlertResult:
    """Send the alert, forward its attachments and store every id.

    Nothing is marked confirmed until the transport returns an id: the
    state change and the alert event are written after the text send.
    """
    already = ticket.state == TicketState.NEEDS_YOU.value
    text = (
        compose_delta(store, ticket, contact)
        if already
        else compose_alert(store, ticket, contact, language, translator)
    )
    sent_id = waha.send_text(session, destination, text)
    if not sent_id:
        return AlertResult(text=text, message_ids=[])
    ids = [sent_id]
    failed = forward_attachments(
        store,
        waha,
        session,
        destination,
        ticket,
        already=already,
        only_message_ids=only_message_ids,
    )
    ids.extend(failed.ids)
    store.append_alert_ids(ticket.id, ids)
    store.add_event(
        ticket.id,
        Actor.ASSISTANT.value,
        EventKind.ALERT.value,
        text,
        ids,
    )
    updated = store.update_ticket(
        ticket.id,
        TicketUpdate(state=TicketState.NEEDS_YOU.value),
        actor=Actor.ASSISTANT.value,
        kind=EventKind.STATE.value,
        body="alert sent to the operator",
    )
    if failed.count:
        notice = _ALERT_FAILED.format(count=failed.count)
        try:
            store.append_alert_ids(
                ticket.id, [waha.send_text(session, destination, notice)]
            )
        except Exception as exc:
            logger.warning("Attachment-failure notice failed: {exc}", exc=exc)
    ticket.state = updated.state
    return AlertResult(text=text, message_ids=ids, failed_attachments=failed.count)


@dataclass(slots=True)
class ForwardResult:
    """The forwarded ids and the number of failures."""

    ids: list[str]
    count: int = 0


def forward_attachments(
    store: TicketStore,
    waha: Transport,
    session: str,
    destination: str,
    ticket: Ticket,
    *,
    already: bool,
    only_message_ids: list[str] | None = None,
) -> ForwardResult:
    """Forward new attachments, falling back to the stored local file."""
    ids: list[str] = []
    failed = 0
    selected = set(only_message_ids) if only_message_ids else None
    for attachment in store.attachments(ticket.id):
        if selected is not None and attachment.message_id not in selected:
            continue
        if already and attachment.message_id in ticket.alert_message_ids:
            continue
        message_id = forward_one(waha, session, destination, attachment)
        if message_id:
            ids.extend([attachment.message_id, message_id])
        else:
            failed += 1
    return ForwardResult(ids=ids, count=failed)


def forward_one(
    waha: Transport, session: str, destination: str, attachment: Attachment
) -> str:
    """Forward one attachment, falling back to a local media send."""
    try:
        return waha.forward_message(session, destination, attachment.message_id)
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
