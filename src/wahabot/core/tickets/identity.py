"""Customer JID identity, aliases and LID-to-phone resolution."""

from collections.abc import Callable
from typing import cast

from loguru import logger

from wahabot.core.models import WahaEvent
from wahabot.core.tickets.models import text_field
from wahabot.core.tickets.store import TicketStore

_PHONE_DOMAINS = ("c.us", "s.whatsapp.net")
_ALIAS_KEYS = (
    "remoteJidAlt",
    "senderLid",
    "senderPn",
    "participantAlt",
    "participantPn",
)


def jid_text(value: object) -> str:
    """A JID from a plain string or a WAHA ``{_serialized: ...}`` object."""
    if isinstance(value, dict):
        return str(cast("dict[str, object]", value).get("_serialized") or "")
    return str(value or "")


def jid_parts(value: object) -> tuple[str, str]:
    """``(user, domain)`` for a JID, stripping any linked-device suffix."""
    user, _, domain = jid_text(value).partition("@")
    return user.split(":", 1)[0], domain


def normalize_phone(value: object) -> str:
    """A canonical phone JID from *value*, or "" when it is not one."""
    user, domain = jid_parts(value)
    return f"{user}@{domain}" if domain in _PHONE_DOMAINS else ""


def normalize_lid(value: object) -> str:
    """A canonical LID from *value*, or "" when it is not one."""
    user, domain = jid_parts(value)
    return f"{user}@lid" if domain == "lid" else ""


def alias_jids(event: WahaEvent) -> list[str]:
    """Alternate JIDs WAHA sometimes includes beside the chat and sender."""
    data = event.payload.get("_data")
    if not isinstance(data, dict):
        return []
    fields = cast("dict[str, object]", data)
    found: list[str] = []
    for container in (fields, fields.get("key")):
        if not isinstance(container, dict):
            continue
        values = cast("dict[str, object]", container)
        for key in _ALIAS_KEYS:
            value = values.get(key)
            if value:
                found.append(jid_text(value))
    return found


def primary_identity(chat_id: str, sender: str) -> tuple[str, str]:
    """``(phone, lid)`` from the chat and sender JIDs."""
    sender_user, sender_domain = jid_parts(sender)
    chat_user, chat_domain = jid_parts(chat_id)
    phone = f"{chat_user}@{chat_domain}" if chat_domain in _PHONE_DOMAINS else ""
    if sender_domain in _PHONE_DOMAINS:
        phone = phone or f"{sender_user}@{sender_domain}"
    lid = ""
    if sender_domain == "lid":
        lid = f"{sender_user}@lid"
    elif chat_domain == "lid":
        lid = f"{chat_user}@lid"
    return phone, lid


def merge_alias_identity(phone: str, lid: str, event: WahaEvent) -> tuple[str, str]:
    """Fill missing phone/lid values from WAHA's alternate JIDs."""
    for candidate in alias_jids(event):
        user, domain = jid_parts(candidate)
        if domain in _PHONE_DOMAINS:
            phone = phone or f"{user}@{domain}"
        elif domain == "lid":
            lid = lid or f"{user}@lid"
    return phone, lid


def customer_details(event: WahaEvent) -> tuple[str, str, str, str]:
    """``(chat_id, phone, lid, display_name)`` from one customer event."""
    chat_id = jid_text(event.payload.get("from"))
    sender = jid_text(event.payload.get("participant")) or chat_id
    phone, lid = primary_identity(chat_id, sender)
    phone, lid = merge_alias_identity(phone, lid, event)
    name = text_field(event.payload.get("_data"), "notifyName") or str(
        event.payload.get("notifyName") or ""
    )
    return chat_id, phone, lid, name


def phone_alias(
    store: TicketStore,
    event: WahaEvent,
    resolve: Callable[[str, str], str],
) -> str:
    """The phone JID behind a LID chat, via the payload or WAHA's mapping."""
    chat_id, phone, _, _ = customer_details(event)
    if phone or not chat_id.endswith("@lid"):
        return phone
    existing = store.contact(chat_id)
    if existing is not None and existing.phone:
        return existing.phone
    try:
        return resolve(event.session, chat_id)
    except Exception as exc:
        logger.debug("LID lookup failed for {chat}: {exc}", chat=chat_id, exc=exc)
        return ""
