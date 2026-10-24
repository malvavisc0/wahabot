"""Find an operator alert already delivered by an earlier attempt."""

from loguru import logger

from wahabot.core.tickets.models import AlertIntent
from wahabot.core.tickets.store import TicketStore
from wahabot.core.transport import Transport

ALERT_LOOKBACK_S = 300


def waha_message_id(message: dict[str, object]) -> str:
    value = message.get("id")
    if isinstance(value, dict):
        return str(value.get("_serialized") or "")
    return str(value or "")


def waha_message_body(message: dict[str, object]) -> str:
    body = str(message.get("body") or "")
    if body:
        return body
    data = message.get("_data")
    return str(data.get("body") or "") if isinstance(data, dict) else ""


def waha_message_time(message: dict[str, object]) -> int | None:
    value = message.get("timestamp")
    if value is None:
        data = message.get("_data")
        if isinstance(data, dict):
            value = data.get("messageTimestamp")
    if not isinstance(value, int | float | str):
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def ledger_alert_id(
    store: TicketStore,
    waha: Transport,
    destination: str,
    intent: AlertIntent,
) -> str:
    """A sent alert found in the transport's own intent ledger."""
    intents = getattr(waha, "intents", None)
    if intents is None:
        return ""
    found = intents.find_sent_text(
        destination, intent.text, float(intent.created_at) - ALERT_LOOKBACK_S
    )
    if found and not store.alert_id_used(found, intent.key):
        return found
    return ""


def history_alert_id(
    store: TicketStore,
    waha: Transport,
    session: str,
    destination: str,
    intent: AlertIntent,
) -> str:
    """A sent alert found in the operator chat's WAHA history."""
    try:
        messages = waha.fetch_chat_messages(session, destination, limit=100)
    except Exception as exc:
        logger.debug("Alert reconciliation fetch failed: {exc}", exc=exc)
        return ""
    for message in messages:
        sent_id = history_match(store, message, intent)
        if sent_id:
            return sent_id
    return ""


def history_match(store: TicketStore, message: object, intent: AlertIntent) -> str:
    """The id of one history message when it is this intent's alert."""
    if not isinstance(message, dict):
        return ""
    if message.get("fromMe") is False:
        return ""
    if waha_message_body(message) != intent.text:
        return ""
    moment = waha_message_time(message)
    if moment is not None and moment < intent.created_at - ALERT_LOOKBACK_S:
        return ""
    sent_id = waha_message_id(message)
    if sent_id and not store.alert_id_used(sent_id, intent.key):
        return sent_id
    return ""


def find_delivered_alert(
    store: TicketStore,
    waha: Transport,
    session: str,
    destination: str,
    intent: AlertIntent,
) -> str:
    """The transport id of an alert already delivered for *intent*, or ""."""
    return ledger_alert_id(store, waha, destination, intent) or history_alert_id(
        store, waha, session, destination, intent
    )
