import time

from wahabot.core.cache import TtlCache
from wahabot.core.models import WahaEvent

_MAX_MESSAGE_AGE_S = 300

_SEE_TTL_S = _MAX_MESSAGE_AGE_S

_MAX_SEEN_IDS = 10_000

_seen_ids: TtlCache[str, bool] = TtlCache(_SEE_TTL_S, _MAX_SEEN_IDS)


def seen_recently(message_id: str) -> bool:
    """Return True if this message id was already handled in the last TTL.

    The id is marked seen on first sight, so a duplicate redelivery
    racing the in-flight run is also deduplicated. A run that later
    fails must call :func:`forget_seen` to allow a retry.
    """
    if not message_id or message_id in _seen_ids:
        return bool(message_id)
    _seen_ids.put(message_id, True)
    return False


def forget_seen(message_id: str) -> None:
    """Drop a message id's seen marker so a redelivery is reprocessed."""
    _seen_ids.drop(message_id)


def is_stale(event: WahaEvent, started_at: float) -> bool:
    """True when the message is replayed backlog rather than a live turn.

    WhatsApp redelivers undelivered messages when the WAHA session or the
    phone reconnects, and WAHA forwards them as fresh ``message`` events.
    Two guards: anything sent before this process started is definitionally
    backlog, and anything older than ``_MAX_MESSAGE_AGE_S`` is stale even
    mid-run (phone reconnect flush). Unknown timestamps pass — better one
    late reply than silence.
    """
    ts = event.payload.get("timestamp")
    if not isinstance(ts, (int, float)):
        return False
    return ts < started_at or time.time() - ts > _MAX_MESSAGE_AGE_S
