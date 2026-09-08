"""FastAPI app exposing the WAHA webhook endpoint and agent dispatch."""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Header, HTTPException, Request
from loguru import logger

from wahabot.core.hmac import verify_hmac
from wahabot.core.journal import save_event
from wahabot.core.models import WahaEvent

app = FastAPI(title="wahabot webhook server")

Handler = Callable[[WahaEvent], Awaitable[None]]

#: Handler registries keyed by event type; ``on_event`` registers,
#: ``dispatch`` runs every handler of the event's type.
_registries: dict[str, list[Handler]] = {
    "message": [],
    "message.reaction": [],
    "command": [],
    "forget": [],
    "session.status": [],
}


def on_event(event_type: str, handler: Handler) -> None:
    """Register a coroutine invoked for every incoming *event_type* event."""
    _registries[event_type].append(handler)


def on_message(handler: Handler) -> None:
    """Register a coroutine invoked for every incoming `message` event."""
    on_event("message", handler)


def on_reaction(handler: Handler) -> None:
    """Register a coroutine invoked for every incoming `message.reaction` event."""
    on_event("message.reaction", handler)


def on_command(handler: Handler) -> None:
    """Register a coroutine invoked for every incoming `command` event."""
    on_event("command", handler)


def on_forget(handler: Handler) -> None:
    """Register a coroutine invoked for every incoming `forget` event."""
    on_event("forget", handler)


def on_session_status(handler: Handler) -> None:
    """Register a coroutine invoked for every incoming `session.status` event."""
    on_event("session.status", handler)


async def dispatch(event: WahaEvent) -> None:
    """Run all registered handlers for the event's type.

    ``message*`` events other than ``message.reaction`` (``message``,
    ``message.any``, engine-specific variants) share the ``message``
    registry, mirroring the original catch-all routing.
    """
    event_type = event.event
    if event_type.startswith("message") and event_type != "message.reaction":
        event_type = "message"
    for handler in _registries[event_type]:
        await handler(event)


@app.post("/api/webhook/{session}")
async def waha_webhook(
    session: str,
    request: Request,
    x_webhook_hmac: str | None = Header(default=None),
    x_webhook_hmac_algorithm: str | None = Header(default=None),
) -> WahaEvent:
    """Receive WAHA webhook events, dispatching messages to registered handlers."""
    from wahabot.settings import get_settings

    settings = get_settings()
    if session != settings.session:
        raise HTTPException(status_code=404, detail=f"Unknown session {session!r}")

    body = await request.body()
    verify_hmac(settings, body, x_webhook_hmac, x_webhook_hmac_algorithm)
    event = parse_event(session, body)
    save_event(settings.journal_dir, session, body)
    log_event(event)
    await dispatch(event)
    return event


def parse_event(session: str, body: bytes) -> WahaEvent:
    """Validate the event body; reject malformed or foreign-session events."""
    try:
        event = WahaEvent.model_validate_json(body)
    except (UnicodeDecodeError, ValueError) as exc:
        logger.error("Rejecting malformed event body: {exc}", exc=exc)
        raise HTTPException(status_code=400, detail="Malformed event body") from exc
    if event.session != session:
        logger.error(
            "Event session {event_session} does not match webhook route {route_session}",
            event_session=event.session,
            route_session=session,
        )
        raise HTTPException(status_code=404, detail=f"Unknown session {event.session!r}")
    return event


def log_event(event: WahaEvent) -> None:
    """Log the incoming event briefly."""
    logger.info(
        "Received {event} event from session {session}",
        event=event.event,
        session=event.session,
    )
    logger.debug("Event payload: {payload}", payload=event.payload)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
