"""FastAPI app exposing the WAHA webhook endpoint and agent dispatch."""

import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable

from fastapi import FastAPI, Header, HTTPException, Request
from loguru import logger

from wahabot.core.hmac import verify_hmac
from wahabot.core.journal import save_event
from wahabot.core.models import WahaEvent

#: Coroutines awaited once as the app shuts down. Bounded background
#: work (e.g. detached escalation sends) drains here instead of being
#: dropped when the process exits.
_shutdown_hooks: list[Callable[[], Awaitable[None]]] = []


def on_shutdown(hook: Callable[[], Awaitable[None]]) -> None:
    """Register *hook* to be awaited once when the app shuts down."""
    _shutdown_hooks.append(hook)


async def run_shutdown_hooks() -> None:
    """Await every shutdown hook; one failure never blocks the rest."""
    for hook in tuple(_shutdown_hooks):
        try:
            await hook()
        except Exception as exc:
            logger.warning("Shutdown hook failed: {exc}", exc=exc)


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Drain bounded background work as the app shuts down."""
    yield
    await run_shutdown_hooks()


app = FastAPI(title="wahabot webhook server", lifespan=_lifespan)

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

    A handler exception is contained: it is logged once, in full, and
    the remaining handlers for the event still run. An event that one
    handler failed on must not become a webhook 500 — WAHA treats a
    non-200 as a failed delivery and re-sends the event, turning a
    transient handler error into an endless redelivery loop, and a
    crash mid-dispatch would silently drop the other handlers.
    """
    event_type = event.event
    if event_type.startswith("message") and event_type != "message.reaction":
        event_type = "message"
    for handler in _registries[event_type]:
        try:
            await handler(event)
        except Exception as exc:
            logger.error(
                "Handler {handler} failed for {event} event {id}: "
                + "{error_type}: {error}",
                handler=getattr(handler, "__name__", handler),
                event=event.event,
                id=event.payload.get("id", event.id),
                error_type=type(exc).__name__,
                error=exc,
            )
            logger.opt(exception=exc).debug("Handler failure detail")


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
    except ValueError as exc:
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
