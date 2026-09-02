"""FastAPI app exposing the WAHA webhook endpoint and agent dispatch."""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Header, HTTPException, Request
from loguru import logger

from whabot.core.hmac import verify_hmac
from whabot.core.journal import save_event
from whabot.core.models import WahaEvent

app = FastAPI(title="whabot webhook server")

Handler = Callable[[WahaEvent], Awaitable[None]]
_message_handlers: list[Handler] = []


def on_message(handler: Handler) -> None:
    """Register a coroutine invoked for every incoming `message` event."""
    _message_handlers.append(handler)


async def dispatch(event: WahaEvent) -> None:
    """Run all registered message handlers for the event."""
    for handler in _message_handlers:
        await handler(event)


@app.post("/api/webhook/{session}")
async def waha_webhook(
    session: str,
    request: Request,
    x_webhook_hmac: str | None = Header(default=None),
    x_webhook_hmac_algorithm: str | None = Header(default=None),
) -> WahaEvent:
    """Receive WAHA webhook events, dispatching messages to registered handlers."""
    from whabot.settings import get_settings

    settings = get_settings()
    if session != settings.session:
        raise HTTPException(status_code=404, detail=f"Unknown session {session!r}")

    body = await request.body()
    verify_hmac(settings, body, x_webhook_hmac, x_webhook_hmac_algorithm)

    save_event(settings.journal_dir, session, body)

    event = WahaEvent.model_validate_json(body)
    logger.info(
        "Received {event} event from session {session}",
        event=event.event,
        session=event.session,
    )
    logger.debug("Event payload: {payload}", payload=event.payload)
    if event.event == "message":
        await dispatch(event)
    return event


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
