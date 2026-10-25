"""Setup: create the WAHA session, show the QR and wait for the first test."""

import contextlib

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from wahabot.settings import Settings
from wahabot.web.auth import csrf_matches, forbidden, origin_ok
from wahabot.web.deps import (
    current_settings,
    media_auth,
    page_auth,
    render,
    ticket_store_for,
    waha_for,
)
from wahabot.web.services.setup import SetupService, SetupState

router = APIRouter()


def service_for(request: Request, settings: Settings) -> SetupService:
    """Build the Setup service around the request's WAHA client."""
    return SetupService(settings, ticket_store_for(settings), waha_for(request, settings))


def setup_context(
    request: Request, settings: Settings, state: SetupState, token: str
) -> dict[str, object]:
    """The template values every Setup render shares."""
    service = service_for(request, settings)
    return {
        "csrf_token": token,
        "status": state.status,
        "qr_data": state.qr_data,
        "qr_value": state.qr_value,
        "exchange": state.exchange,
        "completed": state.completed,
        "support_number": service.support_number(),
        "pairing_code": "",
        "error": state.error,
    }


@router.get("/setup")
def setup_page(request: Request, reconnect: str = "") -> Response:
    """Open the session, show the QR, and finish once the test is answered."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    service = service_for(request, settings)
    try:
        state = service.state()
    except Exception as exc:
        state = SetupState(
            status="UNKNOWN",
            qr_data="",
            qr_value="",
            exchange=None,
            completed=False,
            error=f"WAHA is not reachable: {type(exc).__name__}",
        )
    if state.completed and not reconnect:
        return RedirectResponse("/app/history", status_code=303)
    return render(
        request, "setup.html", setup_context(request, settings, state, auth.csrf)
    )


@router.post("/setup/pairing")
async def setup_pairing(request: Request) -> Response:
    """Request a pairing code; the QR remains the fallback."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    if not origin_ok(request, settings):
        return forbidden()
    form = await request.form()
    if not csrf_matches(request, settings, str(form.get("csrf_token", ""))):
        return forbidden()
    service = service_for(request, settings)
    pairing_code = ""
    error = ""
    try:
        pairing_code = service.pairing_code(str(form.get("phone_number", "")))
    except ValueError as exc:
        error = str(exc)
    except Exception as exc:
        error = f"WAHA refused the pairing code: {type(exc).__name__}"
    state = service.state()
    context = setup_context(request, settings, state, auth.csrf)
    context.update({"pairing_code": pairing_code, "error": error})
    return render(request, "setup.html", context)


@router.post("/setup/restart")
async def setup_restart(request: Request) -> Response:
    """Restart a failed session in place."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    if not origin_ok(request, settings):
        return forbidden()
    form = await request.form()
    if not csrf_matches(request, settings, str(form.get("csrf_token", ""))):
        return forbidden()
    with contextlib.suppress(Exception):
        service_for(request, settings).restart()
    return RedirectResponse("/setup", status_code=303)


@router.get("/setup/status")
def setup_status(request: Request) -> Response:
    """A small JSON poll target for the QR and first-test state."""
    settings = current_settings()
    refused = media_auth(request, settings)
    if refused is not None:
        return refused
    state = service_for(request, settings).state()
    exchange = None
    if state.exchange is not None:
        exchange = {
            "ticket_number": state.exchange.ticket_number,
            "question": state.exchange.question,
            "answer": state.exchange.answer,
        }
    return JSONResponse(
        {
            "status": state.status,
            "qr_data": state.qr_data,
            "qr_value": state.qr_value,
            "completed": state.completed,
            "exchange": exchange,
        }
    )
