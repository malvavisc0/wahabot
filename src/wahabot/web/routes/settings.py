"""Settings: four independent section saves over the session config."""

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import Response

from wahabot.core.waha import WahaClient
from wahabot.settings import Settings
from wahabot.web.auth import csrf_matches, forbidden, origin_ok
from wahabot.web.deps import (
    current_settings,
    page_auth,
    render,
    ticket_store_for,
    waha_for,
)
from wahabot.web.services.settings import (
    DAY_LABELS,
    away_preview,
    display_numbers,
    hours_values,
    load_config,
    read_raw,
    save_assistant,
    save_hours,
    save_who,
    team_groups_view,
)
from wahabot.web.services.setup import SetupService

router = APIRouter()


def whatsapp_state(settings: Settings, waha: WahaClient) -> dict[str, str]:
    """The connection block: status, number and, when unlinked, the QR."""
    service = SetupService(settings, ticket_store_for(settings), waha)
    status = service.status()
    number = service.support_number()
    qr_data = ""
    if status != "WORKING":
        try:
            qr_data, _value = service.qr()
        except Exception:
            qr_data = ""
    return {"status": status, "number": number, "qr_data": qr_data}


def page_context(
    settings: Settings,
    token: str,
    waha: WahaClient,
    *,
    form: dict[str, str] | None = None,
    feedback: dict[str, dict[str, str]] | None = None,
    section: str = "assistant",
) -> dict[str, Any]:
    """The full Settings page values, with optional unsaved overrides."""
    raw = read_raw(settings)
    config = load_config(settings)
    saved = {"message": "Saved", "class": "saved"}
    messages = feedback or {
        "assistant": saved,
        "who": saved,
        "hours": saved,
        "whatsapp": {"message": "", "class": ""},
    }
    return {
        "csrf_token": token,
        "raw": raw,
        "config": config,
        "form": form or {},
        "feedback": messages,
        "current_section": section,
        "days": DAY_LABELS,
        "hours_values": hours_values(raw),
        "away_preview": away_preview(raw),
        "whitelist_text": display_numbers(list(raw.get("whitelist") or [])),
        "blacklist_text": display_numbers(list(raw.get("blacklist") or [])),
        "night_number_text": raw.get("night_number", ""),
        "team_groups": team_groups_view(
            waha, settings.session, list(raw.get("team_groups") or [])
        ),
        "tts_configured": bool(settings.tts_url),
        "whatsapp": whatsapp_state(settings, waha),
    }


def form_values(form: Any) -> dict[str, str]:
    """A form as plain strings, without the CSRF token."""
    return {
        key: str(value)
        for key, value in form.items()
        if key not in {"csrf_token", "team_groups"}
    }


def valid_write(request: Request, settings: Settings, form: Any) -> bool:
    """Origin and CSRF for one settings write."""
    if not origin_ok(request, settings):
        return False
    return csrf_matches(request, settings, str(form.get("csrf_token", "")))


@router.get("/app/settings")
def settings_page(request: Request, section: str = "") -> Response:
    """All four sections, each with its own save action."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    return render(
        request,
        "settings.html",
        page_context(
            settings,
            auth.csrf,
            waha_for(request, settings),
            section=section or "assistant",
        ),
    )


@router.post("/app/settings/assistant")
async def settings_assistant(request: Request) -> Response:
    """Save the Assistant section only."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    form = await request.form()
    if not valid_write(request, settings, form):
        return forbidden()
    values = form_values(form)
    try:
        save_assistant(settings, values)
    except ValueError as exc:
        return render(
            request,
            "settings.html",
            page_context(
                settings,
                auth.csrf,
                waha_for(request, settings),
                form=values,
                feedback={
                    "assistant": {"message": f"Not saved: {exc}", "class": "error"}
                },
                section="assistant",
            ),
            status_code=400,
        )
    return render(
        request,
        "settings.html",
        page_context(
            settings,
            auth.csrf,
            waha_for(request, settings),
            feedback={"assistant": {"message": "Saved", "class": "saved"}},
            section="assistant",
        ),
    )


@router.post("/app/settings/who")
async def settings_who(request: Request) -> Response:
    """Save only allowed/blocked numbers and team groups."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    form = await request.form()
    if not valid_write(request, settings, form):
        return forbidden()
    values = form_values(form)
    groups = [str(item) for item in form.getlist("team_groups") if str(item)]
    try:
        save_who(settings, form, groups, waha_for(request, settings))
    except ValueError as exc:
        return render(
            request,
            "settings.html",
            page_context(
                settings,
                auth.csrf,
                waha_for(request, settings),
                form=values,
                feedback={"who": {"message": f"Not saved: {exc}", "class": "error"}},
                section="who",
            ),
            status_code=400,
        )
    return render(
        request,
        "settings.html",
        page_context(
            settings,
            auth.csrf,
            waha_for(request, settings),
            feedback={"who": {"message": "Saved", "class": "saved"}},
            section="who",
        ),
    )


@router.post("/app/settings/hours")
async def settings_hours(request: Request) -> Response:
    """Save only timezone, hours, away message and the reply window."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    form = await request.form()
    if not valid_write(request, settings, form):
        return forbidden()
    values = form_values(form)
    try:
        save_hours(settings, values)
    except ValueError as exc:
        return render(
            request,
            "settings.html",
            page_context(
                settings,
                auth.csrf,
                waha_for(request, settings),
                form=values,
                feedback={"hours": {"message": f"Not saved: {exc}", "class": "error"}},
                section="hours",
            ),
            status_code=400,
        )
    return render(
        request,
        "settings.html",
        page_context(
            settings,
            auth.csrf,
            waha_for(request, settings),
            feedback={"hours": {"message": "Saved", "class": "saved"}},
            section="hours",
        ),
    )


@router.post("/app/settings/whatsapp")
async def settings_whatsapp(request: Request) -> Response:
    """Reconnect the session and show a fresh QR when it is unlinked."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    form = await request.form()
    if not valid_write(request, settings, form):
        return forbidden()
    service = SetupService(
        settings, ticket_store_for(settings), waha_for(request, settings)
    )
    message = "Reconnect requested."
    classes = "saved"
    try:
        service.ensure_session()
    except Exception as exc:
        message = f"Not saved: {type(exc).__name__}"
        classes = "error"
    return render(
        request,
        "settings.html",
        page_context(
            settings,
            auth.csrf,
            waha_for(request, settings),
            feedback={"whatsapp": {"message": message, "class": classes}},
            section="whatsapp",
        ),
    )
