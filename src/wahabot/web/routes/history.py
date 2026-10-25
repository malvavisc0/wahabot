"""History list and read-only ticket detail."""

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import Response

from wahabot.core.skills.store import SkillStore
from wahabot.web.deps import (
    current_settings,
    database_for,
    error_page,
    page_auth,
    render,
    ticket_store_for,
)
from wahabot.web.services.history import list_view, safe_back, ticket_page

router = APIRouter()


def identity_translation(text: str, target: str) -> str:
    """No translator configured: keep the source text."""
    del target
    return text


def translator(request: Request) -> Any:
    """The configured display translator, or an identity fallback."""
    return getattr(request.app.state, "web_translate", identity_translation)


@router.get("/app/history")
def history_list(request: Request, q: str = "", show: str = "") -> Response:
    """Search, filter and group the tickets."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    store = ticket_store_for(settings)
    view = list_view(store, q=q.strip(), show=show)
    return render(
        request,
        "history_list.html",
        {"csrf_token": auth.csrf, "view": view},
    )


@router.get("/app/history/{number}")
def history_ticket(
    request: Request, number: int, lang: str = "", back: str = ""
) -> Response:
    """One ticket's chronological, read-only thread."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    store = ticket_store_for(settings)
    ticket = store.ticket_by_number(number)
    if ticket is None:
        return error_page(request, "That ticket was not found.", 404)
    skill_store = SkillStore(database_for(settings).path)
    view = ticket_page(
        store,
        skill_store,
        ticket,
        lang=lang or "en",
        translate=translator(request),
        back=safe_back(back),
    )
    return render(
        request,
        "history_ticket.html",
        {"csrf_token": auth.csrf, "view": view},
    )
