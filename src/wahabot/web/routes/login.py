"""Sign in, sign out and the app's entry route."""

import secrets

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from wahabot.settings import Settings
from wahabot.web.auth import (
    clear_session_cookie,
    client_key,
    configured,
    csrf_matches,
    csrf_token,
    forbidden,
    is_secure,
    issue_csrf_cookie,
    issue_session_cookie,
    origin_ok,
    read_session,
    verify_passphrase,
)
from wahabot.web.deps import (
    current_settings,
    insecure,
    limiter_for,
    not_configured,
    page_auth,
    render,
    setup_is_complete,
)

router = APIRouter()


def home_target(settings: Settings) -> str:
    """Setup while it is unfinished, History afterwards."""
    return "/app/history" if setup_is_complete(settings) else "/setup"


@router.get("/")
def home(request: Request) -> Response:
    """Route the front door through login, Setup or History."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    return RedirectResponse(home_target(settings), status_code=303)


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request) -> Response:
    """Show the one passphrase field and issue a CSRF-only cookie."""
    settings = current_settings()
    if not configured(settings):
        return not_configured(request)
    if not is_secure(request, settings):
        return insecure(request)
    existing = read_session(request, settings)
    if existing is not None and existing.authenticated:
        return RedirectResponse(home_target(settings), status_code=303)
    token = existing.csrf if existing is not None else secrets.token_urlsafe(24)
    response = render(
        request,
        "login.html",
        {"csrf_token": token, "error": "", "chromeless": True, "show_header": False},
    )
    if existing is None:
        issue_csrf_cookie(request, response, settings, token)
    return response


@router.post("/login")
async def login_submit(request: Request) -> Response:
    """Verify the passphrase, enforce CSRF and rate limit failed attempts."""
    settings = current_settings()
    if not configured(settings):
        return not_configured(request)
    if not is_secure(request, settings):
        return insecure(request)
    if not origin_ok(request, settings):
        return forbidden()
    form = await request.form()
    submitted = str(form.get("csrf_token", ""))
    if not csrf_matches(request, settings, submitted):
        return forbidden()
    limiter = limiter_for(request)
    key = client_key(request)
    retry = limiter.retry_after(key)
    token = csrf_token(request, settings)
    if retry is not None:
        response = render(
            request,
            "login.html",
            {
                "csrf_token": token,
                "error": f"Too many attempts. Try again in {retry} seconds.",
                "chromeless": True,
                "show_header": False,
            },
            status_code=429,
        )
        response.headers["Retry-After"] = str(retry)
        return response
    passphrase = str(form.get("passphrase", ""))
    if not verify_passphrase(passphrase, settings.web_passphrase_hash):
        limiter.record_failure(key)
        return render(
            request,
            "login.html",
            {
                "csrf_token": token,
                "error": "Wrong passphrase. Try again.",
                "chromeless": True,
                "show_header": False,
            },
            status_code=401,
        )
    limiter.reset(key)
    response = RedirectResponse(home_target(settings), status_code=303)
    issue_session_cookie(request, response, settings, token)
    return response


@router.post("/logout")
async def logout(request: Request) -> Response:
    """Clear the cookie after the CSRF and origin checks."""
    settings = current_settings()
    if not configured(settings):
        return not_configured(request)
    if not origin_ok(request, settings):
        return forbidden()
    form = await request.form()
    if not csrf_matches(request, settings, str(form.get("csrf_token", ""))):
        return forbidden()
    response = RedirectResponse("/login", status_code=303)
    clear_session_cookie(response)
    return response


@router.get("/logout")
def logout_get() -> Response:
    """GET never mutates the session; use the visible Sign out button."""
    return HTMLResponse(
        "<p>Use the Sign out button to end the session.</p>",
        status_code=405,
    )
