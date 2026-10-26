"""Shared web dependencies kept free of route imports."""

from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.templating import Jinja2Templates

from wahabot.core.answers.service import AnswerService
from wahabot.core.answers.store import AnswerStore
from wahabot.core.audit import save_action
from wahabot.core.db import Database
from wahabot.core.skills.drafts import SkillDraftService
from wahabot.core.skills.secrets import SecretStore
from wahabot.core.skills.service import SkillService
from wahabot.core.skills.store import SkillStore
from wahabot.core.tickets.store import TicketStore
from wahabot.core.transport import WahaTransport
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings
from wahabot.web.auth import (
    AuthSession,
    RateLimiter,
    configured,
    is_secure,
    read_session,
)
from wahabot.web.i18n import _ as translate

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
_globals: dict[str, Any] = templates.env.globals
_globals["_"] = translate
NAV_ITEMS = (
    ("/app/history", "History"),
    ("/app/teach", "Teach"),
    ("/app/skills", "Skills"),
    ("/app/settings", "Settings"),
)
WEB_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def current_settings() -> Settings:
    """The process settings, looked up so tests and reloads can replace them."""
    from wahabot import settings as settings_module

    return settings_module.get_settings()


def render(
    request: Request,
    name: str,
    context: dict[str, Any] | None = None,
    *,
    status_code: int = 200,
) -> HTMLResponse:
    """Render one template with the shared request and navigation context."""
    values: dict[str, Any] = {
        "request": request,
        "nav_items": NAV_ITEMS,
        "current_path": request.url.path,
    }
    if context:
        values.update(context)
    values.setdefault(
        "session",
        bool(values.get("csrf_token")) and not values.get("chromeless", False),
    )
    return templates.TemplateResponse(
        request=request,
        name=name,
        context=values,
        status_code=status_code,
    )


def error_page(request: Request, message: str, status_code: int) -> HTMLResponse:
    """A standalone page for auth and configuration failures."""
    return render(request, "error.html", {"message": message}, status_code=status_code)


def not_configured(request: Request) -> HTMLResponse:
    """Fail closed until both web secrets are configured."""
    return error_page(
        request,
        "Web access is not configured. Set WAHABOT_WEB_PASSPHRASE_HASH and "
        + "WAHABOT_WEB_SESSION_KEY on the server.",
        503,
    )


def insecure(request: Request) -> HTMLResponse:
    """Plain LAN HTTP may show instructions but never a login form."""
    return error_page(
        request,
        "Open the encrypted address for this box. Sign-in is refused over plain HTTP.",
        403,
    )


def page_auth(request: Request, settings: Settings) -> AuthSession | Response:
    """Return the authenticated session or the response that refuses the page."""
    if not configured(settings):
        return not_configured(request)
    if not is_secure(request, settings):
        return insecure(request)
    session = read_session(request, settings)
    if session is None or not session.authenticated:
        return RedirectResponse("/login", status_code=303)
    return session


def media_auth(request: Request, settings: Settings) -> Response | None:
    """Media and poll endpoints: auth-gated like pages, without redirects."""
    if not configured(settings):
        return Response(status_code=401)
    if not is_secure(request, settings):
        return Response(status_code=403)
    session = read_session(request, settings)
    if session is None or not session.authenticated:
        return Response(status_code=401)
    return None


def limiter_for(request: Request) -> RateLimiter:
    """The app's failed-login limiter, rebuilt when its settings change."""
    settings = current_settings()
    current = getattr(request.app.state, "web_limiter", None)
    if (
        not isinstance(current, RateLimiter)
        or current.limit != settings.web_login_attempts
        or current.window_s != settings.web_login_window_s
    ):
        current = RateLimiter(settings.web_login_attempts, settings.web_login_window_s)
        request.app.state.web_limiter = current
    return current


def setup_is_complete(settings: Settings) -> bool:
    """Whether Setup finished for this session; a missing database is unfinished."""
    path = settings.data_dir / "wahabot.db"
    if not path.exists():
        return False
    try:
        return TicketStore(path).setup_completed(settings.session)
    except Exception:
        return False


def journal_web_write(request: Request, response: Response) -> None:
    """Record one web write with principal ``web`` and no request body."""
    path = request.url.path
    if request.method not in WEB_WRITE_METHODS:
        return
    if not path.startswith(("/app", "/setup", "/login", "/logout")):
        return
    settings = current_settings()
    save_action(
        settings.data_dir,
        settings.session,
        "web_write",
        principal="web",
        method=request.method,
        path=path,
        status=response.status_code,
    )


def security_headers(response: Response) -> None:
    """Conservative headers for the operator app."""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("Cache-Control", "no-store")


def waha_for(request: Request, settings: Settings) -> WahaClient:
    """The request's WAHA client, with a test override on app.state."""
    override = getattr(request.app.state, "web_waha_client", None)
    if isinstance(override, WahaClient):
        return override
    return WahaClient(base_url=settings.waha_url, api_key=settings.waha_api_key)


def database_for(settings: Settings) -> Database:
    """Open and migrate the platform database for one web request."""
    database = Database(settings.data_dir / "wahabot.db")
    database.migrate()
    return database


def ticket_store_for(settings: Settings) -> TicketStore:
    """A ticket store over the migrated database."""
    return TicketStore(database_for(settings).path)


def answer_service_for(settings: Settings) -> AnswerService:
    """The shared answer service for Teach."""
    return AnswerService(AnswerStore(database_for(settings).path))


def skill_runtime_for(
    settings: Settings,
) -> tuple[SkillStore, SecretStore, SkillDraftService, SkillService]:
    """The skill stores and draft service for the Skills pages."""
    database = database_for(settings)
    store = SkillStore(database.path)
    secrets = SecretStore(database.path, settings.secret_key)
    client = WahaClient(base_url=settings.waha_url, api_key=settings.waha_api_key)
    service = SkillService(
        store,
        secrets,
        settings.skills_dir,
        settings.session,
        WahaTransport(client),
        data_dir=settings.data_dir,
        ticket_store=TicketStore(database.path),
    )
    drafts = SkillDraftService(store, secrets, settings.skills_dir, service)
    return store, secrets, drafts, service
