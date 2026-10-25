"""The operator web back office mounted in the existing FastAPI process."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from wahabot.web.deps import STATIC_DIR

__all__ = ["mount_routes"]


def mount_routes(app: FastAPI) -> None:
    """Mount the web app routes, static files and templates on *app*."""
    if getattr(app.state, "web_mounted", False):
        return
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="web-static")
    from wahabot.web.routes import history, login, media, settings, setup, skills, teach

    for router in (
        login.router,
        setup.router,
        history.router,
        media.router,
        teach.router,
        skills.router,
        settings.router,
    ):
        app.include_router(router)
    app.state.web_mounted = True
