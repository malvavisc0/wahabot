"""Web-only middleware: security headers and the principal-web journal."""

import contextlib
from collections.abc import Awaitable, Callable

from fastapi import Request, Response


async def web_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Add web headers and journal every web write without caching its body."""
    response = await call_next(request)
    if request.url.path.startswith(("/app", "/setup", "/login", "/logout")):
        import importlib

        deps = importlib.import_module("wahabot.web.deps")
        deps.security_headers(response)
        with contextlib.suppress(Exception):
            deps.journal_web_write(request, response)
    return response
