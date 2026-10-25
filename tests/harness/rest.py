"""A local REST service for skill HTTP tests and wire runs."""

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse


@dataclass
class FakeRest:
    """A local REST service and the calls it received."""

    base: str
    other: str
    writes: list[dict[str, Any]] = field(default_factory=list)
    hooks: list[dict[str, Any]] = field(default_factory=list)

    def lead_count(self) -> int:
        """How many writes reached the service."""
        return len(self.writes)


def run(app: FastAPI) -> tuple[uvicorn.Server, threading.Thread, int]:
    """Serve *app* on an ephemeral loopback port."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, port


def stop(server: uvicorn.Server, thread: threading.Thread) -> None:
    """Stop one served app."""
    server.should_exit = True
    thread.join(timeout=5)


def app(rest: FakeRest, other_port: dict[str, int]) -> FastAPI:
    """The fake customer service with reads, writes, redirects and a hook."""
    application = FastAPI()

    @application.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @application.get("/unhealthy")
    def unhealthy() -> JSONResponse:
        return JSONResponse({"ok": False}, status_code=500)

    @application.get("/records")
    def records(phone: str = "") -> dict[str, object]:
        return {"phone": phone, "records": [{"id": "r-1", "amount": "23.40"}]}

    @application.get("/secret-echo")
    def secret_echo(request: Request) -> dict[str, str]:
        return {"authorization": request.headers.get("authorization", "")}

    @application.get("/large")
    def large() -> dict[str, str]:
        return {"filler": "x" * 200_000}

    @application.get("/slow")
    def slow() -> dict[str, bool]:
        time.sleep(0.5)
        return {"ok": True}

    @application.post("/leads")
    async def leads(request: Request) -> dict[str, str]:
        rest.writes.append(await request.json())
        return {"id": "lead-1"}

    @application.get("/redirect-same")
    def redirect_same() -> RedirectResponse:
        return RedirectResponse("/health", status_code=302)

    @application.get("/redirect-other")
    def redirect_other() -> RedirectResponse:
        return RedirectResponse(
            f"http://127.0.0.1:{other_port['port']}/health", status_code=302
        )

    @application.post("/hook")
    async def hook(request: Request) -> dict[str, bool]:
        rest.hooks.append(
            {
                "body": await request.body(),
                "signature": request.headers.get("x-wahabot-signature", ""),
                "event_id": request.headers.get("x-wahabot-event-id", ""),
                "timestamp": request.headers.get("x-wahabot-timestamp", ""),
            }
        )
        return {"ok": True}

    return application


def serve() -> tuple[
    FakeRest,
    tuple[uvicorn.Server, threading.Thread, int],
    tuple[uvicorn.Server, threading.Thread, int],
]:
    """Start the two services a redirect test needs."""
    rest = FakeRest(base="", other="")
    other_port = {"port": 0}
    other_server, other_thread, other = run(app(rest, other_port))
    main_server, main_thread, port = run(app(rest, {"port": other}))
    rest.base = f"http://127.0.0.1:{port}"
    rest.other = f"http://127.0.0.1:{other}"
    return rest, (main_server, main_thread, port), (other_server, other_thread, other)
