"""Shared fixtures and helpers for the phase 5 web smoke checks."""

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API_FAILURE = "phase 5 web modules are not implemented"

PASSPHRASE = "bridge-club-open-sesame"
SESSION_KEY = "smoke-session-key-32-bytes-long!!"

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    """Record one check; failures accumulate for the summary."""
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}{f' — {detail}' if detail and not ok else ''}")
    if not ok:
        failures.append(f"{label}{f' — {detail}' if detail else ''}")


def guarded(label: str, fn: Any) -> None:
    """Run one check, turning a runtime failure into a FAIL line."""
    try:
        fn()
    except Exception as exc:
        check(label, False, f"{type(exc).__name__}: {exc}")


def fake_waha() -> Any:
    """A local WAHA-shaped app for session, QR and pairing calls."""
    from fastapi import FastAPI, HTTPException

    app = FastAPI()
    state: dict[str, Any] = {"created": False, "status": "SCAN_QR_CODE", "restarts": 0}

    @app.get("/api/sessions/{session}")
    def session_status(session: str) -> dict[str, str]:
        if not state["created"]:
            raise HTTPException(status_code=404)
        return {"name": session, "status": state["status"]}

    @app.post("/api/sessions", status_code=201)
    def create_session(body: dict[str, Any]) -> dict[str, str]:
        state["created"] = True
        state["body"] = body
        return {"name": body.get("name", ""), "status": "SCAN_QR_CODE"}

    @app.get("/api/sessions/{session}/me")
    def me(session: str) -> dict[str, str]:
        return {"id": "491555000000@c.us", "name": "Bridge Club"}

    @app.post("/api/sessions/{session}/start", status_code=201)
    def start_session(session: str) -> dict[str, str]:
        state["created"] = True
        state["status"] = "WORKING"
        return {"name": session, "status": "WORKING"}

    @app.post("/api/sessions/{session}/restart", status_code=201)
    def restart_session(session: str) -> dict[str, str]:
        state["restarts"] += 1
        state["status"] = "SCAN_QR_CODE"
        return {"name": session, "status": "SCAN_QR_CODE"}

    @app.post("/api/{session}/chats/overview")
    def chats_overview(session: str, body: dict[str, Any]) -> list[dict[str, str]]:
        ids = body.get("filter", {}).get("ids", [])
        return [{"id": str(item)} for item in ids]

    @app.get("/api/{session}/auth/qr")
    def qr(session: str, format: str = "image") -> Any:
        if format == "raw":
            return {"value": "smoke-qr-value"}
        from tests.harness import SMOKE_PNG

        return __import__("fastapi").Response(SMOKE_PNG, media_type="image/png")

    @app.post("/api/{session}/auth/request-code", status_code=201)
    def request_code(session: str, body: dict[str, Any]) -> dict[str, str]:
        state["pairing_phone"] = body.get("phoneNumber", "")
        return {"code": "BRIDGE-42"}

    return app


def fake_rest() -> Any:
    """A local REST service for the Add-skill safe check."""
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    return app


def settings_for(data_dir: Path, *, configured: bool = True, **overrides: Any) -> Any:
    """Settings shaped like production, pinned to the temporary paths."""
    from wahabot.settings import Settings

    values: dict[str, Any] = {
        "webhook_hmac_key": "smoke-hmac-key",
        "waha_url": "http://waha.invalid",
        "waha_api_key": "waha-key",
        "llm_api_base": "http://llm.invalid/v1",
        "llm_api_key": "sk-smoke",
        "llm_model": "smoke-model",
        "session": "default",
        "data_dir": data_dir,
        "timezone": "UTC",
        "triage_gate": False,
        "secret_key": "d2FoYWJvdC10ZXN0LTMyLWJ5dGUtc2VjcmV0LWtleSE=",
        "langfuse_public_key": "",
        "langfuse_secret_key": "",
        "langfuse_tracing_environment": "test",
        "_env_file": None,
    }
    if configured:
        from wahabot.web.auth import hash_passphrase

        values.update(
            {
                "web_passphrase_hash": hash_passphrase(PASSPHRASE),
                "web_session_key": SESSION_KEY,
                "web_login_attempts": 3,
                "web_login_window_s": 60,
            }
        )
    values.update(overrides)
    return Settings(**values)


def write_session_config(settings: Any) -> None:
    """A minimal session config, as provisioning would write it."""
    path = settings.access_config
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "whitelist": [],
                "blacklist": [],
                "goal": "",
                "system_prompt": "You are {{bot_name}} texting on WhatsApp.",
                "bot_name": "Bridge Club",
                "operator_name": "Sam",
                "group_participation": "mentioned",
            }
        )
    )


def client_for(settings: Any) -> Any:
    """A TestClient over the real app, pinned to *settings* for the call."""
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    import wahabot.webhook as webhook_module
    from wahabot.web import mount_routes

    mount_routes(webhook_module.app)
    limiter = getattr(webhook_module.app.state, "web_limiter", None)
    if limiter is not None:
        limiter.reset()
    patcher = patch("wahabot.settings.get_settings", return_value=settings)
    patcher.start()
    client = TestClient(webhook_module.app, base_url="https://testserver")
    client.patcher = patcher  # type: ignore[attr-defined]
    return client


def close(client: Any) -> None:
    """Stop the settings patch and close the client."""
    client.close()
    client.patcher.stop()  # type: ignore[attr-defined]


def csrf_of(html: str) -> str:
    """The first CSRF token rendered into a page."""
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    return match.group(1) if match else ""


def login(client: Any) -> Any:
    """Sign in and return the authenticated response."""
    page = client.get("/login")
    token = csrf_of(page.text)
    return client.post(
        "/login",
        data={"passphrase": PASSPHRASE, "csrf_token": token},
        follow_redirects=False,
    )


def make_db(settings: Any) -> Any:
    """A migrated ticket store for the fixture rows."""
    from wahabot.core.db import Database
    from wahabot.core.tickets.store import TicketStore

    database = Database(settings.data_dir / "wahabot.db")
    database.migrate()
    return TicketStore(database.path)


def seed_ticket(settings: Any, *, number_hint: str = "1") -> tuple[Any, Any]:
    """One ticket with a customer message and an assistant reply."""
    store = make_db(settings)
    phone = f"49155500000{number_hint}@c.us"
    contact = store.upsert_contact(
        phone,
        phone=phone,
        display_name="Bridge Club",
        language="es",
        last_inbound_at=int(time.time()),
    )
    ticket = store.create_ticket(
        contact.id, phone, summary="Bridge Club opening times", urgent=number_hint == "2"
    )
    store.add_event(ticket.id, "customer", "customer_message", "hola, abren el lunes?")
    store.add_event(ticket.id, "operator", "operator_reply", "internal note")
    store.add_event(ticket.id, "assistant", "reply", "si, de 9 a 17")
    return store, ticket
