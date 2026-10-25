"""Smoke test: the v1.2.0 phase 5 web back office acceptance.

The repo suite proves the behaviour; this script is the phase's live
checklist. It uses a temporary data directory, a local fake WAHA and a
local fake REST service, fictional identities only, and never prints a
passphrase, secret or chat id.

Run:  uv run python scripts/smoke_phase05_web.py
Exit: 0 = smoke passed, 1 = failed.
"""

import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API_FAILURE = "phase 5 web modules are not implemented"

PASSPHRASE = "bridge-club-open-sesame"
SESSION_KEY = "smoke-session-key-32-bytes-long!!"
SOURCE_URL = "http://127.0.0.1:9/docs"

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

    @app.post("/api/sessions/{session}/start", status_code=201)
    def start_session(session: str) -> dict[str, str]:
        state["created"] = True
        state["status"] = "SCAN_QR_CODE"
        return {"name": session, "status": "SCAN_QR_CODE"}

    @app.post("/api/sessions/{session}/restart", status_code=201)
    def restart_session(session: str) -> dict[str, str]:
        state["restarts"] += 1
        state["status"] = "SCAN_QR_CODE"
        return {"name": session, "status": "SCAN_QR_CODE"}

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

    app.state.smoke_waha = state
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
    from wahabot.web import mount_routes

    import wahabot.webhook as webhook_module

    mount_routes(webhook_module.app)
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
    store.add_event(ticket.id, "assistant", "reply", "si, de 9 a 17")
    return store, ticket


def check_hash_command() -> None:
    """The hash-passphrase command emits a verifiable scrypt line."""
    from typer.testing import CliRunner
    from wahabot.web.auth import verify_passphrase

    from wahabot.cli import app

    result = CliRunner().invoke(app, ["hash-passphrase"], input=PASSPHRASE + "\n")
    line = result.stdout.strip()
    check("hash-passphrase prints an scrypt line", line.startswith("scrypt$"))
    check("the printed hash verifies", verify_passphrase(PASSPHRASE, line))
    check("a wrong passphrase does not verify", not verify_passphrase("wrong", line))


def check_fail_closed(root: Path) -> None:
    """Without hash and key the app refuses access; health stays up."""
    settings = settings_for(root / "unconfigured", configured=False)
    write_session_config(settings)
    client = client_for(settings)
    try:
        health = client.get("/health")
        login_page = client.get("/login")
        protected = client.get("/app/history")
        post = client.post("/login", data={"passphrase": PASSPHRASE, "csrf_token": "x"})
        check("health still answers", health.status_code == 200)
        check("login is refused when unconfigured", login_page.status_code == 503)
        check(
            "protected pages are refused when unconfigured", protected.status_code == 503
        )
        check("login POST is refused when unconfigured", post.status_code == 503)
        check(
            "the failure page carries no passphrase",
            PASSPHRASE not in login_page.text,
        )
    finally:
        close(client)


def check_login(root: Path) -> None:
    """Correct/wrong passphrase, CSRF, origin, rate limit and logout."""
    settings = settings_for(root / "login")
    write_session_config(settings)
    make_db(settings)
    client = client_for(settings)
    try:
        page = client.get("/login")
        token = csrf_of(page.text)
        check("login page renders", page.status_code == 200 and bool(token))
        wrong = client.post("/login", data={"passphrase": "not-it", "csrf_token": token})
        check("a wrong passphrase is refused", wrong.status_code != 303)
        check("the wrong passphrase is not echoed", "not-it" not in wrong.text)
        good = client.post(
            "/login",
            data={"passphrase": PASSPHRASE, "csrf_token": token},
            follow_redirects=False,
        )
        check("a correct passphrase signs in", good.status_code == 303)
        check("sign-in redirects to Setup", good.headers.get("location") == "/setup")
    finally:
        close(client)

    parallel = client_for(settings)
    try:
        page = parallel.get("/login")
        token = csrf_of(page.text)
        no_token = parallel.post(
            "/login", data={"passphrase": PASSPHRASE}, follow_redirects=False
        )
        check("a login POST without CSRF is refused", no_token.status_code == 403)
        cross = parallel.post(
            "/login",
            data={"passphrase": PASSPHRASE, "csrf_token": token},
            headers={"Origin": "https://evil.invalid"},
            follow_redirects=False,
        )
        check("a cross-site login POST is refused", cross.status_code == 403)
        fetch_cross = parallel.post(
            "/login",
            data={"passphrase": PASSPHRASE, "csrf_token": token},
            headers={"Sec-Fetch-Site": "cross-site"},
            follow_redirects=False,
        )
        check("Sec-Fetch-Site cross-site is refused", fetch_cross.status_code == 403)
        replay = parallel.post(
            "/login",
            data={"passphrase": PASSPHRASE, "csrf_token": token + "x"},
            follow_redirects=False,
        )
        check("a mismatched CSRF token is refused", replay.status_code == 403)
    finally:
        close(parallel)

    limited = client_for(settings)
    limited.app_state = None
    try:
        page = limited.get("/login")
        token = csrf_of(page.text)
        for _ in range(4):
            limited.post("/login", data={"passphrase": "wrong", "csrf_token": token})
        blocked = limited.post(
            "/login", data={"passphrase": PASSPHRASE, "csrf_token": token}
        )
        check("repeated failures are rate-limited", blocked.status_code == 429)
        check(
            "the retry delay is stated",
            "try again" in blocked.text.lower() and "retry-after" in blocked.headers,
        )
    finally:
        close(limited)


def check_cookies(root: Path) -> None:
    """A tampered or expired cookie does not grant access."""
    from wahabot.web.auth import SESSION_COOKIE, sign_payload

    settings = settings_for(root / "cookies")
    write_session_config(settings)
    make_db(settings)
    client = client_for(settings)
    try:
        login(client)
        fresh = client.get("/setup", follow_redirects=False)
        check("a signed-in cookie reaches Setup", fresh.status_code == 200)
        client.cookies.set(SESSION_COOKIE, "tampered.value", domain="testserver")
        tampered = client.get("/app/history", follow_redirects=False)
        check("a tampered cookie redirects to login", tampered.status_code == 303)
        expired = sign_payload(
            SESSION_KEY,
            {"v": 1, "auth": True, "exp": int(time.time()) - 60, "csrf": "x"},
        )
        client.cookies.set(SESSION_COOKIE, expired, domain="testserver")
        stale = client.get("/app/history", follow_redirects=False)
        check("an expired cookie redirects to login", stale.status_code == 303)
        check(
            "media without a cookie is 401",
            client.get("/app/media/1", cookies={SESSION_COOKIE: ""}).status_code
            in {401, 403},
        )
    finally:
        close(client)


def check_logout(root: Path) -> None:
    """POST logs out; GET does not mutate the session."""
    settings = settings_for(root / "logout")
    write_session_config(settings)
    make_db(settings)
    client = client_for(settings)
    try:
        login(client)
        page = client.get("/app/teach")
        token = csrf_of(page.text)
        get_logout = client.get("/logout", follow_redirects=False)
        still_in = client.get("/app/history", follow_redirects=False)
        check("GET /logout does not mutate", get_logout.status_code in {200, 405})
        check("GET /logout leaves the session valid", still_in.status_code == 200)
        posted = client.post(
            "/logout", data={"csrf_token": token}, follow_redirects=False
        )
        check("POST /logout redirects", posted.status_code == 303)
        after = client.get("/app/history", follow_redirects=False)
        check("POST /logout clears access", after.status_code == 303)
    finally:
        close(client)


def check_setup(root: Path, waha_port: int) -> None:
    """Session creation, QR, pairing, restart, self-chat and completion."""
    import httpx

    settings = settings_for(
        root / "setup",
        waha_url=f"http://127.0.0.1:{waha_port}",
        web_trusted_proxy=False,
    )
    write_session_config(settings)
    store = make_db(settings)
    client = client_for(settings)
    try:
        login(client)
        page = client.get("/setup")
        check("Setup creates the WAHA session", page.status_code == 200)
        check("Setup shows the QR code", "smoke-qr" in page.text or "<img" in page.text)
        pairing = client.post(
            "/setup/pairing", data={"phone_number": "491555000000", "csrf_token": ""}
        )
        token = csrf_of(page.text)
        pairing = client.post(
            "/setup/pairing",
            data={"phone_number": "491555000000", "csrf_token": token},
        )
        check("pairing code is requested", pairing.status_code == 200)
        check("pairing feedback is shown", "BRIDGE-42" in pairing.text)

        with httpx.Client(base_url=f"http://127.0.0.1:{waha_port}") as probe:
            probe.post("/api/sessions/default/restart")
            probe.get("/api/sessions/default")
        restarted = client.get("/setup")
        check("a failed session restarts in place", restarted.status_code == 200)

        own = store.upsert_contact(
            "491555000000@c.us", phone="491555000000@c.us", display_name="Support"
        )
        self_ticket = store.create_ticket(own.id, own.chat_id, summary="self chat")
        store.add_event(self_ticket.id, "customer", "customer_message", "hi")
        store.add_event(self_ticket.id, "assistant", "reply", "hello")
        waiting = client.get("/setup", follow_redirects=False)
        check("a self-chat message is not a customer test", waiting.status_code == 200)

        _store, _ = seed_ticket(settings, number_hint="1")
        with httpx.Client(base_url=f"http://127.0.0.1:{waha_port}") as probe:
            probe.post("/api/sessions/default/start")
        completed = client.get("/setup", follow_redirects=False)
        check(
            "Setup completes after an answered customer message",
            completed.status_code == 303,
        )
        check(
            "completion redirects to History",
            completed.headers.get("location") == "/app/history",
        )
        check("setup_completed is persisted", store.setup_completed("default"))
    finally:
        close(client)


def check_history(root: Path) -> None:
    """List, search, filters, ticket thread, toggle and media auth."""
    settings = settings_for(root / "history")
    write_session_config(settings)
    store, first = seed_ticket(settings, number_hint="1")
    second_phone = "491555000002@c.us"
    second_contact = store.upsert_contact(
        second_phone, phone=second_phone, display_name="Riverside Rowing", language="en"
    )
    second = store.create_ticket(
        second_contact.id, second_phone, summary="Late order", urgent=True
    )
    store.add_event(second.id, "customer", "customer_message", "where is it?")
    from wahabot.core.tickets.models import TicketUpdate

    store.add_event(second.id, "operator", "operator_reply", "internal note")
    store.update_ticket(second.id, TicketUpdate(state="needs_you"))

    client = client_for(settings)
    try:
        login(client)
        listing = client.get("/app/history")
        check(
            "History lists tickets",
            "Bridge Club" in listing.text and "Riverside" in listing.text,
        )
        check(
            "Needs you is first",
            listing.text.find("Late order") < listing.text.find("Bridge Club"),
        )
        check("row and section counts are shown", "tickets shown" in listing.text.lower())
        search = client.get("/app/history", params={"q": "Riverside"})
        check(
            "search narrows the list",
            "Riverside" in search.text and "opening times" not in search.text,
        )
        filtered = client.get("/app/history", params={"show": "needs_you"})
        check(
            "the Show filter applies",
            "Late order" in filtered.text and "Bridge Club" not in filtered.text,
        )
        zero = client.get("/app/history", params={"q": "nothing-matches"})
        check("zero results are stated", "0 tickets shown" in zero.text.lower())

        detail = client.get(f"/app/history/{first.number}")
        check("ticket detail renders", detail.status_code == 200)
        check(
            "the thread is chronological",
            detail.text.find("hola") < detail.text.find("si, de 9"),
        )
        check("internal is labelled", "not sent to customer" in detail.text.lower())
        check(
            "the reply stays in WhatsApp",
            "reply to" in detail.text.lower() and "whatsapp" in detail.text.lower(),
        )
        original = client.get(
            f"/app/history/{first.number}", params={"lang": "originals"}
        )
        check(
            "the language toggle is offered",
            "Show originals" in detail.text or "Show English" in detail.text,
        )
        check("originals render", "hola" in original.text)

        image = settings.data_dir / "media"
        image.mkdir(parents=True, exist_ok=True)
        path = image / "smoke.png"
        from tests.harness import SMOKE_PNG

        path.write_bytes(SMOKE_PNG)
        attachment = store.add_attachment(
            first.id,
            "msg-smoke-1",
            "image",
            mime="image/png",
            size=len(SMOKE_PNG),
            path=str(path),
        )
        store.extend_event_message_ids(store.events(first.id)[0].id, ["msg-smoke-1"])
        media = client.get(f"/app/media/{attachment.id}")
        check(
            "media needs the cookie",
            client.get(
                f"/app/media/{attachment.id}", cookies={"wahabot_web": ""}
            ).status_code
            == 401,
        )
        check(
            "media streams with the stored type",
            media.status_code == 200 and media.headers.get("content-type") == "image/png",
        )
        missing = client.get("/app/media/9999")
        check("a missing attachment is 404", missing.status_code == 404)
        check("no path leaks from media", str(settings.data_dir) not in missing.text)
        traversal = client.get("/app/media/../../etc/passwd")
        check("media traversal is refused", traversal.status_code == 404)
    finally:
        close(client)


def check_teach(root: Path) -> None:
    """Answer CRUD, on/off, delete and latest undo."""
    settings = settings_for(root / "teach")
    write_session_config(settings)
    make_db(settings)
    client = client_for(settings)
    try:
        login(client)
        page = client.get("/app/teach")
        token = csrf_of(page.text)
        add = client.post(
            "/app/teach/add",
            data={
                "question": "When is the bridge club open?",
                "answer": "Mondays, 9 to 17.",
                "csrf_token": token,
            },
            follow_redirects=True,
        )
        check("an answer is added", "When is the bridge club open?" in add.text)
        answer_id = re.search(r'data-answer-id="(\d+)"', add.text)
        check("the answer has an id", answer_id is not None)
        if answer_id is None:
            return
        edited = client.post(
            f"/app/teach/{answer_id.group(1)}/edit",
            data={
                "question": "When is the bridge club open?",
                "answer": "Mondays and Wednesdays, 9 to 17.",
                "csrf_token": token,
            },
            follow_redirects=True,
        )
        check("an answer is edited", "Mondays and Wednesdays" in edited.text)
        toggled = client.post(
            f"/app/teach/{answer_id.group(1)}/toggle",
            data={"csrf_token": token},
            follow_redirects=True,
        )
        check("an answer is switched off", "Off" in toggled.text)
        deleted = client.post(
            f"/app/teach/{answer_id.group(1)}/delete",
            data={"csrf_token": token},
            follow_redirects=True,
        )
        check("an answer is deleted", "When is the bridge club open?" not in deleted.text)
        undo = client.post(
            "/app/teach/undo", data={"csrf_token": token}, follow_redirects=True
        )
        check(
            "the latest deletion is undone", "When is the bridge club open?" in undo.text
        )
    finally:
        close(client)


def check_skills(root: Path, rest_port: int) -> None:
    """Draft, review, test gating, stale and write-only key."""
    settings = settings_for(root / "skills")
    write_session_config(settings)
    make_db(settings)
    client = client_for(settings)
    try:
        login(client)
        listing = client.get("/app/skills")
        check(
            "the empty state is purposeful",
            "works without skills" in listing.text.lower(),
        )
        token = csrf_of(listing.text)
        created = client.post(
            "/app/skills/new",
            data={
                "source_url": f"http://127.0.0.1:{rest_port}/docs",
                "csrf_token": token,
            },
            follow_redirects=True,
        )
        check(
            "one URL creates a draft",
            created.status_code == 200 and "Review" in created.text,
        )
        path = re.search(r"/app/skills/([a-z0-9-]+)", str(created.url))
        check("the draft has a page", path is not None)
        if path is None:
            return
        name = path.group(1)
        premature = client.post(
            f"/app/skills/{name}/enable",
            data={"csrf_token": token},
            follow_redirects=True,
        )
        check("enable is refused before a test", "Passed" not in premature.text)
        saved = client.post(
            f"/app/skills/{name}/save",
            data={
                "base_url": f"http://127.0.0.1:{rest_port}",
                "test_path": "/health",
                "customer_chats": "read",
                "writes_need_operator": "on",
                "instructions": "Look up records.",
                "api_key": "sk-fictional-bridge-key",
                "csrf_token": token,
                "stage": "review",
            },
            follow_redirects=True,
        )
        check("the key is saved write-only", "sk-fictional-bridge-key" not in saved.text)
        check("a saved key shows set", "set" in saved.text.lower())
        tested = client.post(
            f"/app/skills/{name}/test", data={"csrf_token": token}, follow_redirects=True
        )
        check("the prepared test passes", "Passed" in tested.text)
        enabled = client.post(
            f"/app/skills/{name}/enable",
            data={"csrf_token": token},
            follow_redirects=True,
        )
        check("enable follows a current pass", "Enabled" in enabled.text)
        stale = client.post(
            f"/app/skills/{name}/save",
            data={
                "base_url": f"http://127.0.0.1:{rest_port}",
                "test_path": "/health",
                "customer_chats": "read",
                "instructions": "Changed instructions.",
                "csrf_token": token,
                "stage": "review",
            },
            follow_redirects=True,
        )
        check("a changed configuration is stale", "Stale" in stale.text)
        refused = client.post(
            f"/app/skills/{name}/enable",
            data={"csrf_token": token},
            follow_redirects=True,
        )
        check("a stale test refuses enable", "Enabled" not in refused.text)
    finally:
        close(client)


def check_settings(root: Path, waha_port: int) -> None:
    """Four independent saves, validation and normalization."""
    settings = settings_for(root / "settings", waha_url=f"http://127.0.0.1:{waha_port}")
    write_session_config(settings)
    make_db(settings)
    client = client_for(settings)
    try:
        login(client)
        page = client.get("/app/settings")
        token = csrf_of(page.text)
        for section in ("Assistant", "Who it answers", "Hours", "WhatsApp"):
            check(f"Settings shows {section}", section in page.text)
        saved = client.post(
            "/app/settings/assistant",
            data={
                "bot_name": "Bridge Club Desk",
                "goal": "Answer opening-hours questions.",
                "instructions": "Be brief.",
                "operator_name": "Sam",
                "csrf_token": token,
            },
            follow_redirects=True,
        )
        check(
            "Assistant saves", "Saved" in saved.text and "Bridge Club Desk" in saved.text
        )
        config = json.loads(settings.access_config.read_text())
        check(
            "the mention pattern is stored invisibly",
            bool(config.get("bot_mention_regex")),
        )
        check(
            "the mention pattern is not rendered",
            config.get("bot_mention_regex") not in saved.text,
        )
        who = client.post(
            "/app/settings/who",
            data={
                "whitelist": "+1 646 555 0177\n00 49 1555 000002",
                "blacklist": "0049 1555 000003",
                "team_groups": "",
                "csrf_token": token,
            },
            follow_redirects=True,
        )
        check("Who it answers saves", "Saved" in who.text)
        config = json.loads(settings.access_config.read_text())
        check(
            "numbers normalize and resolve",
            config.get("whitelist") == ["16465550177@c.us", "491555000002@c.us"],
        )
        check(
            "the blacklist resolves too", config.get("blacklist") == ["491555000003@c.us"]
        )
        invalid = client.post(
            "/app/settings/who",
            data={"whitelist": "123", "blacklist": "", "csrf_token": token},
            follow_redirects=True,
        )
        check("an invalid number is refused", "Not saved" in invalid.text)
        check(
            "the invalid number is not stored",
            json.loads(settings.access_config.read_text()).get("whitelist")
            == ["16465550177@c.us", "491555000002@c.us"],
        )
        hours = client.post(
            "/app/settings/hours",
            data={
                "timezone": "Europe/Berlin",
                "mon_open": "09:00",
                "mon_close": "17:00",
                "mon_closed": "",
                "away_message": "We are away; next opening {next_opening}.",
                "reply_window_days": "7",
                "night_number": "",
                "csrf_token": token,
            },
            follow_redirects=True,
        )
        check("Hours saves", "Saved" in hours.text)
        config = json.loads(settings.access_config.read_text())
        check("the timezone is stored", config.get("timezone") == "Europe/Berlin")
        check(
            "the hours are stored",
            config.get("hours", {}).get("mon", {}).get("open") == "09:00",
        )
        check("the hour preview uses placeholders", "next opening" in hours.text.lower())
        whatsapp = client.post(
            "/app/settings/whatsapp", data={"csrf_token": token}, follow_redirects=True
        )
        check(
            "WhatsApp reports the connection",
            "WORKING" in whatsapp.text or "Reconnect" in whatsapp.text,
        )
        check(
            "a section save does not clear another",
            json.loads(settings.access_config.read_text()).get("bot_name")
            == "Bridge Club Desk",
        )
    finally:
        close(client)


def check_journal(root: Path) -> None:
    """Every write is journaled as web; no secret reaches a log or page."""
    settings = settings_for(root / "journal")
    write_session_config(settings)
    make_db(settings)
    client = client_for(settings)
    try:
        login(client)
        page = client.get("/app/teach")
        token = csrf_of(page.text)
        client.post(
            "/app/teach/add",
            data={"question": "Q", "answer": "A", "csrf_token": token},
            follow_redirects=True,
        )
    finally:
        close(client)
    rows = [
        json.loads(line)
        for path in (settings.data_dir / "audit").rglob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    web_rows = [row for row in rows if row.get("principal") == "web"]
    check("writes are journaled with principal web", bool(web_rows))
    text = json.dumps(rows)
    check("no passphrase reaches the journal", PASSPHRASE not in text)
    check("no session key reaches the journal", SESSION_KEY not in text)
    check("no skill secret reaches the journal", "sk-fictional-bridge-key" not in text)


def main() -> int:
    """Run the phase checks and report."""
    print("wahabot phase 5 web smoke")
    try:
        import wahabot.web  # noqa: F401
    except Exception as exc:
        print(f"[FAIL] {API_FAILURE} — {type(exc).__name__}: {exc}")
        return 1
    from tests.harness import serving

    waha_app = fake_waha()
    rest_app = fake_rest()
    with tempfile.TemporaryDirectory(prefix="wahabot-smoke-web-") as raw:
        root = Path(raw)
        with serving(waha_app) as waha_port, serving(rest_app) as rest_port:
            guarded("hash command", check_hash_command)
            guarded("fail closed", lambda: check_fail_closed(root))
            guarded("login", lambda: check_login(root))
            guarded("cookies", lambda: check_cookies(root))
            guarded("logout", lambda: check_logout(root))
            guarded("setup", lambda: check_setup(root, waha_port))
            guarded("history", lambda: check_history(root))
            guarded("teach", lambda: check_teach(root))
            guarded("skills", lambda: check_skills(root, rest_port))
            guarded("settings", lambda: check_settings(root, waha_port))
            guarded("journal", lambda: check_journal(root))
    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("\nAll phase 5 web smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
