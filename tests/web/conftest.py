"""Fixtures for the operator web tests: real app, bot settings, fake WAHA."""

import re
import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.conftest import Bot, webhook_app

from wahabot.core.tickets.models import TicketUpdate
from wahabot.web import mount_routes
from wahabot.web.auth import hash_passphrase

PASSPHRASE = "bridge-club-open-sesame"
SESSION_KEY = "test-session-key-32-bytes-long!!!"


def csrf_of(html: str) -> str:
    """The first CSRF token rendered into a page."""
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    return match.group(1) if match else ""


def sign_in(client: TestClient, passphrase: str = PASSPHRASE) -> Any:
    """Sign in with one fresh login page and return the POST response."""
    token = csrf_of(client.get("/login").text)
    return client.post(
        "/login",
        data={"passphrase": passphrase, "csrf_token": token},
        follow_redirects=False,
    )


@pytest.fixture()
def web_client(bot: Bot) -> Iterator[TestClient]:
    """A TestClient over the real app with this test's settings and WAHA."""
    bot.settings.web_passphrase_hash = hash_passphrase(PASSPHRASE)
    bot.settings.web_session_key = SESSION_KEY
    bot.settings.web_login_attempts = 3
    bot.settings.web_login_window_s = 60
    app = webhook_app()
    mount_routes(app)
    app.state.web_waha_client = bot.waha
    app.state.web_limiter = None
    if hasattr(app.state, "web_translate"):
        delattr(app.state, "web_translate")
    client = TestClient(app, base_url="https://testserver")
    try:
        yield client
    finally:
        client.close()
        app.state.web_limiter = None
        if hasattr(app.state, "web_waha_client"):
            delattr(app.state, "web_waha_client")
        if hasattr(app.state, "web_translate"):
            delattr(app.state, "web_translate")


@pytest.fixture()
def auth_client(web_client: TestClient) -> TestClient:
    """A client that already signed in."""
    response = sign_in(web_client)
    assert response.status_code == 303
    return web_client


@pytest.fixture()
def store(bot: Bot) -> Any:
    """The per-test ticket store used to seed rows."""
    return bot.handlers.ticket_store


def seed_ticket(
    store: Any,
    *,
    phone: str = "491555000001@c.us",
    display: str = "Bridge Club",
    language: str = "es",
    summary: str = "Opening times",
    urgent: bool = False,
    state: str = "open",
) -> Any:
    """Create one contact and ticket with a customer message and a reply."""
    contact = store.upsert_contact(
        phone,
        phone=phone,
        display_name=display,
        language=language,
        last_inbound_at=int(time.time()),
    )
    ticket = store.create_ticket(contact.id, phone, summary=summary, urgent=urgent)
    store.add_event(ticket.id, "customer", "customer_message", "hola, abren el lunes?")
    store.add_event(ticket.id, "assistant", "reply", "si, de 9 a 17")
    if state != "open":
        store.update_ticket(ticket.id, TicketUpdate(state=state))
    return store.ticket_by_number(ticket.number)
