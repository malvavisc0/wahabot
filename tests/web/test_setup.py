"""Setup: session creation, QR, pairing, restart and first-exchange completion."""

from fastapi.testclient import TestClient
from tests.harness.constants import ME_JID
from tests.web.conftest import csrf_of, seed_ticket

from wahabot.web.routes.setup import QR_MAX_POLLS


def test_setup_creates_session_and_shows_qr(auth_client: TestClient, store, bot) -> None:
    """Opening Setup creates the WAHA session with the bot webhook and a QR."""
    bot.waha.session_status = "NOT_CREATED"
    page = auth_client.get("/setup")
    assert page.status_code == 200
    assert len(bot.waha.created_sessions) == 1
    created = bot.waha.created_sessions[0]
    assert created["webhook_url"].endswith("/api/webhook/default")
    assert created["hmac_key"] == bot.settings.webhook_hmac_key
    assert "message" in created["events"]
    assert bot.waha.qr_calls == ["default"]
    assert "data:image/png;base64," in page.text


def test_setup_pairing_code(auth_client: TestClient, bot) -> None:
    """A pairing code is requested for the normalized support number."""
    token = csrf_of(auth_client.get("/setup").text)
    response = auth_client.post(
        "/setup/pairing",
        data={"phone_number": "+49 1555 000000", "csrf_token": token},
    )
    assert response.status_code == 200
    assert "BRIDGE-42" in response.text
    assert bot.waha.pairing_requests[-1] == ("default", "491555000000")


def test_setup_restart_records_the_call(auth_client: TestClient, bot) -> None:
    """Restart uses WAHA's restart endpoint, not a browser reload."""
    token = csrf_of(auth_client.get("/setup").text)
    response = auth_client.post(
        "/setup/restart", data={"csrf_token": token}, follow_redirects=False
    )
    assert response.status_code == 303
    assert bot.waha.restarted_sessions == ["default"]


def test_self_chat_does_not_complete_setup(auth_client: TestClient, store, bot) -> None:
    """A conversation with the support phone is not a customer test."""
    seed_ticket(store, phone=ME_JID, display="Support")
    bot.waha.session_status = "WORKING"
    response = auth_client.get("/setup", follow_redirects=False)
    assert response.status_code == 200
    assert "not a customer conversation" in response.text
    assert not store.setup_completed(bot.settings.session)


def test_completion_after_answered_customer_message(
    auth_client: TestClient, store, bot
) -> None:
    """A WORKING session plus one answered customer message finishes Setup."""
    seed_ticket(store)
    bot.waha.session_status = "WORKING"
    response = auth_client.get("/setup", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/app/history"
    assert store.setup_completed(bot.settings.session)
    again = auth_client.get("/setup", follow_redirects=False)
    assert again.status_code == 303


def test_setup_status_json_and_media_auth(auth_client: TestClient, store, bot) -> None:
    """The poll target returns state; it needs the cookie like media."""
    seed_ticket(store)
    bot.waha.session_status = "WORKING"
    payload = auth_client.get("/setup/status").json()
    assert payload["status"] == "WORKING"
    assert payload["completed"] is True
    assert payload["exchange"]["question"].startswith("hola")
    auth_client.cookies.clear()
    assert auth_client.get("/setup/status").status_code == 401


def test_qr_poll_budget_restarts_the_session(auth_client: TestClient, bot) -> None:
    """Six expiry windows of polling restart the session in place."""
    bot.waha.session_status = "SCAN_QR_CODE"
    auth_client.app.state.web_qr_polls = QR_MAX_POLLS - 1
    payload = auth_client.get("/setup/status").json()
    assert payload["restarted"] is True
    assert bot.waha.restarted_sessions == ["default"]


def test_setup_write_requires_csrf(auth_client: TestClient) -> None:
    """No CSRF token means no pairing request."""
    assert (
        auth_client.post(
            "/setup/pairing", data={"phone_number": "491555000000"}
        ).status_code
        == 403
    )
