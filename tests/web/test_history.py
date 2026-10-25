"""History list, ticket thread, translation cache and media auth."""

import json
from pathlib import Path

from fastapi.testclient import TestClient
from tests.harness.constants import SMOKE_PNG
from tests.web.conftest import csrf_of, seed_ticket

from wahabot.core.tickets.models import TicketUpdate
from wahabot.web.auth import SESSION_COOKIE


def test_list_groups_and_counts(auth_client: TestClient, store) -> None:
    """Sections put Needs you first and never hide Open."""
    seed_ticket(store, summary="Opening times")
    urgent = seed_ticket(
        store,
        phone="491555000002@c.us",
        display="Riverside Rowing",
        summary="Late order",
        urgent=True,
    )
    store.update_ticket(urgent.id, TicketUpdate(state="needs_you"))
    response = auth_client.get("/app/history")
    assert response.status_code == 200
    assert "2 tickets shown" in response.text.lower()
    assert response.text.find("Late order") < response.text.find("Opening times")
    assert "Open" in response.text
    assert "Needs you" in response.text
    assert "Waiting for the customer" in response.text


def test_search_and_filters(auth_client: TestClient, store) -> None:
    """Search narrows rows and Show applies its state filter."""
    seed_ticket(store, summary="Opening times")
    other = seed_ticket(
        store,
        phone="491555000002@c.us",
        display="Riverside Rowing",
        summary="Late order",
    )
    store.update_ticket(other.id, TicketUpdate(state="needs_you"))
    search = auth_client.get("/app/history", params={"q": "Riverside"})
    assert "Riverside Rowing" in search.text
    assert "Opening times" not in search.text
    filtered = auth_client.get("/app/history", params={"show": "needs_you"})
    assert "Late order" in filtered.text
    assert "Opening times" not in filtered.text
    zero = auth_client.get("/app/history", params={"q": "nothing-matches"})
    assert "0 tickets shown" in zero.text.lower()
    assert "No tickets match" in zero.text


def test_ticket_thread_and_internal_label(auth_client: TestClient, store) -> None:
    """The thread is chronological, distinguishes audiences and stays read-only."""
    ticket = seed_ticket(store)
    store.add_event(ticket.id, "operator", "operator_reply", "internal note")
    response = auth_client.get(
        f"/app/history/{ticket.number}", params={"back": "/app/history?show=needs_you"}
    )
    assert response.status_code == 200
    assert response.text.find("hola") < response.text.find("si, de 9")
    assert "Internal / not sent to customer" in response.text
    assert "Sent to customer" in response.text
    assert "Reply to the Ticket" in response.text
    assert "in WhatsApp" in response.text
    assert "/app/history?show=needs_you" in response.text


def test_back_parameter_cannot_leave_the_app(auth_client: TestClient, store) -> None:
    """An external or malformed back target falls back to History."""
    ticket = seed_ticket(store)
    response = auth_client.get(
        f"/app/history/{ticket.number}", params={"back": "https://evil.invalid/x"}
    )
    assert "evil.invalid" not in response.text
    assert 'href="/app/history"' in response.text


def test_translation_is_cached_and_toggleable(
    auth_client: TestClient, store, bot
) -> None:
    """English is translated once, cached by source hash, and toggled."""
    ticket = seed_ticket(store, language="es")
    store.add_attachment(
        ticket.id, "msg-voice", "voice", derived_text="hola voice transcript"
    )
    store.extend_event_message_ids(
        store.events(ticket.id)[0].id, ["msg-voice"]
    )
    calls: list[str] = []

    def translate(text: str, target: str) -> str:
        calls.append(target)
        return f"EN: {text}"

    auth_client.app.state.web_translate = translate
    first = auth_client.get(f"/app/history/{ticket.number}")
    assert "EN: hola" in first.text
    assert "EN: hola voice transcript" in first.text
    assert calls
    auth_client.app.state.web_translate = lambda text, target: (_ for _ in ()).throw(
        AssertionError("cache miss")
    )
    second = auth_client.get(f"/app/history/{ticket.number}")
    assert "EN: hola" in second.text
    originals = auth_client.get(
        f"/app/history/{ticket.number}", params={"lang": "originals"}
    )
    assert "hola, abren" in originals.text
    assert "hola voice transcript" in originals.text
    assert "Show English translation" in originals.text


def test_media_requires_cookie_and_streams(
    auth_client: TestClient, store, bot, tmp_path: Path
) -> None:
    """Media is 401 without a cookie, then streams the stored type."""
    ticket = seed_ticket(store)
    path = bot.settings.data_dir / "media" / "smoke.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(SMOKE_PNG)
    attachment = store.add_attachment(
        ticket.id, "msg-1", "image", mime="image/png", size=len(SMOKE_PNG), path=str(path)
    )
    store.extend_event_message_ids(store.events(ticket.id)[0].id, ["msg-1"])
    unauthenticated = TestClient(auth_client.app, base_url="https://testserver")
    try:
        response = unauthenticated.get(f"/app/media/{attachment.id}")
        assert response.status_code == 401
        assert response.content == b""
    finally:
        unauthenticated.close()
    response = auth_client.get(f"/app/media/{attachment.id}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == SMOKE_PNG
    assert "inline" in response.headers["content-disposition"]
    head = auth_client.head(f"/app/media/{attachment.id}")
    assert head.status_code == 200
    ranged = auth_client.get(
        f"/app/media/{attachment.id}", headers={"Range": "bytes=0-3"}
    )
    assert ranged.status_code == 206
    assert ranged.content == SMOKE_PNG[:4]
    assert auth_client.get("/app/media/9999").status_code == 404
    traversal = auth_client.get("/app/media/../../etc/passwd")
    assert traversal.status_code == 404


def test_media_outside_data_dir_is_404(
    auth_client: TestClient, store, bot, tmp_path: Path
) -> None:
    """A stored path outside the data directory never streams."""
    ticket = seed_ticket(store)
    outside = tmp_path / "outside.png"
    outside.write_bytes(SMOKE_PNG)
    attachment = store.add_attachment(
        ticket.id,
        "msg-2",
        "image",
        mime="image/png",
        size=len(SMOKE_PNG),
        path=str(outside),
    )
    response = auth_client.get(f"/app/media/{attachment.id}")
    assert response.status_code == 404
    assert str(outside) not in response.text


def test_every_write_is_journaled_as_web(auth_client: TestClient, bot) -> None:
    """A web write leaves one principal=web audit row and no request body."""
    token = csrf_of(auth_client.get("/app/teach").text)
    auth_client.post(
        "/app/teach/add",
        data={"question": "Q", "answer": "A", "csrf_token": token},
        follow_redirects=True,
    )
    rows = [
        json.loads(line)
        for path in (bot.settings.data_dir / "audit").rglob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    web_rows = [row for row in rows if row.get("principal") == "web"]
    assert web_rows
    assert all("passphrase" not in json.dumps(row).lower() for row in web_rows)
    assert any(row.get("path") == "/app/teach/add" for row in web_rows)
    assert SESSION_COOKIE not in json.dumps(rows)
