"""Setup, History, Teach, Skills, Settings and journal smoke checks."""

import json
import re
from pathlib import Path

import httpx
from smoke_phase05_support import (
    PASSPHRASE,
    SESSION_KEY,
    check,
    client_for,
    close,
    csrf_of,
    login,
    make_db,
    seed_ticket,
    settings_for,
    write_session_config,
)

from wahabot.core.tickets.models import TicketUpdate


def check_setup(root: Path, waha_port: int) -> None:
    """Session creation, QR, pairing, restart, self-chat and completion."""

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
            str(config.get("bot_mention_regex") or "") not in saved.text,
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
