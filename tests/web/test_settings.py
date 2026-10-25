"""Settings: independent saves, validation, normalization and hot reload."""

import json

from fastapi.testclient import TestClient
from tests.web.conftest import csrf_of

from wahabot.web.services.settings import draft_mention, normalize_number


def read_config(bot) -> dict:
    """The current session JSON."""
    return json.loads(bot.settings.access_config.read_text())


def save_assistant(client: TestClient, token: str, name: str = "Bridge Club Desk"):
    """Post the Assistant section."""
    return client.post(
        "/app/settings/assistant",
        data={
            "bot_name": name,
            "goal": "Answer opening-hours questions.",
            "instructions": "Be brief.",
            "operator_name": "Sam",
            "csrf_token": token,
        },
    )


def test_page_shows_four_sections(auth_client: TestClient) -> None:
    """Each section has its own form and purpose."""
    response = auth_client.get("/app/settings")
    assert response.status_code == 200
    for section in ("Assistant", "Who it answers", "Hours", "WhatsApp"):
        assert section in response.text
    assert "next message your assistant handles" in response.text


def test_assistant_save_is_independent(auth_client: TestClient, bot) -> None:
    """Assistant save writes only its fields and stores the mention pattern."""
    token = csrf_of(auth_client.get("/app/settings").text)
    before = read_config(bot)
    response = save_assistant(auth_client, token)
    assert response.status_code == 200
    assert "Saved" in response.text
    assert "Bridge Club Desk" in response.text
    config = read_config(bot)
    assert config["bot_name"] == "Bridge Club Desk"
    assert config["system_prompt"] == "Be brief."
    assert config["bot_mention_regex"]
    assert config["bot_mention_regex"] not in response.text
    assert config["whitelist"] == before["whitelist"]
    assert config.get("hours", {}) == before.get("hours", {})


def test_assistant_rejects_empty_instructions(auth_client: TestClient, bot) -> None:
    """Empty instructions are a validation error, not a saved config."""
    before = read_config(bot)
    token = csrf_of(auth_client.get("/app/settings").text)
    response = auth_client.post(
        "/app/settings/assistant",
        data={
            "bot_name": "Bridge Club",
            "goal": "",
            "instructions": "",
            "operator_name": "",
            "csrf_token": token,
        },
    )
    assert response.status_code == 400
    assert "Not saved" in response.text
    assert read_config(bot) == before


def test_who_normalizes_and_resolves(auth_client: TestClient, bot) -> None:
    """International formats become resolved @c.us ids."""
    token = csrf_of(auth_client.get("/app/settings").text)
    response = auth_client.post(
        "/app/settings/who",
        data={
            "whitelist": "+1 646 555 0177\n00 49 1555 000002",
            "blacklist": "0049 1555 000003",
            "csrf_token": token,
        },
    )
    assert response.status_code == 200
    assert "Saved" in response.text
    config = read_config(bot)
    assert config["whitelist"] == ["16465550177@c.us", "491555000002@c.us"]
    assert config["blacklist"] == ["491555000003@c.us"]


def test_assistant_knowledge_only_toggle(auth_client: TestClient, bot) -> None:
    """The answers-only switch persists without changing other sections."""
    token = csrf_of(auth_client.get("/app/settings").text)
    response = auth_client.post(
        "/app/settings/assistant",
        data={
            "bot_name": "Bridge Club",
            "goal": "",
            "instructions": "Stay brief.",
            "operator_name": "",
            "knowledge_only": "on",
            "csrf_token": token,
        },
    )
    assert response.status_code == 200
    assert read_config(bot)["knowledge_only"] is True


def test_who_saves_group_behaviour(auth_client: TestClient, bot) -> None:
    """Group behaviour is validated and saved with the number lists."""
    token = csrf_of(auth_client.get("/app/settings").text)
    response = auth_client.post(
        "/app/settings/who",
        data={
            "whitelist": "",
            "blacklist": "",
            "group_participation": "never",
            "csrf_token": token,
        },
    )
    assert response.status_code == 200
    assert read_config(bot)["group_participation"] == "never"


def test_who_rejects_invalid_number_without_writing(auth_client: TestClient, bot) -> None:
    """A too-short number is reported and the prior list is untouched."""
    token = csrf_of(auth_client.get("/app/settings").text)
    auth_client.post(
        "/app/settings/who",
        data={"whitelist": "491555000002", "blacklist": "", "csrf_token": token},
    )
    before = read_config(bot)
    response = auth_client.post(
        "/app/settings/who",
        data={"whitelist": "123", "blacklist": "", "csrf_token": token},
    )
    assert response.status_code == 400
    assert "Not saved" in response.text
    assert read_config(bot) == before


def test_hours_save_and_preview(auth_client: TestClient, bot) -> None:
    """Hours, timezone and reply window persist and the preview fills slots."""
    token = csrf_of(auth_client.get("/app/settings").text)
    response = auth_client.post(
        "/app/settings/hours",
        data={
            "timezone": "Europe/Berlin",
            "mon_open": "09:00",
            "mon_close": "17:00",
            "tue_closed": "on",
            "away_message": "We are away; next opening {next_opening}.",
            "reply_window_days": "7",
            "night_number": "00 49 1555 000004",
            "csrf_token": token,
        },
    )
    assert response.status_code == 200
    assert "Saved" in response.text
    assert "next opening" in response.text.lower()
    config = read_config(bot)
    assert config["timezone"] == "Europe/Berlin"
    assert config["hours"]["mon"] == {"open": "09:00", "close": "17:00", "closed": False}
    assert config["hours"]["tue"]["closed"] is True
    assert config["reply_window_days"] == 7
    assert config["night_number"] == "491555000004"


def test_hours_reject_invalid_window(auth_client: TestClient, bot) -> None:
    """A reply window outside 1-30 is refused and not written."""
    before = read_config(bot)
    token = csrf_of(auth_client.get("/app/settings").text)
    response = auth_client.post(
        "/app/settings/hours",
        data={
            "timezone": "UTC",
            "away_message": "Away",
            "reply_window_days": "99",
            "csrf_token": token,
        },
    )
    assert response.status_code == 400
    assert "Not saved" in response.text
    assert read_config(bot) == before


def test_whatsapp_reconnect(auth_client: TestClient, bot) -> None:
    """Reconnect starts the session and shows the status."""
    bot.waha.session_status = "STOPPED"
    token = csrf_of(auth_client.get("/app/settings").text)
    response = auth_client.post("/app/settings/whatsapp", data={"csrf_token": token})
    assert response.status_code == 200
    assert bot.waha.started_sessions == ["default"]
    assert "WORKING" in response.text


def test_section_save_does_not_clear_another(auth_client: TestClient, bot) -> None:
    """Assistant and Who are written independently."""
    token = csrf_of(auth_client.get("/app/settings").text)
    save_assistant(auth_client, token, "Bridge Club Desk")
    auth_client.post(
        "/app/settings/who",
        data={"whitelist": "491555000002", "blacklist": "", "csrf_token": token},
    )
    config = read_config(bot)
    assert config["bot_name"] == "Bridge Club Desk"
    assert config["whitelist"] == ["491555000002@c.us"]


def test_settings_write_requires_csrf(auth_client: TestClient) -> None:
    """A settings POST without CSRF is refused."""
    response = auth_client.post(
        "/app/settings/assistant",
        data={"bot_name": "X", "instructions": "Y"},
    )
    assert response.status_code == 403


def test_mention_and_number_helpers() -> None:
    """Draft patterns match the name and normalize drops international prefixes."""
    pattern = draft_mention("Kai")
    assert pattern
    assert normalize_number("+49 1555 000002") == "491555000002"
    assert normalize_number("00 49 1555 000002") == "491555000002"
