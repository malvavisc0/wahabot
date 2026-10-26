"""Skills: one-URL draft, review, safe test gating, write-only key and delete."""

import json
import re

from fastapi.testclient import TestClient
from tests.harness.rest import FakeRest
from tests.web.conftest import csrf_of

from wahabot.core.skills.secrets import SecretStore
from wahabot.core.skills.store import SkillStore


def create_draft(client: TestClient, source: str) -> str:
    """Create a draft from one URL and return its name."""
    token = csrf_of(client.get("/app/skills").text)
    response = client.post(
        "/app/skills/new",
        data={"source_url": source, "csrf_token": token},
        follow_redirects=True,
    )
    assert response.status_code == 200
    match = re.search(r"/app/skills/([a-z0-9-]+)", str(response.url))
    assert match is not None
    return match.group(1)


def test_empty_state(auth_client: TestClient) -> None:
    """Skills are optional and the empty page says so."""
    response = auth_client.get("/app/skills")
    assert response.status_code == 200
    assert "works without skills" in response.text.lower()
    assert "Add skill" in response.text


def test_create_draft_opens_review(auth_client: TestClient, fake_rest: FakeRest) -> None:
    """One documentation URL creates an inactive Review draft."""
    name = create_draft(auth_client, fake_rest.base + "/docs")
    page = auth_client.get(f"/app/skills/{name}")
    assert page.status_code == 200
    assert "Review" in page.text
    assert fake_rest.base in page.text
    assert "Enable skill" not in page.text


def test_create_draft_keeps_the_source_url(
    auth_client: TestClient, fake_rest: FakeRest, bot
) -> None:
    """The documentation URL is not replaced by the service destination."""
    name = create_draft(auth_client, fake_rest.base + "/docs")

    skill = SkillStore(bot.database.path).get(name)

    assert skill is not None
    assert skill.source_url == fake_rest.base + "/docs"


def test_save_draft_stays_inactive_and_never_stores_the_key(
    auth_client: TestClient, fake_rest: FakeRest, bot
) -> None:
    """Save draft persists the fields without configuring or writing the key."""
    name = create_draft(auth_client, fake_rest.base + "/docs")
    token = csrf_of(auth_client.get(f"/app/skills/{name}").text)
    response = auth_client.post(
        f"/app/skills/{name}/save",
        data={
            "base_url": fake_rest.base,
            "test_path": "/health",
            "customer_chats": "read",
            "instructions": "Look up records.",
            "api_key": "sk-fictional-draft-key",
            "action": "save",
            "csrf_token": token,
        },
    )

    assert response.status_code == 200
    assert "Saved draft." in response.text
    assert "Delete" in response.text
    store = SkillStore(bot.database.path)
    skill = store.get(name)
    assert skill is not None
    assert skill.lifecycle == "draft"
    assert skill.draft_stage == "review"
    assert not (bot.settings.skills_dir / name).exists()
    secret_ref = f"{name.upper().replace('-', '_')}_KEY"
    secrets = SecretStore(bot.database.path, bot.settings.secret_key)
    assert store.secret_exists(secret_ref) is False
    assert not secrets.ciphertext(secret_ref)


def test_invalid_source_url_is_refused(auth_client: TestClient) -> None:
    """A non-HTTP URL and a credential-bearing URL are refused."""
    token = csrf_of(auth_client.get("/app/skills/new").text)
    for source in (
        "ftp://service.invalid/docs",
        "https://user:pass@service.invalid/docs",
    ):
        response = auth_client.post(
            "/app/skills/new",
            data={"source_url": source, "csrf_token": token},
        )
        assert response.status_code == 400
        assert "http(s)" in response.text or "credentials" in response.text


def test_save_key_is_write_only(
    auth_client: TestClient, fake_rest: FakeRest, bot
) -> None:
    """A saved key appears only as set and never in HTML or audit rows."""
    name = create_draft(auth_client, fake_rest.base + "/docs")
    token = csrf_of(auth_client.get(f"/app/skills/{name}").text)
    response = auth_client.post(
        f"/app/skills/{name}/save",
        data={
            "base_url": fake_rest.base,
            "test_path": "/health",
            "customer_chats": "read",
            "instructions": "Look up records.",
            "api_key": "sk-fictional-bridge-key",
            "csrf_token": token,
        },
    )
    assert response.status_code == 200
    assert "sk-fictional-bridge-key" not in response.text
    assert "set" in response.text.lower()
    assert "Saved" in response.text
    secrets = SecretStore(bot.database.path, bot.settings.secret_key)
    assert (
        secrets.get(f"{name.upper().replace('-', '_')}_KEY") == "sk-fictional-bridge-key"
    )
    rows = [
        json.loads(line)
        for path in (bot.settings.data_dir / "audit").rglob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert "sk-fictional-bridge-key" not in json.dumps(rows)


def test_test_gating_and_stale_configuration(
    auth_client: TestClient, fake_rest: FakeRest
) -> None:
    """Enable needs a current pass; a changed configuration turns it stale."""
    name = create_draft(auth_client, fake_rest.base + "/docs")
    token = csrf_of(auth_client.get(f"/app/skills/{name}").text)
    premature = auth_client.post(
        f"/app/skills/{name}/enable",
        data={"csrf_token": token},
        follow_redirects=True,
    )
    assert "Not enabled" in premature.text
    auth_client.post(
        f"/app/skills/{name}/save",
        data={
            "base_url": fake_rest.base,
            "test_path": "/health",
            "customer_chats": "read",
            "instructions": "Look up records.",
            "csrf_token": token,
        },
    )
    tested = auth_client.post(f"/app/skills/{name}/test", data={"csrf_token": token})
    assert "Passed" in tested.text
    assert "Tested at" in tested.text
    enabled = auth_client.post(f"/app/skills/{name}/enable", data={"csrf_token": token})
    assert "Enabled" in enabled.text
    disabled = auth_client.post(f"/app/skills/{name}/disable", data={"csrf_token": token})
    assert "Paused" in disabled.text
    stale = auth_client.post(
        f"/app/skills/{name}/save",
        data={
            "base_url": fake_rest.base,
            "test_path": "/health",
            "customer_chats": "read",
            "instructions": "Changed instructions.",
            "csrf_token": token,
        },
    )
    assert "Stale" in stale.text
    refused = auth_client.post(f"/app/skills/{name}/enable", data={"csrf_token": token})
    assert "Not enabled" in refused.text


def test_failed_safe_check_is_recorded(
    auth_client: TestClient, fake_rest: FakeRest
) -> None:
    """An unhealthy prepared check fails and never enables."""
    name = create_draft(auth_client, fake_rest.base + "/docs")
    token = csrf_of(auth_client.get(f"/app/skills/{name}").text)
    auth_client.post(
        f"/app/skills/{name}/save",
        data={
            "base_url": fake_rest.base,
            "test_path": "/unhealthy",
            "customer_chats": "read",
            "instructions": "Look up records.",
            "csrf_token": token,
        },
    )
    failed = auth_client.post(f"/app/skills/{name}/test", data={"csrf_token": token})
    assert "Failed" in failed.text
    assert "Tested at" in failed.text
    enabled = auth_client.post(f"/app/skills/{name}/enable", data={"csrf_token": token})
    assert "Not enabled" in enabled.text
    page = auth_client.get("/app/skills")
    assert "Inactive draft" not in page.text
    assert "Paused" in page.text


def test_delete_removes_row_and_folder(
    auth_client: TestClient, fake_rest: FakeRest, bot
) -> None:
    """Delete removes the row and its saved folder, then returns to the list."""
    name = create_draft(auth_client, fake_rest.base + "/docs")
    token = csrf_of(auth_client.get(f"/app/skills/{name}").text)
    auth_client.post(
        f"/app/skills/{name}/save",
        data={
            "base_url": fake_rest.base,
            "test_path": "/health",
            "customer_chats": "read",
            "instructions": "Look up records.",
            "csrf_token": token,
        },
    )
    response = auth_client.post(
        f"/app/skills/{name}/delete",
        data={"csrf_token": token},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert name not in response.text
    assert not (bot.settings.skills_dir / name).exists()


def test_write_requires_csrf(auth_client: TestClient, fake_rest: FakeRest) -> None:
    """No CSRF token means no draft is created."""
    response = auth_client.post(
        "/app/skills/new", data={"source_url": fake_rest.base + "/docs"}
    )
    assert response.status_code == 403


def test_delete_removes_staged_requests(auth_client: TestClient, bot) -> None:
    """Deleting a skill drops its pending approval prompts with it."""
    token = csrf_of(auth_client.get("/app/skills").text)
    created = auth_client.post(
        "/app/skills/new",
        data={"source_url": "https://docs.example.com/api", "csrf_token": token},
        follow_redirects=True,
    )
    name = re.search(r"/app/skills/([a-z0-9-]+)", str(created.url))
    assert name is not None
    skills = SkillStore(bot.settings.data_dir / "wahabot.db")
    skills.create_request(
        skill=name.group(1),
        ticket_id=None,
        chat_id="491555000001@c.us",
        request_json="{}",
        summary="staged write",
    )
    assert skills.latest_pending() is not None

    deleted = auth_client.post(
        f"/app/skills/{name.group(1)}/delete",
        data={"csrf_token": token},
        follow_redirects=False,
    )

    assert deleted.status_code == 303
    assert skills.latest_pending() is None
