"""Shared fixtures for the skills runtime tests."""

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from wahabot.core.db import Database
from wahabot.core.skills.secrets import SecretStore
from wahabot.core.skills.store import SkillStore

SENDER = "491555000001@c.us"
OTHER = "491555000002@c.us"
TEAM_GROUP = "1234567890-1234567890@g.us"
SECRET_VALUE = "sk-fictional-smoke-key"
SIGNING_SECRET = "fictional-signing-secret"
TEST_KEY = base64.urlsafe_b64encode(b"wahabot-test-32-byte-secret-key!").decode()


@pytest.fixture()
def skill_db(tmp_path: Path) -> Path:
    """A migrated database file for one skills test."""
    database = Database(tmp_path / "wahabot.db")
    database.migrate()
    return database.path


@pytest.fixture()
def skill_store(skill_db: Path) -> SkillStore:
    """The skill rows for one test."""
    return SkillStore(skill_db)


@pytest.fixture()
def secret_store(skill_db: Path) -> SecretStore:
    """A Fernet store with the test key and one fictional secret."""
    store = SecretStore(skill_db, TEST_KEY)
    store.put("SMOKE_KEY", SECRET_VALUE)
    return store


def write_skill(
    root: Path,
    name: str,
    base_url: str,
    *,
    kind: str = "http",
    auth: str = "{type: bearer, secret: SMOKE_KEY}",
    customer_chats: str = "read",
    writes_need_operator: bool = True,
    test_check: str = "{method: GET, path: /health}",
    events: dict[str, bool] | None = None,
    event_path: str = "/hook",
    sign_events: bool = False,
    signing_secret: str = "",
    sql: str = "",
    docs: str = "Records API: GET /records?phone=...",
) -> Path:
    """Write one skill folder and return it."""
    folder = root / "source" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(f"# {name}\n\n{docs}\n", encoding="utf-8")
    lines = [
        f"name: {name}",
        f"kind: {kind}",
        f'base_url: "{base_url}"',
        f"auth: {auth}",
        f"customer_chats: {customer_chats}",
        f"writes_need_operator: {str(writes_need_operator).lower()}",
        f"test: {test_check}",
    ]
    if sql:
        lines.append(f"sql: {sql}")
    if events is not None:
        flags = ", ".join(f"{key}: {str(value).lower()}" for key, value in events.items())
        lines.append(f"events: {{{flags}}}")
        lines.append(f"event_path: {event_path}")
    if sign_events:
        lines.append("sign_events: true")
    if signing_secret:
        lines.append(f"signing_secret: {signing_secret}")
    (folder / "skill.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder


def write_skill_docs(folder: Path, docs: str) -> None:
    """Replace one skill's docs."""
    (folder / "SKILL.md").write_text(f"# docs\n\n{docs}\n", encoding="utf-8")


def json_body(response: str) -> dict[str, Any]:
    """Parse a tool envelope."""
    return json.loads(response)
