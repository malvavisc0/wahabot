from pathlib import Path

import pytest
from tests.skills.conftest import SECRET_VALUE

from wahabot.core.skills.secrets import (
    SecretKeyError,
    SecretStore,
    redact_secrets,
)


def test_a_secret_round_trips(secret_store: SecretStore) -> None:
    assert secret_store.get("SMOKE_KEY") == SECRET_VALUE


def test_a_missing_key_refuses_a_write(skill_db: Path) -> None:
    store = SecretStore(skill_db, "")

    with pytest.raises(SecretKeyError, match="WAHABOT_SECRET_KEY"):
        store.put("SMOKE_KEY", SECRET_VALUE)


def test_a_wrong_key_fails_clearly(skill_db: Path, secret_store: SecretStore) -> None:
    secret_store.put("SMOKE_KEY", SECRET_VALUE)
    other = SecretStore(
        skill_db,
        "d2FoYWJvdC13cm9uZy0zMi1ieXRlLXNlY3JldC1rZXk=",
    )

    with pytest.raises(SecretKeyError, match="SECRET_KEY"):
        other.get("SMOKE_KEY")


def test_redaction_removes_every_secret_value() -> None:
    text = "token sk-one here and sk-two there"

    assert redact_secrets(text, ["sk-one", "sk-two"]) == "token *** here and *** there"


def test_secret_values_never_appear_in_the_row(secret_store: SecretStore) -> None:
    with secret_store.connect() as conn:
        row = conn.execute(
            "SELECT encrypted_value FROM secrets WHERE name = 'SMOKE_KEY'"
        ).fetchone()

    assert row is not None
    assert SECRET_VALUE not in str(row[0])
    assert str(row[0]).startswith("gAAAA")
