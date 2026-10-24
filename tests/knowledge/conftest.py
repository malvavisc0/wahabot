"""Fixtures for the knowledge/document/schedule test families."""

import pytest
from tests.harness import smoke_settings

from wahabot.core.db import Database
from wahabot.core.tickets.store import TicketStore
from wahabot.settings import Settings


@pytest.fixture()
def store(tmp_path) -> TicketStore:
    """A migrated ticket store in an isolated data directory."""
    db = Database(tmp_path / "wahabot.db")
    db.migrate()
    return TicketStore(db.path)


@pytest.fixture()
def settings(tmp_path) -> Settings:
    """Settings shaped like production, pinned to an isolated data dir."""
    return smoke_settings(tmp_path, "http://127.0.0.1:1/v1")
