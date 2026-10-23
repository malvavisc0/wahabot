from pathlib import Path

import pytest

from wahabot.core.db import Database


@pytest.fixture()
def database(tmp_path: Path) -> Database:
    return Database(tmp_path / "wahabot.db")
