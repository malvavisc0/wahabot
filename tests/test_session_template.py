"""The starter config written by ``wahabot sessions init``."""

from __future__ import annotations

import json
from pathlib import Path

from wahabot.core.access import SessionConfig, parse_session_config
from wahabot.core.cli import session_template


def test_template_lists_every_session_field() -> None:
    template = session_template()
    assert set(template) == set(SessionConfig.model_fields)


def test_template_parses_and_starts_with_groups_off() -> None:
    config = parse_session_config(
        json.dumps(session_template()), Path("data/sessions/default.json")
    )
    assert config.group_participation == "never"
    assert config.bot_name is None
    assert config.bot_mention_regex is None
