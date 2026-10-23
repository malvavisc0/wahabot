import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from wahabot.core.access import SessionConfigReloader
from wahabot.handlers.config_mixin import GATE_BRIEF_CHARS, ConfigMixin


class BriefHandler(ConfigMixin):
    def __init__(self, reloader: SessionConfigReloader) -> None:
        self.config_reloader = reloader
        self.settings = SimpleNamespace(timezone="UTC")


def brief_reloader(tmp_path: Path, **overrides: Any) -> SessionConfigReloader:
    """A file-backed reloader around one config, hot-reloadable per test."""
    config = {"system_prompt": "p", **overrides}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return SessionConfigReloader(path)


def test_gate_brief_names_assistant_operator_mode_and_goal(tmp_path: Path) -> None:
    handler = BriefHandler(
        brief_reloader(
            tmp_path,
            group_participation="judicious",
            bot_name="Kai",
            operator_name="Sam",
            goal="help with laundry bookings",
        )
    )

    brief = handler.render_gate_brief()

    assert "Assistant name: Kai." in brief
    assert "Timezone: UTC." in brief
    assert "Human operator: Sam." in brief
    assert "judge each message" in brief
    assert "personal life" in brief
    assert "Goal: help with laundry bookings" in brief


def test_gate_brief_omits_what_the_config_leaves_unset(tmp_path: Path) -> None:
    handler = BriefHandler(brief_reloader(tmp_path))

    brief = handler.render_gate_brief()

    assert "the assistant" in brief
    assert "Human operator" not in brief
    assert "Goal:" not in brief


def test_gate_brief_is_bounded_and_hot_reloadable(tmp_path: Path) -> None:
    reloader = brief_reloader(tmp_path, goal="plans " * 2000)
    handler = BriefHandler(reloader)

    assert len(handler.render_gate_brief()) <= GATE_BRIEF_CHARS

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"system_prompt": "p", "operator_name": "Sam"}))

    assert "Human operator: Sam." in handler.render_gate_brief()
