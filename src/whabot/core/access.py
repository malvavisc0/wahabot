"""Per-session config (access lists + system prompt) loaded from a JSON file."""

import json
from pathlib import Path

from loguru import logger
from pydantic import BaseModel


class SessionConfig(BaseModel):
    """Per-session settings read from `<data>/sessions/<session>.json`.

    An empty whitelist means answer everybody; a blacklist entry is
    always ignored, regardless of the whitelist. ``system_prompt``, when
    set, overrides the agent's default system prompt.

    ``bot_name`` is the bot's display name used to detect when a group
    message is aimed at it (e.g. "Kai"). ``group_participation``
    controls whether/how the bot joins group conversations:

    - ``never`` — never reply in groups.
    - ``mentioned`` (default) — reply only when the bot is mentioned by
      name/regex, or replies to a bot message.
    - ``judicious`` — also run the agent on unmentioned group messages,
      which may choose to reply based on its judgment.
    """

    whitelist: set[str] = set()
    blacklist: set[str] = set()
    system_prompt: str | None = None
    bot_name: str | None = None
    bot_mention_regex: str | None = None
    group_participation: str = "mentioned"


def load_session_config(path: Path) -> SessionConfig:
    """Read the session config file.

    The file may be absent or empty; both mean "no filtering" and no
    system prompt override.
    """
    if not path.exists():
        logger.info("No session config at {path}; using defaults", path=path)
        return SessionConfig()
    config = SessionConfig.model_validate(json.loads(path.read_text()))
    logger.info(
        "Loaded session config from {path}: {n_white} whitelisted, "
        "{n_black} blacklisted, system_prompt={has_prompt}, "
        "group_participation={participation}",
        path=path,
        n_white=len(config.whitelist),
        n_black=len(config.blacklist),
        has_prompt=bool(config.system_prompt),
        participation=config.group_participation,
    )
    return config
