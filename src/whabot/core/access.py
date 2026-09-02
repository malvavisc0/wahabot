"""Per-session config (access lists + system prompt) loaded from a JSON file."""

import json
from pathlib import Path

from loguru import logger
from pydantic import BaseModel


class SessionConfig(BaseModel):
    """Per-session settings read from `<data>/sessions/<session>.json`.

    An empty whitelist means answer everybody; a blacklist entry is
    always ignored, regardless of the whitelist.

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
    system_prompt: str
    bot_name: str | None = None
    bot_mention_regex: str | None = None
    group_participation: str = "mentioned"


def load_session_config(path: Path) -> SessionConfig:
    """Read the session config file; fail fast when it is unusable.

    The file must exist and define a ``system_prompt`` — a bot without
    instructions must not start.
    """
    if not path.exists():
        raise FileNotFoundError(f"Session config not found: {path}")
    config = SessionConfig.model_validate(json.loads(path.read_text()))
    if not config.system_prompt.strip():
        raise ValueError(f"Session config {path} has an empty system_prompt")
    summary = (
        f"Loaded session config from {path}: {len(config.whitelist)} whitelisted, "
        f"{len(config.blacklist)} blacklisted, "
        f"group_participation={config.group_participation}"
    )
    logger.info(summary)
    return config
