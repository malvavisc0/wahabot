"""Per-session config (access lists + system prompt) loaded from a JSON file."""

import json
import threading
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
    config = parse_session_config(path.read_text(), path)
    log_config_load(path, config)
    return config


def parse_session_config(raw: str, path: Path) -> SessionConfig:
    """Parse and validate session config text; raise when unusable."""
    config = SessionConfig.model_validate(json.loads(raw))
    if not config.system_prompt.strip():
        raise ValueError(f"Session config {path} has an empty system_prompt")
    return config


def log_config_load(path: Path, config: SessionConfig) -> None:
    """Log a successful config load in one summary line."""
    logger.info(
        "Loaded session config from {path}: {wl} whitelisted, {bl} blacklisted, "
        "group_participation={mode}",
        path=path,
        wl=len(config.whitelist),
        bl=len(config.blacklist),
        mode=config.group_participation,
    )


class SessionConfigReloader:
    """Reloads the session config when its file changes, per event.

    ``current_config`` stats the file on every call — microseconds — and
    only re-reads/parses when mtime or size changed, so config edits
    (whitelist, system prompt, participation mode) apply without a
    restart. A file that becomes unreadable or invalid mid-edit keeps
    the last good config and logs the problem; the next event retries.
    Safe for concurrent handler use via a lock (stats are cheap).
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._config: SessionConfig | None = None
        self._fingerprint: tuple[float, int] | None = None

    def current_config(self) -> SessionConfig:
        """The newest valid config, reloading only when the file changed."""
        with self._lock:
            try:
                stat = self.path.stat()
            except OSError:
                logger.warning(
                    "Session config {path} is gone; keeping last good", path=self.path
                )
                return self._cached()
            fingerprint = (stat.st_mtime, stat.st_size)
            if fingerprint != self._fingerprint:
                try:
                    config = parse_session_config(self.path.read_text(), self.path)
                except Exception as exc:
                    logger.error(
                        "Session config {path} failed to reload: {exc}; "
                        "keeping last good",
                        path=self.path,
                        exc=exc,
                    )
                    self._fingerprint = fingerprint
                    return self._cached()
                self._config = config
                self._fingerprint = fingerprint
                log_config_load(self.path, config)
            return self._cached()

    def _cached(self) -> SessionConfig:
        """The last good config; raises when none was ever loaded."""
        if self._config is None:
            raise FileNotFoundError(f"Session config not found: {self.path}")
        return self._config
