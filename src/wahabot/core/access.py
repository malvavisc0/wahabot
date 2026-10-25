"""Per-session config (access lists + system prompt) loaded from a JSON file."""

import json
import threading
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from wahabot.core.schedule.models import DAY_KEYS, DayHours
from wahabot.core.tokens import text_tokens

CONFIG_MEMORY_TOKEN_LIMIT = 1024


class SessionConfig(BaseModel):
    """Per-session settings read from `<data>/sessions/<session>.json`.

    An empty whitelist means answer everybody; a blacklist entry is
    always ignored, regardless of the whitelist.

    ``bot_name`` is the bot's display name used to detect when a group
    message is aimed at it (e.g. "Kai"). ``operator_name`` is the
    human who owns the WhatsApp account the bot runs on; exposed to
    the prompt via the ``{{operator_name}}`` placeholder so the bot
    can defer questions meant for that human (empty renders "the
    operator"). ``goal`` is an optional
    statement of what the bot should aim for; when set it is prepended
    to the rendered system prompt. ``memory`` is an optional
    free-text list of durable facts the operator maintains; when set it
    is rendered as a ``# Memory`` section after the goal, before the
    system prompt. ``group_participation``
    controls whether/how the bot joins group conversations:

    - ``never`` — never reply in groups.
    - ``mentioned`` (default) — reply only when the bot is mentioned by
      name/regex, or replies to a bot message.
    - ``judicious`` — also run the agent on unmentioned group messages,
      which may choose to reply based on its judgment.
    """

    whitelist: set[str] = set()
    blacklist: set[str] = set()
    goal: str = ""
    memory: str = ""
    system_prompt: str
    bot_name: str | None = None
    bot_mention_regex: str | None = None
    group_participation: str = "mentioned"
    operator_name: str = ""
    #: How many days after the customer last wrote a reply may still be
    #: delivered; older work becomes operator-required instead.
    reply_window_days: int = Field(default=7, ge=1, le=30)
    #: How many days after a ticket is done it can be reopened by a new
    #: customer message; after that it closes and the next message opens
    #: a new ticket linked to it.
    close_after_days: int = Field(default=5, ge=1, le=365)
    #: How many hours a human-operator message pauses the assistant in
    #: that chat.
    pause_after_human_hours: int = Field(default=2, ge=0, le=168)
    #: Operator-facing alert language; empty falls back to English.
    operator_language: str = ""
    #: Session timezone; empty falls back to WAHABOT_TIMEZONE.
    timezone: str = ""
    #: Opening hours per weekday (mon..sun); an empty mapping is always open.
    hours: dict[str, DayHours] = Field(default_factory=dict)
    #: Customer-facing message sent outside hours; {ticket} and
    #: {next_opening} are filled by code.
    away_message: str = ""
    #: The after-hours contact; it is only ever messaged after it wrote first.
    night_number: str = ""
    #: Group chat ids that run as the internal team surface.
    team_groups: list[str] = Field(default_factory=list)
    #: Answers-only support mode; opt-in, never the personal default.
    knowledge_only: bool = False
    #: Answer a voice note with a synthesized voice note when TTS is configured.
    voice_replies: bool = False

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        """Accept an empty fallback or a zoneinfo-resolvable name."""
        name = value.strip()
        if not name:
            return ""
        try:
            ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
            raise ValueError(f"unknown timezone {name!r}") from exc
        return name

    @field_validator("hours")
    @classmethod
    def validate_hours(cls, value: dict[str, DayHours]) -> dict[str, DayHours]:
        """Accept only the seven weekday keys, in a stable order."""
        unknown = set(value) - set(DAY_KEYS)
        if unknown:
            raise ValueError(f"unknown hours day(s): {', '.join(sorted(unknown))}")
        return {day: value[day] for day in DAY_KEYS if day in value}

    @field_validator("team_groups")
    @classmethod
    def validate_team_groups(cls, value: list[str]) -> list[str]:
        """Keep the allowlist as distinct, non-empty chat ids."""
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))


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
    if text_tokens(config.memory) > CONFIG_MEMORY_TOKEN_LIMIT:
        raise ValueError(
            f"Session config {path} memory exceeds {CONFIG_MEMORY_TOKEN_LIMIT} tokens"
        )
    return config


def log_config_load(path: Path, config: SessionConfig) -> None:
    """Log a successful config load in one summary line."""
    logger.info(
        (
            "Loaded session config from {path}: {wl} whitelisted, {bl} blacklisted, "
            "group_participation={mode}"
        ),
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
                return self.cached_config()
            fingerprint = (stat.st_mtime, stat.st_size)
            if fingerprint != self._fingerprint:
                try:
                    config = parse_session_config(self.path.read_text(), self.path)
                except Exception as exc:
                    logger.error(
                        (
                            "Session config {path} failed to reload: {exc}; "
                            "keeping last good"
                        ),
                        path=self.path,
                        exc=exc,
                    )
                    self._fingerprint = fingerprint
                    return self.cached_config()
                self._config = config
                self._fingerprint = fingerprint
                log_config_load(self.path, config)
            return self.cached_config()

    def cached_config(self) -> SessionConfig:
        """The last good config; raises when none was ever loaded."""
        if self._config is None:
            raise FileNotFoundError(f"Session config not found: {self.path}")
        return self._config
