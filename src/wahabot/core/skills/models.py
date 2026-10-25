"""Domain models for the skill runtime."""

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import SplitResult, unquote, urlsplit

from pydantic import BaseModel, Field, model_validator

from wahabot.core.skills.errors import SkillError
from wahabot.core.tickets.store import TicketStore

SAFE_NAME = re.compile(r"^[a-z0-9-]+$")
READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
MAX_SKILL_CALLS_PER_TURN = 5
SKILL_DOC_TOKEN_BUDGET = 3000


class AuthConfig(BaseModel):
    """How one skill's secret is attached to a request."""

    type: Literal["none", "bearer", "header"] = "none"
    secret: str = ""
    header: str = "X-API-Key"


class TestCheck(BaseModel):
    """One prepared, non-mutating check from ``skill.yaml``."""

    method: str = "GET"
    path: str = ""

    @property
    def configured(self) -> bool:
        """Whether the file names a usable safe check."""
        return bool(self.path.strip())


class SqlConfig(BaseModel):
    """The read-only database behind a SQL skill."""

    engine: Literal["sqlite", "postgres"] = "sqlite"
    sqlite_path: str = ""
    dsn_secret: str = ""


class SkillConfig(BaseModel):
    """The validated contents of one skill folder."""

    name: str
    kind: Literal["http", "sql"] = "http"
    base_url: str = ""
    base_urls: list[str] = Field(default_factory=list)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    customer_chats: Literal["none", "read", "read_write"] = "none"
    writes_need_operator: bool = True
    test: TestCheck | None = None
    events: dict[str, bool] = Field(default_factory=dict)
    event_path: str = "/events"
    sign_events: bool = False
    signing_secret: str = ""
    sql: SqlConfig | None = None
    instructions: str = ""
    skill_md: str = ""

    @model_validator(mode="after")
    def validate_skill(self) -> SkillConfig:
        """Enforce the folder contract shared by import and enable."""
        if not SAFE_NAME.fullmatch(self.name):
            raise ValueError("skill name must match [a-z0-9-]+")
        validate_http(self)
        validate_auth(self)
        validate_events(self)
        validate_sql(self)
        return self

    @property
    def effective_base_urls(self) -> list[str]:
        """Every configured base URL, with the singular shorthand included."""
        values = list(self.base_urls)
        if self.base_url and self.base_url not in values:
            values.insert(0, self.base_url)
        return [value.rstrip("/") for value in values if value.strip()]

    @property
    def secret_refs(self) -> list[str]:
        """Every secret name the folder actually uses for its kind."""
        refs: list[str] = []
        if self.kind == "http":
            if self.auth.type != "none":
                refs.append(self.auth.secret)
            if self.sign_events and self.signing_secret:
                refs.append(self.signing_secret)
        elif self.sql is not None and self.sql.dsn_secret:
            refs.append(self.sql.dsn_secret)
        return list(dict.fromkeys(ref for ref in refs if ref))


def validate_http(config: SkillConfig) -> None:
    """One base URL contract for HTTP skills; SQL skills need their block."""
    if config.kind != "http":
        if config.sql is None:
            raise ValueError("a SQL skill needs a sql block")
        return
    urls = config.effective_base_urls
    if not urls:
        raise ValueError("an HTTP skill needs a base_url")
    for url in urls:
        check_base_url(url)
    if config.test is not None and config.test.path:
        check_relative_path(config.test.path)


def validate_auth(config: SkillConfig) -> None:
    """A non-none auth type must name the secret it needs."""
    if config.auth.type != "none" and not config.auth.secret.strip():
        raise ValueError("the auth type needs a secret reference")


def validate_events(config: SkillConfig) -> None:
    """Only HTTP skills may subscribe, with the four known event names."""
    unknown = set(config.events) - {"opened", "needs_you", "done", "closed"}
    if unknown:
        raise ValueError("unknown event(s): " + ", ".join(sorted(unknown)))
    if config.kind != "http":
        if any(config.events.values()) or config.sign_events:
            raise ValueError("only HTTP skills can send ticket events")
        return
    if config.event_path:
        if not config.event_path.startswith("/"):
            raise ValueError("event_path must start with /")
        check_relative_path(config.event_path)
    if config.sign_events and not config.signing_secret.strip():
        raise ValueError("signed events need a signing_secret reference")


def check_relative_path(value: str) -> None:
    """Refuse an absolute, traversal, backslash or control path."""
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or parsed.fragment or "#" in value:
        raise ValueError("skill paths must be relative to the base URL")
    decoded = repeat_unquote(value)
    if ".." in decoded.split("/") or value.startswith("//"):
        raise ValueError("skill paths must stay under the base URL")
    check_base_path(decoded)


def repeat_unquote(value: str) -> str:
    """Decode repeatedly so double-encoded escapes are visible."""
    decoded = unquote(value)
    for _ in range(4):
        next_value = unquote(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    return decoded


def validate_sql(config: SkillConfig) -> None:
    """A SQL block needs the path or DSN secret its engine reads."""
    if config.sql is None:
        return
    if config.sql.engine == "sqlite" and not config.sql.sqlite_path.strip():
        raise ValueError("the SQLite skill needs sqlite_path")
    if config.sql.engine == "postgres" and not config.sql.dsn_secret.strip():
        raise ValueError("the Postgres skill needs dsn_secret")


def check_base_url(value: str) -> None:
    """Validate one base URL: HTTP(S), a host, no credentials, no fragment."""
    parsed = urlsplit(value)
    check_base_parts(parsed)
    require_valid_port(parsed)


def check_base_parts(parsed: SplitResult) -> None:
    """Validate a base URL's scheme, host and path."""
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("base URL must be http(s) and name a host")
    if parsed.username or parsed.password:
        raise ValueError("base URL must not carry credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base URL must not carry a query or fragment")
    check_base_host(parsed.hostname)
    check_base_path(parsed.path)


def check_base_host(hostname: str) -> None:
    """Refuse whitespace or control characters in a configured host."""
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in hostname):
        raise ValueError("base URL host must not contain whitespace or controls")


def check_base_path(path: str) -> None:
    """Refuse backslashes or control characters in a configured base path."""
    decoded = unquote(path)
    if "\\" in decoded:
        raise ValueError("base URL path must not contain a backslash")
    if any(ord(char) < 32 or ord(char) == 127 for char in decoded):
        raise ValueError("base URL path must not contain control characters")


def require_valid_port(parsed: SplitResult) -> None:
    """Refuse a malformed or zero port on a configured base URL."""
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("base URL has an invalid port") from exc
    if port == 0:
        raise ValueError("base URL has an invalid port")


@dataclass(slots=True)
class Skill:
    """One ``skills`` row."""

    name: str
    lifecycle: str
    enabled: bool = False
    base_urls: list[str] = field(default_factory=list)
    auth_type: str = "none"
    secret_ref: str = ""
    customer_chats: str = "none"
    writes_need_operator: bool = True
    source_url: str = ""
    instructions: str = ""
    skill_md: str = ""
    draft_stage: str = "connect"
    draft_inputs: dict[str, object] = field(default_factory=dict)
    test_config_hash: str = ""
    test_state: str = "untested"
    test_reason: str = ""
    last_tested_at: int | None = None
    failure_count: int = 0
    created_by: str = "system"
    created_at: int = 0
    updated_at: int = 0


@dataclass(slots=True)
class SkillCall:
    """One ``skill_calls`` row."""

    id: int
    skill: str
    ticket_id: int | None
    chat_id: str
    started_by: str
    method: str
    path: str
    status: int | None
    duration_ms: int
    outcome: str
    error: str
    request_json: str
    created_at: int


@dataclass(slots=True)
class SkillRequest:
    """One staged write awaiting the operator's yes."""

    id: int
    skill: str
    ticket_id: int | None
    chat_id: str
    request_json: str
    summary: str
    state: str
    operator_message_id: str
    created_at: int
    resolved_at: int | None


@dataclass(slots=True)
class SkillRun:
    """The run-scoped values one skill call needs.

    ``RunTarget`` satisfies the same shape; this dataclass keeps the
    skills package independent of the agent's target module.
    """

    session: str = ""
    chat_id: str = ""
    sender_id: str = ""
    sender_phone: str = ""
    surface: str = ""
    armed: bool = False
    ticket_id: int = 0
    ticket_store: TicketStore | None = None
    skill_calls: int = 0
    skill_call_keys: list[str] = field(default_factory=list)
    skill_docs: dict[str, str] = field(default_factory=dict)


class SkillRuntime(Protocol):
    """The run-scoped state a skill call reads and advances."""

    session: str
    chat_id: str
    sender_id: str
    sender_phone: str
    surface: str
    armed: bool
    ticket_id: int
    ticket_store: TicketStore | None
    skill_calls: int
    skill_call_keys: list[str]
    skill_docs: dict[str, str]


@dataclass(slots=True)
class PreparedRequest:
    """The exact request a staged write will execute."""

    method: str
    path: str
    query: dict[str, str]
    body: object
    url: str


@dataclass(slots=True)
class HttpResult:
    """One HTTP response after the guardrails and caps."""

    status: int
    text: str
    truncated: bool


@dataclass(slots=True)
class SkillTestResult:
    """The outcome of one prepared safe check."""

    passed: bool
    reason: str


def skill_from_row(row: sqlite3.Row) -> Skill:
    """Map one ``skills`` row to its model."""
    mapping = dict(row)
    return Skill(
        name=str(mapping["name"]),
        lifecycle=str(mapping["lifecycle"]),
        enabled=bool(mapping["enabled"]),
        base_urls=json_list(str(mapping["base_urls_json"])),
        auth_type=str(mapping["auth_type"]),
        secret_ref=str(mapping["secret_ref"]),
        customer_chats=str(mapping["customer_chats"]),
        writes_need_operator=bool(mapping["writes_need_operator"]),
        source_url=str(mapping["source_url"]),
        instructions=str(mapping["instructions"]),
        skill_md=str(mapping["skill_md"]),
        draft_stage=str(mapping["draft_stage"]),
        draft_inputs=json_object(str(mapping["draft_inputs_json"])),
        test_config_hash=str(mapping["test_config_hash"]),
        test_state=str(mapping["test_state"]),
        test_reason=str(mapping["test_reason"]),
        last_tested_at=(
            int(mapping["last_tested_at"])
            if mapping["last_tested_at"] is not None
            else None
        ),
        failure_count=int(mapping["failure_count"]),
        created_by=str(mapping["created_by"]),
        created_at=int(mapping["created_at"]),
        updated_at=int(mapping["updated_at"]),
    )


def json_list(raw: str) -> list[str]:
    """Decode a JSON list of strings, failing soft to empty."""
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def json_object(raw: str) -> dict[str, object]:
    """Decode a JSON object, failing soft to empty."""
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def skill_folder(skills_dir: Path, name: str) -> Path:
    """The folder of one validated skill name."""
    if not SAFE_NAME.fullmatch(name):
        raise SkillError(f"invalid skill name: {name!r}")
    return skills_dir / name
