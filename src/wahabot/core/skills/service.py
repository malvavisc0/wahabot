"""The skill runtime: guarded calls, staging and confirmation."""

import json
import time
from pathlib import Path
from urllib.parse import parse_qsl

from loguru import logger

from wahabot.core.skills.capabilities import (
    active_skills as load_active_skills,
)
from wahabot.core.skills.capabilities import (
    capability_summary as capability_text,
)
from wahabot.core.skills.capabilities import (
    enabled_config,
)
from wahabot.core.skills.capabilities import (
    prompt_section as prompt_text,
)
from wahabot.core.skills.capabilities import (
    read_skill as read_skill_text,
)
from wahabot.core.skills.errors import SkillError
from wahabot.core.skills.execution import (
    CALL_PATH_CAP,
    REQUEST_JSON_CAP,
    note_failure,
    record_call,
    record_confirmed_ticket,
    record_ticket,
)
from wahabot.core.skills.guardrails import (
    check_surface_access,
    foreign_identifier_error,
    is_write_method,
    join_skill_url,
    needs_confirmation,
    substitute_sender,
)
from wahabot.core.skills.http import MAX_REQUEST_BYTES, decode_response, http_call
from wahabot.core.skills.loader import load_skill
from wahabot.core.skills.models import (
    HttpResult,
    PreparedRequest,
    Skill,
    SkillConfig,
    SkillRun,
    SkillRuntime,
    envelope_error,
    envelope_ok,
)
from wahabot.core.skills.secrets import SecretStore
from wahabot.core.skills.sql import execute_query
from wahabot.core.skills.store import SkillStore
from wahabot.core.skills.support import (
    body_bytes,
    body_text,
    claim_attempt,
    render_result,
    request_json,
    secret_error,
    stored_request,
    substitute_body,
    summary_text,
)
from wahabot.core.tickets.store import TicketStore
from wahabot.core.transport import Transport
from wahabot.status import state as status_state


def _request_error(method: str, body: object) -> str | None:
    allowed = {"GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"}
    if method not in allowed:
        return f"unsupported HTTP method: {method}"
    if len(body_bytes(body) or b"") > MAX_REQUEST_BYTES:
        return f"the request body exceeds {MAX_REQUEST_BYTES} bytes"
    return None


class SkillService:
    """Run one skill call, enforcing every code-owned guardrail."""

    def __init__(
        self,
        store: SkillStore,
        secrets: SecretStore,
        skills_dir: Path,
        session: str,
        transport: Transport,
        data_dir: Path | None = None,
        ticket_store: TicketStore | None = None,
    ) -> None:
        self.store = store
        self.secrets = secrets
        self.skills_dir = skills_dir
        self.session = session
        self.transport = transport
        self.data_dir = data_dir
        self.ticket_store = ticket_store

    def active_skills(
        self, run: SkillRuntime | None = None
    ) -> list[tuple[Skill, SkillConfig]]:
        """Every enabled skill the run's surface may use."""
        return load_active_skills(self.store, self.skills_dir, run)

    def prompt_section(self, run: SkillRuntime) -> str:
        """The active capability list and, when they fit, the skill docs."""
        return prompt_text(self.store, self.skills_dir, run)

    def capability_summary(self) -> str:
        """Plain-language active skills for the operator's self-chat."""
        return capability_text(self.store, self.skills_dir)

    def read_skill(self, name: str, run: SkillRuntime) -> str:
        """Read docs on demand when they exceeded the prompt budget."""
        return read_skill_text(self.store, self.skills_dir, name, run)

    def execute_http(
        self,
        name: str,
        method: str,
        path: str,
        query: dict[str, str] | None,
        body: object,
        run: SkillRuntime,
    ) -> str:
        """Run one HTTP skill call, or stage a customer write for the operator."""
        started = time.monotonic()
        method = method.strip().upper()
        config, prepared, error = self._http_attempt(name, method, path, query, body, run)
        if error is not None:
            return error
        assert config is not None and prepared is not None
        if is_write_method(method) and needs_confirmation(config, run):
            return self._stage(name, config, prepared, run, started)
        return self._perform_envelope(name, config, prepared, run, started)

    def _http_attempt(
        self,
        name: str,
        method: str,
        path: str,
        query: dict[str, str] | None,
        body: object,
        run: SkillRuntime,
    ) -> tuple[SkillConfig | None, PreparedRequest | None, str | None]:
        config, error = self._usable_config(name, run, write=is_write_method(method))
        if error is not None:
            return None, None, error
        assert config is not None
        request_error = _request_error(method, body)
        if request_error is not None:
            return None, None, envelope_error(request_error, skill=name)
        key_error = secret_error(self.secrets, config)
        if key_error is not None:
            return None, None, key_error
        try:
            prepared = self._prepare(config, method, path, query, body, run)
        except SkillError as exc:
            return None, None, envelope_error(self._safe(str(exc)), skill=name)
        claim_error = self._claim_request(name, prepared, run)
        if claim_error is not None:
            return None, None, envelope_error(claim_error, skill=name)
        return config, prepared, None

    def _claim_request(
        self, name: str, prepared: PreparedRequest, run: SkillRuntime
    ) -> str | None:
        repeat = (
            f"{name}:{prepared.method}:{prepared.path}:"
            + f"{json.dumps(prepared.query, sort_keys=True)}:{body_text(prepared.body)}"
        )
        refused = claim_attempt(run, repeat)
        if refused is not None:
            return refused
        return foreign_identifier_error(
            " ".join(
                [
                    prepared.path,
                    json.dumps(prepared.query, sort_keys=True),
                    body_text(prepared.body),
                ]
            ),
            run,
        )

    def execute_sql(self, name: str, query: str, run: SkillRuntime) -> str:
        """Run an operator-only read query against a SQL skill."""
        started = time.monotonic()
        config, error = self._sql_config(name, run)
        if error is not None:
            return error
        assert config is not None and config.sql is not None
        refused = claim_attempt(run, f"{name}:SQL:{query}")
        if refused is not None:
            return envelope_error(refused, skill=name)
        try:
            secret = (
                self.secrets.get(config.sql.dsn_secret) if config.sql.dsn_secret else ""
            )
            result = execute_query(config, query, secret)
        except SkillError as exc:
            return self._fail_sql(name, run, str(exc), started)
        self._record(
            name=name,
            run=run,
            method="SQL",
            path=query[:CALL_PATH_CAP],
            status=None,
            duration=int((time.monotonic() - started) * 1000),
            outcome="ok",
            error="",
            request=query[:REQUEST_JSON_CAP],
        )
        self.store.reset_failures(name)
        return envelope_ok(skill=name, **result)

    def _sql_config(
        self, name: str, run: SkillRuntime
    ) -> tuple[SkillConfig | None, str | None]:
        if not run.armed and run.surface not in ("operator", "system"):
            return None, envelope_error(
                "SQL skills are operator-only in this version", skill=name
            )
        skill = self.store.get(name)
        if skill is None or not skill.enabled:
            return None, envelope_error("that skill is not enabled", skill=name)
        try:
            config = load_skill(name, self.skills_dir)
        except SkillError as exc:
            return None, envelope_error(str(exc), skill=name)
        if config.kind != "sql" or config.sql is None:
            return None, envelope_error("that skill is not a SQL skill", skill=name)
        return config, None

    def confirm(self, request_id: int, *, approve: bool) -> str:
        """Execute one staged write exactly once, or decline it."""
        request = self.store.claim_request(request_id)
        if request is None:
            return "That request was already handled."
        if not approve:
            self.store.resolve_request(request_id, "refused")
            return "Request declined."
        try:
            config, prepared = stored_request(self.store, self.skills_dir, request)
            run = SkillRun(
                session=self.session,
                chat_id=request.chat_id,
                surface="operator",
                armed=True,
                ticket_id=request.ticket_id or 0,
                ticket_store=self.ticket_store,
            )
            result = self._perform(config, prepared, run)
        except (SkillError, KeyError, ValueError, TypeError) as exc:
            self.store.resolve_request(request_id, "failed")
            return f"Request failed: {self._safe(str(exc))}"
        self.store.resolve_request(request_id, "done")
        record_confirmed_ticket(self.ticket_store, self.session, request, "ok")
        return f"Done: {self._safe(render_result(result))}"

    def send_event(self, name: str, body: bytes, headers: dict[str, str]) -> HttpResult:
        """Post one event payload from the system, without staging."""
        config = enabled_config(self.store, self.skills_dir, name)
        url = join_skill_url(config, config.event_path or "/events")
        prepared = PreparedRequest(
            method="POST",
            path=config.event_path or "/events",
            query={},
            body=body,
            url=url,
        )
        run = SkillRun(session=self.session, surface="system", armed=True)
        try:
            return self._perform(config, prepared, run, extra_headers=headers)
        except SkillError as exc:
            self._fail_event(name, run, prepared, str(exc))
            raise

    def test_probe(self, name: str, method: str, path: str) -> HttpResult:
        """Run a configured safe check against a disabled draft."""
        skill = self.store.get(name)
        if skill is None:
            raise SkillError("that skill is not configured")
        config = load_skill(name, self.skills_dir)
        prepared = self._prepare(config, method, path, None, None, SkillRun(armed=True))
        run = SkillRun(
            session=self.session,
            surface="system",
            armed=True,
            ticket_id=0,
            ticket_store=self.ticket_store,
        )
        return self._perform(config, prepared, run)

    def _usable_config(
        self, name: str, run: SkillRuntime, *, write: bool
    ) -> tuple[SkillConfig | None, str | None]:
        skill = self.store.get(name)
        if skill is None or not skill.enabled:
            return None, envelope_error("that skill is not enabled", skill=name)
        try:
            config = load_skill(name, self.skills_dir)
        except SkillError as exc:
            return None, envelope_error(str(exc), skill=name)
        if config.kind != "http":
            return None, envelope_error("that skill is not an HTTP skill", skill=name)
        blocked = check_surface_access(config, run, write=write)
        if blocked is not None:
            return None, envelope_error(blocked, skill=name)
        return config, None

    def _prepare(
        self,
        config: SkillConfig,
        method: str,
        path: str,
        query: dict[str, str] | None,
        body: object,
        run: SkillRuntime,
    ) -> PreparedRequest:
        rendered = substitute_sender(path, run)
        raw_path, _, embedded = rendered.partition("?")
        values: dict[str, str] = {
            str(key): substitute_sender(str(value), run)
            for key, value in (query or {}).items()
        }
        if embedded:
            for key, value in parse_qsl(embedded, keep_blank_values=True):
                values[key] = value
        clean_body = substitute_body(body, run)
        url = join_skill_url(config, raw_path, values)
        return PreparedRequest(
            method=method,
            path=raw_path,
            query=values,
            body=clean_body,
            url=url,
        )

    def _stage(
        self,
        name: str,
        config: SkillConfig,
        prepared: PreparedRequest,
        run: SkillRuntime,
        started: float,
    ) -> str:
        operator = status_state.operator_jid
        if not operator:
            return envelope_error(
                "the operator self-chat is not known; the write was not staged",
                skill=name,
            )
        summary = self.secrets.redact(summary_text(name, prepared, run))
        request = self.store.create_request(
            skill=name,
            ticket_id=run.ticket_id or None,
            chat_id=run.chat_id,
            request_json=self.secrets.redact(
                json.dumps(
                    {
                        "method": prepared.method,
                        "path": prepared.path,
                        "query": prepared.query,
                        "body": prepared.body,
                        "url": prepared.url,
                    },
                    sort_keys=True,
                )
            ),
            summary=summary,
        )
        try:
            message_id = self.transport.send_text(self.session, operator, summary)
        except Exception as exc:
            logger.warning(
                "Skill confirmation could not be sent: {error}",
                error=type(exc).__name__,
            )
            self.store.resolve_request(request.id, "refused")
            return envelope_error(
                "the confirmation message could not be sent; the write was not staged",
                skill=name,
            )
        self.store.set_operator_message(request.id, message_id)
        self._record(
            name=name,
            run=run,
            method=prepared.method,
            path=prepared.path,
            status=None,
            duration=int((time.monotonic() - started) * 1000),
            outcome="staged",
            error="",
            request=request.request_json,
        )
        return envelope_ok(
            skill=name,
            staged=True,
            request_id=request.id,
            note="waiting for the operator's yes in the self-chat",
        )

    def _perform_envelope(
        self,
        name: str,
        config: SkillConfig,
        prepared: PreparedRequest,
        run: SkillRuntime,
        started: float,
    ) -> str:
        try:
            result = self._perform(config, prepared, run)
        except SkillError as exc:
            return self._fail_http(name, run, prepared, str(exc), started)
        fields: dict[str, object] = {
            "skill": name,
            "status": result.status,
            "body": decode_response(result),
            "duration_ms": int((time.monotonic() - started) * 1000),
        }
        if result.truncated:
            fields["truncated"] = True
            fields["note"] = "the response was cut at the byte cap"
        return envelope_ok(**fields)

    def _perform(
        self,
        config: SkillConfig,
        prepared: PreparedRequest,
        run: SkillRuntime,
        extra_headers: dict[str, str] | None = None,
    ) -> HttpResult:
        headers: dict[str, str] = {"Accept": "application/json"}
        if config.auth.type != "none":
            secret = self.secrets.get(config.auth.secret)
            if config.auth.type == "bearer":
                headers["Authorization"] = f"Bearer {secret}"
            else:
                headers[config.auth.header or "X-API-Key"] = secret
        if extra_headers:
            headers.update(extra_headers)
        payload = body_bytes(prepared.body)
        if payload is not None:
            headers.setdefault("Content-Type", "application/json")
        started = time.monotonic()
        result = http_call(
            config,
            method=prepared.method,
            url=prepared.url,
            body=payload,
            headers=headers,
        )
        duration = int((time.monotonic() - started) * 1000)
        safe_body = self.secrets.redact(result.text)
        redacted = HttpResult(
            status=result.status,
            text=safe_body,
            truncated=result.truncated,
        )
        self._record(
            name=config.name,
            run=run,
            method=prepared.method,
            path=prepared.path,
            status=result.status,
            duration=duration,
            outcome="ok",
            error="",
            request=request_json(prepared),
        )
        record_ticket(
            run,
            f"{config.name} {prepared.method} {prepared.path} -> {result.status}",
            "ok",
        )
        self.store.reset_failures(config.name)
        return redacted

    def _fail_http(
        self,
        name: str,
        run: SkillRuntime,
        prepared: PreparedRequest,
        error: str,
        started: float,
    ) -> str:
        safe = self._safe(error)
        self._record(
            name=name,
            run=run,
            method=prepared.method,
            path=prepared.path,
            status=None,
            duration=int((time.monotonic() - started) * 1000),
            outcome="failed",
            error=safe,
            request=request_json(prepared),
        )
        record_ticket(run, f"{name} {prepared.method} {prepared.path} failed", "failed")
        self._note_failure(name, safe)
        return envelope_error(safe, skill=name)

    def _fail_event(
        self, name: str, run: SkillRuntime, prepared: PreparedRequest, error: str
    ) -> None:
        safe = self._safe(error)
        self._record(
            name=name,
            run=run,
            method="POST",
            path=prepared.path,
            status=None,
            duration=0,
            outcome="failed",
            error=safe,
            request=request_json(prepared),
        )
        self._note_failure(name, safe)

    def _fail_sql(self, name: str, run: SkillRuntime, error: str, started: float) -> str:
        safe = self._safe(error)
        self._record(
            name=name,
            run=run,
            method="SQL",
            path=safe[:CALL_PATH_CAP],
            status=None,
            duration=int((time.monotonic() - started) * 1000),
            outcome="failed",
            error=safe,
            request=safe[:REQUEST_JSON_CAP],
        )
        record_ticket(run, f"{name} SQL failed", "failed")
        self._note_failure(name, safe)
        return envelope_error(safe, skill=name)

    def _record(
        self,
        *,
        name: str,
        run: SkillRuntime,
        method: str,
        path: str,
        status: int | None,
        duration: int,
        outcome: str,
        error: str,
        request: str,
    ) -> None:
        record_call(
            store=self.store,
            data_dir=self.data_dir,
            session=self.session,
            name=name,
            run=run,
            method=method,
            path=path,
            status=status,
            duration_ms=duration,
            outcome=outcome,
            error=error,
            request_json=request,
            redact=self.secrets.redact,
        )

    def _note_failure(self, name: str, error: str) -> None:
        note_failure(
            store=self.store,
            transport=self.transport,
            session=self.session,
            name=name,
            error=error,
            redact=self.secrets.redact,
        )

    def _safe(self, text: str) -> str:
        try:
            return self.secrets.redact(text)
        except SkillError:
            return text
