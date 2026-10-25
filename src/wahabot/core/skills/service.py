"""The skill runtime: guarded calls, staging and confirmation."""

import json
import time
from pathlib import Path
from typing import cast

from loguru import logger

from wahabot.core.envelope import error as envelope_error
from wahabot.core.envelope import ok as envelope_ok
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
from wahabot.core.skills.configuration import configuration_changed
from wahabot.core.skills.errors import QueryRefused, SkillError
from wahabot.core.skills.execution import (
    CALL_PATH_CAP,
    REQUEST_JSON_CAP,
    record_ticket,
)
from wahabot.core.skills.guardrails import (
    check_surface_access,
    is_write_method,
    join_skill_url,
    needs_confirmation,
)
from wahabot.core.skills.http import decode_response, http_call
from wahabot.core.skills.loader import load_skill
from wahabot.core.skills.models import (
    HttpResult,
    PreparedRequest,
    Skill,
    SkillConfig,
    SkillRun,
    SkillRuntime,
    SkillTestResult,
)
from wahabot.core.skills.secrets import SecretStore
from wahabot.core.skills.service_audit import SkillAuditMixin
from wahabot.core.skills.sql import execute_query, validate_read_query
from wahabot.core.skills.store import SkillStore
from wahabot.core.skills.support import (
    body_bytes,
    claim_attempt,
    claim_request,
    prepare_request,
    redact_data,
    render_result,
    request_error,
    request_json,
    secret_error,
    stored_request,
    summary_text,
)
from wahabot.core.tickets.store import TicketStore
from wahabot.core.transport import Transport
from wahabot.status import state as status_state


class SkillService(SkillAuditMixin):
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
        return load_active_skills(self.store, self.skills_dir, run, self.secrets)

    def prompt_section(self, run: SkillRuntime) -> str:
        """The active capability list and, when they fit, the skill docs."""
        return self.safe_text(prompt_text(self.store, self.skills_dir, run, self.secrets))

    def capability_summary(self) -> str:
        """Plain-language active skills for the operator's self-chat."""
        return self.safe_text(capability_text(self.store, self.skills_dir, self.secrets))

    def read_skill(self, name: str, run: SkillRuntime) -> str:
        """Read docs on demand when they exceeded the prompt budget."""
        return self.safe_text(
            read_skill_text(self.store, self.skills_dir, name, run, self.secrets)
        )

    def configuration_problem(self, name: str, skill: Skill) -> str | None:
        """Refuse an enabled skill whose live configuration moved past its test."""
        if not configuration_changed(self.store, self.secrets, self.skills_dir, name):
            return None
        self.store.set_test(
            name,
            state="stale",
            reason="the skill configuration changed after its last test",
            config_hash=skill.test_config_hash,
        )
        return "the skill configuration changed; test it again"

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
        config, prepared, error = self.http_attempt(name, method, path, query, body, run)
        if error is not None:
            self.record_refusal(name, run, method, path, error, started)
            return error
        assert config is not None and prepared is not None
        if is_write_method(method) and needs_confirmation(config, run):
            return self.stage_request(name, config, prepared, run, started)
        return self.perform_envelope(name, config, prepared, run, started)

    def http_attempt(
        self,
        name: str,
        method: str,
        path: str,
        query: dict[str, str] | None,
        body: object,
        run: SkillRuntime,
    ) -> tuple[SkillConfig | None, PreparedRequest | None, str | None]:
        """Preflight one HTTP call: config, caps, secret, URL and sender scope."""
        config, error = self.usable_config(name, run, write=is_write_method(method))
        if error is not None:
            return None, None, error
        assert config is not None
        request_problem = request_error(method, body)
        if request_problem is not None:
            return None, None, envelope_error(request_problem, skill=name)
        key_error = secret_error(self.secrets, config)
        if key_error is not None:
            return None, None, key_error
        try:
            prepared = prepare_request(config, method, path, query, body, run)
        except SkillError as exc:
            return None, None, envelope_error(self.safe_text(str(exc)), skill=name)
        claim_error = claim_request(name, prepared, run)
        if claim_error is not None:
            return None, None, envelope_error(claim_error, skill=name)
        return config, prepared, None

    def execute_sql(self, name: str, query: str, run: SkillRuntime) -> str:
        """Run an operator-only read query against a SQL skill."""
        started = time.monotonic()
        config, error = self.sql_config(name, run)
        if error is not None:
            self.record_refusal(name, run, "SQL", query, error, started)
            return error
        assert config is not None and config.sql is not None
        refused = claim_attempt(run, f"{name}:SQL:{query}")
        if refused is not None:
            error = envelope_error(refused, skill=name)
            self.record_refusal(name, run, "SQL", query, error, started)
            return error
        try:
            validate_read_query(query)
        except QueryRefused as exc:
            error = envelope_error(str(exc), skill=name)
            self.record_refusal(name, run, "SQL", query, error, started)
            return error
        try:
            secret = (
                self.secrets.get(config.sql.dsn_secret) if config.sql.dsn_secret else ""
            )
            result = execute_query(config, query, secret)
        except SkillError as exc:
            return self.fail_sql(name, run, query, str(exc), started)
        result = cast("dict[str, object]", redact_data(result, self.secrets.redact))
        self.record_call(
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
        record_ticket(run, f"{name} SQL query -> ok", "ok")
        self.store.reset_failures(name)
        return envelope_ok(skill=name, **result)

    def sql_config(
        self, name: str, run: SkillRuntime
    ) -> tuple[SkillConfig | None, str | None]:
        """The enabled SQL skill config, or the operator-only refusal."""
        if not run.armed and run.surface not in ("operator", "system"):
            return None, envelope_error(
                "SQL skills are operator-only in this version", skill=name
            )
        skill = self.store.get(name)
        if skill is None or not skill.enabled:
            return None, envelope_error("that skill is not enabled", skill=name)
        changed = self.configuration_problem(name, skill)
        if changed is not None:
            return None, envelope_error(changed, skill=name)
        try:
            config = load_skill(name, self.skills_dir)
        except SkillError as exc:
            return None, envelope_error(self.safe_text(str(exc)), skill=name)
        if config.kind != "sql" or config.sql is None:
            return None, envelope_error("that skill is not a SQL skill", skill=name)
        return config, None

    def confirm(self, request_id: int, *, approve: bool) -> str:
        """Execute one staged write exactly once, or decline it."""
        started = time.monotonic()
        request = self.store.claim_request(request_id)
        if request is None:
            return "That request was already handled."
        run = SkillRun(
            session=self.session,
            chat_id=request.chat_id,
            surface="operator",
            armed=True,
            ticket_id=request.ticket_id or 0,
            ticket_store=self.ticket_store,
        )
        if not approve:
            self.record_call(
                name=request.skill,
                run=run,
                method="CONFIRM",
                path="",
                status=None,
                duration=int((time.monotonic() - started) * 1000),
                outcome="refused",
                error="the operator declined the staged write",
                request=request.request_json,
            )
            record_ticket(run, f"{request.skill} staged write declined", "refused")
            self.store.resolve_request(request_id, "refused")
            return "Request declined."
        try:
            config, prepared = stored_request(self.store, self.skills_dir, request)
            skill = self.store.get(request.skill)
            problem = (
                self.configuration_problem(request.skill, skill)
                if skill is not None
                else None
            )
            if problem is not None:
                raise SkillError(problem)
            result = self.perform_request(config, prepared, run)
        except (SkillError, KeyError, ValueError, TypeError) as exc:
            safe = self.safe_text(str(exc))
            self.record_call(
                name=request.skill,
                run=run,
                method="CONFIRM",
                path="",
                status=None,
                duration=int((time.monotonic() - started) * 1000),
                outcome="failed",
                error=safe,
                request=request.request_json,
            )
            record_ticket(run, f"{request.skill} staged write failed", "failed")
            self.count_failure(request.skill, safe)
            self.store.resolve_request(request_id, "failed")
            return f"Request failed: {safe}"
        if not 200 <= result.status < 300:
            self.store.resolve_request(request_id, "failed")
            return f"Request failed: HTTP {result.status}"
        self.store.resolve_request(request_id, "done")
        return f"Done: {self.safe_text(render_result(result))}"

    def send_event(self, name: str, body: bytes, headers: dict[str, str]) -> HttpResult:
        """Post one event payload from the system, without staging."""
        skill = self.store.get(name)
        if skill is not None:
            problem = self.configuration_problem(name, skill)
            if problem is not None:
                raise SkillError(problem)
        config = enabled_config(self.store, self.skills_dir, name)
        if config.kind != "http":
            raise SkillError("only HTTP skills can send ticket events")
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
            result = self.perform_request(config, prepared, run, extra_headers=headers)
        except SkillError as exc:
            self.fail_event(name, run, prepared, str(exc))
            raise
        if not 200 <= result.status < 300:
            raise SkillError(f"the event endpoint returned HTTP {result.status}")
        return result

    def test_probe(self, name: str, method: str, path: str) -> HttpResult:
        """Run a configured safe check against a disabled draft."""
        skill = self.store.get(name)
        if skill is None:
            raise SkillError("that skill is not configured")
        config = load_skill(name, self.skills_dir)
        if config.kind != "http":
            raise SkillError("safe HTTP probes are only available for HTTP skills")
        prepared = prepare_request(config, method, path, None, None, SkillRun(armed=True))
        run = SkillRun(
            session=self.session,
            surface="system",
            armed=True,
            ticket_id=0,
            ticket_store=self.ticket_store,
        )
        return self.perform_request(config, prepared, run, note_failures=False)

    def test_sql_probe(self, name: str) -> SkillTestResult:
        """Run a harmless read-only query against one draft SQL skill."""
        skill = self.store.get(name)
        if skill is None:
            raise SkillError("that skill is not configured")
        config = load_skill(name, self.skills_dir)
        if config.kind != "sql" or config.sql is None:
            raise SkillError("that skill is not a SQL skill")
        secret = self.secrets.get(config.sql.dsn_secret) if config.sql.dsn_secret else ""
        execute_query(config, "SELECT 1", secret)
        self.store.reset_failures(name)
        return SkillTestResult(True, "read-only SQL probe passed")

    def usable_config(
        self, name: str, run: SkillRuntime, *, write: bool
    ) -> tuple[SkillConfig | None, str | None]:
        """The enabled live config, checked against the run's surface."""
        skill = self.store.get(name)
        if skill is None or not skill.enabled:
            return None, envelope_error("that skill is not enabled", skill=name)
        changed = self.configuration_problem(name, skill)
        if changed is not None:
            return None, envelope_error(changed, skill=name)
        try:
            config = load_skill(name, self.skills_dir)
        except SkillError as exc:
            return None, envelope_error(self.safe_text(str(exc)), skill=name)
        if config.kind != "http":
            return None, envelope_error("that skill is not an HTTP skill", skill=name)
        blocked = check_surface_access(config, run, write=write)
        if blocked is not None:
            return None, envelope_error(blocked, skill=name)
        return config, None

    def stage_request(
        self,
        name: str,
        config: SkillConfig,
        prepared: PreparedRequest,
        run: SkillRuntime,
        started: float,
    ) -> str:
        """Write the pending request and ask the operator in the self-chat."""
        operator = status_state.operator_jid
        if not operator:
            error = envelope_error(
                "the operator self-chat is not known; the write was not staged",
                skill=name,
            )
            self.record_refusal(name, run, prepared.method, prepared.path, error, started)
            return error
        summary = self.secrets.redact(summary_text(name, prepared, run))
        request = self.store.create_request(
            skill=name,
            ticket_id=run.ticket_id or None,
            chat_id=run.chat_id,
            request_json=json.dumps(
                {
                    "method": prepared.method,
                    "path": prepared.path,
                    "query": prepared.query,
                    "body": prepared.body,
                    "url": prepared.url,
                },
                sort_keys=True,
                default=str,
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
            error = envelope_error(
                "the confirmation message could not be sent; the write was not staged",
                skill=name,
            )
            safe = self.safe_text(error)
            self.record_call(
                name=name,
                run=run,
                method=prepared.method,
                path=prepared.path,
                status=None,
                duration=int((time.monotonic() - started) * 1000),
                outcome="failed",
                error=safe,
                request=request.request_json,
            )
            record_ticket(
                run, f"{name} {prepared.method} {prepared.path} not staged", "failed"
            )
            return error
        self.store.set_operator_message(request.id, message_id)
        self.record_call(
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
        record_ticket(
            run,
            f"{name} {prepared.method} {prepared.path} staged for the operator",
            "staged",
        )
        return envelope_ok(
            skill=name,
            staged=True,
            request_id=request.id,
            note="waiting for the operator's yes in the self-chat",
        )

    def perform_envelope(
        self,
        name: str,
        config: SkillConfig,
        prepared: PreparedRequest,
        run: SkillRuntime,
        started: float,
    ) -> str:
        """Send the prepared call and wrap the result into the tool envelope."""
        try:
            result = self.perform_request(config, prepared, run)
        except SkillError as exc:
            return self.fail_http(name, run, prepared, str(exc), started)
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

    def perform_request(
        self,
        config: SkillConfig,
        prepared: PreparedRequest,
        run: SkillRuntime,
        extra_headers: dict[str, str] | None = None,
        note_failures: bool = True,
    ) -> HttpResult:
        """Add auth, send once, then record and judge the response."""
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
        succeeded = 200 <= result.status < 300
        outcome = "ok" if succeeded else "failed"
        error = "" if succeeded else f"the skill returned HTTP {result.status}"
        self.record_call(
            name=config.name,
            run=run,
            method=prepared.method,
            path=prepared.path,
            status=result.status,
            duration=duration,
            outcome=outcome,
            error=error,
            request=request_json(prepared),
        )
        record_ticket(
            run,
            f"{config.name} {prepared.method} {prepared.path} -> {result.status}",
            outcome,
        )
        if succeeded:
            self.store.reset_failures(config.name)
        elif note_failures:
            self.count_failure(config.name, error)
        return redacted

    def fail_http(
        self,
        name: str,
        run: SkillRuntime,
        prepared: PreparedRequest,
        error: str,
        started: float,
    ) -> str:
        """Audit a transport failure, count it, and return the error envelope."""
        safe = self.safe_text(error)
        self.record_call(
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
        self.count_failure(name, safe)
        return envelope_error(safe, skill=name)
