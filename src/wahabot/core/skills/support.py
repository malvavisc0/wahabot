"""Small request helpers shared by the skill runtime and its tools."""

import json
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qsl, unquote

from wahabot.core.envelope import error as envelope_error
from wahabot.core.skills.errors import SkillError
from wahabot.core.skills.guardrails import (
    decoded_path,
    foreign_identifier_error,
    join_skill_url,
    substitute_path,
    substitute_sender,
)
from wahabot.core.skills.http import MAX_REQUEST_BYTES, decode_response
from wahabot.core.skills.loader import load_skill
from wahabot.core.skills.models import (
    MAX_SKILL_CALLS_PER_TURN,
    HttpResult,
    PreparedRequest,
    SkillConfig,
    SkillRequest,
    SkillRuntime,
)
from wahabot.core.skills.secrets import SecretStore
from wahabot.core.skills.store import SkillStore
from wahabot.core.tickets.models import Ticket


def prepare_request(
    config: SkillConfig,
    method: str,
    path: str,
    query: dict[str, str] | None,
    body: object,
    run: SkillRuntime,
) -> PreparedRequest:
    """Resolve sender placeholders and build the exact request URL."""
    raw_path, _, embedded = path.partition("?")
    path_text = substitute_path(raw_path, run)
    values: dict[str, str] = {
        substitute_sender(str(key), run): substitute_sender(str(value), run)
        for key, value in (query or {}).items()
    }
    for key, value in parse_qsl(embedded, keep_blank_values=True):
        values[substitute_sender(key, run)] = substitute_sender(value, run)
    clean_body = substitute_body(body, run)
    return PreparedRequest(
        method=method,
        path=path_text,
        query=values,
        body=clean_body,
        url=join_skill_url(config, path_text, values),
    )


def claim_request(name: str, prepared: PreparedRequest, run: SkillRuntime) -> str | None:
    """Count the attempt, then refuse a repeated or foreign-scoped request."""
    repeat = (
        f"{name}:{prepared.method}:{decoded_path(prepared.path)}:"
        + f"{json.dumps(prepared.query, sort_keys=True)}:{body_text(prepared.body)}"
    )
    refused = claim_attempt(run, repeat)
    if refused is not None:
        return refused
    text = decoded_path(
        " ".join(
            [
                prepared.path,
                json.dumps(prepared.query, sort_keys=True),
                body_text(prepared.body),
            ]
        )
    )
    return foreign_identifier_error(text, run)


def substitute_body(body: object, run: SkillRuntime) -> object:
    """Fill sender placeholders in a JSON body without changing its shape."""
    if body is None or isinstance(body, bytes):
        return body
    if isinstance(body, str):
        return substitute_sender(body, run)
    if isinstance(body, dict):
        return {
            substitute_sender(str(key), run): substitute_body(value, run)
            for key, value in body.items()
        }
    if isinstance(body, list):
        return [substitute_body(value, run) for value in body]
    return body


def body_text(body: object) -> str:
    """A JSON or raw body as text for logs and repeat keys."""
    if body is None:
        return ""
    if isinstance(body, str):
        return body
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    return json.dumps(body, sort_keys=True, default=str)


def body_bytes(body: object) -> bytes | None:
    """The exact bytes sent for one request body."""
    if body is None:
        return None
    if isinstance(body, bytes):
        return body
    return body_text(body).encode()


def request_error(method: str, body: object) -> str | None:
    """Refuse unsupported methods and oversized request bodies."""
    allowed = {"GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"}
    if method not in allowed:
        return f"unsupported HTTP method: {method}"
    if len(body_bytes(body) or b"") > MAX_REQUEST_BYTES:
        return f"the request body exceeds {MAX_REQUEST_BYTES} bytes"
    return None


def request_json(prepared: PreparedRequest) -> str:
    """The redactable request shape stored with a call."""
    return json.dumps(
        {
            "method": prepared.method,
            "path": prepared.path,
            "query": prepared.query,
            "body": prepared.body,
        },
        sort_keys=True,
        default=str,
    )


def render_result(result: HttpResult) -> str:
    """A bounded, plain rendering for the operator's confirmation reply."""
    body = decode_response(result)
    rendered = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
    note = " [truncated]" if result.truncated else ""
    return f"{result.status}{note} {rendered}"[:1000]


def claim_attempt(run: SkillRuntime, key: str) -> str | None:
    """Count one attempt or refuse the turn limit / a repeated call."""
    if run.skill_calls >= MAX_SKILL_CALLS_PER_TURN:
        return f"skill call limit reached ({MAX_SKILL_CALLS_PER_TURN} per turn)"
    if key in run.skill_call_keys:
        return "refused a repeated identical skill call in the same turn"
    run.skill_calls += 1
    run.skill_call_keys.append(key)
    return None


def summary_text(name: str, prepared: PreparedRequest, run: SkillRuntime) -> str:
    """The exact prepared request the operator must approve."""
    ticket = (
        run.ticket_store.ticket(run.ticket_id)
        if run.ticket_store is not None and run.ticket_id
        else None
    )
    person = person_label(ticket, run)
    lines = [f"{name}: {prepared.method} {unquote(prepared.path)}"]
    if prepared.query:
        lines.append("Query: " + json.dumps(prepared.query, sort_keys=True))
    if prepared.body is not None:
        lines.append("Body: " + body_text(prepared.body))
    where = f" for {person}" if person else ""
    ticket_text = f" (ticket {ticket.number})" if ticket is not None else ""
    lines.append(f"Approve this write{where}{ticket_text}?")
    lines.append("Reply yes.")
    return "\n".join(lines)


def person_label(ticket: Ticket | None, run: SkillRuntime) -> str:
    """The contact display name for a human-readable confirmation."""
    if ticket is None or run.ticket_store is None:
        return ""
    contact = run.ticket_store.contact(ticket.chat_id)
    return contact.display if contact is not None else ""


def secret_error(secrets: SecretStore, config: SkillConfig) -> str | None:
    """Verify every referenced secret decrypts before a request is staged or sent."""
    for ref in config.secret_refs:
        try:
            secrets.get(ref)
        except SkillError as exc:
            return envelope_error(str(exc), skill=config.name)
    return None


def stored_request(
    store: SkillStore, skills_dir: Path, request: SkillRequest
) -> tuple[SkillConfig, PreparedRequest]:
    """Rehydrate the exact request a staged write will execute."""
    skill = store.get(request.skill)
    if skill is None or not skill.enabled:
        raise SkillError("that skill is no longer enabled")
    config = load_skill(request.skill, skills_dir)
    payload = json.loads(request.request_json)
    query = dict(payload.get("query") or {})
    path = str(payload["path"])
    expected = join_skill_url(config, path, query)
    stored = str(payload.get("url") or "")
    if stored != expected:
        raise SkillError(
            "the staged request no longer matches the current skill configuration"
        )
    return config, PreparedRequest(
        method=str(payload["method"]),
        path=path,
        query=query,
        body=payload.get("body"),
        url=expected,
    )


def redact_data(value: object, redact: Callable[[str], str]) -> object:
    """Redact secret values inside nested SQL result data."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [redact_data(item, redact) for item in value]
    if isinstance(value, dict):
        return {str(key): redact_data(item, redact) for key, item in value.items()}
    return value
