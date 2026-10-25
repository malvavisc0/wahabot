"""Small request helpers shared by the skill runtime and its tools."""

import json
from pathlib import Path

from wahabot.core.skills.errors import SkillError
from wahabot.core.skills.guardrails import join_skill_url, substitute_sender
from wahabot.core.skills.http import MAX_REQUEST_BYTES, decode_response
from wahabot.core.skills.loader import load_skill
from wahabot.core.skills.models import (
    MAX_SKILL_CALLS_PER_TURN,
    HttpResult,
    PreparedRequest,
    SkillConfig,
    SkillRequest,
    SkillRuntime,
    envelope_error,
)
from wahabot.core.skills.secrets import SecretStore
from wahabot.core.skills.store import SkillStore
from wahabot.core.tickets.models import Ticket


def substitute_body(body: object, run: SkillRuntime) -> object:
    """Fill sender placeholders in a JSON body without changing its shape."""
    if body is None:
        return None
    if isinstance(body, str):
        return substitute_sender(body, run)
    rendered = substitute_sender(json.dumps(body, default=str), run)
    try:
        return json.loads(rendered)
    except json.JSONDecodeError:
        return rendered


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
    person = _person(ticket, run)
    lines = [f"{name}: {prepared.method} {prepared.path}"]
    if prepared.query:
        lines.append("Query: " + json.dumps(prepared.query, sort_keys=True))
    if prepared.body is not None:
        lines.append("Body: " + body_text(prepared.body))
    where = f" for {person}" if person else ""
    ticket_text = f" (ticket {ticket.number})" if ticket is not None else ""
    lines.append(f"{where}{ticket_text}?".strip())
    lines.append("Reply yes.")
    return "\n".join(lines)


def _person(ticket: Ticket | None, run: SkillRuntime) -> str:
    if ticket is None or run.ticket_store is None:
        return ""
    contact = run.ticket_store.contact(ticket.chat_id)
    return contact.display if contact is not None else ""


def secret_error(secrets: SecretStore, config: SkillConfig) -> str | None:
    """Verify the key exists before a request is staged or sent."""
    if config.auth.type == "none":
        return None
    try:
        secrets.get(config.auth.secret)
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
    stored = str(payload.get("url") or "") or expected
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
