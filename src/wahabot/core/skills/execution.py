"""Call logging, ticket events and consecutive-failure notifications."""

from collections.abc import Callable
from pathlib import Path

from loguru import logger

from wahabot.core.audit import save_action
from wahabot.core.skills.models import SkillCall, SkillRequest, SkillRun, SkillRuntime
from wahabot.core.skills.store import SkillStore
from wahabot.core.tickets.models import Actor, EventKind
from wahabot.core.tickets.store import TicketStore
from wahabot.core.transport import Transport
from wahabot.status import state as status_state

FAILURE_NOTIFY_THRESHOLD = 3
CALL_PATH_CAP = 300
REQUEST_JSON_CAP = 4000


def record_call(
    *,
    store: SkillStore,
    data_dir: Path | None,
    session: str,
    name: str,
    run: SkillRuntime,
    method: str,
    path: str,
    status: int | None,
    duration_ms: int,
    outcome: str,
    error: str,
    request_json: str,
    redact: Callable[[str], str],
) -> None:
    """Append the skill call and its audit entry."""
    safe = _safe(redact, request_json)
    safe_error = _safe(redact, error)[:CALL_PATH_CAP]
    initiator = _initiator(run)
    store.add_call(
        SkillCall(
            id=0,
            skill=name,
            ticket_id=run.ticket_id or None,
            chat_id=run.chat_id,
            started_by=initiator,
            method=method,
            path=path[:CALL_PATH_CAP],
            status=status,
            duration_ms=duration_ms,
            outcome=outcome,
            error=safe_error,
            request_json=safe[:REQUEST_JSON_CAP],
            created_at=store.now(),
        )
    )
    if data_dir is not None:
        save_action(
            data_dir,
            session,
            "skill_call",
            chat_id=run.chat_id,
            skill=name,
            method=method,
            path=path[:CALL_PATH_CAP],
            status=status,
            duration_ms=duration_ms,
            started_by=initiator,
            ticket_id=run.ticket_id or None,
            outcome=outcome,
            error=safe_error,
        )


def record_ticket(run: SkillRuntime, body: str, outcome: str) -> None:
    """Add one skill event to the ticket inside the ticket path."""
    if run.ticket_store is None or not run.ticket_id:
        return
    try:
        run.ticket_store.add_event(
            run.ticket_id,
            Actor.SKILL.value,
            EventKind.SKILL_CALL.value,
            body=f"{body} [{outcome}]",
        )
    except Exception:
        logger.exception("Could not write the skill ticket event")


def record_confirmed_ticket(
    ticket_store: TicketStore | None, session: str, request: SkillRequest, outcome: str
) -> None:
    """Add the confirmation outcome to the staged write's ticket."""
    run = SkillRun(
        session=session,
        chat_id=request.chat_id,
        surface="operator",
        armed=True,
        ticket_id=request.ticket_id or 0,
        ticket_store=ticket_store,
    )
    record_ticket(run, f"{request.skill} staged write [{outcome}]", outcome)


def note_failure(
    *,
    store: SkillStore,
    transport: Transport,
    session: str,
    name: str,
    error: str,
    redact: Callable[[str], str],
) -> None:
    """Count a failure; on the third, alert the operator and mark the test failed."""
    count = store.record_failure(name)
    if count != FAILURE_NOTIFY_THRESHOLD:
        return
    safe = _safe(redact, error)
    skill = store.get(name)
    store.set_test(
        name,
        state="failed",
        reason=safe,
        config_hash=skill.test_config_hash if skill is not None else "",
    )
    operator = status_state.operator_jid
    if not operator:
        return
    try:
        transport.send_text(
            session,
            operator,
            f"Skill {name} failed {count} times in a row. Last error: {safe}",
        )
    except Exception:
        logger.warning("Could not notify the skill failure")


def _initiator(run: SkillRuntime) -> str:
    if run.armed or run.surface in ("operator", "system"):
        return "operator" if run.surface == "operator" or run.armed else "system"
    return "assistant"


def _safe(redact: Callable[[str], str], text: str) -> str:
    try:
        return redact(text)
    except Exception:
        return text
