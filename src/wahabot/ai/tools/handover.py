"""Handover: validate, stage needs_you, alert and defer outside hours."""

from collections.abc import Callable
from dataclasses import replace

from llama_index.core.tools import BaseTool, FunctionTool
from loguru import logger

from wahabot.ai.tools.envelope import error, ok
from wahabot.ai.tools.runcontext import (
    RunTarget,
    current_target,
    fenced_message_id,
    log_action_reason,
)
from wahabot.ai.tools.schemas import HandToOperatorSchema
from wahabot.ai.tools.ticket_common import (
    OperatorLanguage,
    merge_update,
    ticket_context,
)
from wahabot.ai.tools.whatsapp.targets import resolve_message_token
from wahabot.core.schedule.hours import render_away_message
from wahabot.core.schedule.models import OpeningState
from wahabot.core.teams.groups import team_capability_error
from wahabot.core.tickets.alerts import (
    AlertResult,
    Translator,
    alert_intent_key,
    compose_alert,
    operator_language,
    send_alert,
)
from wahabot.core.tickets.models import (
    Actor,
    Contact,
    EventKind,
    Ticket,
    TicketState,
    TicketUpdate,
)
from wahabot.core.tickets.store import TicketStore
from wahabot.core.transport import Transport
from wahabot.settings import Settings
from wahabot.status.state import state as status_state


def handover_preflight(
    target: RunTarget, settings: Settings, attach_message_ids: list[str]
) -> tuple[TicketStore, Ticket, Contact, list[str]] | str:
    """Validate the run, ticket, attachments and operator target."""
    if refusal := team_capability_error(target, "ticket_write"):
        return error(refusal)
    store: TicketStore | None = target.ticket_store
    if store is None:
        return error("tickets are not available for this run")
    ticket, contact = ticket_context(store, target.ticket_id, target.chat_id)
    if ticket is None or contact is None:
        return error("no ticket exists for this chat")
    if ticket.state == TicketState.CLOSED.value:
        return error("ticket is closed; reopen it before handing over")
    selected, selection_error = resolve_attach_ids(
        attach_message_ids, target.chat_id, target.session, settings, target
    )
    if selection_error:
        return error(selection_error)
    if status_state.operator_jid == "":
        return error("no operator target is known; the alert was not sent")
    return store, ticket, contact, selected


def staged_ticket(ticket: Ticket, target: RunTarget, urgent: bool) -> Ticket:
    """The ticket as the alert should show it, including this run's stages."""
    update = target.ticket_update
    details = dict(ticket.details)
    if update is not None and update.details is not None:
        details.update(update.details)
    return replace(
        ticket,
        summary=(
            update.summary
            if update is not None and update.summary is not None
            else ticket.summary
        ),
        details=details,
        missing=(
            list(update.missing)
            if update is not None and update.missing is not None
            else list(ticket.missing)
        ),
        urgent=urgent
        or (update.urgent if update is not None else False)
        or ticket.urgent,
    )


def persist_handover_values(store: TicketStore, ticket: Ticket) -> None:
    """Make the values the operator saw durable before the run ends."""
    store.update_ticket(
        ticket.id,
        TicketUpdate(
            summary=ticket.summary,
            details=ticket.details,
            missing=ticket.missing,
            urgent=ticket.urgent,
        ),
        actor=Actor.ASSISTANT.value,
        kind=EventKind.STATE.value,
        body="handover details",
    )


def stage_handover(target: RunTarget, ticket: Ticket, urgent: bool) -> None:
    """Stage needs_you only after the transport confirmed the alert."""
    target.ticket_id = ticket.id
    staged = merge_update(target.ticket_update, state=TicketState.NEEDS_YOU.value)
    target.ticket_update = replace(staged, urgent=urgent or ticket.urgent)


def send_handover_alert(
    store: TicketStore,
    waha: Transport,
    target: RunTarget,
    ticket: Ticket,
    contact: Contact,
    view: Ticket,
    selected: list[str] | None,
    language_value: str,
    translator: Translator | None,
    *,
    urgent: bool,
    suffix: str,
    label: str,
) -> str:
    """Send one handover alert and stage needs_you unless it was a duplicate."""
    try:
        result = send_alert(
            store,
            waha,
            target.session,
            status_state.operator_jid,
            view,
            contact,
            language=operator_language(language_value),
            translator=translator,
            only_message_ids=selected,
            alert_key=alert_intent_key(store, ticket, suffix),
        )
    except Exception as exc:
        return error(f"{label} alert failed — the operator was not notified: {exc}")
    if result.pending:
        persist_handover_values(store, view)
        stage_handover(target, view, urgent)
        return ok(
            ticket=ticket.number,
            state=TicketState.NEEDS_YOU.value,
            alert_ids=[],
            pending=True,
        )
    if not result.delivered:
        return error(f"{label} alert returned no message id; not confirmed")
    persist_handover_values(store, view)
    stage_handover(target, view, urgent)
    return handover_envelope(view, result)


def duplicate_handover(
    store: TicketStore,
    waha: Transport,
    target: RunTarget,
    ticket: Ticket,
    contact: Contact,
    view: Ticket,
    selected: list[str] | None,
    language_value: str,
    translator: Translator | None,
    urgent: bool,
) -> str:
    """Escalate an already-alerted ticket, or confirm nothing is new."""
    if urgent and not ticket.urgent:
        return send_handover_alert(
            store,
            waha,
            target,
            ticket,
            contact,
            view,
            selected,
            language_value,
            translator,
            urgent=urgent,
            suffix="urgent",
            label="urgent",
        )
    persist_handover_values(store, view)
    stage_handover(target, ticket, urgent)
    return ok(ticket=ticket.number, state=ticket.state, alert_ids=[], duplicate=True)


def defer_handover(
    store: TicketStore,
    target: RunTarget,
    ticket: Ticket,
    contact: Contact,
    view: Ticket,
    state: OpeningState,
    language_value: str,
    translator: Translator | None,
    away_template: str,
) -> str:
    """Queue a non-urgent alert until opening and tell the model what to say."""
    text = compose_alert(
        store,
        view,
        contact,
        operator_language(language_value),
        translator,
    )
    moment = store.now()
    not_before = state.next_opening or moment
    store.defer_notice(
        view.id,
        target.session,
        status_state.operator_jid,
        text,
        not_before=not_before,
        at=moment,
    )
    persist_handover_values(store, view)
    store.update_ticket(
        view.id,
        TicketUpdate(
            state=TicketState.NEEDS_YOU.value,
            summary=view.summary,
            details=view.details,
            missing=view.missing,
            urgent=view.urgent,
        ),
        actor=Actor.ASSISTANT.value,
        kind=EventKind.STATE.value,
        body="deferred handover",
    )
    stage_handover(target, view, urgent=False)
    away = render_away_message(
        away_template, view.number, state.next_opening, state.timezone
    )
    return ok(
        ticket=view.number,
        state=TicketState.NEEDS_YOU.value,
        alert_ids=[],
        deferred=True,
        not_before=not_before,
        away_message=away,
        message=(
            "Outside opening hours: the operator alert is queued for the next "
            "opening. Tell the customer when to expect a reply using away_message."
        ),
    )


def notify_night_number(
    store: TicketStore,
    waha: Transport,
    session: str,
    ticket: Ticket,
    number: str,
) -> bool:
    """Send the after-hours contact a short urgent note, never first contact."""
    if not number.strip() or not store.night_ready(number):
        if number.strip():
            logger.warning(
                "Night number is configured but has not written yet; "
                + "no urgent message was sent"
            )
        return False
    contact = store.contact_by_phone(number)
    if contact is None:
        return False
    text = (
        f"URGENT ticket {ticket.number}: {ticket.summary or 'a customer needs help'}. "
        "Open the operator chat for the full alert."
    )
    try:
        return bool(waha.send_text(session, contact.chat_id, text))
    except Exception as exc:
        logger.warning("Night-number alert failed: {exc}", exc=exc)
        return False


def should_defer(state: OpeningState | None, urgent: bool) -> bool:
    """Whether a non-urgent handover waits for the next opening."""
    return state is not None and not state.open and not urgent


def should_notify_night(urgent: bool, night: str, state: OpeningState | None) -> bool:
    """Whether an urgent handover should also page the night number."""
    return bool(urgent and night and state is not None and not state.open)


def run_handover(
    store: TicketStore,
    waha: Transport,
    target: RunTarget,
    ticket: Ticket,
    contact: Contact,
    selected: list[str],
    urgent: bool,
    language_value: str,
    translator: Translator | None,
    state: OpeningState | None,
    night: str,
    away_template: str,
) -> str:
    """Stage, defer or send one validated handover."""
    view = staged_ticket(ticket, target, urgent)
    if should_defer(state, urgent):
        assert state is not None
        return defer_handover(
            store,
            target,
            ticket,
            contact,
            view,
            state,
            language_value,
            translator,
            away_template,
        )
    if no_new_handover(store, ticket):
        envelope = duplicate_handover(
            store,
            waha,
            target,
            ticket,
            contact,
            view,
            selected,
            language_value,
            translator,
            urgent,
        )
        if not (urgent and not ticket.urgent):
            return envelope
    else:
        envelope = send_handover_alert(
            store,
            waha,
            target,
            ticket,
            contact,
            view,
            selected,
            language_value,
            translator,
            urgent=urgent,
            suffix="handover",
            label="handover",
        )
    if should_notify_night(urgent, night, state):
        notify_night_number(store, waha, target.session, view, night)
    return envelope


def hand_to_operator(
    settings: Settings,
    waha: Transport,
    language: OperatorLanguage = "",
    translator: Translator | None = None,
    *,
    schedule: Callable[[], OpeningState] | None = None,
    night_number: Callable[[], str] | None = None,
    away_message: Callable[[], str] | None = None,
) -> BaseTool:
    """Build the handover tool that stages needs_you and sends the alert."""

    def hand_to_operator_fn(
        reason: str, urgent: bool = False, attach_message_ids: list[str] | None = None
    ) -> str:
        target = current_target()
        prepared = handover_preflight(target, settings, attach_message_ids or [])
        if isinstance(prepared, str):
            return prepared
        store, ticket, contact, selected = prepared
        language_value = language() if callable(language) else language
        log_action_reason("hand_to_operator", reason, ticket=ticket.number)
        return run_handover(
            store,
            waha,
            target,
            ticket,
            contact,
            selected,
            urgent,
            language_value,
            translator,
            schedule() if schedule is not None else None,
            night_number() if night_number is not None else "",
            away_message() if away_message is not None else "",
        )

    return FunctionTool.from_defaults(
        fn=hand_to_operator_fn,
        fn_schema=HandToOperatorSchema,
        name="hand_to_operator",
        description=(
            "Hand this ticket to the operator: stage needs_you, copy the listed "
            "attachments and send the alert. If the ticket is already needs_you, "
            "only new information is sent. There is no cooldown and no silent drop."
        ),
    )


def no_new_handover(store: TicketStore, ticket: Ticket) -> bool:
    """Whether a needs-you ticket has nothing new since its last alert.

    A recovered or retried run calls the tool again with the same
    customer message; re-alerting would page the operator twice for one
    event. A genuinely newer customer message still goes out.
    """
    if ticket.state != TicketState.NEEDS_YOU.value:
        return False
    alert = store.last_event(ticket.id, EventKind.ALERT.value)
    customer = store.last_event(ticket.id, EventKind.CUSTOMER_MESSAGE.value)
    return alert is not None and (customer is None or customer.id <= alert.id)


def handover_envelope(ticket: Ticket, result: AlertResult) -> str:
    """The confirmed handover's tool envelope."""
    return ok(
        ticket=ticket.number,
        state=TicketState.NEEDS_YOU.value,
        alert_ids=result.message_ids,
        attachments_failed=result.failed_attachments,
    )


def resolve_attach_ids(
    message_ids: list[str],
    chat_id: str,
    session: str,
    settings: Settings,
    target: RunTarget,
) -> tuple[list[str], str | None]:
    """Resolve `#m<n>` tokens and refuse paths or foreign ids."""
    resolved: list[str] = []
    for value in message_ids:
        if "/" in value or "\\" in value:
            return [], "raw file paths are invalid; pass a message id or #m<n> token"
        token, token_error = resolve_message_token(
            value, session=session, chat_id=chat_id, settings=settings, allow_search=False
        )
        if token_error:
            return [], token_error
        _, fence_error = fenced_message_id(token, target)
        if fence_error:
            return [], fence_error
        resolved.append(token)
    return resolved, None
