"""Ticket tools: support state, updates, handover and splitting."""

from collections.abc import Callable
from dataclasses import replace

from llama_index.core.tools import BaseTool, FunctionTool

from wahabot.ai.tools.envelope import error, ok
from wahabot.ai.tools.runcontext import (
    RunTarget,
    current_target,
    fenced_message_id,
    log_action_reason,
    operator_run,
)
from wahabot.ai.tools.schemas import (
    HandToOperatorSchema,
    NewTicketSchema,
    SupportStateSchema,
    UpdateTicketSchema,
)
from wahabot.ai.tools.whatsapp.targets import resolve_message_token
from wahabot.core.tickets.alerts import (
    AlertResult,
    Translator,
    operator_language,
    send_alert,
)
from wahabot.core.tickets.attachments import attachment_metadata
from wahabot.core.tickets.models import (
    Contact,
    DoneBy,
    EventKind,
    Ticket,
    TicketState,
    TicketUpdate,
)
from wahabot.core.tickets.router import contacts_named
from wahabot.core.tickets.store import TicketStore
from wahabot.core.transport import Transport
from wahabot.settings import Settings
from wahabot.status.state import state as status_state

_ALLOWED_STATES = frozenset(
    {
        TicketState.OPEN.value,
        TicketState.WAITING_CUSTOMER.value,
        TicketState.NEEDS_YOU.value,
        TicketState.DONE.value,
        TicketState.CLOSED.value,
    }
)
_CUSTOMER_DONE_BY = frozenset({"assistant", "customer"})

OperatorLanguage = str | Callable[[], str]


def ticket_context(
    store: TicketStore | None, ticket_id: int, chat_id: str
) -> tuple[Ticket | None, Contact | None]:
    """The current ticket and contact for a run, falling back to the chat."""
    if store is None:
        return None, None
    ticket = store.ticket(ticket_id) if ticket_id else store.active_ticket(chat_id)
    if ticket is None:
        ticket = store.latest_ticket(chat_id)
    contact = store.contact_by_id(ticket.contact_id) if ticket is not None else None
    return ticket, contact


def support_payload(
    store: TicketStore, ticket: Ticket | None, contact: Contact | None
) -> dict[str, object]:
    """The ticket, contact and attachment facts the model may see."""
    payload: dict[str, object] = {
        "contact": contact_payload(contact),
        "ticket": ticket_payload(store, ticket),
    }
    if ticket is not None:
        payload["recent"] = [
            ticket_payload(store, item)
            for item in store.chat_tickets(ticket.chat_id, limit=5)
        ]
    return payload


def contact_payload(contact: Contact | None) -> dict[str, object] | None:
    """Contact facts without paths or secrets."""
    if contact is None:
        return None
    return {
        "name": contact.display_name,
        "phone": contact.phone.split("@", 1)[0]
        if "@" in contact.phone
        else contact.phone,
        "language": contact.language,
        "external_user_ref": contact.external_user_ref,
        "paused_until": contact.paused_until,
    }


def ticket_payload(store: TicketStore, ticket: Ticket | None) -> dict[str, object] | None:
    """Ticket facts and attachment metadata, never file paths."""
    if ticket is None:
        return None
    return {
        "number": ticket.number,
        "state": ticket.state,
        "urgent": ticket.urgent,
        "done_by": ticket.done_by,
        "summary": ticket.summary,
        "details": ticket.details,
        "missing": ticket.missing,
        "previous_ticket_id": ticket.previous_ticket_id,
        "created_at": ticket.created_at,
        "closed_at": ticket.closed_at,
        "attachments": attachment_metadata(store.attachments(ticket.id)),
    }


def support_state() -> BaseTool:
    """Build the read-only ticket state tool."""

    def support_state_fn(
        number: int | None = None, name: str = "", reason: str = ""
    ) -> str:
        target = current_target()
        store: TicketStore | None = target.ticket_store
        if store is None:
            return error("tickets are not available for this run")
        log_action_reason("support_state", reason, number=number, name=name)
        ticket: Ticket | None = None
        contact: Contact | None = None
        if operator_run(target) and (number is not None or name.strip()):
            ticket, contact = lookup_ticket(store, number, name)
        else:
            ticket, contact = ticket_context(store, target.ticket_id, target.chat_id)
        return ok(**support_payload(store, ticket, contact))

    return FunctionTool.from_defaults(
        fn=support_state_fn,
        fn_schema=SupportStateSchema,
        name="support_state",
        description=(
            "Read the current contact, active ticket, recent tickets, details, "
            "missing items and attachment metadata. Customer runs see only their "
            "own chat; operator runs may look up a ticket by number or unique name."
        ),
    )


def lookup_ticket(
    store: TicketStore, number: int | None, name: str
) -> tuple[Ticket | None, Contact | None]:
    """Resolve an operator lookup by ticket number or unique contact name."""
    if number is not None:
        ticket = store.ticket_by_number(number)
        contact = store.contact_by_id(ticket.contact_id) if ticket is not None else None
        return ticket, contact
    matches = contacts_named(store, name)
    if len(matches) != 1:
        return None, None
    contact = matches[0]
    ticket = store.active_ticket(contact.chat_id) or store.latest_ticket(contact.chat_id)
    return ticket, contact


def update_ticket() -> BaseTool:
    """Build the staging tool for ticket values."""

    def update_ticket_fn(
        state: str | None = None,
        summary: str | None = None,
        details: dict[str, str] | None = None,
        missing: list[str] | None = None,
        done_by: str | None = None,
        reason: str = "",
    ) -> str:
        target = current_target()
        store: TicketStore | None = target.ticket_store
        if store is None:
            return error("tickets are not available for this run")
        if state is not None and state not in _ALLOWED_STATES:
            return error(f"unknown ticket state {state!r}")
        if state == TicketState.NEEDS_YOU.value:
            return error(
                "use hand_to_operator for needs_you; it stages and alerts the operator"
            )
        if done_by is not None and not done_by_allowed(
            done_by, target_operator=operator_run(target)
        ):
            return error("done_by=operator is reserved for operator runs")
        ticket, _ = ticket_context(store, target.ticket_id, target.chat_id)
        if ticket is None:
            return error("no ticket exists for this chat")
        log_action_reason("update_ticket", reason, ticket=ticket.number)
        target.ticket_id = ticket.id
        target.ticket_update = merge_update(
            target.ticket_update,
            state=state,
            summary=summary,
            details=details,
            missing=missing,
            done_by=done_by,
        )
        return ok(ticket=ticket.number, staged=True)

    return FunctionTool.from_defaults(
        fn=update_ticket_fn,
        fn_schema=UpdateTicketSchema,
        name="update_ticket",
        description=(
            "Stage ticket state, summary, collected details, missing items or who "
            "resolved it. The runtime applies them only after a confirmed delivery. "
            "An answer with no missing items defaults to done by assistant."
        ),
    )


def done_by_allowed(done_by: str, *, target_operator: bool) -> bool:
    """Whether this run may stage the given resolver."""
    if done_by in _CUSTOMER_DONE_BY:
        return True
    return target_operator and done_by == DoneBy.OPERATOR.value


def merge_update(
    existing: TicketUpdate | None,
    *,
    state: str | None = None,
    summary: str | None = None,
    details: dict[str, str] | None = None,
    missing: list[str] | None = None,
    done_by: str | None = None,
) -> TicketUpdate:
    """Merge newly staged values into the run's update."""
    base = existing or TicketUpdate()
    return TicketUpdate(
        state=state if state is not None else base.state,
        summary=summary if summary is not None else base.summary,
        details=dict(details) if details is not None else base.details,
        missing=list(missing) if missing is not None else base.missing,
        done_by=done_by if done_by is not None else base.done_by,
        urgent=base.urgent,
    )


def handover_preflight(
    target: RunTarget, settings: Settings, attach_message_ids: list[str]
) -> tuple[TicketStore, Ticket, Contact, list[str]] | str:
    """Validate the run, ticket, attachments and operator target."""
    store: TicketStore | None = target.ticket_store
    if store is None:
        return error("tickets are not available for this run")
    ticket, contact = ticket_context(store, target.ticket_id, target.chat_id)
    if ticket is None or contact is None:
        return error("no ticket exists for this chat")
    selected, selection_error = resolve_attach_ids(
        attach_message_ids, target.chat_id, target.session, settings, target
    )
    if selection_error:
        return error(selection_error)
    if status_state.operator_jid == "":
        return error("no operator target is known; the alert was not sent")
    return store, ticket, contact, selected


def stage_handover(target: RunTarget, ticket: Ticket, urgent: bool) -> None:
    """Stage needs_you only after the transport confirmed the alert."""
    target.ticket_id = ticket.id
    staged = merge_update(target.ticket_update, state=TicketState.NEEDS_YOU.value)
    target.ticket_update = replace(staged, urgent=urgent or ticket.urgent)


def hand_to_operator(
    settings: Settings,
    waha: Transport,
    language: OperatorLanguage = "",
    translator: Translator | None = None,
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
        if no_new_handover(store, ticket):
            stage_handover(target, ticket, urgent)
            return ok(
                ticket=ticket.number, state=ticket.state, alert_ids=[], duplicate=True
            )
        language_value = language() if callable(language) else language
        log_action_reason("hand_to_operator", reason, ticket=ticket.number)
        try:
            result = send_alert(
                store,
                waha,
                target.session,
                status_state.operator_jid,
                ticket,
                contact,
                language=operator_language(language_value),
                translator=translator,
                only_message_ids=selected,
            )
        except Exception as exc:
            return error(f"handover alert failed — the operator was not notified: {exc}")
        if not result.delivered:
            return error("handover alert returned no message id; not confirmed")
        stage_handover(target, ticket, urgent)
        return handover_envelope(ticket, result)

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


def new_ticket() -> BaseTool:
    """Build the split tool that stages a new ticket after delivery."""

    def new_ticket_fn(summary: str, reason: str = "") -> str:
        target = current_target()
        store: TicketStore | None = target.ticket_store
        if store is None:
            return error("tickets are not available for this run")
        ticket, _ = ticket_context(store, target.ticket_id, target.chat_id)
        if ticket is None:
            return error("no ticket exists for this chat")
        if ticket.state not in (TicketState.OPEN.value, TicketState.DONE.value):
            return error("cannot split: only an open or done ticket can be split")
        log_action_reason("new_ticket", reason, ticket=ticket.number)
        target.ticket_id = ticket.id
        target.ticket_split = summary.strip()
        return ok(ticket=ticket.number, staged=True)

    return FunctionTool.from_defaults(
        fn=new_ticket_fn,
        fn_schema=NewTicketSchema,
        name="new_ticket",
        description=(
            "Stage a split into a new ticket for a different issue. Only an open or "
            "done ticket can split; a needs-you or waiting-customer ticket belongs to "
            "the operator. The runtime creates it after a confirmed delivery."
        ),
    )


def build_ticket_tools(
    store: TicketStore | None,
    settings: Settings,
    waha: Transport,
    *,
    operator_language: OperatorLanguage = "",
    translator: Translator | None = None,
) -> list[BaseTool]:
    """The ticket toolset, or [] when no ticket store is wired."""
    if store is None:
        return []
    return [
        support_state(),
        update_ticket(),
        hand_to_operator(settings, waha, operator_language, translator),
        new_ticket(),
    ]
