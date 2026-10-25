"""Ticket tools: support state, updates, handover and splitting."""

from collections.abc import Callable

from llama_index.core.tools import BaseTool, FunctionTool

from wahabot.ai.tools.envelope import error, ok
from wahabot.ai.tools.handover import hand_to_operator
from wahabot.ai.tools.runcontext import (
    RunTarget,
    current_target,
    log_action_reason,
    operator_run,
)
from wahabot.ai.tools.schemas import (
    LinkTicketUserSchema,
    NewTicketSchema,
    SupportStateSchema,
    UpdateTicketSchema,
)
from wahabot.ai.tools.ticket_common import (
    OperatorLanguage,
    merge_update,
    ticket_context,
)
from wahabot.core.schedule.models import OpeningState
from wahabot.core.teams.groups import team_capability_error
from wahabot.core.tickets.alerts import Translator
from wahabot.core.tickets.attachments import attachment_metadata
from wahabot.core.tickets.models import Contact, DoneBy, Ticket, TicketState
from wahabot.core.tickets.router import contacts_named
from wahabot.core.tickets.store import TicketStore
from wahabot.core.transport import Transport
from wahabot.settings import Settings

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
        if target.surface == "team_group":
            return team_support_payload(store, number, name)
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


def team_support_payload(store: TicketStore, number: int | None, name: str) -> str:
    """Read-only ticket facts for a team-group run: open work or one lookup."""
    if number is not None or name.strip():
        ticket, contact = lookup_ticket(store, number, name)
        return ok(**support_payload(store, ticket, contact))
    return ok(
        open_tickets=[
            {
                "number": ticket.number,
                "person": (
                    contact.display
                    if (contact := store.contact_by_id(ticket.contact_id)) is not None
                    else "unknown"
                ),
                "state": ticket.state,
                "urgent": ticket.urgent,
                "done_by": ticket.done_by,
                "updated_at": ticket.updated_at,
            }
            for ticket in store.tickets_in_states()
        ]
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
        if refusal := team_capability_error(target, "ticket_write"):
            return error(refusal)
        store: TicketStore | None = target.ticket_store
        if store is None:
            return error("tickets are not available for this run")
        staging_error = update_staging_error(target, state, done_by)
        if staging_error:
            return error(staging_error)
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


def link_ticket_user() -> BaseTool:
    """Build the operator-only tool that writes the ticket/contact link atomically."""

    def link_ticket_user_fn(user_ref: str, number: int = 0, reason: str = "") -> str:
        target = current_target()
        if not target.armed and target.surface != "operator":
            return error("linking an external user reference is operator-only")
        if not user_ref.strip():
            return error("user_ref must not be empty")
        store: TicketStore | None = target.ticket_store
        if store is None:
            return error("tickets are not available for this run")
        ticket = (
            store.ticket_by_number(number) if number else store.ticket(target.ticket_id)
        )
        if ticket is None:
            return error("no matching ticket found")
        log_action_reason("link_ticket_user", reason, ticket=ticket.number)
        linked = store.link_ticket_user(ticket.id, user_ref)
        return ok(ticket=linked.number, external_user_ref=linked.external_user_ref)

    return FunctionTool.from_defaults(
        fn=link_ticket_user_fn,
        fn_schema=LinkTicketUserSchema,
        name="link_ticket_user",
        description=(
            "Operator only: atomically link a ticket and its contact to the "
            "external system's stable user reference. Customer runs cannot set it."
        ),
    )


def update_staging_error(
    target: RunTarget, state: str | None, done_by: str | None
) -> str | None:
    """Validate a staged update before touching the ticket."""
    if state is not None and state not in _ALLOWED_STATES:
        return f"unknown ticket state {state!r}"
    if state == TicketState.NEEDS_YOU.value:
        return "use hand_to_operator for needs_you; it stages and alerts the operator"
    if target.ticket_split and state in (
        TicketState.WAITING_CUSTOMER.value,
        TicketState.NEEDS_YOU.value,
        TicketState.CLOSED.value,
    ):
        return "a split ticket ends done; it cannot also be waiting, needs_you or closed"
    if done_by is not None and not done_by_allowed(
        done_by, target_operator=operator_run(target)
    ):
        return "done_by=operator is reserved for operator runs"
    return None


def done_by_allowed(done_by: str, *, target_operator: bool) -> bool:
    """Whether this run may stage the given resolver."""
    if done_by in _CUSTOMER_DONE_BY:
        return True
    return target_operator and done_by == DoneBy.OPERATOR.value


def new_ticket() -> BaseTool:
    """Build the split tool that stages a new ticket after delivery."""

    def new_ticket_fn(summary: str, reason: str = "") -> str:
        target = current_target()
        if refusal := team_capability_error(target, "ticket_write"):
            return error(refusal)
        store: TicketStore | None = target.ticket_store
        if store is None:
            return error("tickets are not available for this run")
        ticket, _ = ticket_context(store, target.ticket_id, target.chat_id)
        if ticket is None:
            return error("no ticket exists for this chat")
        if ticket.state not in (TicketState.OPEN.value, TicketState.DONE.value):
            return error("cannot split: only an open or done ticket can be split")
        staged_state = target.ticket_update.state if target.ticket_update else None
        if staged_state in (
            TicketState.WAITING_CUSTOMER.value,
            TicketState.NEEDS_YOU.value,
            TicketState.CLOSED.value,
        ):
            return error(
                "cannot split: this run staged a state a person must handle first"
            )
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
    schedule: Callable[[], OpeningState] | None = None,
    night_number: Callable[[], str] | None = None,
    away_message: Callable[[], str] | None = None,
) -> list[BaseTool]:
    """The ticket toolset, or [] when no ticket store is wired."""
    if store is None:
        return []
    return [
        support_state(),
        update_ticket(),
        link_ticket_user(),
        hand_to_operator(
            settings,
            waha,
            operator_language,
            translator,
            schedule=schedule,
            night_number=night_number,
            away_message=away_message,
        ),
        new_ticket(),
    ]
