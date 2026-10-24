"""Route self-chat messages into ticket commands or reply candidates."""

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

from wahabot.ai.messages import bot_jids, jid_string
from wahabot.core.models import WahaEvent
from wahabot.core.tickets.models import ACTIVE_STATES, Contact, Ticket
from wahabot.core.tickets.store import TicketStore

_WHATS_OPEN = re.compile(r"^\s*what(?:'s| is)\s+open\??\s*$", re.IGNORECASE)
_CLOSE = re.compile(r"^\s*close\s+(?:ticket\s*)?#?(\d+)\s*$", re.IGNORECASE)
_REOPEN = re.compile(r"^\s*reopen\s+(?:ticket\s*)?#?(\d+)\s*$", re.IGNORECASE)
_PAUSE = re.compile(r"^\s*pause(?:\s+(.+))?\s*$", re.IGNORECASE)
_RESUME = re.compile(r"^\s*resume(?:\s+(.+))?\s*$", re.IGNORECASE)
_NUMBERED = re.compile(
    r"^\s*(?:ticket\s*)?#?(\d+)\s*(?:[:.\-]\s*|\s+)(.+)$", re.IGNORECASE
)
_NAMED = re.compile(r"^\s*([^:\n]{1,60}):\s*(.+)$")
_QUOTED_ID_KEYS = ("id", "_serialized")


class RouteKind(StrEnum):
    """The known self-chat shapes."""

    QUOTE_REPLY = "quote_reply"
    TICKET_REPLY = "ticket_reply"
    WHATS_OPEN = "whats_open"
    CLOSE = "close"
    REOPEN = "reopen"
    PAUSE = "pause"
    RESUME = "resume"
    ASK = "ask"
    NONE = "none"


@dataclass(slots=True)
class Route:
    """One parsed self-chat message."""

    kind: RouteKind
    ticket: Ticket | None = None
    contact: Contact | None = None
    text: str = ""
    candidates: list[Ticket] = field(default_factory=list)


def is_self_chat_event(event: WahaEvent) -> bool:
    """Whether the event is a ``fromMe`` message in the bot's own chat."""
    payload = event.payload
    if not payload.get("fromMe"):
        return False
    own = bot_jids(event)
    if jid_string(payload.get("from")) not in own:
        return False
    to = jid_string(payload.get("to"))
    return not to or to in own


def quoted_id(event: WahaEvent) -> str:
    """The id of the message this event quotes, or ""."""
    reply = event.payload.get("replyTo")
    if not isinstance(reply, dict):
        raw = event.payload.get("_data")
        data: dict[str, Any] = raw if isinstance(raw, dict) else {}
        reply = data.get("quotedMsg")
    if not isinstance(reply, dict):
        return ""
    quoted = cast(dict[str, Any], reply)
    for key in _QUOTED_ID_KEYS:
        value = quoted.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            nested = cast(dict[str, Any], value)
            if nested.get("_serialized"):
                return str(nested["_serialized"])
    return ""


def route_message(store: TicketStore, event: WahaEvent) -> Route:
    """Parse one operator message into a known shape or ``NONE``."""
    body = str(event.payload.get("body", "") or "").removeprefix("[voice note]").strip()
    if not body:
        return Route(RouteKind.NONE)
    alert = store.ticket_for_alert_message(quoted_id(event))
    if alert is not None:
        return Route(RouteKind.QUOTE_REPLY, ticket=alert, text=body)
    command = route_command(store, body)
    if command is not None:
        return command
    return route_ticket_text(store, body)


def route_command(store: TicketStore, body: str) -> Route | None:
    """Recognize the fixed command words."""
    if _WHATS_OPEN.match(body):
        return Route(RouteKind.WHATS_OPEN)
    for pattern, kind in ((_CLOSE, RouteKind.CLOSE), (_REOPEN, RouteKind.REOPEN)):
        match = pattern.match(body)
        if match:
            ticket = store.ticket_by_number(int(match.group(1)))
            return Route(kind, ticket=ticket, text=body)
    pause = _PAUSE.match(body)
    if pause:
        return pause_route(store, RouteKind.PAUSE, pause.group(1), body)
    resume = _RESUME.match(body)
    if resume:
        return pause_route(store, RouteKind.RESUME, resume.group(1), body)
    return None


def pause_route(
    store: TicketStore, kind: RouteKind, name: str | None, body: str
) -> Route:
    """A global pause/resume, or one uniquely named contact."""
    if not name or not name.strip():
        return Route(kind, text=body)
    matches = contacts_named(store, name.strip())
    if not matches:
        return Route(RouteKind.ASK, text=f"No contact named {name.strip()}.")
    if len(matches) > 1:
        return Route(
            RouteKind.ASK,
            text=f"More than one contact is named {name.strip()}; nothing changed.",
        )
    return Route(kind, contact=matches[0], text=body)


def route_ticket_text(store: TicketStore, body: str) -> Route:
    """Recognize a numbered or uniquely named ticket reply."""
    numbered = _NUMBERED.match(body)
    if numbered:
        ticket = store.ticket_by_number(int(numbered.group(1)))
        if ticket is not None:
            return Route(
                RouteKind.TICKET_REPLY, ticket=ticket, text=numbered.group(2).strip()
            )
        return Route(RouteKind.ASK, text=f"No ticket #{numbered.group(1)}.")
    named = _NAMED.match(body)
    if named:
        matches = contacts_named(store, named.group(1))
        if len(matches) == 1:
            ticket = active_for(store, matches[0])
            if ticket is not None:
                return Route(
                    RouteKind.TICKET_REPLY, ticket=ticket, text=named.group(2).strip()
                )
            return Route(
                RouteKind.ASK,
                text=f"No active ticket for {matches[0].display}.",
            )
        if len(matches) > 1:
            tickets = [
                ticket for contact in matches if (ticket := active_for(store, contact))
            ]
            return Route(RouteKind.ASK, candidates=tickets, text=body)
    return Route(RouteKind.NONE)


def contacts_named(store: TicketStore, name: str) -> list[Contact]:
    """Contacts whose display name matches *name* case-insensitively."""
    wanted = name.strip().casefold()
    return [
        contact
        for contact in store.contacts()
        if contact.display_name and contact.display_name.casefold() == wanted
    ]


def active_for(store: TicketStore, contact: Contact) -> Ticket | None:
    """The contact's active ticket."""
    return store.active_ticket(contact.chat_id)


def whats_open_text(store: TicketStore, now: int | None = None) -> str:
    """List active tickets newest first: number, person, age, state, urgent."""
    moment = store.now() if now is None else now
    tickets = store.tickets_in_states(ACTIVE_STATES)
    if not tickets:
        return "No open tickets."
    lines = ["Open tickets:"]
    for ticket in tickets:
        contact = store.contact_by_id(ticket.contact_id)
        person = contact.display if contact is not None else "unknown"
        since = ticket.last_customer_at or ticket.updated_at
        age_hours = max(0, moment - since) // 3600
        flag = " URGENT" if ticket.urgent else ""
        lines.append(f"- {ticket.number}: {person}, {age_hours}h, {ticket.state}{flag}")
    return "\n".join(lines)


def ask_which_ticket(candidates: list[Ticket]) -> str:
    """The clarifying question for an ambiguous target."""
    if not candidates:
        return "Which ticket? I could not match that name to an active ticket."
    numbers = ", ".join(str(ticket.number) for ticket in candidates)
    return f"Which ticket? I matched {numbers}; reply with the number."
