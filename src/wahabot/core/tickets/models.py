"""Ticket domain models shared by the store, services and tools."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class TicketState(StrEnum):
    """The ticket lifecycle states."""

    OPEN = "open"
    WAITING_CUSTOMER = "waiting_customer"
    NEEDS_YOU = "needs_you"
    DONE = "done"
    CLOSED = "closed"


class DoneBy(StrEnum):
    """Who resolved the ticket."""

    ASSISTANT = "assistant"
    OPERATOR = "operator"
    CUSTOMER = "customer"


ACTIVE_STATES = (
    TicketState.OPEN.value,
    TicketState.WAITING_CUSTOMER.value,
    TicketState.NEEDS_YOU.value,
)


class Actor(StrEnum):
    """Who wrote a ticket event."""

    ASSISTANT = "assistant"
    OPERATOR = "operator"
    CUSTOMER = "customer"
    SYSTEM = "system"
    SKILL = "skill"


class EventKind(StrEnum):
    """The ticket-event kinds the runtime writes."""

    OPENED = "opened"
    CUSTOMER_MESSAGE = "customer_message"
    REPLY = "reply"
    ALERT = "alert"
    OPERATOR_REPLY = "operator_reply"
    SENT = "sent"
    HELD = "held"
    HOLDING = "holding"
    STATE = "state"
    ATTACHMENT = "attachment"
    NUDGE = "nudge"
    SKILL_CALL = "skill_call"


@dataclass(slots=True)
class Contact:
    """One customer's 1:1 chat and its routing facts."""

    id: int
    chat_id: str
    phone: str = ""
    lid: str = ""
    display_name: str = ""
    language: str = ""
    external_user_ref: str = ""
    last_inbound_at: int | None = None
    paused_until: int | None = None

    @property
    def display(self) -> str:
        """The operator-facing name, never a raw chat JID."""
        if self.display_name:
            return self.display_name
        for value in (self.phone, self.chat_id):
            if value:
                return value.split("@", 1)[0]
        return ""


@dataclass(slots=True)
class Ticket:
    """One customer conversation's work item."""

    id: int
    number: int
    contact_id: int
    chat_id: str
    state: str
    urgent: bool = False
    done_by: str | None = None
    summary: str = ""
    details: dict[str, object] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    external_user_ref: str = ""
    alert_message_ids: list[str] = field(default_factory=list)
    previous_ticket_id: int | None = None
    created_at: int = 0
    updated_at: int = 0
    done_at: int | None = None
    closed_at: int | None = None
    last_customer_at: int | None = None


@dataclass(slots=True)
class TicketEvent:
    """One durable entry in a ticket's timeline."""

    id: int
    ticket_id: int
    at: int
    actor: str
    kind: str
    body: str = ""
    message_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Attachment:
    """One local copy of a customer attachment, linked to its message."""

    id: int
    ticket_id: int
    message_id: str
    kind: str
    mime: str = ""
    bytes: int = 0
    sha256: str = ""
    path: str = ""
    derived_text: str = ""
    page_count: int = 0


@dataclass(slots=True)
class TicketUpdate:
    """Values an agent run stages and the runtime applies after delivery."""

    state: str | None = None
    summary: str | None = None
    details: dict[str, object] | None = None
    missing: list[str] | None = None
    done_by: str | None = None
    urgent: bool | None = None


def json_list(raw: str) -> list[str]:
    """Decode a JSON list of strings, failing soft to an empty list."""
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def as_mapping(value: object) -> Mapping[str, object]:
    """A string-keyed view of a JSON-like object, or an empty mapping."""
    return value if isinstance(value, Mapping) else {}


def text_field(value: object, key: str) -> str:
    """A string field from a JSON-like object, or ""."""
    found = as_mapping(value).get(key)
    return "" if found is None else str(found)


def json_object(raw: str) -> dict[str, object]:
    """Decode a JSON object, failing soft to an empty mapping."""
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return dict(as_mapping(value))


ALERT_SENDING = "sending"
ALERT_SENT = "sent"
ALERT_FAILED = "failed"


@dataclass(slots=True)
class AlertIntent:
    """One operator alert claimed before its transport send."""

    key: str
    ticket_id: int
    session: str
    chat_id: str
    text: str
    state: str
    sent_id: str
    created_at: int
    updated_at: int
