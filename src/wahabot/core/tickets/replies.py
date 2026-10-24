"""Classify, draft, fact-check and send an operator's ticket reply."""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from loguru import logger

from wahabot.core.tickets.models import (
    Actor,
    Contact,
    EventKind,
    Ticket,
    TicketState,
    TicketUpdate,
    text_field,
)
from wahabot.core.tickets.store import TicketStore
from wahabot.core.transport import Transport

Completer = Callable[[str], str]
Remember = Callable[[str], Awaitable[None]]

RESOLUTION = "resolution"
INSTRUCTION = "instruction"
AMBIGUOUS = "ambiguous"
KINDS = frozenset({RESOLUTION, INSTRUCTION, AMBIGUOUS})

_CLASSIFY_PROMPT = (
    "Classify one operator reply for a support ticket. Return strict JSON only: "
    '{{"kind": "resolution"|"instruction"|"ambiguous"}}. '
    "resolution means the operator says what was done or decided for the customer. "
    "instruction means the operator asks the customer to do or answer something. "
    "ambiguous means neither is clear.\n\nOperator reply:\n{text}"
)

_DRAFT_PROMPT = (
    "Draft one WhatsApp message to a customer in {language}. Rephrase only the "
    "operator's words below; do not add facts, amounts, dates, times or promises. "
    "Return only the message text.\n\nOperator words:\n{text}"
)

_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
}

_UNITS = (
    "days?|weeks?|months?|years?|hours?|hrs?|minutes?|mins?|seconds?|"
    "euros?|eur|usd|dollars?|pounds?|gbp|kg|grams?|g|km|metres?|meters?|%"
)
_FACT_PATTERNS = (
    re.compile(rf"\d+(?:[.,]\d+)?\s*(?:{_UNITS})\b", re.IGNORECASE),
    re.compile(r"[€$£]\s*\d+(?:[.,]\d+)?"),
    re.compile(r"\d+(?:[.,]\d+)?\s*(?:€|\$|£)"),
    re.compile(r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b"),
    re.compile(r"\b\d{1,2}:\d{2}\b"),
    re.compile(r"\b\d{1,2}(?:am|pm)\b"),
    re.compile(r"\b(?:order|ticket|invoice|ref(?:erence)?)\s*#?\s*\d+\b", re.IGNORECASE),
)
_NUMBER_RANGE = re.compile(r"(\d+)\s*(?:-|to|until)\s*(\d+)")
_UNIT_CANON = {
    "euro": "eur",
    "euros": "eur",
    "dollar": "usd",
    "dollars": "usd",
    "pound": "gbp",
    "pounds": "gbp",
    "day": "days",
    "week": "weeks",
    "month": "months",
    "year": "years",
    "hour": "hours",
    "hr": "hours",
    "hrs": "hours",
    "minute": "minutes",
    "min": "minutes",
    "mins": "minutes",
    "second": "seconds",
    "gram": "grams",
    "metre": "metres",
    "meter": "metres",
}


@dataclass(slots=True)
class ReplyOutcome:
    """The operator-facing result of one routed reply."""

    confirmation: str
    sent: bool = False
    kind: str = AMBIGUOUS
    draft: str = ""
    mismatches: list[str] = field(default_factory=list)


def parse_kind(raw: str) -> str:
    """Read a strict JSON classification, failing to ``ambiguous``."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match is None:
        return AMBIGUOUS
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return AMBIGUOUS
    if not isinstance(value, dict):
        return AMBIGUOUS
    kind = text_field(value, "kind").strip().lower()
    return kind if kind in KINDS else AMBIGUOUS


async def classify_reply(complete: Completer, operator_text: str) -> str:
    """Classify the operator's words; never guess on a parse failure."""
    raw = await asyncio.to_thread(complete, _CLASSIFY_PROMPT.format(text=operator_text))
    return parse_kind(raw)


async def draft_message(complete: Completer, operator_text: str, language: str) -> str:
    """Draft the customer message from the operator's words only."""
    raw = await asyncio.to_thread(
        complete, _DRAFT_PROMPT.format(language=language or "en", text=operator_text)
    )
    return raw.strip().strip('"')


def normalize_text(text: str) -> str:
    """Normalize number words, ranges and whitespace for fact comparison."""
    lowered = text.casefold()
    for word, digit in _NUMBER_WORDS.items():
        lowered = re.sub(rf"\b{word}\b", digit, lowered)
    lowered = _NUMBER_RANGE.sub(r"\1-\2", lowered)
    for word, canon in _UNIT_CANON.items():
        lowered = re.sub(rf"\b{word}\b", canon, lowered)
    lowered = re.sub(r"(\d)\s+(am|pm)\b", r"\1\2", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def fact_tokens(text: str) -> set[str]:
    """Every amount, date, time, duration and reference in *text*."""
    normalized = normalize_text(text)
    found: set[str] = set()
    for pattern in _FACT_PATTERNS:
        found.update(match.group(0).strip() for match in pattern.finditer(normalized))
    return found


def mismatch_tokens(draft: str, operator_text: str) -> list[str]:
    """Facts in the draft that the operator never wrote."""
    allowed = fact_tokens(operator_text)
    return sorted(fact_tokens(draft) - allowed)


def window_open(last_customer_at: int | None, now: int, window_days: int) -> bool:
    """Whether the reply window still allows a customer send."""
    return last_customer_at is not None and now - last_customer_at <= window_days * 86400


def held_payload(kind: str, draft: str) -> str:
    """Store the held draft with the classification that will resume it."""
    return json.dumps({"kind": kind, "draft": draft})


def read_held(body: str) -> tuple[str, str]:
    """Decode a held event body back into ``(kind, draft)``."""
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return RESOLUTION, body
    if not isinstance(value, dict):
        return RESOLUTION, body
    kind = text_field(value, "kind") or RESOLUTION
    return (kind if kind in KINDS else RESOLUTION, text_field(value, "draft"))


def confirmation(contact: Contact, ticket: Ticket, kind: str, draft: str) -> str:
    """The short confirmation that names recipient and ticket state."""
    if kind == RESOLUTION:
        return f"Sent to {contact.display}: {draft} Ticket {ticket.number} done."
    return f"Asked {contact.display}: {draft} Ticket {ticket.number} waiting for them."


async def handle_operator_reply(
    store: TicketStore,
    waha: Transport,
    session: str,
    ticket: Ticket,
    contact: Contact,
    operator_text: str,
    *,
    complete: Completer,
    reply_window_days: int,
    remember: Remember | None = None,
    now: int | None = None,
) -> ReplyOutcome:
    """Classify, draft, verify and deliver one operator reply."""
    moment = store.now() if now is None else now
    kind = await classify_reply(complete, operator_text)
    if kind == AMBIGUOUS:
        return ReplyOutcome(
            confirmation=(
                "I could not tell whether that was a resolution or an instruction. "
                f"Ticket {ticket.number} is unchanged; rephrase it for the customer."
            )
        )
    draft = await draft_message(complete, operator_text, contact.language)
    if not draft:
        return ReplyOutcome(
            confirmation=(
                f"Nothing was sent: the draft was empty. "
                f"Ticket {ticket.number} unchanged."
            ),
            kind=kind,
        )
    mismatches = mismatch_tokens(draft, operator_text)
    if mismatches:
        return ReplyOutcome(
            confirmation=(
                "Not sent: the draft adds values the operator did not give ("
                + ", ".join(mismatches)
                + f").\nDraft: {draft}"
            ),
            kind=kind,
            draft=draft,
            mismatches=mismatches,
        )
    if not window_open(contact.last_inbound_at, moment, reply_window_days):
        store.add_event(
            ticket.id,
            Actor.OPERATOR.value,
            EventKind.HELD.value,
            held_payload(kind, draft),
            at=moment,
        )
        days = (
            (moment - contact.last_inbound_at) // 86400
            if contact.last_inbound_at is not None
            else 0
        )
        return ReplyOutcome(
            confirmation=(
                f"Can't send yet: {contact.display} last wrote {days} days ago. "
                "It goes out when they write again."
            ),
            kind=kind,
            draft=draft,
        )
    if not send_reply(
        store, waha, session, ticket, draft, kind, moment, operator_text=operator_text
    ):
        return ReplyOutcome(
            confirmation=(
                f"Send failed — not confirmed. Ticket {ticket.number} unchanged; "
                "the message did not go out."
            ),
            kind=kind,
            draft=draft,
        )
    if remember is not None:
        await remember(draft)
    return ReplyOutcome(
        confirmation=confirmation(contact, ticket, kind, draft),
        sent=True,
        kind=kind,
        draft=draft,
    )


def send_reply(
    store: TicketStore,
    waha: Transport,
    session: str,
    ticket: Ticket,
    draft: str,
    kind: str,
    moment: int,
    *,
    operator_text: str = "",
) -> bool:
    """Send one confirmed relay and update the ticket only on success."""
    try:
        sent_id = waha.send_text(session, ticket.chat_id, draft)
    except Exception as exc:
        logger.warning("Ticket {n} relay failed: {exc}", n=ticket.number, exc=exc)
        return False
    if not sent_id:
        logger.warning("Ticket {n} relay returned no message id", n=ticket.number)
        return False
    store.add_event(
        ticket.id,
        Actor.OPERATOR.value,
        EventKind.SENT.value,
        draft,
        [sent_id],
        at=moment,
    )
    state = (
        TicketState.DONE.value
        if kind == RESOLUTION
        else TicketState.WAITING_CUSTOMER.value
    )
    done_by = "operator" if kind == RESOLUTION else None
    store.update_ticket(
        ticket.id,
        TicketUpdate(state=state, done_by=done_by),
        actor=Actor.OPERATOR.value,
        kind=EventKind.OPERATOR_REPLY.value,
        body=operator_text or draft,
        at=moment,
    )
    return True


async def deliver_held(
    store: TicketStore,
    waha: Transport,
    session: str,
    ticket: Ticket,
    contact: Contact,
    *,
    remember: Remember | None = None,
    now: int | None = None,
) -> ReplyOutcome | None:
    """Send a held draft when the customer writes again."""
    event = store.last_event(ticket.id, EventKind.HELD.value)
    if event is None:
        return None
    kind, draft = read_held(event.body)
    moment = store.now() if now is None else now
    if not send_reply(store, waha, session, ticket, draft, kind, moment):
        return ReplyOutcome(
            confirmation=f"Held draft for ticket {ticket.number} could not be sent.",
            kind=kind,
            draft=draft,
        )
    if remember is not None:
        await remember(draft)
    return ReplyOutcome(
        confirmation=confirmation(contact, ticket, kind, draft),
        sent=True,
        kind=kind,
        draft=draft,
    )


def decision_fact(ticket: Ticket, operator_text: str, draft: str) -> str:
    """The typed fact text recording the operator's decision."""
    return (
        f"Ticket {ticket.number} decision by the operator: {operator_text.strip()}. "
        f"Customer message sent: {draft.strip()}."
    )
