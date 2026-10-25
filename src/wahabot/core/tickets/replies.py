"""Classify, draft, fact-check and send an operator's ticket reply."""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from loguru import logger

from wahabot.core.answers.service import AnswerService
from wahabot.core.tickets.facts import (
    fact_tokens,
    mismatch_tokens,
    omitted_tokens,
)
from wahabot.core.tickets.language import detect_language
from wahabot.core.tickets.models import (
    Actor,
    Contact,
    DoneBy,
    EventKind,
    Ticket,
    TicketEvent,
    TicketState,
    TicketUpdate,
    text_field,
)
from wahabot.core.tickets.store import TicketStore
from wahabot.core.transport import Transport

__all__ = [
    "fact_tokens",
    "mismatch_tokens",
    "omitted_tokens",
]

Completer = Callable[[str], str]
Remember = Callable[[str], Awaitable[None]]

RESOLUTION = "resolution"
INSTRUCTION = "instruction"
ANSWER = "answer"
AMBIGUOUS = "ambiguous"
KINDS = frozenset({RESOLUTION, INSTRUCTION, ANSWER, AMBIGUOUS})

_CLASSIFY_PROMPT = (
    "Classify one operator reply for a support ticket. Return strict JSON only: "
    '{{"kind": "resolution"|"instruction"|"answer"|"ambiguous"}}. '
    "resolution means the operator did a business action for the customer. "
    "instruction means the operator asks the customer to do or answer something. "
    "answer means the operator explains something, with no business action. "
    "ambiguous means none is clear.\n\nOperator reply:\n{text}"
)

_DRAFT_PROMPT = (
    "Draft one WhatsApp message to a customer {language_hint}. Rephrase only the "
    "operator's words below; do not add facts, amounts, dates, times or promises. "
    "Match the business's plain, human tone. Return only the message text.\n\n"
    "Customer's last message (language and context only, never content):\n"
    "{customer}\n\nOperator words:\n{text}"
)


@dataclass(slots=True)
class ReplyOutcome:
    """The operator-facing result of one routed reply."""

    confirmation: str
    detected: str = ""
    sent: bool = False
    kind: str = AMBIGUOUS
    draft: str = ""
    save_ticket: int | None = None
    mismatches: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def should_offer_save(kind: str) -> bool:
    """Whether an answer-shaped reply may be saved as a published answer."""
    return kind == ANSWER


SAVE_HINT_PREFIX = "Should I answer this myself next time?"
_SAVE_HINT_SUFFIX = re.compile(
    r"\n" + re.escape(SAVE_HINT_PREFIX) + r" Reply save \d+\.\s*$"
)


def save_hint(ticket_number: int) -> str:
    """The one-line save offer, with the literal command preserved."""
    return f"{SAVE_HINT_PREFIX} Reply save {ticket_number}."


def strip_save_hint(confirmation: str) -> str:
    """Remove a trailing save offer so it can be rebuilt after translation."""
    return _SAVE_HINT_SUFFIX.sub("", confirmation).rstrip()


def localized_confirmation(
    outcome: ReplyOutcome,
    language: str,
    translator: Callable[[str, str], str] | None,
) -> str:
    """Translate the human confirmation while keeping the save command literal."""
    base = (
        strip_save_hint(outcome.confirmation)
        if outcome.save_ticket is not None
        else outcome.confirmation
    )
    if language == "en" or translator is None:
        translated = base
    else:
        try:
            translated = translator(base, language)
        except Exception:
            translated = base
    if outcome.save_ticket is None:
        return translated
    hint = SAVE_HINT_PREFIX
    if language != "en" and translator is not None:
        try:
            hint = translator(SAVE_HINT_PREFIX, language) or SAVE_HINT_PREFIX
        except Exception:
            hint = SAVE_HINT_PREFIX
    return f"{translated}\n{hint} Reply save {outcome.save_ticket}."


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


async def draft_message(
    complete: Completer, operator_text: str, language: str, customer_words: str = ""
) -> str:
    """Draft the customer message from the operator's words only."""
    hint = (
        f"in {language}"
        if language
        else "in the customer's language (English if unclear)"
    )
    raw = await asyncio.to_thread(
        complete,
        _DRAFT_PROMPT.format(
            language_hint=hint,
            text=operator_text,
            customer=customer_words or "(no message available)",
        ),
    )
    return raw.strip().strip('"')


def window_open(last_customer_at: int | None, now: int, window_days: int) -> bool:
    """Whether the reply window still allows a customer send."""
    return last_customer_at is not None and now - last_customer_at <= window_days * 86400


def customer_words(store: TicketStore, ticket: Ticket) -> str:
    """The newest customer message on a ticket, or its summary."""
    event = store.last_event(ticket.id, EventKind.CUSTOMER_MESSAGE.value)
    return event.body if event is not None else ticket.summary


def held_payload(kind: str, draft: str, question: str = "", language: str = "") -> str:
    """Store the held draft with the classification that will resume it."""
    return json.dumps(
        {"kind": kind, "draft": draft, "question": question, "language": language}
    )


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


def held_question(body: str, fallback: str) -> str:
    """The customer message a held draft answered, or the current one."""
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return fallback
    if not isinstance(value, dict):
        return fallback
    return text_field(value, "question") or fallback


def held_language(body: str) -> str:
    """The operator's detected language when the draft was held."""
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return ""
    if not isinstance(value, dict):
        return ""
    return text_field(value, "language")


def confirmation(contact: Contact, ticket: Ticket, kind: str, draft: str) -> str:
    """The short confirmation that names recipient and ticket state."""
    if kind in (RESOLUTION, ANSWER):
        return f"Sent to {contact.display}: {draft} Ticket {ticket.number} done."
    return f"Asked {contact.display}: {draft} Ticket {ticket.number} waiting for them."


def fact_check_outcome(kind: str, draft: str, operator_text: str) -> ReplyOutcome | None:
    """The refusal outcome when the draft changes the operator's facts."""
    mismatches = mismatch_tokens(draft, operator_text)
    missing = omitted_tokens(draft, operator_text)
    if not mismatches and not missing:
        return None
    changes: list[str] = []
    if mismatches:
        changes.append(
            "adds values the operator did not give (" + ", ".join(mismatches) + ")"
        )
    if missing:
        changes.append("drops values the operator gave (" + ", ".join(missing) + ")")
    return ReplyOutcome(
        confirmation=(
            "Not sent: the draft " + " and ".join(changes) + f".\nDraft: {draft}"
        ),
        kind=kind,
        draft=draft,
        mismatches=mismatches,
        missing=missing,
    )


def hold_outcome(
    store: TicketStore,
    ticket: Ticket,
    contact: Contact,
    kind: str,
    draft: str,
    moment: int,
    reply_window_days: int,
    *,
    detected: str = "",
) -> ReplyOutcome | None:
    """Hold a draft when the customer has been silent past the window."""
    if window_open(contact.last_inbound_at, moment, reply_window_days):
        return None
    store.add_event(
        ticket.id,
        Actor.OPERATOR.value,
        EventKind.HELD.value,
        held_payload(kind, draft, customer_words(store, ticket), detected),
        at=moment,
    )
    if contact.last_inbound_at is None:
        waited = "has not written yet"
    else:
        days = (moment - contact.last_inbound_at) // 86400
        waited = f"last wrote {days} days ago"
    return ReplyOutcome(
        confirmation=(
            f"Can't send yet: {contact.display} {waited}. "
            "It goes out when they write again."
        ),
        kind=kind,
        draft=draft,
        detected=detected,
    )


async def relay_outcome(
    store: TicketStore,
    waha: Transport,
    session: str,
    ticket: Ticket,
    contact: Contact,
    kind: str,
    draft: str,
    operator_text: str,
    moment: int,
    remember: Remember | None,
) -> ReplyOutcome:
    """Send one confirmed relay and build its outcome."""
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
    answer_service: AnswerService | None = None,
    now: int | None = None,
) -> ReplyOutcome:
    """Classify, draft, verify and deliver one operator reply."""
    moment = store.now() if now is None else now
    detected = detect_language(operator_text)
    kind = await classify_reply(complete, operator_text)
    if kind == AMBIGUOUS:
        return ReplyOutcome(
            confirmation=(
                "I could not tell whether that was a resolution or an instruction. "
                f"Ticket {ticket.number} is unchanged; rephrase it for the customer."
            ),
            detected=detected,
        )
    draft = await draft_message(
        complete, operator_text, contact.language, customer_words(store, ticket)
    )
    if not draft:
        return ReplyOutcome(
            confirmation=(
                f"Nothing was sent: the draft was empty. "
                f"Ticket {ticket.number} unchanged."
            ),
            kind=kind,
            detected=detected,
        )
    refused = fact_check_outcome(kind, draft, operator_text)
    if refused is not None:
        refused.detected = detected
        return refused
    held = hold_outcome(
        store,
        ticket,
        contact,
        kind,
        draft,
        moment,
        reply_window_days,
        detected=detected,
    )
    if held is not None:
        return held
    outcome = await relay_outcome(
        store,
        waha,
        session,
        ticket,
        contact,
        kind,
        draft,
        operator_text,
        moment,
        remember,
    )
    outcome.detected = detected
    if outcome.sent and should_offer_save(kind) and answer_service is not None:
        offer_answer_candidate(answer_service, store, ticket, outcome)
    return outcome


def offer_answer_candidate(
    service: AnswerService,
    store: TicketStore,
    ticket: Ticket,
    outcome: ReplyOutcome,
    *,
    question: str | None = None,
) -> None:
    """Store the unsaved candidate and add the one-line save offer."""
    try:
        service.offer_candidate(
            ticket.id,
            customer_words(store, ticket) if question is None else question,
            outcome.draft,
        )
    except Exception as exc:
        logger.warning("Answer candidate was not stored: {exc}", exc=exc)
        return
    outcome.save_ticket = ticket.number
    outcome.confirmation = f"{outcome.confirmation}\n{save_hint(ticket.number)}"


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
    resolved = kind in (RESOLUTION, ANSWER)
    state = TicketState.DONE.value if resolved else TicketState.WAITING_CUSTOMER.value
    done_by = DoneBy.OPERATOR.value if resolved else None
    store.update_ticket(
        ticket.id,
        TicketUpdate(state=state, done_by=done_by),
        actor=Actor.OPERATOR.value,
        kind=EventKind.OPERATOR_REPLY.value,
        body=operator_text or draft,
        at=moment,
    )
    store.cancel_notices(ticket.id, moment)
    return True


def pending_held(store: TicketStore, ticket: Ticket) -> list[TicketEvent]:
    """Held drafts newer than the last confirmed relay, oldest first."""
    sent = store.last_event(ticket.id, EventKind.SENT.value)
    sent_key = (sent.at, sent.id) if sent is not None else None
    return [
        event
        for event in store.events(ticket.id)
        if event.kind == EventKind.HELD.value
        and (sent_key is None or (event.at, event.id) > sent_key)
    ]


async def deliver_held_event(
    store: TicketStore,
    waha: Transport,
    session: str,
    ticket: Ticket,
    contact: Contact,
    event: TicketEvent,
    moment: int,
    remember: Remember | None,
    answer_service: AnswerService | None = None,
) -> ReplyOutcome:
    """Deliver one held draft and report its outcome."""
    kind, draft = read_held(event.body)
    if not send_reply(store, waha, session, ticket, draft, kind, moment):
        return ReplyOutcome(
            confirmation=f"Held draft for ticket {ticket.number} could not be sent.",
            kind=kind,
            draft=draft,
            detected=held_language(event.body),
        )
    if remember is not None:
        await remember(draft)
    outcome = ReplyOutcome(
        confirmation=confirmation(contact, ticket, kind, draft),
        sent=True,
        kind=kind,
        draft=draft,
        detected=held_language(event.body),
    )
    if should_offer_save(kind) and answer_service is not None:
        offer_answer_candidate(
            answer_service,
            store,
            ticket,
            outcome,
            question=held_question(event.body, customer_words(store, ticket)),
        )
    return outcome


async def deliver_held(
    store: TicketStore,
    waha: Transport,
    session: str,
    ticket: Ticket,
    contact: Contact,
    *,
    remember: Remember | None = None,
    answer_service: AnswerService | None = None,
    now: int | None = None,
) -> ReplyOutcome | None:
    """Send every held draft that followed the last confirmed relay."""
    moment = store.now() if now is None else now
    pending = pending_held(store, ticket)
    if not pending:
        return None
    outcome: ReplyOutcome | None = None
    for event in pending:
        outcome = await deliver_held_event(
            store,
            waha,
            session,
            ticket,
            contact,
            event,
            moment,
            remember,
            answer_service,
        )
        if not outcome.sent:
            return outcome
    return outcome


def decision_fact(ticket: Ticket, operator_text: str, draft: str) -> str:
    """The typed fact text recording the operator's decision."""
    return (
        f"Ticket {ticket.number} decision by the operator: {operator_text.strip()}. "
        f"Customer message sent: {draft.strip()}."
    )
