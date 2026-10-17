"""The pinned state block: authoritative facts assembled by code per turn.

The prompt's memory splits into two kinds of information with opposite
rules: *pinned state* — authoritative, assembled by code, never
retrieved — and *recall* — best-effort, a lead, never evidence. This
module renders the pinned side: a plain-text block the runtime
assembles at the start of every turn, in the ticket-shaped field
layout — contact, open tickets (state, summary, details collected,
what is still missing), latest events, session rules — restructured
from what the notebook already injected (notebook facts, operator
policies).

The block is code-injected into the system message by
``FunctionCallingAgentWorkflow.chat_history``, so history trimming can
never evict it — the same guarantee a priority-0 static memory block
gives, without churning the ``ChatMemoryBuffer`` flow.

The support-store fields (contact, tickets, events) come from an
optional :class:`SupportStateProvider`: ``None`` until the support
store exists, and never stub data. Every source is fail-soft — a
notebook or provider that is down renders the block without that
section, never a blocked run. Notebook facts are the bounded part:
whole notes, relevance order (the notebook's newest-first read order)
preserved, selected within a token cap the way ``bounded_notes``
selects for the notebook tool. Policies, support fields, and rules are
authoritative and render in full — bounded at their sources — with
today's budget split intact: an authoritative part that cannot fit the
run's reserved budget raises :class:`ContextBudgetError`, notebook
facts are dropped oldest-first instead.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from loguru import logger

from wahabot.ai.history import ContextBudgetError
from wahabot.core.notebook import (
    NOTEBOOK_SNAPSHOT_TOKEN_LIMIT,
    SESSION_NOTEBOOK_CHAT,
    Note,
    Notebook,
    notebook_operation,
)
from wahabot.core.tokens import text_tokens

#: Whole-block token cap for the notebook-facts section — the same
#: snapshot budget the notebook tool's reads use (``NOTEBOOK_TOKEN_LIMIT``'s
#: role for the old JSON snapshot), so the two surfaces cannot drift.
PINNED_STATE_TOKEN_LIMIT = NOTEBOOK_SNAPSHOT_TOKEN_LIMIT

PINNED_STATE_HEADING = "# Pinned State"
PINNED_PREAMBLE = (
    "Runtime-assembled facts for this conversation, exact as of this "
    "turn. A reply's facts (names, numbers, states, promises, "
    "deadlines) come from this block, never from guesswork, and "
    "nothing listed here may be asked for again."
)
CONTACT_HEADING = "## Contact\n\n"
TICKETS_HEADING = "## Open tickets\n\n"
EVENTS_HEADING = "## Latest events\n\n"
POLICIES_HEADING = "## Operator policies\n\n"
RECAP_HEADING = "## Conversation recap\n\n"
NOTES_HEADING = "## Notebook facts\n\n"
RULES_HEADING = "## Session rules\n\n"
RECALL_HEADING = "## Recalled from earlier in this thread\n\n"
RECALL_PREAMBLE = (
    "Leads, verify before relying on them: older transcript slices, not "
    + "authoritative facts.\n\n"
)


@dataclass(frozen=True, slots=True)
class Contact:
    """The customer record: the customer details."""

    name: str = ""
    phone: str = ""
    language: str = ""
    external_ref: str = ""
    paused_until: str = ""


@dataclass(frozen=True, slots=True)
class Ticket:
    """One open or recent ticket in the ticket field shape."""

    number: str
    state: str
    urgent: bool = False
    summary: str = ""
    details_collected: str = ""
    still_missing: str = ""
    done_by: str = ""
    updated: str = ""


@dataclass(frozen=True, slots=True)
class SupportEvent:
    """One internal event: what happened, by whom, when.

    The provider lists events chronologically (newest last), so the
    latest decision — the operator's
    last instruction — sits closest to the sections that follow.
    """

    at: str
    actor: str
    text: str


@dataclass(frozen=True, slots=True)
class SupportState:
    """The support store's view of one chat: contact, tickets, events."""

    contact: Contact | None = None
    tickets: tuple[Ticket, ...] = ()
    events: tuple[SupportEvent, ...] = ()


class SupportStateProvider(Protocol):
    """Reads ticket-shaped support state for one chat from its store.

    The support store implements this when it exists; until then no
    provider is wired and the block
    renders without the support sections. An implementation bounds what
    it returns (a handful of open tickets and latest events) and
    finishes its SQLite work before returning — the assembler runs it
    under ``notebook_operation`` so a canceled run waits for it
    instead of dropping it mid-read.
    """

    def support_state(self, session: str, chat_id: str) -> SupportState:
        """The ticket-shaped state for one chat."""
        ...


def render_pinned_block(
    *,
    support: SupportState | None = None,
    policies: Sequence[Note] = (),
    notes: Sequence[Note] = (),
    required_notes: Sequence[Note] = (),
    rules: str = "",
    recap: str = "",
    recall: Sequence[str] = (),
    token_budget: int,
    tokenizer: Callable[[str], Sequence[Any]] | None = None,
    prompt_prefix: str = "",
    prompt_budget: int | None = None,
) -> str:
    """Render the pinned state block: plain text, deterministic, bounded.

    *token_budget* caps the notebook facts (whole notes, order
    preserved); the authoritative sections (support fields, policies,
    rules) render in full. When *prompt_budget* is given, the block
    must leave the run's reserved room — the exact split the old
    notebook injection had: an authoritative part that cannot fit
    raises :class:`ContextBudgetError` (permanent for the triggering
    message), while notebook facts are dropped oldest-first instead of
    ever raising. The *recap* (Tier 1, ``core/learned.py``) sits in
    between: it renders whole and is *dropped* when it cannot fit —
    a summary that crowds the reserved room is worth less than the
    room, and its content is also the newest thing in the FIFO.

    *recall* renders last as flattened lead bullets under
    ``## Recalled from earlier in this thread``. It is never authoritative
    and is dropped before the recap when the reserved room runs out, so
    retrieval can never raise or
    crowd out pinned state.

    Returns "" when every source is empty, so an empty block never
    rides the prompt.
    """
    notes = merge_notes(required_notes, notes)
    sections = authoritative_sections(support, policies, rules, recap)
    base = authoritative_sections(support, policies, rules, "")
    recall_block = recall_section(recall)
    if is_empty(sections, notes, recall_block):
        return ""
    sections, recall_block = fit_droppable(
        sections,
        base,
        recall_block,
        recap_section(recap),
        tokenizer,
        prompt_prefix,
        prompt_budget,
    )
    if is_empty(sections, notes, recall_block):
        return ""
    authoritative = joined_sections(sections)
    selected = select_notes(
        notes,
        token_budget,
        tokenizer,
        prompt_prefix,
        authoritative,
        prompt_budget,
    )
    body = assemble_block(authoritative, selected, recall_block)
    return fit_final(
        body,
        authoritative,
        selected,
        recall_block,
        tokenizer,
        prompt_prefix,
        prompt_budget,
    )


def is_empty(sections: Sequence[str], notes: Sequence[Note], recall_section: str) -> bool:
    """Whether nothing (not even a lead) is left to render."""
    return not sections and not notes and not recall_section


def select_notes(
    notes: Sequence[Note],
    token_budget: int,
    tokenizer: Callable[[str], Sequence[Any]] | None,
    prompt_prefix: str,
    authoritative: str,
    prompt_budget: int | None,
) -> list[Note]:
    """Bound the notebook facts by their own cap and the reserved room."""
    selected = fit_notes(notes, token_budget, tokenizer) if token_budget > 0 else []
    if prompt_budget is not None:
        selected = fit_notes(
            selected,
            prompt_budget,
            tokenizer,
            prefix=prompt_prefix + authoritative + "\n\n",
        )
    return selected


def fit_final(
    body: str,
    authoritative: str,
    selected: Sequence[Note],
    recall_section: str,
    tokenizer: Callable[[str], Sequence[Any]] | None,
    prompt_prefix: str,
    prompt_budget: int | None,
) -> str:
    """Drop lead-only recall if notes crowded it, then enforce the budget."""
    if (
        prompt_budget is None
        or text_tokens(prompt_prefix + body, tokenizer) <= prompt_budget
    ):
        return body
    if recall_section:
        logger.info("Recall does not fit the reserved budget; rendering without it")
        body = assemble_block(authoritative, selected, "")
    if text_tokens(prompt_prefix + body, tokenizer) > prompt_budget:
        raise ContextBudgetError("Pinned state and current query exceed token budget")
    return body


def joined_sections(sections: Sequence[str]) -> str:
    """Join the heading, preamble, and authoritative sections."""
    return "\n\n".join((PINNED_STATE_HEADING, PINNED_PREAMBLE, *sections))


def fit_droppable(
    sections: list[str],
    base: list[str],
    recall_section: str,
    recap_section: str,
    tokenizer: Callable[[str], Sequence[Any]] | None,
    prompt_prefix: str,
    prompt_budget: int | None,
) -> tuple[list[str], str]:
    """Drop the lead-only recall first, then the recap, when over budget."""
    if prompt_budget is None:
        return sections, recall_section
    if over_budget(
        prompt_prefix,
        joined_sections(sections) + section_tail(recall_section),
        tokenizer,
        prompt_budget,
    ):
        if recall_section:
            logger.info("Recall does not fit the reserved budget; rendering without it")
        recall_section = ""
    if (
        over_budget(prompt_prefix, joined_sections(sections), tokenizer, prompt_budget)
        and recap_section
    ):
        logger.info("Recap does not fit the reserved budget; rendering without it")
        sections = base
    return sections, recall_section


def section_tail(section: str) -> str:
    """*section* as a joined tail, or "" when there is none."""
    return f"\n\n{section}" if section else ""


def assemble_block(authoritative: str, notes: Sequence[Note], recall_section: str) -> str:
    """Join the authoritative block, bounded notes, and lead-only recall."""
    parts = [authoritative]
    if notes:
        parts.append(notes_text(notes))
    if recall_section:
        parts.append(recall_section)
    return "\n\n".join(part for part in parts if part)


def over_budget(
    prefix: str, text: str, tokenizer: Callable[[str], Sequence[Any]] | None, budget: int
) -> bool:
    """Whether the prefix plus text exceeds the token budget."""
    return text_tokens(prefix + text, tokenizer) > budget


async def pinned_state_block(
    *,
    notebook: Notebook | None,
    provider: SupportStateProvider | None,
    session: str,
    chat_id: str,
    token_budget: int,
    tokenizer: Callable[[str], Sequence[Any]] | None,
    prompt_prefix: str = "",
    prompt_budget: int | None = None,
    rules: str = "",
    recap: str = "",
    recall: Sequence[str] = (),
    required_notes: Sequence[Note] = (),
) -> str:
    """Assemble this run's block: fail-soft reads, then one render.

    Reads the same scopes the old notebook injection read — operator
    policies from the session notebook, chat facts from the current
    chat — plus the optional support provider. Every read is
    fail-soft: a store that is down logs once and renders the block
    without that section, never a blocked run. Cancellation still
    propagates: reads run under ``notebook_operation``.
    """
    support: SupportState | None = None
    if provider is not None:
        support = await _read(provider.support_state, "support state", session, chat_id)
    policies: list[Note] = []
    notes: list[Note] = []
    if notebook is not None:
        policies = (
            await _read(
                notebook.read, "operator policies", session, SESSION_NOTEBOOK_CHAT
            )
            or []
        )
        notes = await _read(notebook.read, "notebook facts", session, chat_id) or []
    return render_pinned_block(
        support=support,
        policies=policies,
        notes=notes,
        rules=rules,
        recap=recap,
        recall=recall,
        required_notes=required_notes,
        token_budget=token_budget,
        tokenizer=tokenizer,
        prompt_prefix=prompt_prefix,
        prompt_budget=prompt_budget,
    )


async def _read[T](operation: Callable[..., T], what: str, *args: Any) -> T | None:
    """One fail-soft store read; None when the store is unavailable.

    ``CancelledError`` is not an ``Exception`` and still propagates, so
    a canceled run is never misread as an outage.
    """
    try:
        return await notebook_operation(operation, *args)
    except Exception:
        logger.warning(
            "Pinned state: {what} unavailable; continuing without it", what=what
        )
        return None


def authoritative_sections(
    support: SupportState | None, policies: Sequence[Note], rules: str, recap: str
) -> list[str]:
    """The block's authoritative sections, empty ones omitted.

    The recap (Tier 1) renders between the policies and the rules: the
    conversation's flow after the durable policies, before the session
    rules that gate behavior. Its lines are flattened to bullets so a
    summary line can never forge the block's section headings.
    """
    sections: list[str] = []
    if support is not None:
        sections.extend(
            section
            for section in (
                contact_section(support.contact),
                tickets_section(support.tickets),
                events_section(support.events),
            )
            if section
        )
    if policies:
        sections.append(
            POLICIES_HEADING + "\n".join(note_line(note) for note in policies)
        )
    recap_block = recap_section(recap)
    if recap_block:
        sections.append(recap_block)
    if rules.strip():
        sections.append(RULES_HEADING + rules.strip())
    return sections


def recap_section(recap: str) -> str:
    """The recap as flat bullet lines, or "" when empty."""
    lines = ["- " + " ".join(line.split()) for line in recap.splitlines() if line.strip()]
    if not lines:
        return ""
    return RECAP_HEADING + "\n".join(lines)


def recall_section(slices: Sequence[str]) -> str:
    """Retrieved slices as flat lead bullets, or "" when empty.

    One bullet per slice: every line is flattened so a retrieved turn can
    never forge a section heading of its own.
    """
    lines = ["- " + " ".join(item.split()) for item in slices if item.strip()]
    if not lines:
        return ""
    return RECALL_HEADING + RECALL_PREAMBLE + "\n".join(lines)


def contact_section(contact: Contact | None) -> str:
    """The contact fields as a section, or "" when empty."""
    if contact is None:
        return ""
    lines = [
        f"{label}: {value.strip()}"
        for label, value in (
            ("Name", contact.name),
            ("Phone", contact.phone),
            ("Language", contact.language),
            ("External ref", contact.external_ref),
            ("Paused until", contact.paused_until),
        )
        if value.strip()
    ]
    if not lines:
        return ""
    return CONTACT_HEADING + "\n".join(lines)


def tickets_section(tickets: Sequence[Ticket]) -> str:
    """Open and recent tickets as a section, or "" when empty."""
    if not tickets:
        return ""
    return TICKETS_HEADING + "\n\n".join(ticket_block(ticket) for ticket in tickets)


def ticket_block(ticket: Ticket) -> str:
    """One ticket as a bullet block."""
    head = f"Ticket {ticket.number.strip()} — {ticket.state.strip()}"
    if ticket.urgent:
        head += " — urgent"
    lines = [head]
    for label, value in (
        ("Summary", ticket.summary),
        ("Details collected", ticket.details_collected),
        ("Still missing", ticket.still_missing),
        ("Done by", ticket.done_by),
        ("Updated", ticket.updated),
    ):
        if value.strip():
            lines.append(f"  {label}: {value.strip()}")
    return "\n".join(lines)


def events_section(events: Sequence[SupportEvent]) -> str:
    """Latest support events as a section, or "" when empty."""
    if not events:
        return ""
    lines = [
        f"{event.at} — {event.actor}: {event.text.strip()}"
        for event in events
        if event.text.strip()
    ]
    if not lines:
        return ""
    return EVENTS_HEADING + "\n".join(lines)


def note_line(note: Note) -> str:
    """One flat ``- key: text`` line for a notebook fact or policy.

    A note is data, never structure: its key and text are flattened to
    one line so a stored fact can never forge the block's section
    headings by starting a line with ``#``.
    """
    key = " ".join(note.key.split())
    text = " ".join(note.text.split())
    return f"- {key}: {text}"


def notes_text(notes: Sequence[Note]) -> str:
    """The notebook facts as flat bullet lines."""
    return NOTES_HEADING + "\n".join(note_line(note) for note in notes)


def merge_notes(required: Sequence[Note], notes: Sequence[Note]) -> list[Note]:
    """Required notes first, then the notebook's notes, one per key."""
    seen: set[str] = set()
    merged: list[Note] = []
    for note in (*required, *notes):
        if note.key in seen:
            continue
        seen.add(note.key)
        merged.append(note)
    return merged


def fit_notes(
    notes: Sequence[Note],
    token_budget: int,
    tokenizer: Callable[[str], Sequence[Any]] | None,
    prefix: str = NOTES_HEADING,
) -> list[Note]:
    """Whole notes whose rendered section, plus *prefix*, fits the budget.

    The plain-text sibling of ``bounded_notes`` (the notebook tool's
    whole-item, order-preserving selection), measuring the ``- key:
    text`` lines the block actually spends: relevance order is
    preserved and a note is never cut mid-text.
    """
    selected: list[Note] = []
    for note in notes:
        candidate = prefix + notes_text([*selected, note])
        if text_tokens(candidate, tokenizer) <= token_budget:
            selected.append(note)
    return selected
