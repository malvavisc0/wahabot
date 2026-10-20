"""The pinned state block: authoritative facts assembled by code per turn.

The prompt's memory splits into two kinds of information with opposite
rules: *pinned state* — authoritative, assembled by code, never
retrieved — and *recall* — best-effort, a lead, never evidence. This
module renders the pinned side: a plain-text block the runtime
assembles at the start of every turn, restructured from what the
notebook already injected (notebook facts, operator policies) plus the
session rules.

The block is code-injected into the system message by
``FunctionCallingAgentWorkflow.chat_history``, so history trimming can
never evict it — the same guarantee a priority-0 static memory block
gives, without churning the ``ChatMemoryBuffer`` flow.

Every source is fail-soft — a notebook that is down renders the block
without that section, never a blocked run. Notebook facts are the
bounded part: whole notes, relevance order (the notebook's
newest-first read order) preserved, selected within a token cap the
way ``bounded_notes`` selects for the notebook tool. Policies and
rules are authoritative and render in full — bounded at their sources
— with today's budget split intact: an authoritative part that cannot
fit the run's reserved budget raises :class:`ContextBudgetError`,
notebook facts are dropped oldest-first instead.
"""

from collections.abc import Callable, Sequence
from typing import Any

from loguru import logger

from wahabot.ai.history import ContextBudgetError
from wahabot.ai.records import record_serials
from wahabot.core.notebook import (
    KIND_RULE,
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
    "Runtime-assembled for this turn; treat it as fact, not memory. "
    "`Session rules` and `Chat rules` are exact and obeyed; "
    "`Operator policies` and `Notebook facts` are the operator's and your "
    "own notes; the recap and recalled slices are leads, never evidence."
)
POLICIES_HEADING = "## Operator policies\n\n"
CHAT_RULES_HEADING = "## Chat rules\n\n"
RECAP_HEADING = "## Conversation recap\n\n"
NOTES_HEADING = "## Notebook facts\n\n"
RULES_HEADING = "## Session rules\n\n"
RECALL_HEADING = "## Recalled from earlier in this thread\n\n"
RECALL_PREAMBLE = (
    "Leads, verify before relying on them: older transcript slices, not "
    + "authoritative facts.\n\n"
)


def render_pinned_block(
    *,
    policies: Sequence[Note] = (),
    chat_rules: Sequence[Note] = (),
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
    preserved); the authoritative sections (policies, chat rules, rules)
    render in full. When *prompt_budget* is given, the block must leave the run's
    reserved room — the exact split the old notebook injection had: an
    authoritative part that cannot fit raises
    :class:`ContextBudgetError` (permanent for the triggering message),
    while notebook facts are dropped oldest-first instead of ever
    raising. The *recap* (Tier 1, ``core/learned.py``) sits in between:
    it renders whole and is *dropped* when it cannot fit — a summary
    that crowds the reserved room is worth less than the room, and its
    content is also the newest thing in the FIFO.

    *recall* renders last as flattened lead bullets under
    ``## Recalled from earlier in this thread``. It is never authoritative
    and is dropped before the recap when the reserved room runs out, so
    retrieval can never raise or
    crowd out pinned state.

    Returns "" when every source is empty, so an empty block never
    rides the prompt.
    """
    notes = merge_notes(required_notes, notes)
    sections = authoritative_sections(policies, chat_rules, rules, recap)
    base = authoritative_sections(policies, chat_rules, rules, "")
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
    chat. Every read is fail-soft: a store that is down logs once and
    renders the block without that section, never a blocked run.
    Cancellation still propagates: reads run under
    ``notebook_operation``.
    """
    policies: list[Note] = []
    chat_rules: list[Note] = []
    notes: list[Note] = []
    if notebook is not None:
        policies = (
            await _read(
                notebook.read, "operator policies", session, SESSION_NOTEBOOK_CHAT
            )
            or []
        )
        stored = await _read(notebook.read, "notebook facts", session, chat_id) or []
        # A chat's `rule` notes are authoritative for that chat; facts and
        # lessons are leads. Rules are read under their own window so a
        # burst of facts can never push a rule out of the block.
        chat_rules = (
            await _read(notebook.read, "chat rules", session, chat_id, kind=KIND_RULE)
            or []
        )
        notes = [note for note in stored if note.kind != KIND_RULE]
    return render_pinned_block(
        policies=policies,
        chat_rules=chat_rules,
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


async def _read[T](
    operation: Callable[..., T], what: str, *args: Any, **kwargs: Any
) -> T | None:
    """One fail-soft store read; None when the store is unavailable.

    ``CancelledError`` is not an ``Exception`` and still propagates, so
    a canceled run is never misread as an outage.
    """
    try:
        return await notebook_operation(operation, *args, **kwargs)
    except Exception:
        logger.warning(
            "Pinned state: {what} unavailable; continuing without it", what=what
        )
        return None


def authoritative_sections(
    policies: Sequence[Note], chat_rules: Sequence[Note], rules: str, recap: str
) -> list[str]:
    """The block's authoritative sections, empty ones omitted.

    Precedence the prompt states: session rules > operator policies >
    chat rules > facts. Policies and the chat's rules render beside
    each other, before the recap (the conversation's flow) and the
    session rules that gate behavior. Every line is flattened to a
    bullet so a note or summary line can never forge a section heading.
    """
    sections = authority_notes(policies, chat_rules)
    recap_block = recap_section(recap)
    if recap_block:
        sections.append(recap_block)
    if rules.strip():
        sections.append(RULES_HEADING + rules.strip())
    return sections


def authority_notes(policies: Sequence[Note], chat_rules: Sequence[Note]) -> list[str]:
    """The operator-policy and chat-rule sections, empty ones omitted."""
    sections: list[str] = []
    for heading, notes in (
        (POLICIES_HEADING, policies),
        (CHAT_RULES_HEADING, chat_rules),
    ):
        if notes:
            sections.append(heading + "\n".join(note_line(note) for note in notes))
    return sections


def recap_section(recap: str) -> str:
    """The recap as flat bullet lines, or "" when empty."""
    lines = ["- " + " ".join(line.split()) for line in recap.splitlines() if line.strip()]
    if not lines:
        return ""
    return RECAP_HEADING + "\n".join(lines)


def dedupe_recall(slices: Sequence[str], live_texts: Sequence[str]) -> list[str]:
    """Drop recalled slices the live window already carries.

    A slice that names a serial of a live record in the turn is the same
    conversation the model can already read verbatim; a slice whose whole
    text is a live line is a duplicate. Partial overlap is kept — a slice
    may carry older neighbours the window does not. Only the serials of
    live record heads count: a bare ``quote=#m<n>`` / ``to: #m<n>``
    reference inside the turn is a pointer, not a verbatim copy — dropping
    a slice over a pointer would delete the only full text of #m<n>.
    """
    live = " ".join(" ".join(text.split()) for text in live_texts if text)
    live_serials = {serial for text in live_texts for serial in record_serials(text)}
    kept: list[str] = []
    for item in slices:
        normalized = " ".join(item.split())
        if not normalized:
            continue
        if any(f"#m{serial}" in normalized for serial in live_serials):
            continue
        if normalized in live:
            continue
        kept.append(item)
    return kept


def recall_section(slices: Sequence[str]) -> str:
    """Retrieved slices as flat lead bullets, or "" when empty.

    One bullet per slice: every line is flattened so a retrieved turn can
    never forge a section heading of its own.
    """
    lines = ["- " + " ".join(item.split()) for item in slices if item.strip()]
    if not lines:
        return ""
    return RECALL_HEADING + RECALL_PREAMBLE + "\n".join(lines)


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
