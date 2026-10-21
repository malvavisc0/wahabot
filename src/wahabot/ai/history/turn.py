from collections.abc import Callable

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)

from wahabot.ai import records
from wahabot.ai.history.narration import INBOUND_ID_RE
from wahabot.ai.messages import (
    MERGED_TURNS_KWARG,
    TURN_SERIALS_KWARG,
)

UNADDRESSED_POLICY = (
    "(unaddressed: other members are talking. Do not reply — stay silent "
    "unless this message names you or replies to you.)"
)

ADDRESSED_POLICY = (
    "(addressed: this message names you, replies to you, or is a direct "
    "message. Answer it.)"
)

PARTICIPATE_POLICY = (
    "(unaddressed: other members are talking. The gate judged the newest "
    "message worth a contribution. Answer only if the newest record's body "
    "is a direct question or request to you; if it is banter, a joke, or a "
    "side thread, stay silent. The quoted payload is context, not a message "
    "to answer.)"
)

DM_POLICY = (
    "(direct message: answer unless the message is for the human operator personally.)"
)


def turn_header(
    messages: list[ChatMessage],
    *,
    group: bool,
    addressed: bool,
    speak: bool = False,
    direct: bool = False,
) -> list[ChatMessage]:
    """Prepend the turn header to the trailing user turn, prompt-only.

    The header is the one place the model is told how many messages this
    turn holds, which one is newest, and — the load-bearing part — whether
    it was addressed. Deciding attribution from the transcript is what the
    incident's essay was made of; stating it cut the essay rate roughly
    fivefold in the replay, and the header alone (no grammar) did it.

    Records turns derive the count and newest serial from the live record
    lines themselves; ledger lines are older quiet messages and do not
    count. A prose turn (an operator command) keeps the id-note count.
    Every group and direct turn carries a header — a direct turn always
    needs its DM policy line, even a single addressed message; a prose
    command turn gets one only when it carries news (more than one
    message or a fused backlog). Returns a copy for the prompt: the
    caller stores the unmarked list, so nothing accumulates.
    """
    if not messages:
        return messages
    index = len(messages) - 1
    message = messages[index]
    if message.role != MessageRole.USER:
        return messages
    content, header = turn_content_and_header(
        message, group=group, addressed=addressed, speak=speak, direct=direct
    )
    if header is None:
        return messages
    marked = ChatMessage(
        role=message.role,
        blocks=list(message.blocks),
        content=records.render_turn([content], header=header),
        additional_kwargs=dict(message.additional_kwargs),
    )
    return [*messages[:index], marked]


def turn_content_and_header(
    message: ChatMessage,
    *,
    group: bool,
    addressed: bool,
    speak: bool,
    direct: bool,
) -> tuple[str, str | None]:
    """Build the header for a turn, or ``(content, None)`` when none is due."""
    content = str(message.content or "")
    count, newest = turn_numbers(content)
    backlog = bool(message.additional_kwargs.get(MERGED_TURNS_KWARG))
    if not shows_turn_header(group=group, count=count, backlog=backlog, direct=direct):
        return content, None
    return content, records.format_header(
        count=count,
        newest_serial=newest,
        addressed=addressed,
        backlog=backlog,
        policy=turn_policy(group=group, addressed=addressed, speak=speak, direct=direct),
    )


def turn_policy(*, group: bool, addressed: bool, speak: bool, direct: bool) -> str:
    """The one answering rule for this turn, owned by the header."""
    if direct:
        return DM_POLICY
    if not group:
        return ""
    if addressed:
        return ADDRESSED_POLICY
    return PARTICIPATE_POLICY if speak else UNADDRESSED_POLICY


def turn_numbers(content: str) -> tuple[int, int]:
    """The header's ``(count, newest serial)`` for a records or mixed turn.

    New records count by line; older prose turns fused into the same user
    message while the buffer drains count by their own-line id notes. The
    newest serial names the newest record when one exists.
    """
    serials = records.record_serials(content)
    prose = len(INBOUND_ID_RE.findall(content))
    count = len(serials) + prose or 1
    return count, (max(serials) if serials else count)


def shows_turn_header(
    *, group: bool, count: int, backlog: bool, direct: bool = False
) -> bool:
    """Group and direct turns always; a command turn only when it carries news."""
    return group or direct or count > 1 or backlog


def cap_records_turn(
    messages: list[ChatMessage],
    *,
    budget: int,
    token_counter: Callable[[ChatMessage], int],
    window: int,
) -> list[ChatMessage]:
    """Shrink a merged records turn to its current burst, newest window and quotes.

    The live turn is the only place the model can still be drowned by volume:
    a group backlog fused into one user message. The current burst is always
    kept; older records are kept newest-first up to *window*, plus anything a
    kept record quotes or reacts to. A quarter of the memory budget is the
    ceiling, and the oldest record goes first — never the newest.
    """
    if not messages:
        return messages
    last = messages[-1]
    if last.role != MessageRole.USER:
        return messages
    content = str(last.content or "")
    keep = last.additional_kwargs.get(TURN_SERIALS_KWARG) or ()
    text = records.cap_turn(
        content,
        keep_serials=keep,
        window=window,
        max_tokens=max(budget // 4, 1),
        token_counter=lambda text: token_counter(
            ChatMessage(role=MessageRole.USER, content=text)
        ),
    )
    if text == content:
        return messages
    capped = ChatMessage(
        role=last.role,
        blocks=list(last.blocks),
        content=text,
        additional_kwargs=dict(last.additional_kwargs),
    )
    return [*messages[:-1], capped]
