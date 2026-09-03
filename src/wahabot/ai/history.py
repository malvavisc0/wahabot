"""Repair conversation history so it is valid for the OpenAI-compatible chat API.

A live agent run can leave ``ChatMemoryBuffer`` in a structurally
invalid state: a failed turn dangles an assistant message advertising
tool calls with the matching ``role="tool"`` replies missing (or only
partially present), and orphans a tool message with no preceding
assistant tool call. The OpenAI-compatible chat API rejects such a
history with "Not the same number of function calls and responses".

The OpenAI-compatible API also requires alternating ``user``/``assistant``
roles for plain turns. Together these invariants are enforced by
:func:`sanitize_chat_history`, which is **tool-call aware**:

1. Collapse consecutive duplicate roles (keep the last of each run) —
   but never collapse ``tool`` messages (parallel tool calls
   legitimately produce several consecutive ``tool`` messages) and never
   collapse an assistant message that carries tool calls.
2. Validate tool groups: for each assistant message advertising ``N``
   tool calls, consume the next ``N`` consecutive ``tool`` messages. If
   fewer than ``N`` follow (a dangling/partial group), drop the whole
   group. Drop orphan ``tool`` messages that have no preceding assistant
   tool call.
3. Drop leading messages until the history starts with a ``user``
   message (without splitting a tool group).
4. Drop a trailing incomplete tool group so the history ends on a clean
   boundary. When *drop_trailing_user* is True (the pre-run path, where a
   new user message is about to be appended), a trailing ``user`` message
   is also removed so the next turn maintains alternation.

Adapted from aria-ai's ``aria.web.session``.
"""

from collections.abc import Callable

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ToolCallBlock,
)

__all__ = [
    "sanitize_chat_history",
    "trim_to_budget",
]

#: Cap for one tool-result message, in estimated tokens (chars).
#: ``fetch_chat_messages``/``search_messages`` embed WAHA's raw
#: ``_data`` blobs — 15 messages ≈ 58k chars — and a group that large
#: breaks the budget twice over: ``trim_to_budget`` keeps it ("a single
#: group larger than the budget is kept"), then ``ChatMemoryBuffer.get``
#: re-trims with the real tokenizer, finds only that group, and drops
#: everything but the tool message — the LLM request goes out with no
#: user turn ("No user query found in messages"). Capping the payload
#: at the source keeps any tool group small enough to coexist with the
#: conversation around it.
MAX_TOOL_RESULT_TOKENS = 2000


def _message_tool_call_count(msg: ChatMessage) -> int:
    """The number of tool calls advertised by an assistant message.

    LlamaIndex carries tool calls in two places (mirroring
    ``to_openai_message_dict``'s precedence): ``ToolCallBlock`` objects in
    ``message.blocks`` (modern path) or ``additional_kwargs["tool_calls"]``
    (legacy/streaming path). Blocks take precedence; the kwargs list is
    only consulted when no blocks are present.
    """
    block_calls = sum(1 for block in msg.blocks if isinstance(block, ToolCallBlock))
    if block_calls:
        return block_calls
    kwarg_calls = msg.additional_kwargs.get("tool_calls")
    if kwarg_calls:
        return len(kwarg_calls)
    return 0


def _is_tool_message(msg: ChatMessage) -> bool:
    """True if *msg* is a tool-result message (``role == TOOL``)."""
    return msg.role == MessageRole.TOOL


def _deduplicate_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Step 1: Collapse consecutive duplicate roles (keep last of each run).

    Never collapse tool messages or assistant-with-tool-call messages.
    """
    deduplicated: list[ChatMessage] = []
    for msg in messages:
        prev = deduplicated[-1] if deduplicated else None
        can_collapse = (
            prev is not None
            and prev.role == msg.role
            and not _is_tool_message(msg)
            and _message_tool_call_count(prev) == 0
            and _message_tool_call_count(msg) == 0
        )
        if can_collapse:
            deduplicated[-1] = msg  # replace — keep latest
        else:
            deduplicated.append(msg)
    return deduplicated


def _validate_tool_groups(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Step 2: Validate tool groups.

    Keep an assistant tool-call message only if exactly N matching tool
    messages follow it; drop orphan tool messages.
    """
    validated: list[ChatMessage] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]

        if _is_tool_message(msg):
            # Orphan tool message — drop.
            i += 1
            continue

        call_count = _message_tool_call_count(msg)
        if call_count > 0:
            # Gather the immediately following tool messages.
            j = i + 1
            tool_msgs: list[ChatMessage] = []
            while j < n and _is_tool_message(messages[j]):
                tool_msgs.append(messages[j])
                j += 1

            if len(tool_msgs) >= call_count:
                # Keep the assistant message plus exactly call_count tool
                # responses.
                validated.append(msg)
                validated.extend(tool_msgs[:call_count])
            # else: dangling/partial group — drop.
            i = j
            continue

        validated.append(msg)
        i += 1

    return validated


def _trim_history(
    messages: list[ChatMessage], drop_trailing_user: bool
) -> list[ChatMessage]:
    """Steps 3 & 4: Trim leading non-user and trailing incomplete.

    Drop leading messages until history starts with user.
    Drop trailing assistant with unfulfilled tool calls.
    Drop trailing user when drop_trailing_user is True.
    """
    # Step 3: Ensure it starts with a user message.
    while messages and messages[0].role != MessageRole.USER:
        messages.pop(0)

    # Step 4: Ensure it ends on a clean boundary.
    while messages:
        last = messages[-1]
        if _message_tool_call_count(last) > 0 or (
            drop_trailing_user and last.role == MessageRole.USER
        ):
            messages.pop()
        else:
            break

    return messages


def sanitize_chat_history(
    chat_history: list[ChatMessage],
    *,
    drop_trailing_user: bool = True,
) -> list[ChatMessage]:
    """Repair chat history so it is valid for the chat completions API.

    Returns a sanitised list whose tool-call/response counts are balanced
    and whose plain turns alternate ``user → assistant``.
    """
    if not chat_history:
        return chat_history

    step1 = _deduplicate_messages(chat_history)
    step2 = _validate_tool_groups(step1)
    step3 = _trim_history(step2, drop_trailing_user)

    return step3


def _group_boundaries(messages: list[ChatMessage]) -> list[int]:
    """Start index of each atomic group (a message plus its tool replies).

    An assistant message advertising ``N`` tool calls owns the next ``N``
    tool messages; the group must be kept or dropped as a unit so a trim
    never splits a tool group.
    """
    starts: list[int] = []
    i = 0
    n = len(messages)
    while i < n:
        starts.append(i)
        call_count = _message_tool_call_count(messages[i])
        i += 1 + call_count if call_count else 1
    return starts


def _split_groups(messages: list[ChatMessage]) -> list[list[ChatMessage]]:
    """Split *messages* into atomic groups (a message plus its tool replies)."""
    if not messages:
        return []
    starts = _group_boundaries(messages)
    ends = [*starts[1:], len(messages)]
    return [messages[a:b] for a, b in zip(starts, ends, strict=True)]


def _newest_within_budget(
    groups: list[list[ChatMessage]],
    budget: int,
    token_counter: Callable[[ChatMessage], int],
) -> list[list[ChatMessage]]:
    """Newest groups whose cumulative token count stays within *budget*.

    A single group larger than the budget is kept so nothing disappears.
    """
    kept: list[list[ChatMessage]] = []
    running = 0
    for group in reversed(groups):
        group_tok = sum(token_counter(m) for m in group)
        if running + group_tok > budget and kept:
            break
        running += group_tok
        kept.append(group)
    return kept


def trim_to_budget(
    messages: list[ChatMessage],
    budget: int,
    token_counter: Callable[[ChatMessage], int],
) -> list[ChatMessage]:
    """Keep only the newest tail of *messages* that fits within *budget* tokens.

    Walks from the newest message backwards and keeps messages whose
    cumulative token count stays within *budget*. Three constraints are
    enforced so the result is safe to feed back into memory:

    1. Tool groups (assistant tool call + its ``tool`` replies) are
       atomic — a trim never leaves a dangling tool call or an orphan
       tool reply.
    2. The returned list must start with a ``user`` message.
    3. The newest user turn is never silently lost: if nothing else
       fits, the result is the last user-led group alone.

    A single group larger than the budget is kept so nothing
    disappears.

    *token_counter* returns the token count for one message.
    """
    groups = _split_groups(messages)
    kept = _newest_within_budget(groups, budget, token_counter)
    if not kept:
        return []

    while kept and kept[-1][0].role != MessageRole.USER:
        kept.pop()
    if not kept:
        return _last_user_turn(groups)

    return [m for group in reversed(kept) for m in group]


def _last_user_turn(groups: list[list[ChatMessage]]) -> list[ChatMessage]:
    """Return the last user-led group alone.

    Fallback for :func:`trim_to_budget` when the budget is so small that
    only non-user groups fit. Only the user group is returned: appending
    the group that follows it can re-add an oversized tool group, and
    ``ChatMemoryBuffer.get`` — trimming with the real tokenizer — would
    then drop everything but that tool message, sending the LLM a
    request with no user turn ("No user query found in messages").
    """
    for group in reversed(groups):
        if group[0].role == MessageRole.USER:
            return list(group)
    return groups[-1] if groups else []
