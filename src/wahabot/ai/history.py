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

import re
from collections.abc import Callable
from typing import Any, NamedTuple, cast

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ToolCallBlock,
)

from wahabot.ai.messages import REACTION_TARGET_KWARG, TURN_HANDLED_KWARG

__all__ = [
    "ToolCall",
    "inbound_message_id",
    "sanitize_chat_history",
    "tool_calls",
    "trim_to_budget",
    "wire_call",
]

#: The serialized id note an inbound turn carries (``context.py``
#: appends it as the turn's last line): the id inside brackets.
_INBOUND_ID_RE = re.compile(r"\[message id: ([^\]]+)\]")


def inbound_message_id(incoming: str) -> str:
    """The serialized id of an inbound turn's ``[message id: …]`` note.

    Empty when the turn carries no note — operator commands have none,
    so they are never treated as redeliveries.
    """
    match = _INBOUND_ID_RE.search(incoming)
    return match.group(1).strip() if match else ""


class ToolCall(NamedTuple):
    """One tool call of an assistant message, either carrier."""

    name: str
    call_id: str


def tool_calls(message: ChatMessage) -> list[ToolCall]:
    """The tool calls an assistant message made (empty for plain text).

    In-memory llama-index carries calls as ``ToolCallBlock`` blocks; the
    OpenAI wire shape (``additional_kwargs["tool_calls"]``) appears in
    serialized histories. Blocks take precedence — the kwargs list is
    only consulted when no blocks are present, mirroring llama-index's
    own serialization.
    """
    blocks = [b for b in message.blocks if isinstance(b, ToolCallBlock)]
    if blocks:
        return [ToolCall(str(b.tool_name), str(b.tool_call_id)) for b in blocks]
    calls: Any = message.additional_kwargs.get("tool_calls") or []
    return [call for call in (wire_call(c) for c in calls) if call is not None]


def wire_call(call: Any) -> ToolCall | None:
    """A wire-shaped tool call dict as a :class:`ToolCall`, else None."""
    if not isinstance(call, dict):
        return None
    entry = cast(dict[str, Any], call)
    function = entry.get("function")
    if not isinstance(function, dict):
        return None
    name = str(cast(dict[str, Any], function).get("name", ""))
    call_id = str(entry.get("id", ""))
    return ToolCall(name, call_id) if name else None


def _message_tool_call_count(msg: ChatMessage) -> int:
    """The number of tool calls advertised by an assistant message."""
    return len(tool_calls(msg))


def _is_tool_message(msg: ChatMessage) -> bool:
    """True if *msg* is a tool-result message (``role == TOOL``)."""
    return msg.role == MessageRole.TOOL


def _deduplicate_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Step 1: Merge consecutive duplicate-role messages (keep every word).

    Out-of-band folds create consecutive same-role turns the chat API
    rejects (the operator's ``fromMe`` text after the model's reply; a
    reaction note before the next user message); merging keeps
    everything the chat actually saw.

    Never merges tool messages or assistant-with-tool-call messages.
    """
    merged: list[ChatMessage] = []
    for msg in messages:
        prev = merged[-1] if merged else None
        can_merge = (
            prev is not None
            and prev.role == msg.role
            and not _is_tool_message(msg)
            and _message_tool_call_count(prev) == 0
            and _message_tool_call_count(msg) == 0
        )
        if prev is not None and can_merge:
            merged[-1] = _merge_pair(prev, msg)
        else:
            merged.append(msg)
    return merged


def _merge_pair(first: ChatMessage, second: ChatMessage) -> ChatMessage:
    """Two same-role messages as one, every word and kwarg of both kept.

    Kwargs merge left-to-right (``second`` wins collisions) so a tag
    carried by the first message — a reaction note's
    ``reaction_target_id`` — survives into the merged turn; the
    superseded-note removal filters on that tag.
    """
    return ChatMessage(
        role=first.role,
        content=f"{first.content or ''}\n{second.content or ''}",
        additional_kwargs={**first.additional_kwargs, **second.additional_kwargs},
    )


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
            i += 1
            continue

        call_count = _message_tool_call_count(msg)
        if call_count > 0:
            j = i + 1
            tool_msgs: list[ChatMessage] = []
            while j < n and _is_tool_message(messages[j]):
                tool_msgs.append(messages[j])
                j += 1

            if len(tool_msgs) >= call_count:
                validated.append(msg)
                validated.extend(tool_msgs[:call_count])
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
    Drop a trailing user turn only when it is unhandled run scaffolding:
    the run that consumed it crashed or was replaced before stamping it
    ``turn_handled``. A *handled* trailing user turn — a message a
    completed run read and (in ``judicious`` mode) chose to answer or
    ignore — is conversation, not scaffolding: it stays, and the
    alternation it would break is fixed by the merge in step 1 instead.
    An out-of-band fold (a reaction note, an operator ``fromMe`` text)
    must survive the same way.
    """
    trimmed = list(messages)

    while trimmed and trimmed[0].role != MessageRole.USER:
        trimmed.pop(0)

    while trimmed:
        last = trimmed[-1]
        if _message_tool_call_count(last) > 0 or (
            drop_trailing_user and last.role == MessageRole.USER and not _is_handled(last)
        ):
            trimmed.pop()
        else:
            break

    return trimmed


def _is_handled(msg: ChatMessage) -> bool:
    """True when *msg*'s run completed — the turn is real conversation.

    Unstamped means the run that appended it never finished, so it is
    the run-scoped turn the next run replaces. A reaction fold carries
    the reaction-target kwarg instead (it has no run at all) and must
    never be dropped here.
    """
    return (
        TURN_HANDLED_KWARG in msg.additional_kwargs
        or REACTION_TARGET_KWARG in msg.additional_kwargs
    )


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

    deduplicated = _deduplicate_messages(chat_history)
    validated = _validate_tool_groups(deduplicated)
    return _trim_history(validated, drop_trailing_user)


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
        i += 1 + call_count
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
