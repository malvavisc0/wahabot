from typing import Any, NamedTuple, cast

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ToolCallBlock,
)

from wahabot.ai.messages import (
    MERGED_TURNS_KWARG,
    OPERATOR_FOLD_KWARG,
    REACTION_TARGET_KWARG,
    TURN_HANDLED_KWARG,
    WRAP_UP_NOTE_KWARG,
)


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


def message_tool_call_count(msg: ChatMessage) -> int:
    """The number of tool calls advertised by an assistant message."""
    return len(tool_calls(msg))


def is_tool_message(msg: ChatMessage) -> bool:
    """True if *msg* is a tool-result message (``role == TOOL``)."""
    return msg.role == MessageRole.TOOL


def is_wrap_up_note(msg: ChatMessage) -> bool:
    """True when *msg* is a private wrap-up note (never sent to the chat)."""
    return bool(msg.additional_kwargs.get(WRAP_UP_NOTE_KWARG))


def merges_across_wrap_up_boundary(first: ChatMessage, second: ChatMessage) -> bool:
    """True when merging *first*/*second* would cross the private boundary.

    A wrap-up note is private by construction; concatenating it with a
    chat turn (in either order) erases the distinction the tag exists to
    preserve and hands the model its own notes as chat style.
    """
    return is_wrap_up_note(first) != is_wrap_up_note(second)


def deduplicate_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Step 1: Merge consecutive duplicate-role messages (keep every word).

    Out-of-band folds create consecutive same-role turns the chat API
    rejects (the operator's ``fromMe`` text after the model's reply; a
    reaction note before the next user message); merging keeps
    everything the chat actually saw.

    Never merges tool messages, assistant-with-tool-call messages, or
    pairs that would fold a private wrap-up note into a chat turn.
    """
    merged: list[ChatMessage] = []
    for msg in messages:
        prev = merged[-1] if merged else None
        can_merge = (
            prev is not None
            and prev.role == msg.role
            and not is_tool_message(msg)
            and message_tool_call_count(prev) == 0
            and message_tool_call_count(msg) == 0
            and not merges_across_wrap_up_boundary(prev, msg)
        )
        if prev is not None and can_merge:
            merged[-1] = merge_pair(prev, msg)
        else:
            merged.append(msg)
    return merged


def merge_pair(first: ChatMessage, second: ChatMessage) -> ChatMessage:
    """Two same-role messages as one, every word and kwarg of both kept.

    Kwargs merge left-to-right (``second`` wins collisions) so a tag
    carried by the first message — a reaction note's
    ``reaction_target_id`` — survives into the merged turn; the
    superseded-note removal filters on that tag.

    Refuses to cross the private boundary: folding a wrap-up note into a
    chat turn (or the reverse) would erase the distinction the tag
    exists to preserve.
    :func:`deduplicate_messages` applies the same rule before calling,
    so a raise here is a violated invariant, never a fallback.
    """
    if merges_across_wrap_up_boundary(first, second):
        raise ValueError("refusing to merge a wrap-up note with a chat turn")
    kwargs: dict[str, Any] = {**first.additional_kwargs, **second.additional_kwargs}
    if is_inbound_chat_turn(first) and is_inbound_chat_turn(second):
        kwargs[MERGED_TURNS_KWARG] = True
    return ChatMessage(
        role=first.role,
        content=f"{first.content or ''}\n{second.content or ''}",
        additional_kwargs=kwargs,
    )


def is_inbound_chat_turn(msg: ChatMessage) -> bool:
    """True for a plain inbound chat turn (not an out-of-band fold).

    A reaction note or an operator ``fromMe`` text is a user-role message
    too, but it is not a turn the group's members spoke; fusing one with
    the next turn must not label the result as a backlog of member
    messages.
    """
    return msg.role == MessageRole.USER and not any(
        key in msg.additional_kwargs
        for key in (REACTION_TARGET_KWARG, OPERATOR_FOLD_KWARG, WRAP_UP_NOTE_KWARG)
    )


def validate_tool_groups(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Step 2: Validate tool groups.

    Keep an assistant tool-call message only if exactly N matching tool
    messages follow it; drop orphan tool messages.
    """
    validated: list[ChatMessage] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]

        if is_tool_message(msg):
            i += 1
            continue

        call_count = message_tool_call_count(msg)
        if call_count > 0:
            j = i + 1
            tool_msgs: list[ChatMessage] = []
            while j < n and is_tool_message(messages[j]):
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


def trim_history(
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
        if message_tool_call_count(last) > 0 or (
            drop_trailing_user and last.role == MessageRole.USER and not is_handled(last)
        ):
            trimmed.pop()
        else:
            break

    return trimmed


def is_handled(msg: ChatMessage) -> bool:
    """True when *msg*'s run completed — the turn is real conversation.

    Unstamped means the run that appended it never finished, so it is
    the run-scoped turn the next run replaces. A reaction fold carries
    the reaction-target kwarg instead (it has no run at all) and must
    never be dropped here.
    """
    return (
        TURN_HANDLED_KWARG in msg.additional_kwargs
        or REACTION_TARGET_KWARG in msg.additional_kwargs
        or OPERATOR_FOLD_KWARG in msg.additional_kwargs
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

    deduplicated = deduplicate_messages(chat_history)
    validated = validate_tool_groups(deduplicated)
    return trim_history(validated, drop_trailing_user)
