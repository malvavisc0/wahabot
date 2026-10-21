from collections.abc import Callable

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)

from wahabot.ai.history.messages import message_tool_call_count


class ContextBudgetError(ValueError):
    """The reserved context (current query, newest tool group, required
    prompt sections) cannot fit the memory token budget.

    Permanent for the triggering message: a redelivery would rebuild
    the same oversized turn and fail identically, so callers must not
    drop their seen markers for this class — they log once and let the
    message go instead of entering a retry loop.
    """


def group_boundaries(messages: list[ChatMessage]) -> list[int]:
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
        call_count = message_tool_call_count(messages[i])
        i += 1 + call_count
    return starts


def split_groups(messages: list[ChatMessage]) -> list[list[ChatMessage]]:
    """Split *messages* into atomic groups (a message plus its tool replies)."""
    if not messages:
        return []
    starts = group_boundaries(messages)
    ends = [*starts[1:], len(messages)]
    return [messages[a:b] for a, b in zip(starts, ends, strict=True)]


def newest_within_budget(
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
    3. The current question and its newest atomic response group are
       never lost. Older groups may be dropped; an oversized minimum
       raises clearly rather than sending an orphan tool response.

    *token_counter* returns the token count for one message.
    """
    if sum(token_counter(message) for message in minimum_turn(messages)) > budget:
        raise ContextBudgetError(
            "Current query and newest tool group exceed the token budget"
        )
    groups = split_groups(messages)
    kept = newest_within_budget(groups, budget, token_counter)
    if not kept:
        return []

    while kept and kept[-1][0].role != MessageRole.USER:
        kept.pop()
    if not kept:
        return last_user_turn(groups, budget, token_counter)

    return [m for group in reversed(kept) for m in group]


def minimum_turn(messages: list[ChatMessage]) -> list[ChatMessage]:
    """The current query plus the newest complete assistant/tool group."""
    groups = split_groups(messages)
    for index in range(len(groups) - 1, -1, -1):
        if groups[index][0].role == MessageRole.USER:
            if index == len(groups) - 1:
                return list(groups[index])
            return [*groups[index], *groups[-1]]
    return []


def last_user_turn(
    groups: list[list[ChatMessage]],
    budget: int,
    token_counter: Callable[[ChatMessage], int],
) -> list[ChatMessage]:
    """Keep the current query with the newest whole groups that fit."""
    for index in range(len(groups) - 1, -1, -1):
        user = groups[index]
        if user[0].role == MessageRole.USER:
            remaining = budget - sum(token_counter(message) for message in user)
            kept = newest_within_budget(groups[index + 1 :], remaining, token_counter)
            return [*user, *(message for group in reversed(kept) for message in group)]
    return []
