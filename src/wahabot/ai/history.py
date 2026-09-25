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

This module also owns the reply-text classification that decides what
is *chat-visible* (:func:`narration_kind` and the three ``is_*``
predicates it backs, :func:`chat_visible_text`) — one definition
shared by delivery (``final_reply``) and storage (``remember``) so the
two filters can never drift apart. It lives here, next to the history
invariants it protects, because ``context`` imports ``workflow`` which
imports this module: the predicates must sit below both.

Adapted from aria-ai's ``aria.web.session``.
"""

import json
import re
from collections.abc import Callable
from typing import Any, NamedTuple, cast

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ThinkingBlock,
    ToolCallBlock,
)

from wahabot.ai.messages import REACTION_TARGET_KWARG, TURN_HANDLED_KWARG

__all__ = [
    "ToolCall",
    "chat_visible_text",
    "degrade_old_history",
    "inbound_message_id",
    "is_emoji_narration",
    "is_error_narration",
    "is_silence_narration",
    "narration_kind",
    "sanitize_chat_history",
    "tool_calls",
    "trim_to_budget",
    "wire_call",
]

#: The serialized id note an inbound turn carries (``context.py``
#: appends it as the turn's last line): the id inside brackets.
INBOUND_ID_RE = re.compile(r"\[message id: ([^\]]+)\]")


#: Emoji-only replies.  Small models often output a lone emoji (👋, 🤣,
#: 😂) as plain text instead of calling ``react_to_message``.  The
#: system prompt says "A lone emoji is a reaction, never a message",
#: so we treat a single-emoji final reply as an implicit reaction or
#: silence.  Multi-emoji strings like ``🤣🤣🤣`` are kept as real
#: messages — those are intentional chat text.
SINGLE_EMOJI_RE = re.compile(
    "".join(
        (
            r"^\s*(?:",
            r"[\U0001F600-\U0001F64F]",  # emoticons
            r"|[\U0001F300-\U0001F5FF]",  # misc symbols & pictographs
            r"|[\U0001F680-\U0001F6FF]",  # transport & map
            r"|[\U0001F1E0-\U0001F1FF]",  # flags (regional indicators)
            r"|[\U00002702-\U000027B0]",  # dingbats
            r"|[\U0000FE00-\U0000FE0F]",  # variation selectors
            r"|[\U0001F900-\U0001F9FF]",  # supplemental symbols
            r"|[\U0001FA00-\U0001FA6F]",  # chess symbols / extended-A
            r"|[\U0001FA70-\U0001FAFF]",  # symbols extended-A (cont.)
            r"|[\U00002600-\U000026FF]",  # misc symbols (☀, ⚡, …)
            r"|[\U0000200D]",  # ZWJ
            r"|[\U0000FE0F]",  # VS-16
            r")\s*$",  # exactly ONE emoji (with optional whitespace)
        )
    )
)


def is_single_emoji(reply: str) -> bool:
    """True when *reply* is exactly one emoji and nothing else."""
    return bool(SINGLE_EMOJI_RE.match(reply))


#: An emoji followed by narration: ``👍 Reaccioné con 👍 a…``,
#: ``🙄 I reacted with 🙄 to…``. The pattern-completion small models
#: produce after (or instead of) a reaction tool call: the emoji is
#: the reaction, the rest is a report to nobody. Language-agnostic by
#: shape, not vocabulary: one lone emoji, then a line that is *about*
#: the emoji or the reaction/send — verbs the model narrates in
#: whatever language the chat uses (reacted/reaccioné/habe reagiert,
#: sent/envié/geschickt, without writing/sin escribir/ohne zu
#: schreiben, …). A real message may open with an emoji, but its
#: second line talks to the chat, not about the bot's own action.
EMOJI_NARRATION_RE = re.compile(
    "".join(
        (
            r"^\s*",
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF",
            r"[\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF",
            r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]",
            r"\U0000FE0F?[\U0001F3FB-\U0001F3FF]?",
            r"\s*(?:[\n·|—-]+\s*)?",
            r"(?:i\s+|ich\s+)?",
            r"(?:already\s+|bereits\s+|schon\s+)?",
            r"(?:habe\s+|hab\s+)?",
            r"(?:mit\s+\S+\s+)?",
            r"(?:",
            r"reacted|reaccion(?:é|o|amos)|reacted with|",
            r"he\s+reacted|he\s+reaccionado|",
            r"envié|envió|enviado|mandé|mandó|mandado|",
            r"sent|answered|respondí|respondió|",
            r"replied|replió|",
            r"reagiert|habe\s+reagiert|",
            r"geschickt|gesendet|",
            r"geantwortet|habe\s+geantwortet|",
            r"reagierte",
            r")\b",
            r".*",
        )
    ),
    re.IGNORECASE | re.DOTALL,
)


def is_emoji_narration(reply: str) -> bool:
    """True when *reply* is one emoji plus a report about that action.

    ``👍\\nReaccioné con 👍 al matiz de Francisco, sin escribir por el
    modo silencio.`` — the model narrated the reaction it meant to
    deliver instead of calling the tool (or after a tool round). The
    emoji half makes it *look* like a lone-emoji reaction; the words
    half is a self-report no chat member asked for, in any language.
    Delivery turns this into silence (the handler's lone-emoji path
    cannot run — the emoji rides narration, not a message), and
    storage filters it so the model's self-history cannot re-teach
    the pattern.
    """
    return bool(EMOJI_NARRATION_RE.match(reply.strip()))


#: Replies that narrate a chosen silence instead of being one. Small
#: models asked to "reply with an empty string to stay silent" often
#: answer with meta-commentary ("I'll stay silent here — ...", "No
#: response.") — matching it here keeps it out of the chat. The reply
#: must *be about* staying silent, not merely contain the word (so
#: "silence is golden, but I'll answer anyway" still goes through).
#: The same happens after a delivered reaction or reply: the model
#: pattern-completes "I already reacted to that message, so I'm done
#: here." instead of going quiet — the reaction/reply already went out
#: via the tool, so the narration is chatter, not an answer. The
#: chat's languages (English, Spanish, German) carry the same anchored
#: shapes: a Spanish "Sin respuesta." or German "Keine Antwort."
#: reaching the chat is the same bug as the English "No response.".
SILENCE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^no response\b",
        r"^no reply\b",
        r"^nothing to (add|say)\b",
        r"^nothing (more|else) to (add|say|do)\b",
        r"^(i'?ll |i will |i'?m )?(stay|staying|remain|choosing to stay)[\s'-]*silent\b",
        r"^stay_silent\b",
        r"^(i'?ll |i will )?(stay|keep) (quiet|out of (this|it|the conversation))\b",
        r"^silence[.!…]?$",
        r"^\(silence\)$",
        r"^not (addressed|directed) (to|at) me\b",
        r"^(i'?ll |i will )?say nothing\b",
        r"^i (already )?(reacted|replied|sent|answered)\b[^.]*(so )?i'?m done\b",
        "".join(
            (
                r"^i (already )?(reacted|replied|sent|answered)\b[^.]*",
                r"(so )?(there'?s?|there is) nothing (more|else|left) (to )?(add|say|do)",
            )
        ),
        r"^i'?m done (here|with this)\b",
        r"^sin respuesta[.!…]?$",
        r"^(no tengo )?nada( más)? que (añadir|decir)[.!…]?$",
        r"^(me |prefiero )?(quedo|quedarme) (callado|callada|en silencio)[.!…]?$",
        r"^silencio[.!…]?$",
        r"^\(silencio\)$",
        r"^no diré nada\b",
        r"^no voy a (responder|contestar)\b",
        r"^keine antwort[.!…]?$",
        r"^keine (antwort|rückmeldung) hierzu[.!…]?$",
        "".join(
            (
                r"^(ich )?(habe )?nichts( (mehr|weiter))?",
                r"( zu (sagen|ergänzen|hinzuzufügen))?[.!…]?$",
            )
        ),
        r"^nichts hinzuzufügen[.!…]?$",
        r"^ich bleibe (still|stumm|leise)[.!…]?$",
        r"^bleib(?:e|t)? (still|stumm)[.!…]?$",
        r"^stille[.!…]?$",
        r"^\(stille\)$",
        r"^ich sage nichts[.!…]?$",
        r"^ich (werde |will )?nicht (antworten|antworten werde|reagieren)\b",
        r"^nicht an mich (gerichtet|addressiert)\b",
        r"^ich bin fertig (hier|damit)\b",
        "".join(
            (
                r"^ich habe (bereits )?(reagiert|geantwortet|geschickt)",
                r"[,.]?\s*(damit )?bin ich fertig\b",
            )
        ),
    )
)


def is_silence_narration(reply: str) -> bool:
    """True when *reply* narrates a silence instead of being one.

    Stripped of surrounding whitespace/quotes/parentheses and matched
    case-insensitively against the silence-meta patterns (English,
    Spanish and German — the chat's languages); anything the model
    actually wanted to say still goes through.
    """
    cleaned = reply.strip().strip("\"'`()").strip()
    return any(pattern.search(cleaned) for pattern in SILENCE_PATTERNS)


def is_error_narration(reply: str) -> bool:
    """True when *reply* is an error payload, not a chat answer.

    Small models sometimes *write* an API error as their reply — e.g.
    a made-up ``{"error": {"message": "resource exhausted …", "type":
    "upstream_error", "code": "resource_exhausted"}}`` naming a provider
    the bot never used. Whatever the model's reason (pattern-
    completing text it has seen), the result must never reach the
    chat. Only near-JSON bodies whose top level is an ``error`` object
    match; genuine prose answers never do.
    """
    stripped = reply.strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError, ValueError:
        return False
    if not isinstance(data, dict):
        return False
    error = cast(dict[str, Any], data).get("error")
    return isinstance(error, dict) and any(
        key in error for key in ("message", "code", "type", "status")
    )


def narration_kind(reply: str) -> str:
    """The narration class of *reply*: "error", "emoji", "silence", or "".

    The one classifier behind :func:`is_error_narration`,
    :func:`is_emoji_narration` and :func:`is_silence_narration` — a
    single dispatch whose *kind* survives to the caller, so delivery
    and storage not only agree that a reply is narration (the drop),
    they can log — and later act on — *which* failure mode the model
    produced. An empty string means the reply is chat-visible text.
    """
    text = str(reply or "").strip()
    if not text:
        return ""
    if is_error_narration(text):
        return "error"
    if is_emoji_narration(text):
        return "emoji"
    if is_silence_narration(text):
        return "silence"
    return ""


def chat_visible_text(content: Any) -> str:
    """The chat-visible text of *content*.

    Empty when the model narrated instead of answering:

    One definition of "this text would reach the chat" shared by both
    ends of a run — delivery (``final_reply``) and storage
    (``remember``) — so a leaked ``stay_silent`` token or an invented
    error payload is filtered at the source and can never drift back
    into the model's self-history (the exact string it pattern-
    completes on). Thinking blocks are ignored: only the *text* blocks
    decide visibility.
    """
    reply = str(content or "").strip() if content is not None else ""
    if not reply:
        return ""
    if narration_kind(reply):
        return ""
    return reply


def inbound_message_id(incoming: str) -> str:
    """The serialized id of an inbound turn's ``[message id: …]`` note.

    Empty when the turn carries no note — operator commands have none,
    so they are never treated as redeliveries.
    """
    match = INBOUND_ID_RE.search(incoming)
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


def message_tool_call_count(msg: ChatMessage) -> int:
    """The number of tool calls advertised by an assistant message."""
    return len(tool_calls(msg))


def is_tool_message(msg: ChatMessage) -> bool:
    """True if *msg* is a tool-result message (``role == TOOL``)."""
    return msg.role == MessageRole.TOOL


def deduplicate_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
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
            and not is_tool_message(msg)
            and message_tool_call_count(prev) == 0
            and message_tool_call_count(msg) == 0
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
    """
    return ChatMessage(
        role=first.role,
        content=f"{first.content or ''}\n{second.content or ''}",
        additional_kwargs={**first.additional_kwargs, **second.additional_kwargs},
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
    3. The newest user turn is never silently lost: if nothing else
       fits, the result is the last user-led group alone.

    A single group larger than the budget is kept so nothing
    disappears.

    *token_counter* returns the token count for one message.
    """
    groups = split_groups(messages)
    kept = newest_within_budget(groups, budget, token_counter)
    if not kept:
        return []

    while kept and kept[-1][0].role != MessageRole.USER:
        kept.pop()
    if not kept:
        return last_user_turn(groups)

    return [m for group in reversed(kept) for m in group]


def last_user_turn(groups: list[list[ChatMessage]]) -> list[ChatMessage]:
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


#: How many of the newest *user turns* stay verbatim. A user turn and
# everything after it up to the next user turn is one run's slice;
#: ``degrade_old_history`` squeezes only the slices that fall outside
#: this window. Two turns: the newest is the current/last run (its tool
#: output may still be the subject of the next message), the one before
#: gives one run of full-fidelity hindsight — past that, the delivered
#: words matter but the *how* (thinking blocks, raw shell output) does
#: not (docs/bug-report-2c665d8.md, bug 7).
FRESH_TURNS = 2

#: Tool-result keys that survive degradation. ``ok``/``error``/``outcome``
#: are the verdict ("it worked", "it failed why"); ``chat`` identifies the
#: delivered conversation; ``tool``/``error`` text keeps failure diagnosis.
#: Everything else — ``stdout`` dumps, message lists, page text — is the
#: payload the model already consumed when the run was live.
VERDICT_KEYS = ("ok", "error", "outcome", "chat", "tool")

#: Tool-call keys that survive degradation. ``reason`` is the audit
#: trail's one-line "what and why" (every tool takes it); everything
#: else — ``command`` bodies, search queries, URLs, page text — is the
#: work the model already did.
CALL_VERDICT_KEYS = ("reason",)

#: Shortest a squeezed tool-call kwargs dict may be, relative to the
#: original — below this the squeeze saves nothing worth the swap.
MIN_SQUEEZE_GAIN = 32


def squeeze_tool_call_kwargs(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """Tool-call kwargs reduced to their ``reason``, or None when unchanged."""
    if not kwargs:
        return None
    verdict = {k: kwargs[k] for k in CALL_VERDICT_KEYS if k in kwargs}
    squeezed = json.dumps(verdict, ensure_ascii=False)
    if len(squeezed) + MIN_SQUEEZE_GAIN > len(json.dumps(kwargs, ensure_ascii=False)):
        return None
    return verdict


def degraded_tool_call_block(block: ToolCallBlock) -> ToolCallBlock:
    """A ``ToolCallBlock`` with its kwargs squeezed to the verdict."""
    kwargs = block.tool_kwargs
    if isinstance(kwargs, str):
        try:
            kwargs = json.loads(kwargs)
        except ValueError:
            return block
    if not isinstance(kwargs, dict):
        return block
    verdict = squeeze_tool_call_kwargs(cast(dict[str, Any], kwargs))
    if verdict is None:
        return block
    return ToolCallBlock(
        tool_call_id=block.tool_call_id,
        tool_name=block.tool_name,
        tool_kwargs=verdict,
    )


def degrade_old_history(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Squeeze old history: keep every word, lose the operational scaffolding.

    Recent conversation is social memory and stays verbatim; old turns
    keep their *conclusions* but drop their *work*:

    - assistant ``ThinkingBlock``s are removed from old slices — the
      deliberation is spent once the run is over, but it dominates
      replayed history (a single reasoning block can be ~1k tokens);
    - old tool results keep only their verdict (``ok``, ``error``, …)
      — the model needs to remember *that* a command worked, not the
      2k characters of ``stdout`` it printed;
    - old ``ToolCallBlock`` kwargs keep only their ``reason`` — the
      meme-script incident (docs/bug-report-2c665d8.md, bug 7: a 3k-char
      ``run_shell_command`` body rode every later prompt as dead
      weight): the model needs to remember *that* it drew a meme, not
      the Python it drew it with.

    Destructive on purpose: the caller (``chat_history``) persists the
    result back into memory, so degradation compounds across runs. A
    run's turns degrade on the *next-next* run, never while fresh — the
    newest ``FRESH_TURNS`` user turns and everything after them are
    untouched. Idempotent: an already-degraded message passes through
    unchanged.
    """
    turn_starts = [i for i, m in enumerate(messages) if m.role == MessageRole.USER]
    if len(turn_starts) <= FRESH_TURNS:
        return messages
    cutoff = turn_starts[-FRESH_TURNS]
    head = [degrade_message(m) for m in messages[:cutoff]]
    return head + messages[cutoff:]


def degrade_message(msg: ChatMessage) -> ChatMessage:
    """One message degraded: thinking stripped, tool work squeezed.

    ``role=TOOL`` results are reduced to their verdict envelope
    (:func:`squeeze_tool_result`); assistant messages lose their
    ``ThinkingBlock``s and have every ``ToolCallBlock``'s kwargs
    squeezed to the call's ``reason`` (:func:`degraded_tool_call_block`).
    A message with nothing to squeeze passes through as the same
    object. The rebuild passes ``content=None``: ``ChatMessage.__init__``
    treats any non-None content — the empty string included — as an
    instruction to replace ``blocks`` with it, which would drop the
    very tool calls being preserved.
    """
    if msg.role == MessageRole.TOOL:
        return squeeze_tool_result(msg)
    blocks = list(msg.blocks)
    if not blocks:
        return msg
    squeezed = [
        degraded_tool_call_block(b) if isinstance(b, ToolCallBlock) else b for b in blocks
    ]
    stripped = [b for b in squeezed if not isinstance(b, ThinkingBlock)]
    if stripped == blocks:
        return msg
    return ChatMessage(
        role=msg.role,
        blocks=stripped,
        content=None,
        additional_kwargs=msg.additional_kwargs,
    )


def squeeze_tool_result(msg: ChatMessage) -> ChatMessage:
    """A tool-result message reduced to its verdict envelope.

    The tool results are JSON envelopes (``wahabot.ai.tools.envelope``);
    keeping the verdict keys preserves the outcome and the failure
    diagnosis while dropping the payload. Non-JSON results pass through
    unchanged — a plain-text tool answer is its own verdict.
    """
    text = str(msg.content or "").strip()
    try:
        payload = json.loads(text)
    except ValueError, TypeError:
        return msg
    if not isinstance(payload, dict):
        return msg
    verdict = {k: payload[k] for k in VERDICT_KEYS if k in payload}
    if not verdict or len(json.dumps(verdict, ensure_ascii=False)) >= len(text):
        return msg
    return ChatMessage(
        role=msg.role,
        content=json.dumps(verdict, ensure_ascii=False),
        additional_kwargs=msg.additional_kwargs,
    )
