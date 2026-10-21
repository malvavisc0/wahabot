import json
from typing import Any, cast

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ThinkingBlock,
    ToolCallBlock,
)

FRESH_TURNS = 2

VERDICT_KEYS = ("ok", "error", "outcome", "chat", "tool", "exit_code", "capture_errors")

CALL_ARG_PREVIEW = 80

MIN_SQUEEZE_GAIN = 32


def squeeze_tool_call_kwargs(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """Tool-call kwargs with long string values trimmed, or None when unchanged.

    Every argument *key* survives, with its value cut to a short preview:
    the model learns how to call its tools from its own replayed calls,
    and an older squeeze that kept only ``reason`` taught it to send
    ``{"reason": …}`` alone — the dominant cause of "missing a required
    argument" failures (16 of 27 replayed calls in one group's history
    had that shape). Trimming the values still drops the bulk (shell
    bodies, page text) that motivated the squeeze.
    """
    if not kwargs:
        return None
    squeezed = {key: trim_call_value(value) for key, value in kwargs.items()}
    before = len(json.dumps(kwargs, ensure_ascii=False))
    if len(json.dumps(squeezed, ensure_ascii=False)) + MIN_SQUEEZE_GAIN > before:
        return None
    return squeezed


def trim_call_value(value: Any) -> Any:
    """A long string cut to :data:`CALL_ARG_PREVIEW` chars plus an ellipsis."""
    if isinstance(value, str) and len(value) > CALL_ARG_PREVIEW:
        return value[:CALL_ARG_PREVIEW] + "…"
    return value


def original_call_kwargs(msg: ChatMessage, call_id: str) -> dict[str, Any] | None:
    """The full arguments a tool call was issued with, from the raw message.

    ``additional_kwargs["tool_calls"]`` keeps the provider's original
    call, untouched by squeezing — the source for repairing calls an
    older squeeze cut down to ``reason`` only.
    """
    calls = cast(list[Any], msg.additional_kwargs.get("tool_calls") or [])
    for call in calls:
        if isinstance(call, dict):
            entry = cast(dict[str, Any], call)
            if entry.get("id") == call_id:
                return parsed_arguments(entry.get("function", {}).get("arguments"))
    return None


def parsed_arguments(raw: Any) -> dict[str, Any] | None:
    """A tool call's arguments (JSON text or dict) as a dict, else None."""
    try:
        parsed: Any = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return None
    return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None


def block_kwargs(block: ToolCallBlock) -> dict[str, Any] | None:
    """A block's kwargs as a dict (they may be stored as JSON text)."""
    return parsed_arguments(block.tool_kwargs)


def repaired_kwargs(
    block: ToolCallBlock, kwargs: dict[str, Any], msg: ChatMessage | None
) -> dict[str, Any] | None:
    """The original arguments of a call an older squeeze cut to ``reason``."""
    if msg is None or not block.tool_call_id or set(kwargs) - {"reason"}:
        return None
    original = original_call_kwargs(msg, block.tool_call_id)
    return original if original and set(original) - {"reason"} else None


def degraded_tool_call_block(
    block: ToolCallBlock, msg: ChatMessage | None = None
) -> ToolCallBlock:
    """A ``ToolCallBlock`` with long argument values trimmed.

    A block an older squeeze reduced to ``{"reason": …}`` is repaired
    from the message's original call (*msg*) when that is available.
    """
    kwargs = block_kwargs(block)
    if kwargs is None:
        return block
    repaired = repaired_kwargs(block, kwargs, msg)
    source = repaired or kwargs
    verdict = squeeze_tool_call_kwargs(source) or (repaired and source)
    if not verdict:
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
    - old ``ToolCallBlock`` kwargs keep every key but trim long values
      to a short preview — the meme-script incident (a 3k-char
      ``run_shell_command`` body rode every later prompt as dead
      weight) — without teaching the model call shapes that lack
      their required arguments.

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
    ``ThinkingBlock``s and have every ``ToolCallBlock``'s long argument
    values trimmed, keys kept (:func:`degraded_tool_call_block`).
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
        degraded_tool_call_block(b, msg) if isinstance(b, ToolCallBlock) else b
        for b in blocks
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
