"""Delivery-group folding and chat-visible text extraction.

The workflow's delivery tools (``send_message``/``send_media``/
``forward_message``/``react_to_message`` — see :data:`DELIVERY_TOOLS`)
fire at most once per run: once one succeeds, everything later writes
the post-delivery wrap-up note. Every helper here supports the
bookkeeping that keeps history mirroring the chat, so a delivered tool
group collapses into plain assistant text at run end.
"""

import json
from typing import Any

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ToolCallBlock,
)

from wahabot.ai.history import tool_calls, wire_call

#: The tools that deliver content to a chat; each latches the shared
#: holder at most once per run. The five media kinds live behind the
#: single ``send_media`` tool (its ``kind`` argument), so the set keys
#: on that name and the one-delivery latch holds across all five kinds
#: (docs/bug-report-2c665d8.md, bug 7b).
DELIVERY_TOOLS = frozenset(
    {
        "send_message",
        "send_media",
        "forward_message",
        "react_to_message",
    }
)

#: Collapse marker for a delivered WebP the chat experiences as a
#: sticker, never a "file".
MIME_MARKERS: dict[str, str] = {"image/webp": "[sticker]"}

#: The tool name the agent uses to say nothing this round; the
#: workflow treats a call to it as a terminal silence.
SILENCE_TOOL = "stay_silent"


def tool_call_names(message: ChatMessage) -> list[str]:
    """Names of the tools an assistant message called (empty for plain text)."""
    return [call.name for call in tool_calls(message)]


def delivered_text(message: ChatMessage) -> str:
    """The delivered content of an ``ok`` delivery tool result, else "".

    A failed delivery left nothing in the chat, so its pair stays as
    the model's record of what went wrong — only a delivered result
    collapses.
    """
    if message.role != MessageRole.TOOL:
        return ""
    try:
        envelope = json.loads(str(message.content or "{}"))
    except json.JSONDecodeError:
        return ""
    if not envelope.get("ok"):
        return ""
    return delivery_content(envelope)


def delivery_content(envelope: dict[str, Any]) -> str:
    """The chat-visible content of a delivered tool envelope, else "".

    Text sends keep their full text; media/file sends and forwards keep
    a short bracketed marker (with the caption when present); reactions
    keep their emoji. Voice notes and stickers get their own markers so
    the model's self-history says what the chat actually heard/saw.
    """
    if text := envelope.get("text"):
        return str(text)
    if "reaction" in envelope:
        return (
            f"[removed reaction from {envelope.get('message_id')}]"
            if envelope.get("removed")
            else str(envelope["reaction"])
        )
    caption = str(envelope.get("caption") or "").strip()
    if url := envelope.get("url"):
        return f"[media: {url}] {caption}".strip()
    return _typed_marker(envelope, caption) or _forwarded_marker(envelope)


def _typed_marker(envelope: dict[str, Any], caption: str) -> str:
    """The collapse marker for a delivered file of a known kind."""
    mimetype = str(envelope.get("mimetype") or "")
    if mimetype.startswith("audio/"):
        return f"[voice note: {mimetype}]"
    if marker := MIME_MARKERS.get(mimetype):
        return marker
    return f"[file: {mimetype}] {caption}".strip() if mimetype else ""


def _forwarded_marker(envelope: dict[str, Any]) -> str:
    """The collapse marker of a forwarded message, else ""."""
    if message_id := envelope.get("message_id"):
        return f"[forwarded {message_id}]"
    return ""


def tool_call_ids(message: ChatMessage) -> list[str]:
    """All tool-call ids of an assistant message."""
    return [call.call_id for call in tool_calls(message) if call.call_id]


def delivery_call_ids(message: ChatMessage) -> list[str]:
    """Tool-call ids of an assistant message's delivery calls only."""
    return [
        call.call_id
        for call in tool_calls(message)
        if call.name in DELIVERY_TOOLS and call.call_id
    ]


def tool_result_indices(messages: list[ChatMessage]) -> dict[str, int]:
    """Tool-call id → index of its result message, across the whole history."""
    return {
        str(message.additional_kwargs.get("tool_call_id", "")): j
        for j, message in enumerate(messages)
        if message.role == MessageRole.TOOL
    }


def batch_closes_history(all_ids: list[str], results: dict[str, int], total: int) -> bool:
    """True when every call's result exists and together they end the history.

    A batch with unfulfilled calls is left for
    :func:`sanitize_chat_history` to drop as a unit; a batch followed by
    more conversation belongs to an earlier run, already collapsed at
    that run's end.
    """
    indices = sorted(results[call_id] for call_id in all_ids if call_id in results)
    return len(indices) == len(all_ids) and indices == list(
        range(total - len(indices), total)
    )


def delivered_texts(
    call_ids: list[str], results: dict[str, int], messages: list[ChatMessage]
) -> list[str]:
    """The delivered texts of *call_ids*' results (skipping failures)."""
    return [
        text
        for call_id in call_ids
        if call_id in results and (text := delivered_text(messages[results[call_id]]))
    ]


def delivery_group_bounds(
    messages: list[ChatMessage],
) -> tuple[int, list[int], str] | None:
    """(assistant index, result indices, delivered text) of the last delivery group.

    A group is an assistant message whose tool calls include a delivery
    plus the ``tool`` result messages of its *delivery* calls — located
    by tool-call id, not position, so a parallel batch mixing delivery
    and research calls collapses correctly (research calls and their
    results stay). Only a group whose *whole* tool batch closes the
    message list qualifies.
    """
    results = tool_result_indices(messages)
    for i in range(len(messages) - 1, -1, -1):
        all_ids = tool_call_ids(messages[i])
        if not all_ids or not set(tool_call_names(messages[i])) & DELIVERY_TOOLS:
            continue
        if not batch_closes_history(all_ids, results, len(messages)):
            return None
        call_ids = delivery_call_ids(messages[i])
        texts = delivered_texts(call_ids, results, messages)
        if not texts:
            return None
        indices = sorted(results[c] for c in call_ids if c in results)
        return i, indices, " ".join(texts)
    return None


def collapse_send_group(messages: list[ChatMessage]) -> list[ChatMessage] | None:
    """*messages* with the last delivery group folded into the assistant turn.

    The delivery calls are stripped from the assistant message and their
    results removed; the delivered text becomes the message's content,
    so the model's self-history records what the chat saw. A parallel
    batch keeps its research calls (on the same message, so the history
    stays API-valid and dedup-safe). Returns None when there is nothing
    to collapse.
    """
    bounds = delivery_group_bounds(messages)
    if bounds is None:
        return None
    i, result_indices, text = bounds
    collapsed = [m for j, m in enumerate(messages) if j not in {i, *result_indices}]
    collapsed.insert(i, strip_delivery_calls(messages[i], text))
    return collapsed


def strip_delivery_calls(message: ChatMessage, text: str) -> ChatMessage:
    """*message* without its delivery calls, carrying the delivered text.

    A parallel batch keeps its research calls (blocks or kwargs, whichever
    the message carries) with the delivered text as content; a
    delivery-only message becomes the plain assistant text.
    """
    if any(isinstance(block, ToolCallBlock) for block in message.blocks):
        return strip_delivery_blocks(message, text)
    return strip_delivery_kwargs(message, text)


def strip_delivery_blocks(message: ChatMessage, text: str) -> ChatMessage:
    """Block-carried *message* without its delivery ``ToolCallBlock``s."""
    remaining = [
        block
        for block in message.blocks
        if isinstance(block, ToolCallBlock) and block.tool_name not in DELIVERY_TOOLS
    ]
    if not remaining:
        return ChatMessage(role=MessageRole.ASSISTANT, content=text)
    return ChatMessage(
        role=MessageRole.ASSISTANT, blocks=[TextBlock(text=text), *remaining]
    )


def strip_delivery_kwargs(message: ChatMessage, text: str) -> ChatMessage:
    """Kwargs-carried *message* without its delivery ``tool_calls``."""
    remaining = [
        call
        for call in message.additional_kwargs.get("tool_calls") or []
        if (parsed := wire_call(call)) and parsed.name not in DELIVERY_TOOLS
    ]
    if not remaining:
        return ChatMessage(role=MessageRole.ASSISTANT, content=text)
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        content=text,
        additional_kwargs={**message.additional_kwargs, "tool_calls": remaining},
    )
