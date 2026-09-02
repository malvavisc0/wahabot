"""LlamaIndex agent wiring for WhatsApp messages.

Split into focused submodules:

- :mod:`whabot.ai.events` — workflow events.
- :mod:`whabot.ai.workflow` — the function calling agent (steps, LLM, build).
- :mod:`whabot.ai.messages` — message classification and extraction.
- :mod:`whabot.ai.context` — reply context rendering and the entrypoint.
- :mod:`whabot.ai.tools` — the bundled WhatsApp tools.
"""

from whabot.ai.context import (
    handle_message,
    reply_context,
    reply_context_section,
    reply_description,
)
from whabot.ai.events import (
    FunctionOutputEvent,
    InputEvent,
    StreamEvent,
    ToolCallEvent,
)
from whabot.ai.messages import (
    NON_REPLYABLE_SUFFIXES,
    bot_mentioned,
    extract_text,
    is_group_addressed,
    is_replyable,
    message_kind,
    message_replies_to,
)
from whabot.ai.tools import (
    build_default_tools,
    fetch_chat_messages,
    forward_message,
    get_chat,
    get_contact,
    react_to_message,
    search_messages,
    send_image,
    send_message,
    send_seen,
    set_typing,
)
from whabot.ai.workflow import (
    FunctionCallingAgentWorkflow,
    build_agent,
    load_llm,
)

__all__ = [
    "NON_REPLYABLE_SUFFIXES",
    "FunctionCallingAgentWorkflow",
    "FunctionOutputEvent",
    "InputEvent",
    "StreamEvent",
    "ToolCallEvent",
    "bot_mentioned",
    "build_agent",
    "build_default_tools",
    "extract_text",
    "fetch_chat_messages",
    "forward_message",
    "get_chat",
    "get_contact",
    "handle_message",
    "is_group_addressed",
    "is_replyable",
    "load_llm",
    "message_kind",
    "message_replies_to",
    "react_to_message",
    "reply_context",
    "reply_context_section",
    "reply_description",
    "search_messages",
    "send_image",
    "send_message",
    "send_seen",
    "set_typing",
]
