"""Stateful function calling agent workflow, split into a package.

The original ``workflow.py`` module grew to ~1400 lines and was broken
into focused submodules:

- :mod:`wahabot.ai.workflow.agent` — the ``FunctionCallingAgentWorkflow``
  class and its steps: history preparation, the LLM round,
  repetition/silence detection, delivery collapse, the wrap-up note.
- :mod:`wahabot.ai.workflow.toolkit` — tool-call execution: signature
  validation, failure envelopes, span error marking, outcome audit.
- :mod:`wahabot.ai.workflow.delivery` — delivery-group folding and
  chat-visible text extraction (history mirroring).
- :mod:`wahabot.ai.workflow.text` — message/prompt sizing (token
  counting for the trim).

The package re-exports everything the caller surface expects, so
``from wahabot.ai.workflow import FunctionCallingAgentWorkflow,
load_llm, ...`` keeps working.
"""

from wahabot.ai.workflow.agent import (
    _EARLY_STOPPING_PROMPT,  # pyright: ignore[reportPrivateUsage]
    _POST_DELIVERY_WRAP_UP_PROMPT,  # pyright: ignore[reportPrivateUsage]
    FunctionCallingAgentWorkflow,
    ObservableOpenAILike,
    build_agent,
    load_llm,
)
from wahabot.ai.workflow.delivery import (
    DELIVERY_TOOLS,
    MIME_MARKERS,
    SILENCE_TOOL,
    collapse_send_group,
    delivered_text,
    delivery_content,
)
from wahabot.ai.workflow.text import message_text, token_count
from wahabot.ai.workflow.toolkit import (
    run_tool_call,
    tool_call_log_extra,
    tool_outcome,
    tool_outcome_ok,
)

#: Everything the old single-module surface exported — tests import
#: helpers directly from ``wahabot.ai.workflow``. Kept together here
#: so any future caller sees the full split surface in one place.
__all__ = [
    "DELIVERY_TOOLS",
    "MIME_MARKERS",
    "SILENCE_TOOL",
    "_EARLY_STOPPING_PROMPT",
    "_POST_DELIVERY_WRAP_UP_PROMPT",
    "FunctionCallingAgentWorkflow",
    "ObservableOpenAILike",
    "build_agent",
    "collapse_send_group",
    "delivered_text",
    "delivery_content",
    "load_llm",
    "message_text",
    "run_tool_call",
    "token_count",
    "tool_call_log_extra",
    "tool_outcome",
    "tool_outcome_ok",
]
