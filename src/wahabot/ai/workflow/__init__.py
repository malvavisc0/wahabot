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

from wahabot.ai.workflow.agent import FunctionCallingAgentWorkflow, build_agent
from wahabot.ai.workflow.delivery import (
    DELIVERY_TOOLS,
    MIME_MARKERS,
    SILENCE_TOOL,
    collapse_send_group,
    delivered_text,
    delivery_content,
)
from wahabot.ai.workflow.prompts import (
    EARLY_STOPPING_PROMPT,
    POST_DELIVERY_WRAP_UP_PROMPT,
    ObservableOpenAILike,
    load_llm,
)
from wahabot.ai.workflow.text import message_text, token_count
from wahabot.ai.workflow.toolkit import (
    run_tool_call,
    tool_call_log_extra,
    tool_outcome,
    tool_outcome_ok,
)

__all__ = [
    "DELIVERY_TOOLS",
    "EARLY_STOPPING_PROMPT",
    "MIME_MARKERS",
    "POST_DELIVERY_WRAP_UP_PROMPT",
    "SILENCE_TOOL",
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
