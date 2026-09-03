"""Workflow events for the function calling agent."""

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection
from llama_index.core.workflow import Event


class InputEvent(Event):
    """Chat history (with the latest user message) ready for the LLM."""

    input: list[ChatMessage]


class ToolCallEvent(Event):
    """The LLM requested tool calls."""

    tool_calls: list[ToolSelection]
