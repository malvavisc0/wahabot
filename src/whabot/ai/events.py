"""Workflow events for the function calling agent."""

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolOutput, ToolSelection
from llama_index.core.workflow import Event


class InputEvent(Event):
    """Chat history (with the latest user message) ready for the LLM."""

    input: list[ChatMessage]


class StreamEvent(Event):
    """A chunk of the streamed LLM response."""

    delta: str


class ToolCallEvent(Event):
    """The LLM requested tool calls."""

    tool_calls: list[ToolSelection]


class FunctionOutputEvent(Event):
    """A single tool output, ready to feed back into the LLM."""

    output: ToolOutput
