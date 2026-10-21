import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import Mock

from llama_index.core.base.llms.types import ChatMessage, LLMMetadata, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool, ToolSelection
from llama_index.core.workflow import Context

from wahabot.ai.tools.whatsapp import (
    RunTarget,
    bind_target,
    reset_target,
)
from wahabot.ai.workflow.agent import FunctionCallingAgentWorkflow
from wahabot.ai.workflow.text import message_text
from wahabot.ai.workflow.toolkit import run_tool_call
from wahabot.core.notebook import (
    Notebook,
)
from wahabot.core.tokens import text_tokens

"""Offline regressions for durable facts, shared policies, and private self-lessons."""


def _note_pairs(payload: str) -> list[dict[str, str]]:
    """The key/text pairs of a notebook read envelope."""
    return [
        {"key": note["key"], "text": note["text"]}
        for note in json.loads(payload)["notes"]
    ]


SESSION = "notebook-test"

OTHER_SESSION = "other-notebook-test"

CHAT = "491555000021@c.us"

OTHER_CHAT = "491555000022@c.us"


@contextmanager
def bound_target(target: RunTarget) -> Iterator[None]:
    token = bind_target(target)
    try:
        yield
    finally:
        reset_target(token)


def notebook_workflow(store: Notebook) -> FunctionCallingAgentWorkflow:
    llm = Mock(spec=FunctionCallingLLM)
    llm.metadata = LLMMetadata(is_function_calling_model=True)
    return FunctionCallingAgentWorkflow(
        llm=llm, notebook=store, system_prompt="Follow configured permissions."
    )


async def notebook_context(
    agent: FunctionCallingAgentWorkflow, messages: list[ChatMessage]
) -> Context:
    ctx = Context(agent)
    memory = ChatMemoryBuffer(
        token_limit=agent.memory_token_limit, tokenizer_fn=str.split
    )
    await memory.aset(messages)
    await ctx.store.set("memory", memory)
    return ctx


async def call_notebook(tool: FunctionTool, **kwargs: Any) -> dict[str, Any]:
    selection = ToolSelection(
        tool_id="notebook-call", tool_name="notebook", tool_kwargs=kwargs
    )
    message = await run_tool_call({"notebook": tool}, selection)
    assert message.role == MessageRole.TOOL
    assert message.additional_kwargs["tool_call_id"] == selection.tool_id
    return json.loads(str(message.content))


def request_tokens(messages: list[ChatMessage]) -> int:
    return sum(text_tokens(message_text(message), str.split) for message in messages)


def system_tokens(agent: FunctionCallingAgentWorkflow) -> int:
    system = agent.system_message()
    assert system is not None
    return request_tokens([system])


def tool_group(
    *,
    query: str = "Which meeting day was saved?",
    result: str = '{"ok": true, "notes": []}',
    arguments: str = '{"mode": "read", "key": "meeting"}',
) -> list[ChatMessage]:
    return [
        ChatMessage(role="user", content=query),
        ChatMessage(
            role="assistant",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "lookup-note",
                        "type": "function",
                        "function": {"name": "notebook", "arguments": arguments},
                    }
                ]
            },
        ),
        ChatMessage(
            role="tool",
            content=result,
            additional_kwargs={"tool_call_id": "lookup-note", "name": "notebook"},
        ),
    ]
