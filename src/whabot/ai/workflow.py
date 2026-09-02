"""LlamaIndex function calling agent workflow.

Implements the reference function calling agent workflow
(https://developers.llamaindex.ai/python/examples/workflow/function_calling_agent/):
user messages are added to a per-chat ``ChatMemoryBuffer``, the LLM
gets tools + history, tool results loop back, and the final text comes
back in ``StopEvent.result``.
"""

from typing import Any

from llama_index.core.llms import ChatMessage
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import BaseTool
from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.openai_like import OpenAILike

from whabot.ai.events import InputEvent, StreamEvent, ToolCallEvent
from whabot.settings import Settings

__all__ = [
    "FunctionCallingAgentWorkflow",
    "build_agent",
    "load_llm",
]


class FunctionCallingAgentWorkflow(Workflow):
    """Stateful function calling agent built from plain workflow steps."""

    def __init__(
        self,
        *args: Any,
        llm: FunctionCallingLLM | None = None,
        tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.llm = llm or OpenAILike(model="gpt-4o-mini", api_base="", api_key="")
        assert self.llm.metadata.is_function_calling_model

    @step
    async def prepare_chat_history(self, ctx: Context, ev: StartEvent) -> InputEvent:
        """Add the incoming user message to memory and fetch chat history."""
        await ctx.store.set("sources", [])
        memory = await ctx.store.get("memory", default=None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=None)
            if self.system_prompt:
                await memory.aput(ChatMessage(role="system", content=self.system_prompt))
        await memory.aput(ChatMessage(role="user", content=str(ev.input)))
        await ctx.store.set("memory", memory)
        return InputEvent(input=await memory.aget())

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        """Call the LLM with tools + history; loop on tool calls."""
        chat_history = ev.input
        response_stream = await self.llm.astream_chat_with_tools(
            self.tools, chat_history=chat_history
        )
        response = None
        async for response in response_stream:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))
        if response is None:
            raise RuntimeError("LLM returned an empty response stream")
        memory = await ctx.store.get("memory")
        await memory.aput(response.message)
        await ctx.store.set("memory", memory)
        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )
        if not tool_calls:
            sources = await ctx.store.get("sources", default=[])
            return StopEvent(result={"response": response, "sources": sources})
        return ToolCallEvent(tool_calls=tool_calls)

    @step
    async def handle_tool_calls(self, ctx: Context, ev: ToolCallEvent) -> InputEvent:
        """Run requested tools, collect outputs, feed them back to memory."""
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}
        tool_msgs: list[ChatMessage] = []
        sources = await ctx.store.get("sources", default=[])
        for tool_call in ev.tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool_call.tool_name,
            }
            if not tool:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Tool {tool_call.tool_name} does not exist",
                        additional_kwargs=kwargs,
                    )
                )
                continue
            try:
                tool_output = tool(**tool_call.tool_kwargs)
            except Exception as exc:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Encountered error in tool call: {exc}",
                        additional_kwargs=kwargs,
                    )
                )
                continue
            sources.append(tool_output)
            tool_msgs.append(
                ChatMessage(
                    role="tool",
                    content=tool_output.content,
                    additional_kwargs=kwargs,
                )
            )
        memory = await ctx.store.get("memory")
        for msg in tool_msgs:
            await memory.aput(msg)
        await ctx.store.set("sources", sources)
        await ctx.store.set("memory", memory)
        return InputEvent(input=await memory.aget())


def load_llm(settings: Settings) -> FunctionCallingLLM:
    """Configure the OpenAI-compatible LLM from settings."""
    return OpenAILike(
        model=settings.llm_model,
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key,
        is_function_calling_model=True,
    )


def build_agent(
    settings: Settings,
    tools: list[BaseTool] | None = None,
    system_prompt: str | None = None,
    timeout: int = 120,
) -> FunctionCallingAgentWorkflow:
    """Build the function calling agent workflow with the configured LLM."""
    return FunctionCallingAgentWorkflow(
        llm=load_llm(settings),
        tools=tools or [],
        system_prompt=system_prompt or settings.agent_system_prompt,
        timeout=timeout,
    )
