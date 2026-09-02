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
from llama_index.core.tools import BaseTool, ToolSelection
from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.openai_like import OpenAILike
from loguru import logger

from whabot.ai.events import InputEvent, ToolCallEvent
from whabot.ai.history import sanitize_chat_history, trim_to_budget
from whabot.settings import Settings

__all__ = [
    "FunctionCallingAgentWorkflow",
    "build_agent",
    "load_llm",
]


def raise_missing_llm() -> FunctionCallingLLM:
    """Fail fast when no llm was passed to the workflow."""
    raise ValueError("FunctionCallingAgentWorkflow requires an llm")


def run_tool_call(
    tools_by_name: dict[str, BaseTool], tool_call: ToolSelection
) -> ChatMessage:
    """Run one tool call, returning its output (or the failure) as a tool message."""
    tool = tools_by_name.get(tool_call.tool_name)
    kwargs = {"tool_call_id": tool_call.tool_id, "name": tool_call.tool_name}
    if tool is None:
        content = f"Tool {tool_call.tool_name} does not exist"
    else:
        try:
            content = tool(**tool_call.tool_kwargs).content
        except Exception as exc:
            content = f"Encountered error in tool call: {exc}"
    return ChatMessage(role="tool", content=content, additional_kwargs=kwargs)


def _token_count(msg: ChatMessage) -> int:
    """Estimate a message's token count from its content length (1 char ≈ 1 token)."""
    return len(str(msg.content or ""))


class FunctionCallingAgentWorkflow(Workflow):
    """Stateful function calling agent built from plain workflow steps."""

    def __init__(
        self,
        *args: Any,
        llm: FunctionCallingLLM | None = None,
        tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        memory_token_limit: int = 8000,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.llm = llm or raise_missing_llm()
        self.memory_token_limit = memory_token_limit
        assert self.llm.metadata.is_function_calling_model

    def _system_message(self) -> ChatMessage | None:
        """The system prompt as a message, or None when unset."""
        if not self.system_prompt:
            return None
        return ChatMessage(role="system", content=self.system_prompt)

    async def _chat_history(self, ctx: Context) -> list[ChatMessage]:
        """The trimmed, structurally valid conversation plus system prompt.

        The conversation is sanitised (balanced tool groups, alternating
        turns) and trimmed to the memory token budget. The system message
        is kept out of the rolling buffer so the token trim can never
        evict it; its cost is accounted via ``initial_token_count`` so the
        whole prompt still fits the budget.
        """
        memory = await ctx.store.get("memory")
        system = self._system_message()
        messages = await memory.aget_all()
        messages = sanitize_chat_history(messages, drop_trailing_user=False)
        messages = trim_to_budget(messages, self.memory_token_limit, _token_count)
        await memory.aset(messages)
        initial = self._system_token_count(memory)
        history = await memory.aget(input=None, initial_token_count=initial)
        return ([system] if system else []) + history

    def _system_token_count(self, memory: ChatMemoryBuffer) -> int:
        """Tokens the system prompt costs, clamped to the memory budget.

        ``ChatMemoryBuffer.get`` raises when ``initial_token_count``
        exceeds its token limit; a huge system prompt would crash every
        run. Clamp so the conversation simply gets no room instead.
        """
        if not self.system_prompt:
            return 0
        tokens = len(memory.tokenizer_fn(self.system_prompt))
        if tokens >= memory.token_limit:
            logger.warning(
                (
                    "System prompt ({tokens} tokens) meets or exceeds "
                    "memory_token_limit ({limit}); conversation history gets no room"
                ),
                tokens=tokens,
                limit=memory.token_limit,
            )
            return max(memory.token_limit - 1, 0)
        return tokens

    async def _repair_memory(self, ctx: Context) -> None:
        """Repair dangling state left by a failed workflow run.

        After an LLM/infrastructure error mid-run, memory may end with a
        dangling user message (no assistant reply) or an assistant
        message advertising tool calls whose matching ``tool`` responses
        are missing. :func:`sanitize_chat_history` fixes these in one
        pass; the repair runs at the start of the next run so a broken
        history never reaches the API.
        """
        memory = await ctx.store.get("memory", default=None)
        if memory is None:
            return
        messages = await memory.aget_all()
        if not messages:
            return
        repaired = sanitize_chat_history(messages, drop_trailing_user=True)
        if repaired != messages:
            await memory.aset(repaired)

    @step
    async def prepare_chat_history(self, ctx: Context, ev: StartEvent) -> InputEvent:
        """Add the incoming user message to memory and fetch chat history."""
        memory = await ctx.store.get("memory", default=None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(
                token_limit=self.memory_token_limit, llm=self.llm
            )
        else:
            await self._repair_memory(ctx)
        await memory.aput(ChatMessage(role="user", content=str(ev.input)))
        await ctx.store.set("memory", memory)
        return InputEvent(input=await self._chat_history(ctx))

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        """Call the LLM with tools + history; loop on tool calls."""
        response = await self.llm.achat_with_tools(self.tools, chat_history=ev.input)
        memory = await ctx.store.get("memory")
        await memory.aput(response.message)
        await ctx.store.set("memory", memory)
        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )
        if not tool_calls:
            return StopEvent(result=response)
        return ToolCallEvent(tool_calls=tool_calls)

    @step
    async def handle_tool_calls(self, ctx: Context, ev: ToolCallEvent) -> InputEvent:
        """Run requested tools, collect outputs, feed them back into memory."""
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}
        tool_msgs = [
            run_tool_call(tools_by_name, tool_call) for tool_call in ev.tool_calls
        ]
        memory = await ctx.store.get("memory")
        for msg in tool_msgs:
            await memory.aput(msg)
        await ctx.store.set("memory", memory)
        return InputEvent(input=await self._chat_history(ctx))


def load_llm(settings: Settings) -> FunctionCallingLLM:
    """Configure the OpenAI-compatible chat LLM from settings."""
    return OpenAILike(
        model=settings.llm_model,
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key,
        is_chat_model=True,
        is_function_calling_model=True,
    )


def build_agent(
    settings: Settings,
    tools: list[BaseTool] | None = None,
    system_prompt: str = "",
    timeout: int = 120,
) -> FunctionCallingAgentWorkflow:
    """Build the function calling agent workflow with the configured LLM."""
    if not system_prompt:
        raise ValueError("build_agent requires a system_prompt")
    return FunctionCallingAgentWorkflow(
        llm=load_llm(settings),
        tools=tools or [],
        system_prompt=system_prompt,
        timeout=timeout,
        memory_token_limit=settings.memory_token_limit,
    )
