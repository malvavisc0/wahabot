"""LlamaIndex function calling agent workflow.

Implements the reference function calling agent workflow
(https://developers.llamaindex.ai/python/examples/workflow/function_calling_agent/):
user messages are added to a per-chat ``ChatMemoryBuffer``, the LLM
gets tools + history, tool results loop back, and the final text comes
back in ``StopEvent.result``.
"""

import asyncio
from collections.abc import Callable
from functools import partial
from typing import Any, cast

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    ToolCallBlock,
)
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

from wahabot.ai.events import InputEvent, ToolCallEvent
from wahabot.ai.history import (
    MAX_TOOL_RESULT_TOKENS,
    sanitize_chat_history,
    trim_to_budget,
)
from wahabot.settings import Settings

__all__ = [
    "FunctionCallingAgentWorkflow",
    "build_agent",
    "load_llm",
]

#: Mirrors LlamaIndex's DEFAULT_EARLY_STOPPING_PROMPT (base_agent.py):
#: when the tool-round budget is spent, the model is asked for a final
#: answer without tools instead of the run being cut off mid-research.
_EARLY_STOPPING_PROMPT = (
    "You have reached the maximum number of tool rounds ({limit}). "
    "Based on the information gathered so far, provide a helpful final "
    "response to the user's original message. Do not attempt to use any "
    "more tools. If you already sent your reply, say nothing more — "
    "answer with an empty message."
)


def raise_missing_llm() -> FunctionCallingLLM:
    """Fail fast when no llm was passed to the workflow."""
    raise ValueError("FunctionCallingAgentWorkflow requires an llm")


async def run_tool_call(
    tools_by_name: dict[str, BaseTool], tool_call: ToolSelection
) -> ChatMessage:
    """Run one tool call, returning its output (or the failure) as a tool message.

    Tool functions are sync and do network I/O (WAHA, web), so they run
    in a worker thread via ``asyncio.to_thread`` to keep the event loop
    responsive. Outputs are capped at ``MAX_TOOL_RESULT_TOKENS`` chars:
    list tools embed WAHA's raw ``_data`` blobs (15 messages ≈ 58k
    chars), and a tool group that large evicts the user turn from the
    prompt when the buffer's real tokenizer re-trims.
    """
    tool = tools_by_name.get(tool_call.tool_name)
    kwargs = {"tool_call_id": tool_call.tool_id, "name": tool_call.tool_name}
    if tool is None:
        content = f"Tool {tool_call.tool_name} does not exist"
    else:
        try:
            fn = partial(tool, **tool_call.tool_kwargs)
            called = await asyncio.to_thread(fn)
            content = called.content
        except Exception as exc:
            content = f"Encountered error in tool call: {exc}"
    if len(content) > MAX_TOOL_RESULT_TOKENS:
        content = content[:MAX_TOOL_RESULT_TOKENS] + "… (truncated)"
    return ChatMessage(role="tool", content=content, additional_kwargs=kwargs)


def _token_count(msg: ChatMessage) -> int:
    """Estimate a message's token count from its content length (1 char ≈ 1 token).

    Tool-call arguments ride on the message as blocks/kwargs, not in
    ``content`` — without them a long tool-call history looks nearly
    free and the true prompt size drifts far past the budget.
    """
    args = "".join(
        str(block.tool_kwargs) for block in msg.blocks if isinstance(block, ToolCallBlock)
    ) or str(msg.additional_kwargs.get("tool_calls", ""))
    return len(str(msg.content or "")) + len(args)


class FunctionCallingAgentWorkflow(Workflow):
    """Stateful function calling agent built from plain workflow steps."""

    def __init__(
        self,
        *args: Any,
        llm: FunctionCallingLLM | None = None,
        tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        prompt_renderer: Callable[[], str] | None = None,
        memory_token_limit: int = 8000,
        tool_round_limit: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.prompt_renderer = prompt_renderer
        self.llm = llm or raise_missing_llm()
        self.memory_token_limit = memory_token_limit
        self.tool_round_limit = tool_round_limit
        assert self.llm.metadata.is_function_calling_model

    def _rendered_system_prompt(self) -> str | None:
        """The system prompt with ``prompt_renderer`` applied when set."""
        if self.prompt_renderer is not None:
            return self.prompt_renderer()
        return self.system_prompt or None

    def _system_message(self) -> ChatMessage | None:
        """The system prompt as a message, or None when unset."""
        prompt = self._rendered_system_prompt()
        if not prompt:
            return None
        return ChatMessage(role="system", content=prompt)

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
        prompt = self._rendered_system_prompt()
        if not prompt:
            return 0
        tokenizer = cast(Callable[[str], list[Any]], memory.tokenizer_fn)
        tokens = len(tokenizer(prompt))
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
            from_defaults = cast(
                Callable[..., ChatMemoryBuffer], ChatMemoryBuffer.from_defaults
            )
            memory = from_defaults(token_limit=self.memory_token_limit, llm=self.llm)
        else:
            await self._repair_memory(ctx)
        await ctx.store.set("image_blocks", getattr(ev, "image_blocks", None))
        await ctx.store.set("tool_rounds", 0)
        await memory.aput(ChatMessage(role="user", content=str(ev.input)))
        await ctx.store.set("memory", memory)
        return InputEvent(input=await self._chat_history(ctx))

    @staticmethod
    def _with_image(
        history: list[ChatMessage], image_blocks: list[Any] | None
    ) -> list[ChatMessage]:
        """Return *history* with image blocks on the newest user message.

        The newest user message is replaced by a copy carrying the
        blocks — the memory-stored originals are never mutated, so
        images ride the first LLM call of a run without ever entering
        the rolling buffer (no megabyte payloads, no re-sends in a
        tool-call loop).
        """
        if not image_blocks or not history:
            return history
        out = list(history)
        for i in range(len(out) - 1, -1, -1):
            if out[i].role == MessageRole.USER:
                msg = out[i]
                out[i] = ChatMessage(
                    role=msg.role,
                    blocks=[*msg.blocks, *image_blocks],
                    additional_kwargs=dict(msg.additional_kwargs),
                )
                break
        return out

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        """Call the LLM with tools + history; loop on tool calls.

        An assistant response with empty content is a chosen silence,
        not data: it is never stored to memory, so the rolling buffer
        never carries empty ``assistant`` turns that some
        OpenAI-compatible providers reject outright.
        """
        rounds = await self._next_round(ctx)
        image_blocks = await ctx.store.get("image_blocks", default=None)
        if image_blocks:
            await ctx.store.set("image_blocks", None)
        chat_history = self._with_image(list(ev.input), image_blocks)
        response = await self.llm.achat_with_tools(self.tools, chat_history=chat_history)
        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )
        memory = await ctx.store.get("memory")
        if tool_calls or str(response.message.content or "").strip():
            await memory.aput(response.message)
        await ctx.store.set("memory", memory)
        if not tool_calls:
            return StopEvent(result=response)
        if rounds >= self.tool_round_limit:
            logger.warning(
                "Tool round limit ({limit}) hit; forcing a final answer",
                limit=self.tool_round_limit,
            )
            return StopEvent(result=await self._early_stopping_response(ctx))
        return ToolCallEvent(tool_calls=tool_calls)

    async def _early_stopping_response(self, ctx: Context) -> ChatResponse:
        """One last tool-free LLM call, mirroring LlamaIndex's
        ``early_stopping_method="generate"``: the model is told the
        iteration budget is spent and asked for a final answer.

        A hard stop would break legitimate research that needs many
        tool calls; instead the model gets one forced chance to wrap
        up. The round counter is already over the limit, so even a
        defiant model requesting tools again cannot continue the run.
        """
        logger.warning(
            "Generating early stopping response after {limit} rounds",
            limit=self.tool_round_limit,
        )
        memory = await ctx.store.get("memory")
        history = await memory.aget_all()
        system = self._system_message()
        messages = ([system] if system else []) + list(history)
        messages.append(
            ChatMessage(
                role="user",
                content=_EARLY_STOPPING_PROMPT.format(limit=self.tool_round_limit),
            )
        )
        return await self.llm.achat(messages)

    async def _next_round(self, ctx: Context) -> int:
        """Increment and return this run's tool-round counter."""
        rounds = await ctx.store.get("tool_rounds", default=0)
        rounds += 1
        await ctx.store.set("tool_rounds", rounds)
        return rounds

    @step
    async def handle_tool_calls(self, ctx: Context, ev: ToolCallEvent) -> InputEvent:
        """Run requested tools, collect outputs, feed them back into memory."""
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}
        tool_msgs = [
            await run_tool_call(tools_by_name, tool_call) for tool_call in ev.tool_calls
        ]
        memory = await ctx.store.get("memory")
        for msg in tool_msgs:
            await memory.aput(msg)
        await ctx.store.set("memory", memory)
        return InputEvent(input=await self._chat_history(ctx))


class ObservableOpenAILike(OpenAILike):
    """OpenAILike whose instrumentation payload names model/temperature.

    The OTel llama-index instrumentor reads ``model_dict["model"]`` and
    ``model_dict["temperature"]`` for the ``gen_ai.request.*`` span
    attributes, but the base ``to_payload`` only exposes metadata
    (``model_name``, no temperature) — leaving both as ``None`` and
    spamming OTel "Invalid type NoneType" warnings per LLM call.
    """

    def to_payload(self) -> dict[str, Any]:
        return {
            **super().to_payload(),
            "model": self.model,
            "temperature": self.temperature,
        }


def load_llm(settings: Settings) -> FunctionCallingLLM:
    """Configure the OpenAI-compatible chat LLM from settings."""
    return ObservableOpenAILike(
        model=settings.llm_model,
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
        # No client retries: a hung endpoint must fail fast within the
        # workflow timeout, not stack 60s attempts until it blows up.
        max_retries=0,
        is_chat_model=True,
        is_function_calling_model=True,
    )


def build_agent(
    settings: Settings,
    tools: list[BaseTool] | None = None,
    system_prompt: str = "",
    prompt_renderer: Callable[[], str] | None = None,
    timeout: int = 120,
) -> FunctionCallingAgentWorkflow:
    """Build the function calling agent workflow with the configured LLM.

    ``prompt_renderer`` (optional) returns the freshly rendered system
    prompt on every run, so date/time placeholders and config edits stay
    current for the life of the process.
    """
    if not system_prompt:
        raise ValueError("build_agent requires a system_prompt")
    return FunctionCallingAgentWorkflow(
        llm=load_llm(settings),
        tools=tools or [],
        system_prompt=system_prompt,
        prompt_renderer=prompt_renderer,
        timeout=timeout,
        memory_token_limit=settings.memory_token_limit,
        tool_round_limit=settings.tool_round_limit,
    )
