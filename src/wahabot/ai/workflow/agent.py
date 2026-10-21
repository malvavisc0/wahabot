"""The function-calling agent workflow."""

import asyncio
from collections.abc import Callable
from typing import Any, cast, override

from llama_index.core.base.llms.types import (
    ChatMessage,
)
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
from loguru import logger

from wahabot.ai.events import InputEvent, ToolCallEvent
from wahabot.ai.messages import (
    TURN_SERIALS_KWARG,
)
from wahabot.ai.tools.notebook import NOTEBOOK_INSTRUCTIONS
from wahabot.ai.tools.whatsapp import (
    current_target,
)
from wahabot.ai.triage import (
    TriageGate,
    TriageResult,
    is_tool_allowed,
)
from wahabot.ai.workflow.correction_mixin import CorrectionMixin
from wahabot.ai.workflow.delivery import DELIVERY_TOOLS
from wahabot.ai.workflow.memory_mixin import MemoryMixin
from wahabot.ai.workflow.prompts import (
    POST_DELIVERY_WRAP_UP_PROMPT,
    build_thread_index,
    load_llm,
    parse_wrap_up,
    raise_missing_llm,
    recap_line,
    topic_slug,
    truncate_note,
    warn_context_mismatch,
)
from wahabot.ai.workflow.response_mixin import ResponseMixin
from wahabot.ai.workflow.step_mixin import StepMixin
from wahabot.ai.workflow.triage_mixin import TriageMixin
from wahabot.core.audit import save_action
from wahabot.core.learned import RecapMerger
from wahabot.core.notebook import (
    Notebook,
)
from wahabot.core.thread_index import (
    ThreadIndexer,
)
from wahabot.settings import Settings

__all__ = [
    "FunctionCallingAgentWorkflow",
    "build_agent",
    "build_thread_index",
    "load_llm",
    "parse_wrap_up",
    "recap_line",
    "topic_slug",
    "truncate_note",
    "warn_context_mismatch",
]


class FunctionCallingAgentWorkflow(
    StepMixin,
    MemoryMixin,
    TriageMixin,
    ResponseMixin,
    CorrectionMixin,
    Workflow,
):
    def __init__(
        self,
        *args: Any,
        llm: FunctionCallingLLM | None = None,
        tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        prompt_renderer: Callable[[], str] | None = None,
        memory_token_limit: int = 8000,
        tool_round_limit: int = 50,
        max_concurrent_llm: int = 4,
        settings: Settings | None = None,
        notebook: Notebook | None = None,
        rules_renderer: Callable[[], str] | None = None,
        thread_index: ThreadIndexer | None = None,
        triage: TriageGate | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.prompt_renderer = prompt_renderer
        self.llm = llm or raise_missing_llm()
        self.memory_token_limit = memory_token_limit
        self.tool_round_limit = tool_round_limit
        self.llm_semaphore = asyncio.Semaphore(max_concurrent_llm)
        self.settings = settings
        self.notebook = notebook
        self.rules_renderer = rules_renderer
        self.triage = triage
        self.triage_tasks: dict[int, asyncio.Task[TriageResult]] = {}
        # Tier 1: the per-chat rolling recap. Only a
        # persisted session can carry one (the recap is a file next to the
        # memory buffer), and memory_persist is the family kill switch.
        self.recap_merger = (
            RecapMerger(self.llm, settings.data_dir, self.llm_semaphore)
            if settings is not None
            and settings.memory_persist
            and settings.learned_memory
            else None
        )
        # Thread recall shares the same file family and kill
        # switches. Without an embeddings URL the index is FTS5-only.
        self.thread_index = build_thread_index(settings, thread_index)
        warn_context_mismatch(
            settings.llm_context_window if settings is not None else 0,
            self.memory_token_limit,
        )
        assert self.llm.metadata.is_function_calling_model

    @override
    def audit(self, kind: str, **fields: Any) -> None:
        """Journal one bot action.

        Fail-soft by construction (``save_action`` swallows disk
        errors), so auditing never breaks a run. No-op without
        settings — bare test workflows stay silent. The chat id comes
        from the run-scoped target binding when one is active; tool
        threads run inside a bound run, so it always is there.
        """
        if self.settings is None:
            return
        try:
            chat_id = current_target().chat_id
        except RuntimeError:
            chat_id = ""
        save_action(
            self.settings.data_dir,
            self.settings.session,
            kind,
            chat_id=chat_id,
            **fields,
        )

    def rendered_system_prompt(self) -> str | None:
        """The system prompt with ``prompt_renderer`` applied when set."""
        if self.prompt_renderer is not None:
            return self.prompt_renderer()
        return self.system_prompt or None

    @override
    def system_message(self) -> ChatMessage | None:
        """The system prompt as a message, or None when unset."""
        prompt = self.rendered_system_prompt()
        if self.notebook is not None:
            prompt = f"{prompt or ''}\n\n# Notebook\n\n{NOTEBOOK_INSTRUCTIONS}".strip()
        if not prompt:
            return None
        return ChatMessage(role="system", content=prompt)

    @override
    def rendered_session_rules(self) -> str:
        """The session rules line, or ""; rules never fail a run."""
        if self.rules_renderer is None:
            return ""
        try:
            return self.rules_renderer().strip()
        except Exception:
            logger.warning("Session rules rendering failed; continuing without rules")
            return ""

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
            await self.repair_memory(ctx)
        await ctx.store.set("image_blocks", getattr(ev, "image_blocks", None))
        await ctx.store.set("tool_rounds", 0)
        await ctx.store.set("last_tool_calls", [])
        serials = await ctx.store.get("turn_serials", default=None)
        if not await self.already_in_buffer(memory, str(ev.input), serials):
            kwargs: dict[str, Any] = (
                {TURN_SERIALS_KWARG: list(serials)} if serials else {}
            )
            await memory.aput(
                ChatMessage(role="user", content=str(ev.input), additional_kwargs=kwargs)
            )
        await ctx.store.set("memory", memory)
        return InputEvent(input=await self.chat_history(ctx))

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        """Call the LLM with tools + history; loop on tool calls.

        An assistant response with empty content is a chosen silence,
        not data: it is never stored to memory, so the rolling buffer
        never carries empty ``assistant`` turns that some
        OpenAI-compatible providers reject outright.

        A final text produced *after* a delivery tool succeeded never
        reached the chat (the one-delivery latch already fired), so it
        is dropped — neither stored nor returned. What the chat did see
        (the sent text, the reaction) is preserved by collapsing the
        delivery tool group into a plain assistant message, so the
        model's self-history mirrors the chat and it can see what it
        already said. The collapse runs on every run-end path that
        follows a delivery — including the early-stop paths (chosen
        silence, repeated tool call): the delivery already happened, so
        its group must never survive raw into the retained history
        (persistence would otherwise snapshot the scaffolding to
        disk). Research runs (no delivery tool involved) keep their
        final answer untouched.
        """
        rounds = await self.next_round(ctx)
        chat_history = await self.populated_history(ctx, ev)
        if self.any_delivery():
            # The delivery already fired: everything from here on can
            # only produce the wrap-up note. Say so instead of letting
            # the model guess and discarding its answer (bug 6).
            chat_history = [
                *chat_history,
                ChatMessage(role="user", content=POST_DELIVERY_WRAP_UP_PROMPT),
            ]
        async with self.llm_semaphore:
            response = await self.llm.achat_with_tools(
                self.tools,
                chat_history=chat_history,
                allow_parallel_tool_calls=True,
            )
        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )
        return await self.route_tool_calls(ctx, response, tool_calls, rounds)

    @step
    async def handle_tool_calls(self, ctx: Context, ev: ToolCallEvent) -> InputEvent:
        """Run requested tools, collect outputs, feed them back into memory."""
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}
        # Execute reads/research before delivery calls in a mixed model batch.
        # Once delivery succeeds, no non-delivery action may run afterward.
        ordered_calls = sorted(
            ev.tool_calls, key=lambda call: call.tool_name in DELIVERY_TOOLS
        )
        verdict = await self.triage_result(ctx)
        hardened = bool(
            verdict is not None and verdict.verdict.adversarial and not verdict.failed
        )
        tool_msgs: list[ChatMessage] = []
        for tool_call in ordered_calls:
            if hardened and not is_tool_allowed(tool_call):
                tool_msgs.append(self.deny_tool_call(tool_call))
            else:
                tool_msgs.append(
                    await self.run_and_audit_tool_call(tools_by_name, tool_call)
                )
        memory = await ctx.store.get("memory")
        for msg in tool_msgs:
            await memory.aput(msg)
        await ctx.store.set("memory", memory)
        return InputEvent(input=await self.chat_history(ctx))


def build_agent(
    settings: Settings,
    tools: list[BaseTool] | None = None,
    system_prompt: str = "",
    prompt_renderer: Callable[[], str] | None = None,
    rules_renderer: Callable[[], str] | None = None,
    triage: TriageGate | None = None,
) -> FunctionCallingAgentWorkflow:
    """Build the function calling agent workflow with the configured LLM.

    ``prompt_renderer`` (optional) returns the freshly rendered system
    prompt on every run, so date/time placeholders and config edits stay
    current for the life of the process. ``rules_renderer`` (optional)
    feeds the pinned state block's session rules the same way.
    """
    if not system_prompt:
        raise ValueError("build_agent requires a system_prompt")
    return FunctionCallingAgentWorkflow(
        llm=load_llm(settings),
        tools=tools or [],
        system_prompt=system_prompt,
        prompt_renderer=prompt_renderer,
        # No workflow timeout: llama-index-workflows 2.25.0 enforces
        # timeout= as a cumulative alive-time budget over the whole
        # Context, and this bot reuses one Context per chat — the
        # budget would silently sum every run's latency and
        # permanently kill busy chats at the cap. WAHABOT_RUN_TIMEOUT
        # is a true per-run wall-clock cap enforced around agent.run()
        # in wahabot.ai.context.handle_message.
        timeout=None,
        memory_token_limit=settings.memory_token_limit,
        tool_round_limit=settings.tool_round_limit,
        settings=settings,
        notebook=Notebook(settings.data_dir),
        rules_renderer=rules_renderer,
        triage=triage,
    )
