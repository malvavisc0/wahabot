"""LlamaIndex function calling agent workflow.

Implements the reference function calling agent workflow
(https://developers.llamaindex.ai/python/examples/workflow/function_calling_agent/):
user messages are added to a per-chat ``ChatMemoryBuffer``, the LLM
gets tools + history, tool results loop back, and the final text comes
back in ``StopEvent.result``.
"""

import asyncio
import json
from collections.abc import Callable
from functools import partial
from typing import Any, cast, override

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    TextBlock,
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
    inbound_message_id,
    sanitize_chat_history,
    tool_calls,
    trim_to_budget,
    wire_call,
)
from wahabot.ai.messages import TURN_HANDLED_KWARG
from wahabot.ai.tools.whatsapp import current_target, log_action_reason
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

#: The tools that deliver content to a chat; each latches the shared
#: holder at most once per run.
DELIVERY_TOOLS = frozenset(
    {"send_message", "send_image", "forward_message", "react_to_message"}
)

SILENCE_TOOL = "stay_silent"


def raise_missing_llm() -> FunctionCallingLLM:
    """Fail fast when no llm was passed to the workflow."""
    raise ValueError("FunctionCallingAgentWorkflow requires an llm")


def tool_call_key(tool_call: ToolSelection) -> str:
    """A comparable identity for one tool call: name + sorted arguments.

    Two calls are "the same" when they target the same tool with the
    same arguments — the loop detector compares consecutive rounds by
    these keys. Arguments are sorted so dict key order in the model's
    JSON does not defeat the comparison.
    """
    kwargs = dict(tool_call.tool_kwargs)
    return f"{tool_call.tool_name}:{sorted(kwargs.items())}"


async def run_tool_call(
    tools_by_name: dict[str, BaseTool], tool_call: ToolSelection
) -> ChatMessage:
    """Run one tool call, returning its output (or the failure) as a tool message.

    Tool functions are sync and do network I/O (WAHA, web), so they run
    in a worker thread via ``asyncio.to_thread`` to keep the event loop
    responsive. Tool outputs are not truncated here: every tool bounds
    its own payload at the source (``visit_url`` capping its text,
    ``web_search`` its per-result snippets, the list tools slimming
    ``_data`` and fitting to a whole-message budget), so a blunt
    workflow-level char cutoff would only mangle already-curated JSON
    envelopes.

    Each call is logged: one INFO line with its outcome (completed,
    unknown), or WARNING with the exception when the tool raised — tool
    failures are what gets grepped for, so they carry the error detail.
    """
    tool = tools_by_name.get(tool_call.tool_name)
    kwargs = {"tool_call_id": tool_call.tool_id, "name": tool_call.tool_name}
    failure: Exception | None = None
    if tool is None:
        content = f"Tool {tool_call.tool_name} does not exist"
        outcome = "unknown"
    else:
        try:
            fn = partial(tool, **tool_call.tool_kwargs)
            called = await asyncio.to_thread(fn)
            content = called.content
            outcome = "completed"
        except Exception as exc:
            content = f"Encountered error in tool call: {exc}"
            outcome = "failed"
            failure = exc
    if failure is not None:
        logger.warning(
            "Tool call {tool} failed: {exc}", tool=tool_call.tool_name, exc=failure
        )
    else:
        logger.info(
            "Tool call {tool}: {outcome}", tool=tool_call.tool_name, outcome=outcome
        )
    return ChatMessage(role="tool", content=content, additional_kwargs=kwargs)


def tool_call_names(message: ChatMessage) -> list[str]:
    """Names of the tools an assistant message called (empty for plain text)."""
    return [call.name for call in tool_calls(message)]


def delivered_text(message: ChatMessage) -> str:
    """The delivered content of an ``ok`` delivery tool result, else "".

    A failed delivery left nothing in the chat, so its pair stays as
    the model's record of what went wrong — only a delivered result
    collapses.
    """
    if message.role != MessageRole.TOOL:
        return ""
    try:
        envelope = json.loads(str(message.content or "{}"))
    except json.JSONDecodeError:
        return ""
    if not envelope.get("ok"):
        return ""
    return delivery_content(envelope)


def delivery_content(envelope: dict[str, Any]) -> str:
    """The chat-visible content of a delivered tool envelope, else "".

    Text sends keep their full text; media/file sends and forwards keep
    a short bracketed marker (with the caption when present); reactions
    keep their emoji.
    """
    if text := envelope.get("text"):
        return str(text)
    if "reaction" in envelope:
        return (
            f"[removed reaction from {envelope.get('message_id')}]"
            if envelope.get("removed")
            else str(envelope["reaction"])
        )
    caption = str(envelope.get("caption") or "").strip()
    if url := envelope.get("url"):
        return f"[media: {url}] {caption}".strip()
    if mimetype := envelope.get("mimetype"):
        return f"[file: {mimetype}] {caption}".strip()
    if message_id := envelope.get("message_id"):
        return f"[forwarded {message_id}]"
    return ""


def tool_call_ids(message: ChatMessage) -> list[str]:
    """All tool-call ids of an assistant message."""
    return [call.call_id for call in tool_calls(message) if call.call_id]


def delivery_call_ids(message: ChatMessage) -> list[str]:
    """Tool-call ids of an assistant message's delivery calls only."""
    return [
        call.call_id
        for call in tool_calls(message)
        if call.name in DELIVERY_TOOLS and call.call_id
    ]


def tool_result_indices(messages: list[ChatMessage]) -> dict[str, int]:
    """Tool-call id → index of its result message, across the whole history."""
    return {
        str(message.additional_kwargs.get("tool_call_id", "")): j
        for j, message in enumerate(messages)
        if message.role == MessageRole.TOOL
    }


def batch_closes_history(all_ids: list[str], results: dict[str, int], total: int) -> bool:
    """True when every call's result exists and together they end the history.

    A batch with unfulfilled calls is left for
    :func:`sanitize_chat_history` to drop as a unit; a batch followed by
    more conversation belongs to an earlier run, already collapsed at
    that run's end.
    """
    indices = sorted(results[call_id] for call_id in all_ids if call_id in results)
    return len(indices) == len(all_ids) and indices == list(
        range(total - len(indices), total)
    )


def delivered_texts(
    call_ids: list[str], results: dict[str, int], messages: list[ChatMessage]
) -> list[str]:
    """The delivered texts of *call_ids*' results (skipping failures)."""
    return [
        text
        for call_id in call_ids
        if call_id in results and (text := delivered_text(messages[results[call_id]]))
    ]


def delivery_group_bounds(
    messages: list[ChatMessage],
) -> tuple[int, list[int], str] | None:
    """(assistant index, result indices, delivered text) of the last delivery group.

    A group is an assistant message whose tool calls include a delivery
    plus the ``tool`` result messages of its *delivery* calls — located
    by tool-call id, not position, so a parallel batch mixing delivery
    and research calls collapses correctly (research calls and their
    results stay). Only a group whose *whole* tool batch closes the
    message list qualifies.
    """
    results = tool_result_indices(messages)
    for i in range(len(messages) - 1, -1, -1):
        all_ids = tool_call_ids(messages[i])
        if not all_ids or not set(tool_call_names(messages[i])) & DELIVERY_TOOLS:
            continue
        if not batch_closes_history(all_ids, results, len(messages)):
            return None
        call_ids = delivery_call_ids(messages[i])
        texts = delivered_texts(call_ids, results, messages)
        if not texts:
            return None
        indices = sorted(results[c] for c in call_ids if c in results)
        return i, indices, " ".join(texts)
    return None


def collapse_send_group(messages: list[ChatMessage]) -> list[ChatMessage] | None:
    """*messages* with the last delivery group folded into the assistant turn.

    The delivery calls are stripped from the assistant message and their
    results removed; the delivered text becomes the message's content,
    so the model's self-history records what the chat saw. A parallel
    batch keeps its research calls (on the same message, so the history
    stays API-valid and dedup-safe). Returns None when there is nothing
    to collapse.
    """
    bounds = delivery_group_bounds(messages)
    if bounds is None:
        return None
    i, result_indices, text = bounds
    collapsed = [m for j, m in enumerate(messages) if j not in {i, *result_indices}]
    collapsed.insert(i, strip_delivery_calls(messages[i], text))
    return collapsed


def strip_delivery_calls(message: ChatMessage, text: str) -> ChatMessage:
    """*message* without its delivery calls, carrying the delivered text.

    A parallel batch keeps its research calls (blocks or kwargs, whichever
    the message carries) with the delivered text as content; a
    delivery-only message becomes the plain assistant text.
    """
    if any(isinstance(block, ToolCallBlock) for block in message.blocks):
        return strip_delivery_blocks(message, text)
    return strip_delivery_kwargs(message, text)


def strip_delivery_blocks(message: ChatMessage, text: str) -> ChatMessage:
    """Block-carried *message* without its delivery ``ToolCallBlock``s."""
    remaining = [
        block
        for block in message.blocks
        if isinstance(block, ToolCallBlock) and block.tool_name not in DELIVERY_TOOLS
    ]
    if not remaining:
        return ChatMessage(role=MessageRole.ASSISTANT, content=text)
    return ChatMessage(
        role=MessageRole.ASSISTANT, blocks=[TextBlock(text=text), *remaining]
    )


def strip_delivery_kwargs(message: ChatMessage, text: str) -> ChatMessage:
    """Kwargs-carried *message* without its delivery ``tool_calls``."""
    remaining = [
        call
        for call in message.additional_kwargs.get("tool_calls") or []
        if (parsed := wire_call(call)) and parsed.name not in DELIVERY_TOOLS
    ]
    if not remaining:
        return ChatMessage(role=MessageRole.ASSISTANT, content=text)
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        content=text,
        additional_kwargs={**message.additional_kwargs, "tool_calls": remaining},
    )


def token_count(msg: ChatMessage) -> int:
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
    """Stateful function calling agent built from plain workflow steps.

    Runs of the same workflow may execute concurrently (different
    chats in parallel): everything a run needs — its delivery holder,
    memory, tool-round counters — lives in its own ``Context`` or in
    the run-scoped target binding (``bind_target``), never on the
    workflow instance.
    """

    #: Bounds concurrent LLM calls across all runs of this workflow: a
    #: busy group burst fans out provider cost, so runs queue here
    #: instead of all hitting the endpoint at once.
    llm_semaphore: asyncio.Semaphore

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
        assert self.llm.metadata.is_function_calling_model

    def rendered_system_prompt(self) -> str | None:
        """The system prompt with ``prompt_renderer`` applied when set."""
        if self.prompt_renderer is not None:
            return self.prompt_renderer()
        return self.system_prompt or None

    def system_message(self) -> ChatMessage | None:
        """The system prompt as a message, or None when unset."""
        prompt = self.rendered_system_prompt()
        if not prompt:
            return None
        return ChatMessage(role="system", content=prompt)

    async def chat_history(self, ctx: Context) -> list[ChatMessage]:
        """The trimmed, structurally valid conversation plus system prompt.

        The conversation is sanitised (balanced tool groups, alternating
        turns) and trimmed to the memory token budget. The system message
        is kept out of the rolling buffer so the token trim can never
        evict it; its cost is accounted via ``initial_token_count`` so the
        whole prompt still fits the budget.
        """
        memory = await ctx.store.get("memory")
        system = self.system_message()
        messages = await memory.aget_all()
        messages = sanitize_chat_history(messages, drop_trailing_user=False)
        messages = trim_to_budget(messages, self.memory_token_limit, token_count)
        await memory.aset(messages)
        initial = self.system_token_count(memory)
        history = await memory.aget(input=None, initial_token_count=initial)
        return ([system] if system else []) + history

    def system_token_count(self, memory: ChatMemoryBuffer) -> int:
        """Tokens the system prompt costs, clamped to the memory budget.

        ``ChatMemoryBuffer.get`` raises when ``initial_token_count``
        exceeds its token limit; a huge system prompt would crash every
        run. Clamp so the conversation simply gets no room instead.
        """
        prompt = self.rendered_system_prompt()
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

    async def repair_memory(self, ctx: Context) -> None:
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
            await self.repair_memory(ctx)
        await ctx.store.set("image_blocks", getattr(ev, "image_blocks", None))
        await ctx.store.set("tool_rounds", 0)
        await ctx.store.set("last_tool_calls", [])
        if not await self.already_in_buffer(memory, str(ev.input)):
            await memory.aput(ChatMessage(role="user", content=str(ev.input)))
        await ctx.store.set("memory", memory)
        return InputEvent(input=await self.chat_history(ctx))

    @staticmethod
    async def already_in_buffer(memory: ChatMemoryBuffer, incoming: str) -> bool:
        """Whether *incoming*'s message-id note already rides the buffer.

        A WAHA redelivery (the handler drops the seen marker so a
        crashed run retries) re-enters ``prepare_chat_history`` with
        the same text. The previous attempt's turn — merged with the
        turns before it, and stamped ``turn_handled`` by an *earlier*
        completed run — would survive the trailing-user drop, so a
        plain re-append duplicates the message once per retry. The
        serialized id in the ``[message id: …]`` note identifies the
        exact event: already present means this is a redelivery whose
        body the buffer already carries, and the run proceeds without
        appending (the agent still gets its fresh run).
        """
        message_id = inbound_message_id(incoming)
        if not message_id:
            return False
        messages = await memory.aget_all()
        for message in reversed(messages):
            if message.role != MessageRole.USER:
                continue
            if message_id in str(message.content or ""):
                return True
            # Only the newest user turn can carry it: older turns are
            # earlier messages, already distinct in the buffer.
            break
        return False

    @staticmethod
    def with_image(
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
        async with self.llm_semaphore:
            response = await self.llm.achat_with_tools(
                self.tools,
                chat_history=chat_history,
                allow_parallel_tool_calls=True,
            )
        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )
        if any(call.tool_name == SILENCE_TOOL for call in tool_calls):
            self.log_silence_reason(tool_calls)
            logger.debug("Stopping run: model chose stay_silent")
            await self.mark_turn_handled(ctx)
            await self.collapse_delivery(ctx)
            return self.stopped_response()
        if tool_calls and await self.repeats_tool_call(ctx, tool_calls):
            await self.mark_turn_handled(ctx)
            await self.collapse_delivery(ctx)
            return self.stopped_response()
        if not tool_calls:
            self.warn_accidental_silence(response, rounds)
            delivered = self.any_delivery()
            await self.remember(ctx, response, tool_calls, skip_text=delivered)
            await self.mark_turn_handled(ctx)
            await self.collapse_delivery(ctx)
            return StopEvent(result=self.drop_post_delivery_text(response))
        if self.delivery_complete(tool_calls) or rounds >= self.tool_round_limit:
            # Do not store calls which will not be executed: the chat API
            # requires every advertised call to have a tool response.
            reason = (
                "non-delivery round after completed delivery"
                if self.delivery_complete(tool_calls)
                else f"round limit {self.tool_round_limit}"
            )
            result = await self.wrap_up_response(ctx, reason)
            await self.mark_turn_handled(ctx)
            await self.collapse_delivery(ctx)
            return StopEvent(result=result)
        await self.remember(ctx, response, tool_calls)
        return ToolCallEvent(tool_calls=tool_calls)

    async def mark_turn_handled(self, ctx: Context) -> None:
        """Stamp the run's inbound user turn as handled conversation.

        ``prepare_chat_history`` appended the turn unstamped; the four
        stop paths above call this once the run has concluded. The
        stamp tells ``history``'s trailing-user drop that this message
        is real conversation a completed run read — in ``judicious``
        group mode a silent run's message must stay in context for the
        next run, not be discarded as scaffolding. Fail-soft: without
        the stamp the turn merely behaves like today (dropped on the
        next run's repair).
        """
        memory = await ctx.store.get("memory", default=None)
        if memory is None:
            return
        messages = await memory.aget_all()
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.role == MessageRole.USER and TURN_HANDLED_KWARG not in (
                msg.additional_kwargs
            ):
                messages[i] = ChatMessage(
                    role=msg.role,
                    blocks=list(msg.blocks),
                    content=msg.content,
                    additional_kwargs={**msg.additional_kwargs, TURN_HANDLED_KWARG: True},
                )
                break
        await memory.aset(messages)
        await ctx.store.set("memory", memory)

    @staticmethod
    def log_silence_reason(tool_calls: list[ToolSelection]) -> None:
        """Surface a ``stay_silent`` call's ``reason`` in the log.

        The workflow stops before executing the call (silence is
        terminal), so the tool's own logging never runs — this is the
        reason's only path to the audit trail. The reason is taken from
        the stay_silent call itself: a parallel batch may put other
        calls (no reason kwarg) first.
        """
        reason = str(
            next(
                (
                    call.tool_kwargs.get("reason", "")
                    for call in tool_calls
                    if call.tool_name == SILENCE_TOOL
                ),
                "",
            )
        )
        log_action_reason(SILENCE_TOOL, reason)

    async def populated_history(self, ctx: Context, ev: InputEvent) -> list[ChatMessage]:
        """The event's history, with one-shot image blocks spliced in.

        Image blocks ride the first LLM call of a run only; consuming
        them here clears the store so later rounds stay text-only.
        """
        image_blocks = await ctx.store.get("image_blocks", default=None)
        if image_blocks:
            await ctx.store.set("image_blocks", None)
        return self.with_image(list(ev.input), image_blocks)

    async def repeats_tool_call(
        self, ctx: Context, tool_calls: list[ToolSelection]
    ) -> bool:
        """True when this round's calls repeat the previous round's.

        A looping model re-issues the *exact same* tool call (same
        name, same arguments) round after round — the tool already
        answered "you already did that", so continuing only burns
        budget. The run stops here with an empty final response.
        """
        previous = await self.tool_calls_seen(ctx)
        current = [tool_call_key(call) for call in tool_calls]
        await ctx.store.set("last_tool_calls", current)
        repeated = bool(previous) and previous == current
        if repeated:
            logger.warning(
                "Stopping run: repeated identical tool call {call}", call=current
            )
        return repeated

    async def tool_calls_seen(self, ctx: Context) -> list[str]:
        """The previous round's tool-call keys (empty before round one)."""
        calls: list[str] | None = await ctx.store.get("last_tool_calls", default=None)
        return calls or []

    def any_delivery(self) -> bool:
        """True when any delivery tool already fired this run."""
        target = current_target()
        return bool(target.sent or target.reacted)

    def delivery_complete(self, tool_calls: list[ToolSelection]) -> bool:
        """True when a delivery already fired and this round adds none.

        Once a reply or reaction has been delivered, further *non-
        delivery* rounds are chatter territory — but a reaction may
        legitimately follow a reply (answer, then react) or vice
        versa, so a round consisting of delivery calls is still
        allowed through. The at-most-once latches in the tools and the
        repeated-call detector bound what a looping model can do here.
        """
        if not self.any_delivery():
            return False
        names = {call.tool_name for call in tool_calls}
        if names <= DELIVERY_TOOLS:
            logger.debug(
                "Delivery done; allowing follow-up delivery call {names}", names=names
            )
            return False
        return True

    async def remember(
        self,
        ctx: Context,
        response: ChatResponse,
        tool_calls: list[ToolSelection],
        *,
        skip_text: bool = False,
    ) -> None:
        """Store an assistant response in memory, unless it is empty chatter.

        *skip_text* drops a post-delivery final text: it was never sent
        to the chat, and memory mirrors the chat — storing it would
        record words nobody saw.

        Thinking models separate the reasoning block from the text with
        a leading blank line inside the text block; that separator is
        stripped before storage — it costs buffer tokens on every turn
        and its recurrence teaches the model to keep emitting it.
        """
        memory = await ctx.store.get("memory")
        message = response.message
        for block in message.blocks:
            if isinstance(block, TextBlock) and block.text:
                block.text = block.text.strip()
        has_text = bool(str(message.content or "").strip())
        if tool_calls or (has_text and not skip_text):
            await memory.aput(message)
        await ctx.store.set("memory", memory)

    async def collapse_delivery(self, ctx: Context) -> None:
        """Replace the last delivered tool group with plain assistant text.

        At run end the delivered pair (empty assistant tool-call plus
        its tool-result envelope) is swapped for one plain assistant
        message holding the delivered content, so the retained history
        reads like the chat it mirrors. The whole group is replaced
        atomically: a dangling tool call without its result is exactly
        what :func:`repair_memory` must never find. No-op when nothing
        was delivered (see docs/agent-workflow.md).
        """
        memory = await ctx.store.get("memory")
        messages = await memory.aget_all()
        collapsed = collapse_send_group(messages)
        if collapsed is not None:
            await memory.aset(collapsed)
            await ctx.store.set("memory", memory)

    @staticmethod
    def stopped_response() -> StopEvent:
        """A StopEvent carrying an empty reply: nothing more to say."""
        empty = ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""))
        return StopEvent(result=empty)

    def warn_accidental_silence(self, response: ChatResponse, rounds: int) -> None:
        """Warn when a run stops empty after research without any delivery.

        A chosen silence is the ``stay_silent`` tool or an empty first
        reply; an empty final answer *after tool rounds* with nothing
        delivered is the reasoning-model glitch where thinking is spent
        but no visible answer is produced. It raises no error and is
        indistinguishable from chosen silence at run time, so this
        warning is the only trace it leaves in the logs.
        """
        if rounds <= 1 or self.any_delivery():
            return
        if str(response.message.content or "").strip():
            return
        logger.warning(
            (
                "Stopping run: empty final answer after {rounds} rounds "
                "(nothing delivered, no stay_silent)"
            ),
            rounds=rounds,
        )

    async def wrap_up_response(self, ctx: Context, reason: str) -> ChatResponse:
        """One last tool-free LLM call after a delivered reply / round limit.

        Mirrors LlamaIndex's ``early_stopping_method="generate"``: the
        model is told the budget is spent and asked for a final answer
        without tools. The round counter is already over the limit, so
        even a defiant model requesting tools again cannot continue.
        A delivered reply makes the wrap-up text post-delivery chatter —
        it is dropped like any other post-delivery final text.
        """
        limit = self.tool_round_limit
        logger.warning("Generating wrap-up response ({reason})", reason=reason)
        messages = await self.chat_history(ctx)
        messages.append(
            ChatMessage(
                role="user",
                content=_EARLY_STOPPING_PROMPT.format(limit=limit),
            )
        )
        async with self.llm_semaphore:
            response = await self.llm.achat(messages)
        return self.drop_post_delivery_text(response)

    def drop_post_delivery_text(self, response: ChatResponse) -> ChatResponse:
        """Empty *response* when a delivery tool already fired this run.

        A final text after ``send_message``/``send_image``/
        ``forward_message``/``react_to_message`` succeeded was never
        sent to the chat — the one-delivery latch already fired — so it
        must not be returned (the handler would send it as a second
        reply) nor stored in memory (memory mirrors the chat; the
        delivered text is preserved by ``collapse_delivery`` instead).
        Research runs (no delivery) pass through untouched.
        """
        if not self.any_delivery() or not str(response.message.content or "").strip():
            return response
        logger.info(
            "Dropping post-delivery final text (reply already delivered via tool)"
        )
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""))

    async def next_round(self, ctx: Context) -> int:
        """Increment and return this run's tool-round counter."""
        rounds = await ctx.store.get("tool_rounds", default=0)
        rounds += 1
        await ctx.store.set("tool_rounds", rounds)
        return rounds

    @step
    async def handle_tool_calls(self, ctx: Context, ev: ToolCallEvent) -> InputEvent:
        """Run requested tools, collect outputs, feed them back into memory."""
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}
        # Execute reads/research before delivery calls in a mixed model batch.
        # Once delivery succeeds, no non-delivery action may run afterward.
        ordered_calls = sorted(
            ev.tool_calls, key=lambda call: call.tool_name in DELIVERY_TOOLS
        )
        tool_msgs = [
            await run_tool_call(tools_by_name, tool_call) for tool_call in ordered_calls
        ]
        memory = await ctx.store.get("memory")
        for msg in tool_msgs:
            await memory.aput(msg)
        await ctx.store.set("memory", memory)
        return InputEvent(input=await self.chat_history(ctx))


class ObservableOpenAILike(OpenAILike):
    """OpenAILike whose instrumentation payload names model/temperature.

    Exists because the OTel llama-index instrumentor reads
    ``model_dict["model"]``/``["temperature"]`` for its span attributes
    but the base ``to_payload`` exposes neither (external constraint;
    see docs/agent-workflow.md).
    """

    @override
    def to_payload(self) -> dict[str, Any]:
        return {
            **super().to_payload(),
            "model": self.model,
            "temperature": self.temperature,
        }


def load_llm(settings: Settings) -> FunctionCallingLLM:
    """Configure the OpenAI-compatible chat LLM from settings.

    Sampling options follow the model card (see ``Settings``).
    ``top_p``/``presence_penalty`` ride ``additional_kwargs`` (merged
    into the request body); ``top_k``/``min_p``/``repetition_penalty``
    ride ``extra_body`` because the OpenAI SDK's typed ``create()``
    signature rejects them (external constraint; see
    docs/agent-workflow.md).
    """
    kwargs: dict[str, Any] = {
        "top_p": settings.llm_top_p,
        "presence_penalty": settings.llm_presence_penalty,
        "extra_body": {
            "top_k": settings.llm_top_k,
            "min_p": settings.llm_min_p,
            "repetition_penalty": settings.llm_repetition_penalty,
        },
    }
    if settings.llm_reasoning_effort:
        kwargs["reasoning_effort"] = settings.llm_reasoning_effort
    return ObservableOpenAILike(
        model=settings.llm_model,
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
        temperature=settings.llm_temperature,
        additional_kwargs=kwargs,
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
        timeout=settings.run_timeout or None,
        memory_token_limit=settings.memory_token_limit,
        tool_round_limit=settings.tool_round_limit,
    )
