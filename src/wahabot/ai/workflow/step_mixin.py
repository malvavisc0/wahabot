from typing import Any, override

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
)
from llama_index.core.tools import BaseTool, ToolSelection
from llama_index.core.workflow import (
    Context,
    StopEvent,
)
from loguru import logger

from wahabot.ai.events import ToolCallEvent
from wahabot.ai.history import (
    chat_visible_text,
)
from wahabot.ai.tools.runcontext import mark_run_outcome
from wahabot.ai.workflow.base import AgentBase
from wahabot.ai.workflow.delivery import SILENCE_TOOL
from wahabot.ai.workflow.toolkit import (
    run_tool_call,
    tool_outcome,
)


def _text_outcome(delivered: bool, visible: str, *, after_tools: bool) -> str:
    """The run outcome for a plain-text round."""
    if delivered:
        return "delivered_via_tool"
    if visible:
        return "reply"
    return "empty" if after_tools else "silence"


class StepMixin(AgentBase):
    @override
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

    async def route_tool_calls(
        self,
        ctx: Context,
        response: ChatResponse,
        tool_calls: list[ToolSelection],
        rounds: int,
    ) -> ToolCallEvent | StopEvent:
        """The post-LLM dispatch: silence, repeat, plain text, or tools.

        Shared stop bookkeeping first — every terminal path stamps the
        user turn handled and collapses a delivered tool group — then
        the branch-specific work: storing the reply (or the wrap-up
        note a delivered run leaves instead), and the wrap-up call for
        a round the loop cannot continue from.
        """
        if any(call.tool_name == SILENCE_TOOL for call in tool_calls):
            self.log_silence_reason(tool_calls)
            logger.debug("Stopping run: model chose stay_silent")
            mark_run_outcome("silence")
            return await self.stop_with(ctx, self.stopped_response(), silent=True)
        if tool_calls and await self.repeats_tool_call(ctx, tool_calls):
            mark_run_outcome("empty")
            return await self.stop_with(ctx, self.stopped_response(), silent=True)
        if not tool_calls:
            return await self.finish_text_round(ctx, response, rounds)
        if self.delivery_complete(tool_calls) or rounds >= self.tool_round_limit:
            return await self.finish_tool_round(ctx, tool_calls)
        await self.remember(ctx, response, tool_calls)
        return ToolCallEvent(tool_calls=tool_calls)

    async def finish_tool_round(
        self, ctx: Context, tool_calls: list[ToolSelection]
    ) -> StopEvent:
        """Wrap up a round whose calls cannot run, and end the run."""
        # Do not store calls which will not be executed: the chat API
        # requires every advertised call to have a tool response.
        delivered_round = self.delivery_complete(tool_calls)
        reason = (
            "non-delivery round after completed delivery"
            if delivered_round
            else f"round limit {self.tool_round_limit}"
        )
        result = await self.wrap_up_response(ctx, reason)
        delivered = self.any_delivery()
        visible = chat_visible_text(str(result.message.content or ""))
        mark_run_outcome(_text_outcome(delivered, visible, after_tools=True))
        return await self.stop_with(ctx, StopEvent(result=result))

    async def stop_with(
        self, ctx: Context, event: StopEvent, *, silent: bool = False
    ) -> StopEvent:
        """End the run with *event*: stamp the turn and collapse the delivery group.

        The wrap-up note is journalled and written to the notebook by
        ``wrap_up_response`` itself — it is a durable lesson
        (``kind=lesson``), never an assistant turn, so nothing is staged
        for storage here.
        """
        await self.mark_turn_handled(ctx, silent=silent)
        await self.collapse_delivery(ctx)
        return event

    async def finish_text_round(
        self, ctx: Context, response: ChatResponse, rounds: int
    ) -> ToolCallEvent | StopEvent:
        """A plain-text round: the reply, or the wrap-up note if delivered."""
        self.warn_accidental_silence(response, rounds)
        delivered = self.any_delivery()
        if not delivered:
            response = await self.rewrite_blame(ctx, response)
        note = str(response.message.content or "").strip() if delivered else ""
        await self.remember(ctx, response, [], skip_text=delivered)
        visible = chat_visible_text(str(response.message.content or ""))
        silent = not delivered and not visible
        mark_run_outcome(_text_outcome(delivered, visible, after_tools=rounds > 1))
        await self.mark_turn_handled(ctx, silent=silent)
        await self.collapse_delivery(ctx)
        if delivered:
            if note:
                await self.store_wrap_up_note(note)
            self.audit_wrap_up_note(note)
        return StopEvent(result=self.drop_post_delivery_text(response))

    async def next_round(self, ctx: Context) -> int:
        """Increment and return this run's tool-round counter."""
        rounds = await ctx.store.get("tool_rounds", default=0)
        rounds += 1
        await ctx.store.set("tool_rounds", rounds)
        return rounds

    async def run_and_audit_tool_call(
        self, tools_by_name: dict[str, BaseTool], tool_call: ToolSelection
    ) -> ChatMessage:
        """Run one tool call and journal it.

        The audit entry carries the tool name, its arguments (the
        model's *why* included), and the outcome — the fields the
        operator console's future audit timeline will read. Tool
        output is capped by the audit module; the full output stays
        in the run's memory.
        """
        message = await run_tool_call(tools_by_name, tool_call)
        kwargs = dict(tool_call.tool_kwargs)
        outcome = tool_outcome(str(message.content or ""))
        self.audit(
            "tool_call",
            tool=tool_call.tool_name,
            args={key: str(value)[:300] for key, value in kwargs.items()},
            ok=outcome == "completed",
            outcome=outcome,
        )
        return message
