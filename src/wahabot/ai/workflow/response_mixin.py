import hashlib
from typing import override

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    TextBlock,
)
from llama_index.core.tools import ToolSelection
from llama_index.core.workflow import (
    Context,
    StopEvent,
)
from loguru import logger

from wahabot.ai import records
from wahabot.ai.events import InputEvent
from wahabot.ai.history import (
    chat_visible_text,
    is_single_emoji,
)
from wahabot.ai.messages import (
    TURN_HANDLED_KWARG,
)
from wahabot.ai.tools.whatsapp import (
    apply_chat_prefix,
    current_prefix,
    current_target,
    log_action_reason,
)
from wahabot.ai.workflow.base import AgentBase
from wahabot.ai.workflow.delivery import DELIVERY_TOOLS, SILENCE_TOOL, collapse_send_group
from wahabot.ai.workflow.prompts import (
    EARLY_STOPPING_PROMPT,
    POST_DELIVERY_WRAP_UP_PROMPT,
    parse_wrap_up,
    truncate_note,
)
from wahabot.ai.workflow.toolkit import (
    denied_tool_message,
    tool_call_key,
)
from wahabot.core.notebook import (
    AUTHOR_SYSTEM,
    KIND_LESSON,
    notebook_operation,
)


class ResponseMixin(AgentBase):
    @override
    async def mark_turn_handled(self, ctx: Context, *, silent: bool = False) -> None:
        """Stamp the run's inbound user turn as handled conversation.

        ``prepare_chat_history`` appended the turn unstamped; the four
        stop paths above call this once the run has concluded. The
        stamp tells ``history``'s trailing-user drop that this message
        is real conversation a completed run read — in ``judicious``
        group mode a silent run's message must stay in context for the
        next run, not be discarded as scaffolding. A silent run rewrites
        its turn as ledger lines, one per message it read and did not
        answer, so the next prompt gets the gist instead of the wall.
        Fail-soft: without the stamp the turn merely behaves like today
        (dropped on the next run's repair).
        """
        memory = await ctx.store.get("memory", default=None)
        if memory is None:
            return
        serials = await ctx.store.get("turn_serials", default=None) or []
        messages = await memory.aget_all()
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.role == MessageRole.USER and TURN_HANDLED_KWARG not in (
                msg.additional_kwargs
            ):
                kwargs = {**msg.additional_kwargs, TURN_HANDLED_KWARG: True}
                content = str(msg.content or "")
                if silent and serials:
                    result = await self.stored_triage_result(ctx)
                    gist = result.verdict.gist if result is not None else ""
                    content = records.ledger_turn(
                        content,
                        gist=gist,
                        newest_serial=max(serials, default=0),
                    )
                messages[i] = ChatMessage(
                    role=msg.role,
                    blocks=list(msg.blocks),
                    content=content,
                    additional_kwargs=kwargs,
                )
                break
        await memory.aset(messages)
        await ctx.store.set("memory", memory)

    @override
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

    @override
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

    @override
    def any_delivery(self) -> bool:
        """True when any delivery tool already fired this run."""
        target = current_target()
        return bool(target.sent or target.reacted)

    @override
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

    @override
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
        record words nobody saw. The same mirroring rule filters
        leaked non-answers at the source: a plain-text round whose
        content is not chat-visible (``chat_visible_text`` — the
        identical predicates ``final_reply`` applies to delivery: a
        leaked ``stay_silent`` token, an invented error payload, a
        private self-report appended to the reply) is never stored, so
        the model's self-history cannot re-teach it its own bugs. A
        lone-emoji final reply is dropped here for the same reason (the
        handler converts it to a reaction or silence — it never lands as
        text), while delivery keeps receiving it so the conversion can
        happen. A message with tool calls always stores: the calls will
        execute and their results belong to the run.

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
        visible = "" if skip_text else chat_visible_text(message.content)
        if visible and is_single_emoji(visible) and not tool_calls:
            visible = ""
        if tool_calls or visible:
            await memory.aput(self.visible_message(message, visible, tool_calls))
        await ctx.store.set("memory", memory)

    @staticmethod
    def visible_message(
        message: ChatMessage, visible: str, tool_calls: list[ToolSelection]
    ) -> ChatMessage:
        """The message to store: *message*, minus a leaked trailing note.

        ``visible`` is the chat-visible text (``chat_visible_text``).
        When it differs from the raw text the model emitted a private
        self-report (a leaked wrap-up note) the chat never saw, so the
        stored turn is rebuilt from *visible*: storing the raw text would
        put the note back in front of the model on the next run and
        re-teach the pattern. Tool-call
        messages store as they are — their words are not what the chat
        saw, and their blocks carry the calls.

        The chat's ``chat-message-prefix`` rule is applied here too, so
        the stored text equals the sent text (``deliver_chat_text``);
        otherwise the model's next-turn view is un-prefixed and the rule
        undoes itself.
        """
        raw = str(message.content or "").strip()
        if tool_calls:
            return message
        stored = apply_chat_prefix(visible, current_prefix())
        if not stored or stored == raw:
            return message
        return ChatMessage(
            role=message.role,
            content=stored,
            additional_kwargs=dict(message.additional_kwargs),
        )

    @override
    async def collapse_delivery(self, ctx: Context) -> None:
        """Replace the last delivered tool group with plain assistant text.

        At run end the delivered pair (empty assistant tool-call plus
        its tool-result envelope) is swapped for one plain assistant
        message holding the delivered content, so the retained history
        reads like the chat it mirrors. The whole group is replaced
        atomically: a dangling tool call without its result is exactly
        what :func:`repair_memory` must never find. No-op when nothing
        was delivered.
        """
        memory = await ctx.store.get("memory")
        messages = await memory.aget_all()
        collapsed = collapse_send_group(messages)
        if collapsed is not None:
            await memory.aset(collapsed)
            await ctx.store.set("memory", memory)

    @override
    @staticmethod
    def stopped_response() -> StopEvent:
        """A StopEvent carrying an empty reply: nothing more to say."""
        empty = ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""))
        return StopEvent(result=empty)

    @override
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

    @override
    async def wrap_up_response(self, ctx: Context, reason: str) -> ChatResponse:
        """One last tool-free LLM call after a delivered reply / round limit.

        Mirrors LlamaIndex's ``early_stopping_method="generate"``: the
        model is told the budget is spent and asked for a final answer
        without tools. The round counter is already over the limit, so
        even a defiant model requesting tools again cannot continue.

        A delivered reply changes the ask: the final text can never be
        sent (the one-delivery latch holds), so the round produces the
        post-delivery wrap-up note instead — one private lesson written
        to the notebook and journalled, never an assistant turn and never
        sent to the chat (the un-nudged model burned the round on
        chatter that could only be discarded).
        """
        limit = self.tool_round_limit
        logger.info("Generating wrap-up response ({reason})", reason=reason)
        delivered = self.any_delivery()
        prompt = (
            POST_DELIVERY_WRAP_UP_PROMPT
            if delivered
            else EARLY_STOPPING_PROMPT.format(limit=limit)
        )
        messages = await self.chat_history(ctx)
        messages.append(ChatMessage(role="user", content=prompt))
        async with self.llm_semaphore:
            response = await self.llm.achat(messages)
        if delivered:
            note = str(response.message.content or "").strip()
            if note:
                await self.store_wrap_up_note(note)
            self.audit_wrap_up_note(note)
        else:
            response = await self.rewrite_blame(ctx, response)
        return self.drop_post_delivery_text(response)

    @override
    async def store_wrap_up_note(self, note: str) -> None:
        """Persist the post-delivery wrap-up note as a durable lesson.

        The note is a private lesson, never a message the chat saw: it is
        written to the notebook — the durable store — and journalled, and
        never stored as an assistant turn. A ``topic:`` line names the
        lesson; an identical topic replaces the prior lesson, and only a
        topicless note falls back to a content-hash key.
        """
        if self.notebook is None or not note.strip():
            return
        try:
            target = current_target()
            slug, lesson = parse_wrap_up(note)
            if not lesson:
                return
            key = (
                f"lesson-{slug}"
                if slug
                else "lesson-" + hashlib.sha256(lesson.encode()).hexdigest()[:12]
            )
            await notebook_operation(
                self.notebook.write,
                target.session,
                target.chat_id,
                key,
                truncate_note(lesson),
                kind=KIND_LESSON,
                author_kind=AUTHOR_SYSTEM,
            )
        except Exception:
            logger.exception("Could not store post-delivery wrap-up note")

    @override
    def audit_wrap_up_note(self, note: str) -> None:
        """Journal the run's self-report (empty note included, visibly so)."""
        self.audit("wrap_up", note=note[:300])

    @override
    def drop_post_delivery_text(self, response: ChatResponse) -> ChatResponse:
        """Empty *response* when a delivery tool already fired this run.

        A final text after a delivery tool (``send_message``/``send_media``/
        ``forward_message``/``react_to_message``) succeeded was never
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

    def deny_tool_call(self, tool_call: ToolSelection) -> ChatMessage:
        """Journal and answer a tool call the hardened policy refused."""
        self.audit(
            "tool_call",
            tool=tool_call.tool_name,
            args={key: str(value)[:300] for key, value in tool_call.tool_kwargs.items()},
            ok=False,
            outcome="denied",
        )
        return denied_tool_message(tool_call, "tool unavailable in this run")
