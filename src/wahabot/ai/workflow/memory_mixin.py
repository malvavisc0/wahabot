import asyncio
from collections.abc import Callable, Sequence
from typing import Any, cast, override

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import (
    Context,
)
from loguru import logger

from wahabot.ai.history import (
    ContextBudgetError,
    cap_records_turn,
    degrade_old_history,
    inbound_message_id,
    minimum_turn,
    sanitize_chat_history,
    trim_to_budget,
    turn_header,
)
from wahabot.ai.messages import (
    TURN_SERIALS_KWARG,
)
from wahabot.ai.pinned_state import (
    PINNED_STATE_TOKEN_LIMIT,
    dedupe_recall,
    pinned_state_block,
)
from wahabot.ai.tools.whatsapp import (
    RunTarget,
    current_target,
)
from wahabot.ai.triage import (
    enrich_recall_query,
)
from wahabot.ai.workflow.base import AgentBase
from wahabot.ai.workflow.prompts import (
    recap_line,
)
from wahabot.ai.workflow.text import message_text
from wahabot.core.learned import load_learned
from wahabot.core.notebook import (
    Note,
)
from wahabot.core.thread_index import (
    OPERATOR_CHAT_ID,
    recall_query,
)
from wahabot.core.tickets.pinned import ticket_marker
from wahabot.core.tokens import text_tokens


class MemoryMixin(AgentBase):
    @override
    async def chat_history(self, ctx: Context) -> list[ChatMessage]:
        """The trimmed, structurally valid conversation plus system prompt.

        Reserve the current query and newest whole tool group before
        assembling the pinned state block and older history. The block
        rides the system message (``system_message`` + block), so it
        sits before the FIFO and history trimming can never evict it.
        Do not let the library trim a second time: its last-message
        fallback can orphan tools.
        """
        memory = await ctx.store.get("memory")
        system = self.system_message()
        messages = await memory.aget_all()
        messages = sanitize_chat_history(messages, drop_trailing_user=False)
        messages = degrade_old_history(messages)
        tokenizer = cast(Callable[[str], list[Any]], memory.tokenizer_fn)

        def count_message_tokens(message: ChatMessage) -> int:
            return text_tokens(message_text(message), tokenizer)

        budget = self.memory_token_limit - (
            count_message_tokens(system) if system is not None else 0
        )
        required = sum(
            count_message_tokens(message) for message in minimum_turn(messages)
        )
        if required > budget:
            raise ContextBudgetError(
                "System prompt and current query/tool group exceed token budget"
            )
        prompt = system.content if system is not None else ""
        triage = await self.stored_triage_result(ctx)
        query = recall_query(messages)
        fact_keys: tuple[str, ...] = ()
        if triage is not None:
            query = enrich_recall_query(query, triage.verdict.needs)
            fact_keys = triage.verdict.fact_keys
        block = await self.pinned_block(
            tokenizer,
            f"{prompt}\n\n",
            self.memory_token_limit - required,
            query=query,
            fact_keys=fact_keys,
            live_messages=messages,
        )
        if block:
            system = ChatMessage(
                role=MessageRole.SYSTEM, content=f"{prompt}\n\n{block}".strip()
            )
            budget = self.memory_token_limit - count_message_tokens(system)
        if self.settings is not None:
            messages = cap_records_turn(
                messages,
                budget=budget,
                token_counter=count_message_tokens,
                window=self.settings.records_window,
            )
        history = trim_to_budget(messages, budget, count_message_tokens)
        await memory.aset(history)
        self.schedule_recap_merge(messages, history)
        self.schedule_thread_index(messages, history)
        # Prompt-only turn header: how many messages this turn holds, which is
        # newest, and — from the wake gate — whether it was addressed. The
        # unmarked history above is what stays in memory.
        group = bool(await ctx.store.get("turn_group", default=False))
        addressed = bool(await ctx.store.get("turn_addressed", default=True))
        speak = bool(await ctx.store.get("turn_speak", default=addressed))
        direct = bool(await ctx.store.get("turn_direct", default=False))
        prompt_history = turn_header(
            history, group=group, addressed=addressed, speak=speak, direct=direct
        )
        return ([system] if system is not None else []) + prompt_history

    def schedule_recap_merge(
        self, messages: list[ChatMessage], history: list[ChatMessage]
    ) -> None:
        """Record the trimmed-away prefix for a later recap merge.

        The trim just evicted the dropped turns from the buffer
        (``memory.aset`` kept only *history*), so this is the only
        moment they exist. Recording is pure state (``RecapMerger.
        schedule`` never calls the LLM); the merge itself runs when
        handlers await :meth:`merge_learned` after the reply is out.
        Operator runs are typed, not conversational — no recap — and
        only user/assistant text merges (never tool results, the
        recap trust boundary).
        """
        if self.recap_merger is None or len(history) >= len(messages):
            return
        try:
            target = current_target()
        except RuntimeError:
            return
        if target.chat_id == "operator":
            return
        dropped = messages[: len(messages) - len(history)]
        lines = [line for message in dropped if (line := recap_line(message))]
        self.recap_merger.schedule(target.session, target.chat_id, "\n".join(lines))

    def schedule_thread_index(
        self, messages: list[ChatMessage], history: list[ChatMessage]
    ) -> None:
        """Record the trimmed-away turns for the per-chat thread index.

        The same trim hook as the recap: ``memory.aset`` kept only
        *history*, so the dropped prefix exists only here. Recording is
        pure state (no I/O, no embedding); :meth:`merge_learned` drains
        it after the reply is out. Operator runs are commands, not
        conversation, and are never indexed.
        """
        if self.thread_index is None or len(history) >= len(messages):
            return
        try:
            target = current_target()
        except RuntimeError:
            return
        dropped = messages[: len(messages) - len(history)]
        self.thread_index.schedule(target.session, target.chat_id, dropped)

    async def merge_learned(self) -> None:
        """Run the post-reply memory jobs: recap merge and index drain.

        Called at the persist points after the reply has been sent —
        and outside the chat lock, so the extraction LLM call and the
        embedding batch never delay this chat's next run (the aria-ai
        lesson: embedding must never run inside the agent lock). The
        two jobs are independent and run concurrently; each is
        single-flight and fail-soft, so one service being down never
        blocks the other. The recap merger serializes internally, so
        overlapping calls from concurrent runs cannot lose a batch.
        """
        tasks: list[Any] = []
        if self.recap_merger is not None:
            tasks.append(self.recap_merger.merge_pending())
        if self.thread_index is not None:
            tasks.append(self.thread_index.drain())
        if not tasks:
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Post-reply memory job failed: {error}", error=result)

    async def recall_slices(self, target: RunTarget, query: str) -> list[str]:
        """Lead slices for this chat; [] when recall is off or empty."""
        if self.thread_index is None or not query or target.chat_id == OPERATOR_CHAT_ID:
            return []
        return await self.thread_index.recall(target.session, target.chat_id, query)

    def fact_notes(self, target: RunTarget, fact_keys: Sequence[str]) -> list[Note]:
        """Notebook notes the gate asked the run to pin into the block."""
        if self.notebook is None:
            return []
        notes: list[Note] = []
        for key in fact_keys:
            notes.extend(
                self.notebook.read(target.session, target.chat_id, key=key, limit=1)
            )
        return notes

    async def pinned_block(
        self,
        tokenizer: Callable[[str], list[Any]],
        prompt_prefix: str,
        prompt_budget: int,
        query: str = "",
        fact_keys: Sequence[str] = (),
        live_messages: Sequence[ChatMessage] = (),
    ) -> str:
        """Assemble this run's pinned state block (fail-soft, bounded).

        Restructures what the notebook already injects — notebook
        facts, operator policies — plus the session rules, into one
        never-evicted block. The lead-only recall section comes from
        this chat's thread index, queried with the latest user text.
        Skipped whole for bare workflows with no source wired, which
        also keeps unbound runs off ``current_target``.
        """
        if (
            self.notebook is None
            and self.rules_renderer is None
            and self.settings is None
            and self.ticket_store is None
        ):
            return ""
        target = current_target()
        recap = (
            load_learned(self.settings.data_dir, target.session, target.chat_id)
            if self.settings is not None
            else ""
        )
        recall = dedupe_recall(
            await self.recall_slices(target, query),
            [str(message.content or "") for message in live_messages],
        )
        required_notes = self.fact_notes(target, fact_keys)
        return await pinned_state_block(
            notebook=self.notebook,
            session=target.session,
            chat_id=target.chat_id,
            token_budget=min(PINNED_STATE_TOKEN_LIMIT, self.memory_token_limit // 4),
            tokenizer=tokenizer,
            prompt_prefix=prompt_prefix,
            prompt_budget=prompt_budget,
            rules=self.rendered_session_rules(),
            recap=recap,
            recall=recall,
            ticket=self.render_ticket_marker(target),
            required_notes=required_notes,
        )

    def render_ticket_marker(self, target: RunTarget) -> str:
        """One-line ticket state for this run, or "" when unavailable."""
        if self.ticket_store is None or not target.chat_id:
            return ""
        try:
            ticket = (
                self.ticket_store.ticket(target.ticket_id)
                if target.ticket_id
                else self.ticket_store.active_ticket(target.chat_id)
            )
        except Exception:
            return ""
        return ticket_marker(ticket)

    @override
    async def ensure_memory(self, ctx: Context) -> ChatMemoryBuffer:
        """The chat's buffer, created with the same defaults as a run.

        A gated group turn can be stored without ever starting the
        agent, so the block path cannot assume ``prepare_chat_history``
        created the buffer first.
        """
        memory = await ctx.store.get("memory", default=None)
        if memory is None:
            from_defaults = cast(
                Callable[..., ChatMemoryBuffer], ChatMemoryBuffer.from_defaults
            )
            memory = from_defaults(token_limit=self.memory_token_limit, llm=self.llm)
            await ctx.store.set("memory", memory)
        return cast(ChatMemoryBuffer, memory)

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

    @staticmethod
    async def already_in_buffer(
        memory: ChatMemoryBuffer,
        incoming: str,
        serials: list[int] | None = None,
    ) -> bool:
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
        appending (the agent still gets its fresh run). A records turn
        carries no id note; its allocated serials are the identity.
        """
        message_id = inbound_message_id(incoming)
        if not message_id and serials:
            messages = await memory.aget_all()
            for message in reversed(messages):
                if message.role != MessageRole.USER:
                    continue
                return message.additional_kwargs.get(TURN_SERIALS_KWARG) == serials
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
