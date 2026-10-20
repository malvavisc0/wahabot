"""Function calling agent workflow steps and construction.

LlamaIndex reference function calling agent workflow
(https://developers.llamaindex.ai/python/examples/workflow/function_calling_agent/):
user messages are added to a per-chat ``ChatMemoryBuffer``, the LLM
gets tools + history, tool results loop back, and the final text comes
back in ``StopEvent.result``.
"""

import asyncio
import hashlib
import re
from collections.abc import Callable, Sequence
from typing import Any, cast, override

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    TextBlock,
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
    INBOUND_ID_RE,
    ContextBudgetError,
    chat_visible_text,
    degrade_old_history,
    inbound_message_id,
    is_single_emoji,
    minimum_turn,
    sanitize_chat_history,
    trim_to_budget,
    turn_header,
)
from wahabot.ai.messages import TURN_HANDLED_KWARG
from wahabot.ai.pinned_state import PINNED_STATE_TOKEN_LIMIT, pinned_state_block
from wahabot.ai.tools.notebook import MAX_NOTE_TOKENS, NOTEBOOK_INSTRUCTIONS
from wahabot.ai.tools.slim import slim_tool_specs
from wahabot.ai.tools.whatsapp import (
    RunTarget,
    apply_chat_prefix,
    current_prefix,
    current_target,
    log_action_reason,
)
from wahabot.ai.triage import (
    MUTE_NOTE_KEY,
    TRIAGE_MESSAGE_CHARS,
    TRIAGE_MUTE_CHARS,
    TriageContext,
    TriageGate,
    TriageResult,
    enrich_recall_query,
    format_turns,
    is_tool_allowed,
    result_from_dict,
    result_to_dict,
)
from wahabot.ai.workflow.delivery import DELIVERY_TOOLS, SILENCE_TOOL, collapse_send_group
from wahabot.ai.workflow.text import message_text
from wahabot.ai.workflow.toolkit import (
    denied_tool_message,
    run_tool_call,
    tool_call_key,
    tool_outcome,
)
from wahabot.core.audit import save_action
from wahabot.core.embeddings import LanEmbedding
from wahabot.core.learned import RecapMerger, load_learned
from wahabot.core.notebook import (
    AUTHOR_SYSTEM,
    KIND_LESSON,
    KIND_RULE,
    MAX_NOTE_LENGTH,
    Note,
    Notebook,
    notebook_operation,
)
from wahabot.core.thread_index import (
    OPERATOR_CHAT_ID,
    ThreadIndexer,
    recall_query,
)
from wahabot.core.tokens import text_tokens, truncate_to_tokens
from wahabot.settings import Settings

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

#: After delivery, extract a reusable private lesson rather than an activity log.
_POST_DELIVERY_WRAP_UP_PROMPT = (
    "You already delivered your reply. Do not send more replies or use any tools. "
    "Review this run for a verified, reusable lesson, not an activity summary. "
    "Record only a new correction, confirmed preference, or demonstrated "
    "workaround missing from the notebook. Return two lines: first "
    "`topic: <short-slug>` naming the lesson's subject in a few hyphenated "
    "words, then the lesson — the situation, what was wrong or clarified, the "
    "evidence, and what to do differently next time, in at most two concise "
    "sentences. Reuse the topic of an existing lesson when the subject matches. "
    "Do not infer success from an attempted action, invent facts, or turn your "
    "inference into operator policy. This is a private chat-scoped lesson for "
    "later runs, never a message to the chat. If nothing new and lasting was "
    "learned, return an empty message."
)

#: Appended to a wrap-up note cut down to the notebook's caps, so a
#: later reader knows the lesson was shortened by storage, not by the
#: model's own wording.
_TRUNCATED_NOTE_SUFFIX = "\n[Truncated wrap-up note]"

_TOPIC_PREFIX = "topic:"


def topic_slug(line: str) -> str:
    """Normalise a wrap-up `topic:` line to a short, stable key slug."""
    value = line.split(":", 1)[1] if ":" in line else line
    value = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return value[:40].strip("-")


def parse_wrap_up(text: str) -> tuple[str, str]:
    """Split a wrap-up reply into ``(topic slug, lesson)``.

    A leading ``topic: <slug>`` line names the lesson; the line names
    the key, never the lesson, so it is dropped from the stored body in
    every outcome. Without a usable topic line the slug is empty and the
    caller keys by content instead; a topic line without a lesson body
    leaves nothing to store (``("", "")``).
    """
    lines = text.strip().splitlines()
    if lines and lines[0].strip().casefold().startswith(_TOPIC_PREFIX):
        slug = topic_slug(lines[0])
        lesson = "\n".join(lines[1:]).strip()
        if slug and lesson:
            return slug, lesson
        return "", lesson
    return "", text.strip()


def truncate_note(note: str) -> str:
    """Cut a note to the notebook's caps, marking that storage shortened it."""
    if text_tokens(note) <= MAX_NOTE_TOKENS and len(note) <= MAX_NOTE_LENGTH:
        return note
    room = MAX_NOTE_LENGTH - len(_TRUNCATED_NOTE_SUFFIX)
    return (
        truncate_to_tokens(
            note[:room], MAX_NOTE_TOKENS - text_tokens(_TRUNCATED_NOTE_SUFFIX)
        )
        + _TRUNCATED_NOTE_SUFFIX
    )


def raise_missing_llm() -> FunctionCallingLLM:
    """Fail fast when no llm was passed to the workflow."""
    raise ValueError("FunctionCallingAgentWorkflow requires an llm")


def recap_line(message: ChatMessage) -> str:
    """One recap line, or "" when the message has no visible text.

    The recap must never carry tool-call arguments or narration: a
    model's tool payload is not conversation, and a private self-report
    was never sent to the chat. ``message_text`` is the token counter's
    view (content plus tool args), not this one. The inbound
    ``[message id: …]`` note is metadata this code stamped on the
    turn, not conversation — it is stripped so the recap carries words
    only.
    """
    if message.role not in (MessageRole.USER, MessageRole.ASSISTANT):
        return ""
    text = INBOUND_ID_RE.sub("", str(message.content or "")).strip()
    if message.role == MessageRole.ASSISTANT:
        text = chat_visible_text(text)
    return f"[{message.role.value}] {text}" if text else ""


def build_thread_index(
    settings: Settings | None, thread_index: ThreadIndexer | None
) -> ThreadIndexer | None:
    """The per-chat recall index, or None when its kill switch is off.

    An injected index wins (tests, alternate wiring). Otherwise the index
    requires ``memory_persist`` (it is a file next to the buffer) and
    ``thread_recall``; a missing ``embed_api_base`` leaves it FTS5-only,
    so the exact-token floor works without the LAN embedding service.
    """
    if thread_index is not None:
        return thread_index
    if settings is None or not settings.memory_persist or not settings.thread_recall:
        return None
    embed_model = (
        LanEmbedding(
            api_base=settings.embed_api_base,
            model_name=settings.embed_model,
            timeout=settings.embed_timeout,
        )
        if settings.embed_api_base
        else None
    )
    return ThreadIndexer(
        settings.data_dir,
        embed_model,
        query_timeout=max(0.1, min(settings.embed_timeout, 2.0)),
    )


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

    #: Settings for the audit journal (data dir + session); None in
    #: tests that build a bare workflow without settings.
    settings: Settings | None

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

    def system_message(self) -> ChatMessage | None:
        """The system prompt as a message, or None when unset."""
        prompt = self.rendered_system_prompt()
        if self.notebook is not None:
            prompt = f"{prompt or ''}\n\n# Notebook\n\n{NOTEBOOK_INSTRUCTIONS}".strip()
        if not prompt:
            return None
        return ChatMessage(role="system", content=prompt)

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
        )
        if block:
            system = ChatMessage(
                role=MessageRole.SYSTEM, content=f"{prompt}\n\n{block}".strip()
            )
            budget = self.memory_token_limit - count_message_tokens(system)
        history = trim_to_budget(messages, budget, count_message_tokens)
        await memory.aset(history)
        self.schedule_recap_merge(messages, history)
        self.schedule_thread_index(messages, history)
        # Prompt-only turn header: how many messages this turn holds, which is
        # newest, and — from the wake gate — whether it was addressed. The
        # unmarked history above is what stays in memory.
        group = bool(await ctx.store.get("turn_group", default=False))
        addressed = bool(await ctx.store.get("turn_addressed", default=True))
        prompt_history = turn_header(history, group=group, addressed=addressed)
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
        ):
            return ""
        target = current_target()
        recap = (
            load_learned(self.settings.data_dir, target.session, target.chat_id)
            if self.settings is not None
            else ""
        )
        recall = await self.recall_slices(target, query)
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
            required_notes=required_notes,
        )

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

    def triage_context(
        self,
        target: RunTarget,
        message: str,
        messages: Sequence[ChatMessage],
        addressed: bool,
    ) -> TriageContext:
        """Build the bounded gate input: message, last turns, mute, rules."""
        mute = ""
        if self.notebook is not None:
            notes = self.notebook.read(
                target.session, target.chat_id, key=MUTE_NOTE_KEY, kind=KIND_RULE
            )
            mute = " | ".join(f"{note.key}: {note.text}" for note in notes)[
                :TRIAGE_MUTE_CHARS
            ]
        return TriageContext(
            session=target.session,
            chat_id=target.chat_id,
            message=INBOUND_ID_RE.sub("", message).strip()[:TRIAGE_MESSAGE_CHARS],
            turns=format_turns(messages),
            mute=mute,
            rules=self.rendered_session_rules(),
            group=target.chat_id.endswith("@g.us"),
            addressed=addressed,
        )

    async def start_triage(
        self,
        ctx: Context,
        target: RunTarget,
        user_msg: str,
        addressed: bool,
        gate_first: bool,
    ) -> bool:
        """Start or await the gate; False means the run must stay silent.

        Race mode (DMs, addressed turns) always returns True: the gate
        can only tighten tools there. Block mode (unaddressed groups)
        runs only on a verdict that grants participation, and a verdict
        that is present but silent (the new default) means silence. A
        gate that is *unavailable* — failed, timed out, unreachable — is
        different: it is a filter, not an authority, so the run degrades
        to the pre-gate behaviour rather than muting the chat and losing
        the chance to answer a real question. The failed verdict is
        stored and audited either way.
        """
        if self.triage is None or target.chat_id == OPERATOR_CHAT_ID:
            return True
        try:
            memory = await self.ensure_memory(ctx)
            prior = await memory.aget_all()
            context = self.triage_context(target, user_msg, prior, addressed)
            await ctx.store.set(
                "triage_audit",
                {"session": context.session, "chat_id": context.chat_id},
            )
            if not gate_first:
                task = asyncio.create_task(self.triage.classify_bounded(context))
                self.set_triage_task(ctx, task)
                return True
            result = await self.triage.classify_bounded(context)
            await ctx.store.set("triage_result", result_to_dict(result))
            if result.failed or result.timed_out:
                # Gate down: degrade to the pre-gate behaviour — wake the
                # agent. An outage must not mute the chat; only a verdict
                # the gate actually produced can decline a turn.
                return True
            return result.verdict.participate
        except Exception as exc:
            logger.warning(
                "Triage gate unavailable for {chat_id}: {error}",
                chat_id=target.chat_id,
                error=exc,
            )
            logger.opt(exception=exc).debug("Triage gate failure detail")
            # Gate unavailable: degrade to the pre-gate behaviour in both
            # modes — the agent runs, unhardened.
            return True

    async def stored_triage_result(self, ctx: Context) -> TriageResult | None:
        """The verdict already in this context, if any, as an object."""
        try:
            stored = await ctx.store.get("triage_result", default=None)
        except Exception as exc:
            logger.warning("Triage state unavailable: {error}", error=exc)
            return None
        if isinstance(stored, TriageResult):
            return stored
        if isinstance(stored, dict):
            try:
                return result_from_dict(stored)
            except Exception as exc:
                logger.warning("Triage verdict unreadable: {error}", error=exc)
        return None

    def set_triage_task(self, ctx: Context, task: asyncio.Task[TriageResult]) -> None:
        """Track the run's gate task outside the serialized context store."""
        self.triage_tasks[id(ctx)] = task

    def pop_triage_task(self, ctx: Context) -> asyncio.Task[TriageResult] | None:
        """Take the run's gate task, if one was started."""
        return self.triage_tasks.pop(id(ctx), None)

    async def cancel_triage(self, ctx: Context) -> None:
        """Cancel any gate task the run left behind.

        A live task is cancelled and its never-produced verdict is
        audited as a timeout fallback — a run that ends before the
        gate answers must not leave a verdict-shaped hole in the
        journal. The escalation a partial verdict already dispatched
        is a separate task and is not touched.
        """
        task = self.pop_triage_task(ctx)
        if task is None or task.done():
            return
        task.cancel()
        if self.triage is None:
            return
        try:
            audit = await ctx.store.get("triage_audit", default=None)
        except Exception as exc:
            logger.warning("Triage audit state unavailable: {error}", error=exc)
            return
        if not isinstance(audit, dict):
            return
        values = cast(dict[str, Any], audit)
        self.triage.audit(
            str(values.get("session", "")),
            str(values.get("chat_id", "")),
            TriageResult.fallback(timed_out=True),
        )

    async def triage_result(self, ctx: Context) -> TriageResult | None:
        """Await the run's gate verdict once; fallback results are audited."""
        if self.triage is None:
            return None
        result = await self.stored_triage_result(ctx)
        if result is not None:
            return result
        task = self.pop_triage_task(ctx)
        if task is None:
            return None
        fallback: TriageResult | None = None
        try:
            result = await asyncio.wait_for(task, timeout=self.triage.timeout + 0.05)
        except TimeoutError:
            task.cancel()
            fallback = TriageResult.fallback(timed_out=True)
            result = fallback
        except Exception as exc:
            self.triage.warn_once("await", "Triage task failed: {exc}", exc=exc)
            fallback = TriageResult.fallback(failed=True)
            result = fallback
        try:
            await ctx.store.set("triage_result", result_to_dict(result))
        except Exception as exc:
            logger.warning("Could not store triage verdict: {error}", error=exc)
        if fallback is not None:
            try:
                audit = await ctx.store.get("triage_audit", default=None)
            except Exception as exc:
                logger.warning("Triage audit state unavailable: {error}", error=exc)
                audit = None
            if isinstance(audit, dict):
                values = cast(dict[str, Any], audit)
                self.triage.audit(
                    str(values.get("session", "")),
                    str(values.get("chat_id", "")),
                    fallback,
                )
        return result

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

    def rendered_session_rules(self) -> str:
        """The session rules line, or ""; rules never fail a run."""
        if self.rules_renderer is None:
            return ""
        try:
            return self.rules_renderer().strip()
        except Exception:
            logger.warning("Session rules rendering failed; continuing without rules")
            return ""

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
        if self.any_delivery():
            # The delivery already fired: everything from here on can
            # only produce the wrap-up note. Say so instead of letting
            # the model guess and discarding its answer (bug 6).
            chat_history = [
                *chat_history,
                ChatMessage(role="user", content=_POST_DELIVERY_WRAP_UP_PROMPT),
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
            return await self.stop_with(ctx, self.stopped_response())
        if tool_calls and await self.repeats_tool_call(ctx, tool_calls):
            return await self.stop_with(ctx, self.stopped_response())
        if not tool_calls:
            return await self.finish_text_round(ctx, response, rounds)
        if self.delivery_complete(tool_calls) or rounds >= self.tool_round_limit:
            # Do not store calls which will not be executed: the chat API
            # requires every advertised call to have a tool response.
            reason = (
                "non-delivery round after completed delivery"
                if self.delivery_complete(tool_calls)
                else f"round limit {self.tool_round_limit}"
            )
            result = await self.wrap_up_response(ctx, reason)
            return await self.stop_with(ctx, StopEvent(result=result))
        await self.remember(ctx, response, tool_calls)
        return ToolCallEvent(tool_calls=tool_calls)

    async def stop_with(self, ctx: Context, event: StopEvent) -> StopEvent:
        """End the run with *event*: stamp the turn and collapse the delivery group.

        The wrap-up note is journalled and written to the notebook by
        ``wrap_up_response`` itself — it is a durable lesson
        (``kind=lesson``), never an assistant turn, so nothing is staged
        for storage here.
        """
        await self.mark_turn_handled(ctx)
        await self.collapse_delivery(ctx)
        return event

    async def finish_text_round(
        self, ctx: Context, response: ChatResponse, rounds: int
    ) -> ToolCallEvent | StopEvent:
        """A plain-text round: the reply, or the wrap-up note if delivered."""
        self.warn_accidental_silence(response, rounds)
        delivered = self.any_delivery()
        note = str(response.message.content or "").strip() if delivered else ""
        await self.remember(ctx, response, [], skip_text=delivered)
        await self.mark_turn_handled(ctx)
        await self.collapse_delivery(ctx)
        if delivered:
            if note:
                await self.store_wrap_up_note(note)
            self.audit_wrap_up_note(note)
        return StopEvent(result=self.drop_post_delivery_text(response))

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
            _POST_DELIVERY_WRAP_UP_PROMPT
            if delivered
            else _EARLY_STOPPING_PROMPT.format(limit=limit)
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
        return self.drop_post_delivery_text(response)

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

    def audit_wrap_up_note(self, note: str) -> None:
        """Journal the run's self-report (empty note included, visibly so)."""
        self.audit("wrap_up", note=note[:300])

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


class ObservableOpenAILike(OpenAILike):
    """OpenAILike whose instrumentation payload names model/temperature.

    Exists because the OTel llama-index instrumentor reads
    ``model_dict["model"]``/``["temperature"]`` for its span attributes
    but the base ``to_payload`` exposes neither (external constraint).
    Also slims the tool specs of every request (``wahabot.ai.tools.slim``):
    the base ``_prepare_chat_with_tools`` builds them from Pydantic's
    JSON schema. Titles and redundant ``strict: false`` are removed, and
    nullable unions are
    compacted without dropping explicit null support.
    """

    @override
    def to_payload(self) -> dict[str, Any]:
        return {
            **super().to_payload(),
            "model": self.model,
            "temperature": self.temperature,
        }

    @override
    def _prepare_chat_with_tools(
        self,
        tools: Sequence[BaseTool],
        user_msg: str | ChatMessage | None = None,
        chat_history: list[ChatMessage] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        tool_choice: str | dict[str, Any] | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        base = cast(
            Callable[..., dict[str, Any]],
            super()._prepare_chat_with_tools,
        )
        prepared = base(
            tools,
            user_msg=user_msg,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            tool_required=tool_required,
            tool_choice=tool_choice,
            strict=strict,
            **kwargs,
        )
        specs = prepared.get("tools")
        if isinstance(specs, list):
            prepared["tools"] = slim_tool_specs(specs)
        return prepared


def load_llm(settings: Settings) -> FunctionCallingLLM:
    """Configure the OpenAI-compatible chat LLM from settings.

    Sampling options follow the model card (see ``Settings``).
    ``top_p``/``presence_penalty`` ride ``additional_kwargs`` (merged
    into the request body); ``top_k``/``min_p``/``repetition_penalty``
    ride ``extra_body`` because the OpenAI SDK's typed ``create()``
    signature rejects them (external constraint).
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
    if settings.llm_auto_cache:
        # Requesty gateway flag: adds explicit cache breakpoints on the
        # largest content blocks, so repeated prefixes (the system
        # prompt, early history) are billed at cached rates.
        extra_body = dict(kwargs.get("extra_body", {}))
        extra_body["requesty"] = {"auto_cache": True}
        kwargs["extra_body"] = extra_body
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


def warn_context_mismatch(context_window: int, memory_token_limit: int) -> bool:
    """Warn loudly when the memory budget exceeds the model's context window.

    The aria-ai lesson: a budget larger than what the server can actually
    hold is silently clamped (or truncated) by the provider, and the
    memory system quietly believes it has room
    it does not. The window is the operator-declared
    ``WAHABOT_LLM_CONTEXT_WINDOW`` — llama-index's metadata default
    (3900) describes no real deployment, so it is never consulted.
    Returns True when the mismatch was found; 0 (unknown) is not one.
    """
    if context_window > 0 and memory_token_limit > context_window:
        logger.warning(
            "Memory token limit {limit} exceeds the model context window {window}",
            limit=memory_token_limit,
            window=context_window,
        )
        return True
    return False


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
