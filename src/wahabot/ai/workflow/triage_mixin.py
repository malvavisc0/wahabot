import asyncio
from collections.abc import Sequence
from typing import Any, cast, override

from llama_index.core.base.llms.types import (
    ChatMessage,
)
from llama_index.core.workflow import (
    Context,
)
from loguru import logger

from wahabot.ai.history import (
    INBOUND_ID_RE,
)
from wahabot.ai.tools.whatsapp import (
    RunTarget,
)
from wahabot.ai.triage import (
    MUTE_NOTE_KEY,
    TRIAGE_MESSAGE_CHARS,
    TRIAGE_MUTE_CHARS,
    TriageContext,
    TriageResult,
    format_turns,
    is_side_thread,
    result_from_dict,
    result_to_dict,
)
from wahabot.ai.workflow.base import AgentBase
from wahabot.core.notebook import (
    KIND_RULE,
)
from wahabot.core.thread_index import (
    OPERATOR_CHAT_ID,
)


class TriageMixin(AgentBase):
    def triage_context(
        self,
        target: RunTarget,
        message: str,
        messages: Sequence[ChatMessage],
        addressed: bool,
        quoted_other: bool = False,
        quoted_bot: bool = False,
        quoted_operator: bool = False,
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
            rules=self.rendered_gate_rules(),
            group=target.chat_id.endswith("@g.us"),
            addressed=addressed,
            quoted_other=quoted_other,
            quoted_bot=quoted_bot,
            quoted_operator=quoted_operator,
        )

    async def start_triage(
        self,
        ctx: Context,
        target: RunTarget,
        user_msg: str,
        addressed: bool,
        gate_first: bool,
        quoted_other: bool = False,
        quoted_bot: bool = False,
        quoted_operator: bool = False,
    ) -> bool:
        """Start or await the gate; False means the run must stay silent.

        Race mode (DMs, addressed turns) always returns True: the gate
        can only tighten tools there. Block mode (unaddressed groups)
        runs only on a verdict that grants participation, and a verdict
        that is present but silent (the new default) means silence. Block
        mode also fails closed: a gate that is unavailable — failed,
        timed out, unreachable — leaves the unaddressed turn silent
        rather than waking the agent over unread chatter. Race mode
        (DMs, addressed turns) always runs. The failed verdict is stored
        and audited either way.
        """
        if is_side_thread(
            target.chat_id.endswith("@g.us"),
            addressed,
            quoted_other,
            quoted_bot,
            quoted_operator,
        ):
            # Code-owned fence even when the gate is disabled: a reply to
            # another member's message is not a turn for the assistant.
            self.audit("side_thread", rule="quote of another member")
            return False
        if self.triage is None or target.chat_id == OPERATOR_CHAT_ID:
            return True
        try:
            memory = await self.ensure_memory(ctx)
            prior = await memory.aget_all()
            context = self.triage_context(
                target,
                user_msg,
                prior,
                addressed,
                quoted_other,
                quoted_bot,
                quoted_operator,
            )
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
                # Block mode fails closed: an unavailable gate declines the
                # unaddressed turn instead of waking the agent over chatter.
                return False
            return result.verdict.participate
        except Exception as exc:
            logger.warning(
                "Triage gate unavailable for {chat_id}: {error}",
                chat_id=target.chat_id,
                error=exc,
            )
            logger.opt(exception=exc).debug("Triage gate failure detail")
            # Block mode fails closed; race mode still runs unhardened.
            return not gate_first

    @override
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
