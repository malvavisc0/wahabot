import asyncio
import time
from dataclasses import replace
from typing import Any

import httpx
from loguru import logger

from wahabot.ai.observability import (
    masked_chat_id,
    observation,
    update_observation,
)
from wahabot.ai.tools.whatsapp import EscalationChannel, forward_escalation
from wahabot.ai.triage.models import TriageContext, TriageResult, TriageVerdict
from wahabot.ai.triage.parsing import parse_verdict, render_context
from wahabot.ai.triage.policy import (
    calm_angry_customer,
    repair_question_gist,
    restore_cross_sender_attack,
)
from wahabot.ai.triage.prompts import (
    ESCALATION_CONCURRENCY,
    GATE_RETRY_S,
    TRIAGE_MAX_TOKENS,
    TRIAGE_SYSTEM_PROMPT,
)
from wahabot.core.audit import save_action
from wahabot.core.transport import Transport
from wahabot.settings import Settings


class TriageGate:
    """An OpenAI-compatible client for the small triage model.

    ``transport`` exists so tests can answer without a network. The
    gate never raises for a service problem: it returns a normal-run
    fallback, audits the outcome, and lets the caller proceed.
    """

    def __init__(
        self,
        settings: Settings,
        waha: Transport | None = None,
        channel: EscalationChannel | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.waha = waha
        self.channel = channel
        self.transport = transport
        self.api_base = settings.gate_api_base.rstrip("/")
        self.model = settings.gate_model
        self.api_key = settings.gate_api_key
        self.timeout = settings.gate_timeout
        self.down_until = 0.0
        self.warned: set[str] = set()
        #: In-flight escalation sends, held so they are neither garbage
        #: collected nor cancelled with the verdict's task.
        self.escalations: set[asyncio.Task[None]] = set()
        #: Bounds concurrent escalation sends; see ESCALATION_CONCURRENCY.
        self._escalation_sem = asyncio.Semaphore(ESCALATION_CONCURRENCY)

    @property
    def url(self) -> str:
        """The chat-completions endpoint, or "" when unconfigured."""
        return f"{self.api_base}/chat/completions" if self.api_base else ""

    def headers(self) -> dict[str, str]:
        """Bearer auth when a gate key is configured."""
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def payload(self, context: TriageContext) -> dict[str, Any]:
        """The OpenAI-compatible request body for one verdict."""
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                {"role": "user", "content": render_context(context)},
            ],
            "temperature": 0.0,
            "max_tokens": TRIAGE_MAX_TOKENS,
        }

    def is_down(self) -> bool:
        """Whether the gate is in its post-failure cooldown."""
        return time.monotonic() < self.down_until

    async def classify_bounded(self, context: TriageContext) -> TriageResult:
        """Classify with a hard timeout; always returns a result and audits it.

        The cooldown path audits too: a verdict suppressed by a gate
        outage is a decision the journal must show, not a silent one.
        A successful verdict clears the warned set so the next outage
        episode logs its failures fresh instead of staying muted for
        the process lifetime.
        """
        if self.is_down():
            result = TriageResult.fallback(failed=True)
            # The cooldown path is awaited before the agent starts, so
            # the journal append runs off the event loop: a synchronous
            # write per group message while the gate is down would block
            # the run path for the whole cooldown window.
            await asyncio.to_thread(self.audit, context.session, context.chat_id, result)
            return result
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(self.classify(context), timeout=self.timeout)
        except TimeoutError:
            result = TriageResult.fallback(timed_out=True)
            self.warn_once(
                "timeout",
                "Triage gate timed out for {chat_id}",
                chat_id=context.chat_id,
            )
        except Exception as exc:
            result = TriageResult.fallback(failed=True)
            self.warn_once(
                "failure",
                "Triage gate failed for {chat_id}: {exc}",
                chat_id=context.chat_id,
                exc=exc,
            )
        latency_ms = int((time.monotonic() - started) * 1000)
        result = TriageResult(
            result.verdict,
            latency_ms=latency_ms,
            timed_out=result.timed_out,
            failed=result.failed,
        )
        if result.failed or result.timed_out:
            self.down_until = time.monotonic() + GATE_RETRY_S
        else:
            self.down_until = 0.0
            self.warned.clear()
        self.audit(context.session, context.chat_id, result)
        return result

    async def classify(self, context: TriageContext) -> TriageResult:
        """One request and parse; service failures return a fallback."""
        try:
            raw = await self.complete(context)
            verdict = parse_verdict(raw)
        except Exception as exc:
            self.warn_once(
                "request",
                "Triage gate request failed for {chat_id}: {exc}",
                chat_id=context.chat_id,
                exc=exc,
            )
            return TriageResult.fallback(failed=True)
        verdict = calm_angry_customer(verdict, context.message)
        verdict = restore_cross_sender_attack(verdict, context.message)
        verdict = repair_question_gist(verdict, context.message)
        if context.side_thread:
            # A reply to another member's message is not a turn for the
            # assistant. This fence is code-owned: a gate verdict cannot
            # grant a reply or an escalation for a side thread.
            verdict = replace(verdict, participate=False, escalate=False)
        # Participation is a group-only label; DMs and addressed turns
        # always run (the plan's rules bypass).
        if not context.group or context.addressed:
            verdict = replace(verdict, participate=True)
        if verdict.escalate:
            # The send is dispatched, not awaited: a slow WAHA send must
            # neither eat the verdict's latency budget nor be cancelled
            # by a run that ends before the gate task completes.
            self.dispatch_escalation(context, verdict)
        return TriageResult(verdict)

    async def complete(self, context: TriageContext) -> str:
        """POST one request and return the assistant's text.

        The gate call is a bare ``httpx`` client, so the LlamaIndex
        instrumentor never saw it: each verdict runs as its own Langfuse
        observation, marked ``guardrail`` so gate calls stay out of
        LLM spend unless asked for. Failures mark the observation
        ``ERROR`` and re-raise; the caller's fallback owns the verdict.
        """
        if not self.url:
            raise ValueError("triage gate has no API base")
        body = self.payload(context)
        async with observation(
            "triage-gate",
            as_type="guardrail",
            model=self.model,
            input={
                "system": body["messages"][0]["content"],
                "user": body["messages"][1]["content"],
            },
            metadata={
                "chat_id": masked_chat_id(context.chat_id),
                "session": context.session,
            },
        ) as span:
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout, transport=self.transport
                ) as client:
                    response = await client.post(
                        self.url, json=body, headers=self.headers()
                    )
                    response.raise_for_status()
                    payload = response.json()
                try:
                    text = str(payload["choices"][0]["message"]["content"])
                except (KeyError, IndexError, TypeError) as exc:
                    raise ValueError("triage response has no message content") from exc
            except Exception as exc:
                update_observation(span, level="ERROR", status_message=str(exc))
                raise
            usage: dict[str, Any] = payload.get("usage") or {}
            update_observation(
                span,
                output=text,
                usage_details={
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                },
            )
            return text

    async def escalate(self, context: TriageContext, verdict: TriageVerdict) -> None:
        """Forward a code-written escalation off the event loop."""
        if self.waha is None or self.channel is None:
            return
        reason = verdict.why or "needs human review"
        await asyncio.to_thread(
            forward_escalation,
            self.waha,
            self.channel,
            self.settings,
            context.session,
            context.chat_id,
            f"triage flagged this chat: {reason}",
        )

    def dispatch_escalation(self, context: TriageContext, verdict: TriageVerdict) -> None:
        """Send the gate's escalation off the verdict's latency path.

        The send runs as its own task, held on the gate so it is never
        garbage collected mid-send and never cancelled with the verdict's
        task — a run that finishes early cannot kill an escalation the
        verdict already asked for.
        """
        task = asyncio.create_task(self.escalate_safely(context, verdict))
        self.escalations.add(task)
        task.add_done_callback(self.escalations.discard)

    async def escalate_safely(
        self, context: TriageContext, verdict: TriageVerdict
    ) -> None:
        """One escalation send; any failure is logged, never raised.

        Bounded by ESCALATION_CONCURRENCY so a burst of escalate
        verdicts queues rather than fanning out into unbounded sends.
        """
        async with self._escalation_sem:
            try:
                await self.escalate(context, verdict)
            except Exception as exc:
                self.warn_once(
                    "escalate",
                    "Triage escalation failed for {chat_id}: {exc}",
                    chat_id=context.chat_id,
                    exc=exc,
                )

    async def close(self) -> None:
        """Await in-flight escalation sends so shutdown never drops one.

        Registered as an app shutdown hook. A failure inside a send is
        already swallowed by ``escalate_safely``; ``gather`` only waits.
        """
        if self.escalations:
            await asyncio.gather(*tuple(self.escalations), return_exceptions=True)

    def audit(self, session: str, chat_id: str, result: TriageResult) -> None:
        """Journal one verdict, including timeouts and parse failures."""
        save_action(
            self.settings.data_dir,
            session,
            "triage",
            chat_id=chat_id,
            adversarial=result.verdict.adversarial,
            escalate=result.verdict.escalate,
            participate=result.verdict.participate,
            why=result.verdict.why,
            gist=result.verdict.gist,
            needs=list(result.verdict.needs),
            latency_ms=result.latency_ms,
            timed_out=result.timed_out,
            failed=result.failed,
        )

    def warn_once(self, kind: str, message: str, **fields: Any) -> None:
        """Log one failure kind per outage episode; every verdict is audited."""
        if kind in self.warned:
            return
        self.warned.add(kind)
        logger.warning(message, **fields)
