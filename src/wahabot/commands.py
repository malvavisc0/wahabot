"""Operator command handling: agent runs over the operator's own history.

A ``command`` event is a wahabot-internal event type (not a WAHA one)
posted to the same HMAC-verified webhook by ``wahabot tell``. Possession
of the HMAC key is the operator credential — it can already forge any
WAHA event — so commands bypass the chat gates (``chat_allowed``,
``is_group_addressed``) that exist to keep *strangers* out.

The command runs the same agent with the same tools over the operator's
own rolling history (context key ``"operator"``): commands share one
conversation, LRU-evicted and persisted like any chat's, so follow-ups
("now send that to the second group") work without restating context.
No chat's memory is touched — the instruction *names* its targets —
and no gates apply. Because that shared history is channel-agnostic,
a command that names a chat also gets that chat's *real* last message
pinned into its turn before the model reasons (see
:mod:`wahabot.ai.resolve`): "the last message in <chat>" is resolved by
code, never inferred from mixed memory.
"""

import asyncio
import time
import uuid

from llama_index.core.workflow import Context
from loguru import logger

from wahabot.ai.context import handle_message
from wahabot.ai.history import ContextBudgetError
from wahabot.ai.observability import chat_trace_attributes
from wahabot.ai.resolve import ResolvedChat, chat_context_note, resolve_last_message
from wahabot.ai.serials import serial_store
from wahabot.ai.tools.runcontext import delivery_path
from wahabot.ai.tools.whatsapp import RunTarget
from wahabot.ai.workflow import FunctionCallingAgentWorkflow
from wahabot.core.audit import save_action, save_run_outcome
from wahabot.core.models import WahaEvent
from wahabot.core.runs import chat_lock, context_for, persist_memory
from wahabot.core.tickets.store import TicketStore
from wahabot.core.transport import Transport
from wahabot.settings import Settings
from wahabot.status import (
    classify_llm_failure,
    llm_failure_class,
    mark_llm_recovered,
    probe_session_recovery,
    session_healthy,
)
from wahabot.webhook import on_command

#: Prefix marking an agent turn as an operator command (the session
#: prompt carries a matching "Operator commands" section).
COMMAND_PREFIX = "[operator command]"

#: The operator's own context key: one shared history for every command
#: channel (``wahabot tell`` from the CLI, self-chat mentions), LRU'd
#: and persisted under ``memory/<session>/operator.json`` like a chat.
OPERATOR_CHAT_ID = "operator"


def pin_serial(resolved: ResolvedChat | None, settings: Settings, session: str) -> int:
    """The ``#m<n>`` of the pinned last message, or 0 when unknown.

    The registry is written at ingest only: a chat the bot has never
    run in has no serials, and the note then offers the raw id alone —
    the one id shape the fence always accepts.
    """
    if resolved is None or resolved.last_message is None:
        return 0
    registry = serial_store(str(settings.data_dir)).for_chat(session, resolved.chat_id)
    return registry.serial_for(str(resolved.last_message.get("id", "")))


def chat_display_name(waha: Transport, session: str, chat_id: str) -> str:
    """A chat's display name for operator notices; the bare JID if unknown.

    The name alone is the notice: the raw JID is a LID-shaped opaque
    string that reads as noise, so it is only the floor for a chat the
    overview cannot name. The delivered-notice is a convenience, not a
    feature — a WAHA hiccup or a nameless chat must never turn it into
    an error, so the overview lookup is best-effort.
    """
    try:
        overview = waha.get_chat_overview(session, chat_id)
        name = str(overview.get("name", "")).strip() if overview else ""
        return name or chat_id
    except Exception:
        return chat_id


def command_outcome(
    reply: str, target: RunTarget, budget_error: ContextBudgetError | None
) -> tuple[str, str]:
    """The run outcome and failure class for one finished command."""
    if budget_error is not None:
        return "failed", "budget"
    if target.sent:
        return "delivered_via_tool", ""
    if target.reacted:
        return "reaction", ""
    if reply.strip():
        return "reply", ""
    return ("empty" if target.outcome == "empty" else "silence"), ""


async def run_command(
    event: WahaEvent,
    agent: FunctionCallingAgentWorkflow,
    settings: Settings,
    waha: Transport,
    ticket_store: TicketStore | None = None,
) -> str:
    """Run the agent over the command instruction, on the operator context.

    No dedup (the command id is unique by construction), no staleness
    gate (no WAHA redelivery for a command the operator just fired), no
    chat gates. The run uses the synthetic "operator" context, not a
    deliverable WhatsApp chat. Delivery tools must receive an explicit
    ``chat`` JID; plain final text is returned to the command caller.
    A delivery latch from another run can never block this command's
    send (each run binds its own target). ``armed=True`` opens the
    cross-chat fence in the WhatsApp tools for this run alone; the
    arming flag rides the run-scoped binding, so a concurrent chat run
    can never inherit it.

    All commands — however issued — share the ``"operator"`` history:
    one rolling conversation across commands, serialized by the
    operator's run lock (a second command waits rather than
    interleaving runs over one buffer) and persisted at run end, so
    follow-ups and refinements work across commands and restarts.

    Returns the run's final text (empty when the run delivered via a
    tool or stayed silent). A `wahabot tell` command logs it — the
    terminal has no chat to land in — while a self-chat command has
    the caller send it back to the operator's "message yourself" chat
    via ``reply_chat_id``. When the run delivered via a tool instead,
    the caller gets a short delivered-notice so the console is never
    silent about where the answer went. Either way the decision is
    journaled to the audit file — an
    operator command is a bot action too.
    """
    instruction = str(event.payload.get("body", "")).strip()
    if not instruction:
        logger.debug("Ignoring empty command {id}", id=event.payload.get("id"))
        return ""
    started = time.monotonic()
    target = RunTarget(
        session=event.session, chat_id=OPERATOR_CHAT_ID, ticket_store=ticket_store
    )
    outcome = "failed"
    failure_class = ""
    try:
        logger.info(
            "Running operator command {id}: {instruction}",
            id=event.payload.get("id"),
            instruction=instruction[:200],
        )
        # Ground truth before the model: when the command names a chat,
        # resolve it against the operator's chat list and pin that chat's
        # real last message into the turn. The shared operator history is
        # channel-agnostic — it cannot answer "the last message in <chat>" —
        # so the note replaces the inference, not supplements it. A command naming no
        # chat resolves to nothing and runs unchanged. Failures are
        # swallowed inside: pinning is best-effort, never a blocker.
        resolved = await asyncio.to_thread(
            resolve_last_message, waha, event.session, instruction
        )
        pinned_note = (
            chat_context_note(resolved, pin_serial(resolved, settings, event.session))
            if resolved
            else ""
        )
        budget_error: ContextBudgetError | None = None
        async with chat_lock(event.session, OPERATOR_CHAT_ID):
            ctx = await context_for(event.session, OPERATOR_CHAT_ID, agent, settings)
            with chat_trace_attributes(
                "operator-command",
                message_ids=[str(event.payload.get("id", ""))],
                reply_to=str(event.payload.get("reply_to", "")),
            ):
                # armed=True: operator commands are the one trusted cross-chat
                # channel; the fence in the WhatsApp tools opens for this run
                # alone (the arming flag rides the run's own target binding,
                # so a concurrent chat run can never inherit it).
                reply, target, budget_error = await run_command_turn(
                    event, agent, ctx, settings, waha, pinned_note, ticket_store
                )
            await persist_memory(settings, event.session, OPERATOR_CHAT_ID, ctx)
        outcome, failure_class = command_outcome(reply, target, budget_error)
        return command_reply(event, settings, reply, target, waha, budget_error)
    except Exception as exc:
        failure_class = llm_failure_class(exc)
        raise
    finally:
        # The outcome row belongs to the operator console, not any chat
        # the command happened to deliver into: its message ids are the
        # command's, and the delivery tools already journalled theirs.
        save_run_outcome(
            settings.data_dir,
            event.session,
            OPERATOR_CHAT_ID,
            outcome=outcome,
            duration_ms=int((time.monotonic() - started) * 1000),
            message_ids=[str(event.payload.get("id", ""))],
            outbound_id=target.sent_id,
            delivered_via=delivery_path(outcome, target),
            failure_class=failure_class,
        )


async def run_command_turn(
    event: WahaEvent,
    agent: FunctionCallingAgentWorkflow,
    ctx: Context,
    settings: Settings,
    waha: Transport,
    pinned_note: str,
    ticket_store: TicketStore | None = None,
) -> tuple[str, RunTarget, ContextBudgetError | None]:
    """Run one operator-command turn, classifying its failures.

    Returns the final text, run target, and a ``ContextBudgetError``
    when the command could never fit the context budget — a permanent
    failure: logging and re-raising would only put a redelivered
    command (or a self-chat retry) back on the same doomed path, so
    the error is reported to the caller for journaling instead. Any
    other exception keeps the shared classification and re-raises:
    commands have no seen marker (their ids are unique), so the duties
    are the notifies, and the self-chat caller drops its own seen
    marker so the command retries on redelivery.
    """
    try:
        reply, target = await handle_message(
            event,
            agent,
            ctx=ctx,
            settings=settings,
            waha=waha,
            armed=True,
            surface="operator",
            pinned_note=pinned_note,
            ticket_store=ticket_store,
        )
    except ContextBudgetError as exc:
        logger.warning(
            "Command {id} exceeds the context budget ({limit} tokens): {exc}",
            id=event.payload.get("id"),
            limit=settings.memory_token_limit,
            exc=exc,
        )
        return (
            "",
            RunTarget(
                session=event.session,
                chat_id=OPERATOR_CHAT_ID,
                surface="operator",
            ),
            exc,
        )
    except Exception as exc:
        await classify_llm_failure(
            waha, event.session, exc, settings.llm_timeout, settings.run_timeout
        )
        raise
    await mark_llm_recovered(waha, event.session)
    return reply, target, None


def command_reply(
    event: WahaEvent,
    settings: Settings,
    reply: str,
    target: RunTarget,
    waha: Transport,
    budget_error: ContextBudgetError | None = None,
) -> str:
    """Journal the command's outcome and return its operator-facing text."""
    reply = (reply or "").strip()
    if budget_error is not None:
        # Permanent failure, its own audit kind: never borrowed silence
        # ("the log must say why"), and the returned text rides the
        # self-chat reply so the operator's phone shows it.
        save_action(
            settings.data_dir,
            event.session,
            "operator_budget_error",
            chat_id=OPERATOR_CHAT_ID,
            error=str(budget_error),
        )
        return (
            f"Command rejected: {budget_error}."
            " Shorten it, lower the pinned context, or raise WAHABOT_MEMORY_TOKEN_LIMIT."
        )
    if not reply and target.sent and target.sent != OPERATOR_CHAT_ID:
        name = chat_display_name(waha, event.session, target.sent)
        reply = f"Delivered to {name}."
    # The audit row names the chat the command acted on — the tool's
    # delivery target when one fired, else the operator context the
    # command ran over.
    acted_chat = target.sent or OPERATOR_CHAT_ID
    if reply:
        save_action(
            settings.data_dir,
            event.session,
            "operator_reply",
            chat_id=acted_chat,
            reply=reply,
        )
        logger.info(
            "Command {id} final reply: {reply}",
            id=event.payload.get("id"),
            reply=reply[:500],
        )
    else:
        save_action(
            settings.data_dir,
            event.session,
            "operator_silence",
            chat_id=acted_chat,
        )
    return reply


def register_command_handler(
    settings: Settings,
    waha: Transport,
    agent: FunctionCallingAgentWorkflow,
    ticket_store: TicketStore | None = None,
) -> None:
    """Register the webhook command handler around the shared agent.

    Commands serialize on the operator's run lock — they share one
    history buffer — but never take a chat's lock, so chat runs are
    never delayed by a command (and vice versa: different keys).
    """

    @on_command
    async def handle_command(event: WahaEvent) -> None:
        """Run one operator command (concurrently with chat runs)."""
        if not session_healthy():
            # Same recovery probe as the message path: a delivered
            # webhook is evidence the session may be back; a WORKING
            # answer from GET /api/sessions unmutes. Commands carry no
            # seen marker, so there is nothing to drop for redelivery.
            logger.info(
                "Muting command {id} while WAHA session is not WORKING",
                id=event.payload.get("id"),
            )
            await probe_session_recovery(waha, event.session)
            return
        if event.session != settings.session:
            return
        await run_command(event, agent, settings, waha, ticket_store)


def build_command_event(session: str, instruction: str) -> dict[str, object]:
    """The command event payload ``wahabot tell`` posts to the webhook.

    Shaped as a message-like event so the standard entrypoint renders
    the turn: ``body`` carries the instruction prefixed with the
    ``[operator command]`` marker (the session prompt keys on it) and
    ``notifyName`` names the sender "operator". The ``instruction``
    field keeps the raw operator intent explicit for the journal and
    future issuers.
    """
    body = f"{COMMAND_PREFIX} {instruction}"
    return {
        "id": f"evt_command_{uuid.uuid4().hex[:12]}",
        "timestamp": int(time.time()),
        "event": "command",
        "session": session,
        "me": None,
        "payload": {
            "id": f"cmd_{uuid.uuid4().hex[:12]}",
            "from": "operator",
            "fromMe": False,
            "body": body,
            "_data": {"type": "chat", "notifyName": "operator"},
            "instruction": instruction,
            "issuer": "operator",
        },
    }
