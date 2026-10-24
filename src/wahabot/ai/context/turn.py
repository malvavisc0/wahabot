import asyncio
import time
import uuid
from collections.abc import Sequence
from typing import Any

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.workflow import Context
from loguru import logger

from wahabot.ai import records
from wahabot.ai.context.participants import chat_rules
from wahabot.ai.context.user_message import final_reply, turn_user_message
from wahabot.ai.intake import IntakeLedger
from wahabot.ai.messages import (
    TURN_HANDLED_KWARG,
    TURN_SERIALS_KWARG,
    conversation_jid,
    newest_quote_flags,
)
from wahabot.ai.tools.whatsapp import (
    RunTarget,
    bind_target,
    jids_in_text,
    reset_target,
)
from wahabot.ai.turn_records import (
    RecordEntry,
)
from wahabot.ai.workflow import FunctionCallingAgentWorkflow
from wahabot.core import images as image_store
from wahabot.core.models import WahaEvent
from wahabot.core.notebook import (
    RULE_HARD_MUTE,
    RULE_MESSAGE_PREFIX,
)
from wahabot.core.recovery.context import apology
from wahabot.core.tickets.store import TicketStore
from wahabot.core.transport import Transport
from wahabot.core.transport.context import bind_intake_ids, reset_intake_ids
from wahabot.settings import Settings


async def record_handled_turn(
    agent: FunctionCallingAgentWorkflow,
    ctx: Context,
    user_msg: str,
    *,
    gist: str = "",
) -> None:
    """Store a quiet turn as ledger lines, stamped handled.

    The gate read these messages and declined them; the next prompt gets one
    line each instead of the wall it already judged. Without the stamp the
    next run would treat the trailing message as scaffolding and drop it,
    vanishing it from the buffer, recap, and thread index alike.
    """
    memory = await agent.ensure_memory(ctx)
    serials = await ctx.store.get("turn_serials", default=None) or []
    content = records.ledger_turn(
        user_msg, gist=gist, newest_serial=max(serials, default=0)
    )
    await memory.aput(
        ChatMessage(
            role=MessageRole.USER,
            content=content,
            additional_kwargs={
                TURN_HANDLED_KWARG: True,
                **({TURN_SERIALS_KWARG: list(serials)} if serials else {}),
            },
        )
    )


async def bind_inbound_image(
    image: dict[str, Any] | None, settings: Settings | None, target: RunTarget
) -> None:
    """Persist this turn's inbound image so ``edit_image`` can repaint it.

    Only when the image service is configured — otherwise nothing could
    edit it and the temp file would be pure litter.
    """
    if image is None or settings is None or not settings.image_api_base:
        return
    data = image.get("data")
    if not data:
        return
    target.image_path = str(await asyncio.to_thread(image_store.save, data))


async def bind_run_rules(
    agent: FunctionCallingAgentWorkflow,
    ctx: Context | None,
    user_msg: str,
    target: RunTarget,
    addressed: bool,
    gate_first: bool,
    quoted_other: bool = False,
    quoted_bot: bool = False,
    quoted_operator: bool = False,
) -> bool:
    """Apply the chat's code-read rules to *target*; False when hard-muted.

    Read before the triage gate so a hard mute holds even when the gate
    is down. A muted chat still runs the gate (so an adversarial read or
    a human request can escalate) but never answers; its turn is
    recorded stamped handled. The prefix applies to everything the run
    delivers.
    """
    if target.armed:
        target.allowed_chats.update(jids_in_text(user_msg))
    rules = await chat_rules(agent.notebook, target.session, target.chat_id)
    if RULE_HARD_MUTE in rules:
        logger.info("Chat is hard-muted; not answering {chat_id}", chat_id=target.chat_id)
        if ctx is not None and await run_gate(
            agent,
            ctx,
            target,
            user_msg,
            addressed,
            gate_first,
            quoted_other,
            quoted_bot,
            quoted_operator,
        ):
            await record_handled_turn(agent, ctx, user_msg)
        return False
    target.prefix = f"{apology.get()}{rules.get(RULE_MESSAGE_PREFIX, '')}"
    return True


async def run_gate(
    agent: FunctionCallingAgentWorkflow,
    ctx: Context,
    target: RunTarget,
    user_msg: str,
    addressed: bool,
    gate_first: bool,
    quoted_other: bool = False,
    quoted_bot: bool = False,
    quoted_operator: bool = False,
) -> bool:
    """Reset triage state and consult the gate; False when it ends the turn.

    A silent gate still records the turn, stamped handled (the same
    stamp a silent run gets): without it the next run's ``repair_memory``
    would drop the trailing user message as scaffolding, vanishing it
    from the buffer, recap, and thread index alike.
    """
    await ctx.store.set("triage_result", None)
    await ctx.store.set("triage_audit", None)
    if await agent.start_triage(
        ctx,
        target,
        user_msg,
        addressed,
        gate_first,
        quoted_other,
        quoted_bot,
        quoted_operator,
    ):
        await ctx.store.set("turn_speak", True)
        return True
    await ctx.store.set("turn_speak", False)
    result = await agent.stored_triage_result(ctx)
    gist = result.verdict.gist if result is not None else ""
    await record_handled_turn(agent, ctx, user_msg, gist=gist)
    return False


async def prepare_turn_state(
    ctx: Context | None,
    event: WahaEvent,
    addressed: bool,
) -> None:
    """Record the header verdict and chat shape before the run."""
    if ctx is not None:
        chat_id = conversation_jid(event)
        group = chat_id.endswith("@g.us")
        # The prompt-time turn header (history.turn_header) reads these back:
        # the wake gate's verdict and whether this is a group/direct chat.
        await ctx.store.set("turn_addressed", addressed)
        await ctx.store.set("turn_speak", addressed)
        await ctx.store.set("turn_group", group)
        await ctx.store.set("turn_direct", event.event != "command" and not group)


def durable_run_owner(
    intake: IntakeLedger | None, turn_ids: Sequence[str], owner: str
) -> str:
    """Resolve the one owner a run's claims and target share.

    A durable run (``intake`` plus ``turn_ids``) starts only with the
    owner its ``claim_turn`` recorded — raising on a missing owner
    keeps the claim, the target and the owner-scoped settle attached to
    the same run by construction. Every other shape mints a fresh
    owner, so this stays the only minting site.
    """
    if not intake or not turn_ids:
        return owner or uuid.uuid4().hex
    if not owner:
        raise ValueError("a durable run needs the owner its claim_turn recorded")
    return owner


async def handle_message(
    event: WahaEvent,
    agent: FunctionCallingAgentWorkflow,
    ctx: Context | None = None,
    image: dict[str, Any] | None = None,
    images: list[dict[str, Any]] | None = None,
    settings: Settings | None = None,
    waha: Transport | None = None,
    armed: bool = False,
    pinned_note: str = "",
    bot_name: str | None = None,
    bot_mention_regex: str | None = None,
    body_scrubbed: bool = False,
    addressed: bool = True,
    gate_first: bool = False,
    record_entries: Sequence[RecordEntry] | None = None,
    intake: IntakeLedger | None = None,
    turn_ids: Sequence[str] = (),
    turn_started_at: float | None = None,
    owner: str = "",
    ticket_store: TicketStore | None = None,
    ticket_id: int = 0,
) -> tuple[str, RunTarget]:
    """Run the agent workflow over an incoming message event.

    Returns ``(reply, target)``: the run's final text and its delivery
    target (``sent``/``reacted`` latches), so the caller can tell a
    delivered run from a text reply without touching run-scoped state.

    ``pinned_note`` carries turn-scoped ground truth the *caller*
    resolved before the run (see :mod:`wahabot.ai.resolve`): it rides
    the turn text — never memory — so the model starts from facts a
    mixed history cannot infer (the real chat a command named, its real
    last message). Empty for every chat-run caller.

    ``bot_name``/``bot_mention_regex`` (the session config's identity
    pair) render the addressed-mention marker into a group turn whose
    text names the bot (:func:`turn_addressed_note`): the wake gate already
    decided the message is for the bot, and the marker tells the model
    so it never re-derives — and misreads — that fact. Absent both, no
    marker (operator commands, tests). An *unaddressed* group turn gets
    the explicit negative marker instead, so group chatter the gate woke
    the agent for is never mistaken for a direct address.

    ``body_scrubbed`` says the caller already scrubbed the body's
    member text (the burst path, per member, before stamping its id
    notes) so :func:`turn_body` must not scrub again and break the
    code-stamped ids.

    ``image`` (single) or ``images`` (an album, already downloaded)
    carry image bytes (``data`` + ``mimetype``); they ride along as
    ``image_blocks`` on the run and are injected into the first LLM
    call only — memory stays text-only, so no megabyte payloads
    accumulate in the rolling buffer. Downloaded images also carry a
    one-line ``caption`` (set by the download path, outside the agent
    lock — see ``wahabot.ai.vision``); it goes *into* the user message
    text — ``(image shows: beer glass with bear foam)`` — so later
    rounds and later turns keep a text anchor once the pixels are
    gone; a missing caption falls back to the bare ``(image)`` marker.

    With ``settings.vision`` enabled, image URLs sniffed from the
    message text are fetched too (see ``wahabot.ai.tools.url_images``), so a
    bare link like "look at this https://host/pic.png" shows the model
    the picture as well.

    A durable run (``intake`` plus ``turn_ids``) must pass the
    ``owner`` that claimed its turn rows (``claim_turn``);
    :func:`durable_run_owner` raises on a missing one so the claim,
    the target and the settle stay attached to one run by construction
    instead of by threading discipline.
    """
    run_owner = durable_run_owner(intake, turn_ids, owner)
    chat_id = str(event.payload.get("from", ""))
    logger.info("Agent handling message from {chat_id}", chat_id=chat_id)
    newest_event = record_entries[-1].event if record_entries else event
    quoted_other, quoted_bot, quoted_operator = newest_quote_flags(newest_event)
    await prepare_turn_state(ctx, event, addressed)
    user_msg, image_blocks, serials = await turn_user_message(
        event,
        image=image,
        images=images,
        settings=settings,
        waha=waha,
        pinned_note=pinned_note,
        bot_name=bot_name,
        bot_mention_regex=bot_mention_regex,
        body_scrubbed=body_scrubbed,
        addressed=addressed,
        record_entries=record_entries,
    )
    if ctx is not None and serials:
        await ctx.store.set("turn_serials", serials)
    # Bind this run's delivery target around the whole run: the workflow
    # engine schedules steps/tasks under this context, so the binding
    # propagates to every step and tool call of THIS run — concurrent
    # runs bind their own targets and never see each other's. The
    # binding resets before returning, so the *target itself* is
    # returned alongside the reply for the caller's latch checks.
    typing = (
        (settings.typing_presence_min_s, settings.typing_presence_max_s)
        if settings is not None
        else None
    )
    target = RunTarget(
        session=event.session,
        chat_id=chat_id,
        armed=armed,
        typing=typing,
        intake=intake,
        owner=run_owner,
        turn_ids=list(turn_ids),
        turn_started_at=(
            time.monotonic() if turn_started_at is None else turn_started_at
        ),
        ticket_store=ticket_store,
        ticket_id=ticket_id,
    )
    token = bind_target(target)
    intake_token = bind_intake_ids(turn_ids)
    try:
        await bind_inbound_image(image, settings, target)
        if not await bind_run_rules(
            agent,
            ctx,
            user_msg,
            target,
            addressed,
            gate_first,
            quoted_other,
            quoted_bot,
            quoted_operator,
        ):
            return "", target
        if ctx is not None and not await run_gate(
            agent,
            ctx,
            target,
            user_msg,
            addressed,
            gate_first,
            quoted_other,
            quoted_bot,
            quoted_operator,
        ):
            return "", target
        result = await run_agent_once(agent, ctx, user_msg, image_blocks, settings)
    finally:
        await release_run(agent, ctx, token, intake_token)
    return final_reply(result), target


async def run_agent_once(
    agent: FunctionCallingAgentWorkflow,
    ctx: Context | None,
    user_msg: str,
    image_blocks: list[Any],
    settings: Settings | None,
) -> Any:
    """Run the agent under the per-run wall-clock cap.

    The library's own ``timeout=`` must stay None: on the reused
    per-chat Context it acts as a cumulative alive-time budget that
    would sum every run's latency and permanently kill busy chats, so
    the cap is enforced here instead; ``wait_for`` raises ``TimeoutError``,
    classified by ``run_timed_out``.
    """
    timeout = settings.run_timeout if settings is not None else None
    return await asyncio.wait_for(
        agent.run(input=user_msg, image_blocks=image_blocks, ctx=ctx),
        timeout=timeout or None,
    )


async def release_run(
    agent: FunctionCallingAgentWorkflow,
    ctx: Context | None,
    token: Any,
    intake_token: Any = None,
) -> None:
    """Cancel any pending gate and unbind this run's delivery context."""
    if ctx is not None:
        await agent.cancel_triage(ctx)
    reset_target(token)
    if intake_token is not None:
        reset_intake_ids(intake_token)
