"""Operator command handling: agent runs without a chat of their own.

A ``command`` event is a wahabot-internal event type (not a WAHA one)
posted to the same HMAC-verified webhook by ``wahabot tell``. Possession
of the HMAC key is the operator credential — it can already forge any
WAHA event — so commands bypass the chat gates (``chat_allowed``,
``is_group_addressed``) that exist to keep *strangers* out.

The command runs the same agent with the same tools, but on a fresh
``Context`` per command: a command has no chat of its own (the
instruction *names* its targets), so no chat's memory is touched and no
gates apply.
"""

import time
import uuid

from llama_index.core.workflow import Context
from loguru import logger

from wahabot.ai.context import handle_message
from wahabot.ai.observability import chat_trace_attributes
from wahabot.ai.workflow import FunctionCallingAgentWorkflow
from wahabot.core.models import WahaEvent
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings
from wahabot.status import session_healthy
from wahabot.webhook import on_command

#: Prefix marking an agent turn as an operator command (the session
#: prompt carries a matching "Operator commands" section).
COMMAND_PREFIX = "[operator command]"


async def run_command(
    event: WahaEvent,
    agent: FunctionCallingAgentWorkflow,
    settings: Settings,
    waha: WahaClient,
) -> str:
    """Run the agent over the command instruction, on a fresh context.

    No dedup (the command id is unique by construction), no staleness
    gate (no WAHA redelivery for a command the operator just fired), no
    chat gates. The run's delivery target is the event's ``from``
    ("operator") so the run behaves like a DM: the model may pass
    ``chat=…`` explicitly (a group or a person resolved via
    ``resolve_chat``) or omit it, exactly as in a normal chat — and a
    delivery latch from another run can never block this command's
    send (each run binds its own target). ``armed=True`` opens the
    cross-chat fence in the WhatsApp tools for this run alone; the
    arming flag rides the run-scoped binding, so a concurrent chat run
    can never inherit it.

    Returns the run's final text (empty when the run delivered via a
    tool or stayed silent). A `wahabot tell` command logs it — the
    terminal has no chat to land in — while a self-chat command has
    the caller send it back to the operator's "message yourself" chat
    via ``reply_chat_id``.
    """
    instruction = str(event.payload.get("body", "")).strip()
    if not instruction:
        logger.debug("Ignoring empty command {id}", id=event.payload.get("id"))
        return ""
    logger.info(
        "Running operator command {id}: {instruction}",
        id=event.payload.get("id"),
        instruction=instruction[:200],
    )
    ctx = Context(agent)
    with chat_trace_attributes("operator-command"):
        # armed=True: operator commands are the one trusted cross-chat
        # channel; the fence in the WhatsApp tools opens for this run
        # alone (the arming flag rides the run's own target binding,
        # so a concurrent chat run can never inherit it).
        reply, _target = await handle_message(
            event, agent, ctx=ctx, settings=settings, waha=waha, armed=True
        )
    reply = (reply or "").strip()
    if reply:
        logger.info(
            "Command {id} final reply: {reply}",
            id=event.payload.get("id"),
            reply=reply[:500],
        )
    return reply


def register_command_handler(
    settings: Settings,
    waha: WahaClient,
    agent: FunctionCallingAgentWorkflow,
) -> None:
    """Register the webhook command handler around the shared agent.

    Commands run without a lock: each binds its own run target and
    fresh ``Context``, so concurrent commands (and commands vs chat
    runs) cannot interfere.
    """

    @on_command
    async def handle_command(event: WahaEvent) -> None:
        """Run one operator command (concurrently with chat runs)."""
        if not session_healthy():
            logger.info(
                "Muting command {id} while WAHA session is not WORKING",
                id=event.payload.get("id"),
            )
            return
        if event.session != settings.session:
            return
        await run_command(event, agent, settings, waha)


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
