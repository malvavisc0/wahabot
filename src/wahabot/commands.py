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

import asyncio
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
) -> None:
    """Run the agent over the command instruction, on a fresh context.

    No dedup (the command id is unique by construction), no staleness
    gate (no WAHA redelivery for a command the operator just fired), no
    chat gates. The shared send-target holder is pointed at the
    event's ``from`` ("operator") so the run behaves like a DM: the
    model may pass ``chat=…`` explicitly (a group or a person resolved
    via ``resolve_chat``) or omit it, exactly as in a normal chat —
    and a delivery latch left over from a previous turn can never
    block this command's send.
    """
    instruction = str(event.payload.get("body", "")).strip()
    if not instruction:
        logger.debug("Ignoring empty command {id}", id=event.payload.get("id"))
        return
    logger.info(
        "Running operator command {id}: {instruction}",
        id=event.payload.get("id"),
        instruction=instruction[:200],
    )
    holder = agent.send_holder
    if holder is not None:
        holder["session"] = event.session
        holder["chat_id"] = str(event.payload.get("from", ""))
        holder["sent"] = ""
        holder["reacted"] = ""
    ctx = Context(agent)
    with chat_trace_attributes("operator-command"):
        reply = await handle_message(event, agent, ctx=ctx, settings=settings, waha=waha)
    if reply and reply.strip():
        # A command's text reply has no chat to land in — the log (and
        # the Langfuse trace) is the only place the operator can read it.
        logger.info(
            "Command {id} final reply: {reply}",
            id=event.payload.get("id"),
            reply=reply[:500],
        )


def register_command_handler(
    settings: Settings,
    waha: WahaClient,
    agent: FunctionCallingAgentWorkflow,
    agent_lock: asyncio.Lock,
) -> None:
    """Register the webhook command handler around the shared agent."""

    @on_command
    async def handle_command(event: WahaEvent) -> None:
        """Run one operator command under the shared agent lock."""
        if not session_healthy():
            logger.info(
                "Muting command {id} while WAHA session is not WORKING",
                id=event.payload.get("id"),
            )
            return
        if event.session != settings.session:
            return
        async with agent_lock:
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
