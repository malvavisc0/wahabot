"""Webhook registration and per-process handler wiring."""

import time

from loguru import logger

from wahabot.ai import bursts
from wahabot.ai.albums import set_completion_handler
from wahabot.ai.intake import IntakeLedger, max_age_for_run_timeout
from wahabot.ai.observability import enable_langfuse
from wahabot.ai.tools import build_default_tools
from wahabot.ai.tools.whatsapp import EscalationChannel
from wahabot.ai.triage import TriageGate
from wahabot.ai.workflow import FunctionCallingAgentWorkflow, build_agent
from wahabot.core import identity
from wahabot.core.access import SessionConfigReloader, load_session_config
from wahabot.core.learned import forget_learned
from wahabot.core.models import WahaEvent
from wahabot.core.notebook import Notebook, notebook_operation
from wahabot.core.persistence import forget_memory
from wahabot.core.runs import chat_lock, contexts
from wahabot.core.thread_index import forget_thread_index
from wahabot.core.waha import WahaClient
from wahabot.handlers.core import Handlers
from wahabot.settings import Settings
from wahabot.webhook import on_forget, on_message, on_shutdown


def register_forget_handler(
    settings: Settings,
    agent: FunctionCallingAgentWorkflow | None = None,
) -> None:
    """Register the ``forget`` webhook handler that wipes a chat's memory.

    The ``wahabot forget`` CLI posts a signed ``forget`` event; the wipe
    runs here, in the bot process, under the chat's run lock so it
    serializes with that chat's runs: an in-flight run finishes and
    saves first, then the live context and the memory file are dropped
    along with the chat's notebook, recap and thread index, with no
    resurrection window. When *agent* is supplied, its in-memory index
    queue is cleared too.
    """
    notebook = Notebook(settings.data_dir)

    @on_forget
    async def handle_forget(event: WahaEvent) -> None:
        chat_id = str(event.payload.get("chat_id", ""))
        if not chat_id:
            logger.warning("Ignoring forget event without a chat_id")
            return
        async with chat_lock(event.session, chat_id):
            live = contexts.pop((event.session, chat_id), None) is not None
            removed = forget_memory(settings.data_dir, event.session, chat_id)
            forgot_learned = forget_learned(settings.data_dir, event.session, chat_id)
            forgot_index = forget_thread_index(settings.data_dir, event.session, chat_id)
            if agent is not None and agent.thread_index is not None:
                agent.thread_index.forget(event.session, chat_id)
            await notebook_operation(notebook.forget, event.session, chat_id)
            logger.info(
                "Forgot {chat_id}: context {live}, file {removed}, "
                + "recap {recap}, index {index}",
                chat_id=chat_id,
                live="present" if live else "absent",
                removed="deleted" if removed else "absent",
                recap="deleted" if forgot_learned else "absent",
                index="deleted" if forgot_index else "absent",
            )


def register_agent_handler(
    settings: Settings, waha: WahaClient
) -> tuple[FunctionCallingAgentWorkflow, SessionConfigReloader]:
    """Build the agent and register the reply handler.

    Returns the built agent and the config reloader so other handlers
    (the command channel) share the exact same agent and hot-reloaded
    config instead of building their own.
    """
    config = load_session_config(settings.access_config)
    config_reloader = SessionConfigReloader(settings.access_config)
    identity.configure(settings.data_dir, settings.session)
    enable_langfuse(settings)
    if settings.burst_enabled:
        bursts.configure(settings.burst_inactivity_s, settings.burst_hold_cap_s)
    else:
        bursts.set_completion_handler(None)

    escalation_channel = EscalationChannel()
    triage = (
        TriageGate(settings, waha, channel=escalation_channel)
        if settings.triage_gate and settings.gate_api_base
        else None
    )
    if triage is not None:
        on_shutdown(triage.close)

    handlers = Handlers(
        settings,
        waha,
        config,
        config_reloader,
        escalation_channel,
        triage,
        IntakeLedger(max_age_s=max_age_for_run_timeout(settings.run_timeout)),
        time.time(),
    )
    agent = build_agent(
        settings,
        tools=build_default_tools(waha, settings, escalation_channel=escalation_channel),
        system_prompt=config.system_prompt,
        prompt_renderer=handlers.render_prompt,
        rules_renderer=handlers.render_session_rules,
        gate_rules_renderer=handlers.render_gate_brief,
        triage=triage,
    )
    handlers.agent = agent
    set_completion_handler(handlers.run_album)
    if settings.burst_enabled:
        bursts.set_completion_handler(handlers.run_burst)
    on_message(handlers.reply_with_agent)
    logger.info(
        "Agent ready: {tools}",
        tools=", ".join(sorted(tool.metadata.get_name() for tool in agent.tools)),
    )
    return agent, config_reloader
