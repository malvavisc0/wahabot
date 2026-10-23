"""Webhook registration and per-process handler wiring."""

import time
from dataclasses import dataclass

from loguru import logger

from wahabot.ai.intake import (
    IntakeLedger,
    claim_lease_for_run_timeout,
    max_age_for_run_timeout,
)
from wahabot.ai.observability import enable_langfuse
from wahabot.ai.tools import build_default_tools
from wahabot.ai.tools.whatsapp import EscalationChannel
from wahabot.ai.triage import TriageGate
from wahabot.ai.workflow import FunctionCallingAgentWorkflow, build_agent
from wahabot.core import identity
from wahabot.core.access import SessionConfigReloader, load_session_config
from wahabot.core.db import Database
from wahabot.core.delivery.intents import IntentStore
from wahabot.core.intake.sqlite import SqliteIntakeStore
from wahabot.core.learned import forget_learned
from wahabot.core.models import WahaEvent
from wahabot.core.notebook import Notebook, notebook_operation
from wahabot.core.persistence import forget_memory
from wahabot.core.recovery import Recovery
from wahabot.core.runs import chat_lock, contexts
from wahabot.core.scheduler import TicketScheduler
from wahabot.core.thread_index import forget_thread_index
from wahabot.core.tickets.store import TicketStore
from wahabot.core.tickets.translate import TicketTranslator
from wahabot.core.transport import WahaTransport
from wahabot.core.waha import WahaClient
from wahabot.handlers.core import Handlers
from wahabot.settings import Settings
from wahabot.webhook import app as webhook_app
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


@dataclass(frozen=True)
class Registration:
    """The objects one agent registration owns."""

    agent: FunctionCallingAgentWorkflow
    config_reloader: SessionConfigReloader
    transport: WahaTransport
    ledger: IntakeLedger
    intents: IntentStore
    database: Database
    handlers: Handlers
    recovery: Recovery


def webhook_app_state(recovery: Recovery) -> None:
    """Make the recovery pass visible to the app's lifespan and ticker."""
    webhook_app.state.recovery = recovery


def register_agent_handler(settings: Settings, waha: WahaClient) -> Registration:
    """Build the durable platform, agent and reply handler."""
    database = Database(settings.data_dir / "wahabot.db")
    database.migrate()
    intents = IntentStore(database.path, max_attempts=settings.intake_max_attempts)
    ledger = IntakeLedger(
        store=SqliteIntakeStore(database.path),
        max_age_s=max_age_for_run_timeout(settings.run_timeout),
        claim_lease_s=claim_lease_for_run_timeout(settings.run_timeout),
    )
    transport = WahaTransport(waha, intents)
    config = load_session_config(settings.access_config)
    config_reloader = SessionConfigReloader(settings.access_config)
    identity.configure(settings.data_dir, settings.session)
    enable_langfuse(settings)
    escalation_channel = EscalationChannel()
    ticket_store = TicketStore(database.path)
    ticket_translator = TicketTranslator()
    scheduler = TicketScheduler(settings, ticket_store, transport, config_reloader)
    triage = (
        TriageGate(settings, transport, channel=escalation_channel)
        if settings.triage_gate and settings.gate_api_base
        else None
    )
    if triage is not None:
        on_shutdown(triage.close)

    handlers = Handlers(
        settings,
        transport,
        config,
        config_reloader,
        escalation_channel,
        triage,
        ledger,
        time.time(),
        intents,
        ticket_store,
        ticket_translator,
        scheduler,
    )
    agent = build_agent(
        settings,
        tools=build_default_tools(
            transport,
            settings,
            escalation_channel=escalation_channel,
            ticket_store=ticket_store,
            ticket_translator=ticket_translator,
            ticket_operator_language=config.operator_language,
        ),
        system_prompt=config.system_prompt,
        prompt_renderer=handlers.render_prompt,
        rules_renderer=handlers.render_session_rules,
        gate_rules_renderer=handlers.render_gate_brief,
        triage=triage,
        ticket_store=ticket_store,
    )
    handlers.agent = agent
    ticket_translator.bind(lambda prompt: str(agent.llm.complete(prompt).text))
    webhook_app.state.ticket_scheduler = scheduler
    webhook_app.state.intake_writer = handlers.record_webhook_intake
    on_message(handlers.reply_with_agent)
    logger.info(
        "Agent ready: {tools}",
        tools=", ".join(sorted(tool.metadata.get_name() for tool in agent.tools)),
    )
    recovery = Recovery(settings, ledger, transport, handlers)
    webhook_app_state(recovery)
    return Registration(
        agent=agent,
        config_reloader=config_reloader,
        transport=transport,
        ledger=ledger,
        intents=intents,
        database=database,
        handlers=handlers,
        recovery=recovery,
    )
