"""Bundled tools for the function calling agent.

``whatsapp`` holds the WhatsApp tools (send/text/react/media/read/
forward/escalate), ``external`` the research and host tools (web
search, page fetch, shell), ``schemas`` the Pydantic parameter schemas
for all of them, and ``envelope`` the unified JSON envelope
(``ok`` / ``error``) every tool returns. The 20 tools were merged into
10: the five media senders became
``send_media`` (``kind``), the five chat readers ``read_chat``
(``mode``), and the two niche research tools were dropped.
"""

from llama_index.core.tools import BaseTool

from wahabot.ai.tools.external import (
    shell_builder,
    visit_url_builder,
    web_search_builder,
)
from wahabot.ai.tools.image import image_builder
from wahabot.ai.tools.notebook import notebook_builder
from wahabot.ai.tools.tickets import OperatorLanguage, build_ticket_tools
from wahabot.ai.tools.whatsapp import (
    EscalationChannel,
    escalate,
    forward_message,
    profile_picture_builder,
    react_to_message,
    read_chat,
    send_media,
    send_message,
    stay_silent,
)
from wahabot.core.tickets.alerts import Translator
from wahabot.core.tickets.store import TicketStore
from wahabot.core.transport import Transport
from wahabot.settings import Settings

__all__ = ["build_default_tools"]


def build_default_tools(
    waha: Transport,
    settings: Settings | None = None,
    escalation_channel: EscalationChannel | None = None,
    *,
    ticket_store: TicketStore | None = None,
    ticket_translator: Translator | None = None,
    ticket_operator_language: OperatorLanguage = "",
) -> list[BaseTool]:
    """Build all bundled tools.

    Tools resolve the current run's session/chat target and delivery
    latches through the run-scoped binding (``bind_target``), so one
    toolset safely serves concurrent runs across different chats.
    ``escalation_channel`` carries in-memory cooldowns and reads the
    current operator target from live status state; a fresh channel is
    created when the caller has none to share (tests, one-off agents).
    """
    if settings is None:
        from wahabot.settings import get_settings

        settings = get_settings()
    channel = escalation_channel or EscalationChannel()
    tools = [
        send_message(waha, settings),
        stay_silent(),
        escalate(waha, channel, settings),
        react_to_message(waha, settings),
        send_media(waha, settings),
        read_chat(waha, settings),
        notebook_builder(settings),
        forward_message(waha),
        web_search_builder(settings),
        visit_url_builder(settings),
    ]
    if settings.shell_tool:
        tools.append(shell_builder(settings))
    tools.extend(
        build_ticket_tools(
            ticket_store,
            settings,
            waha,
            operator_language=ticket_operator_language,
            translator=ticket_translator,
        )
    )
    tools.append(profile_picture_builder(waha))
    if settings.image_api_base:
        tools.extend(image_builder(settings))
    return tools
