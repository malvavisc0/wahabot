"""Bundled tools for the function calling agent.

``whatsapp`` holds the WhatsApp tools (send/text/react/media/read/
forward/escalate), ``external`` the research and host tools (web
search, page fetch, shell), ``schemas`` the Pydantic parameter schemas
for all of them, and ``envelope`` the unified JSON envelope
(``ok`` / ``error``) every tool returns. The 20 tools were merged into
10 (docs/bug-report-2c665d8.md, bug 7b): the five media senders became
``send_media`` (``kind``), the five chat readers ``read_chat``
(``mode``), and the two niche research tools were dropped.
"""

from llama_index.core.tools import BaseTool

from wahabot.ai.tools.external import (
    shell_builder,
    visit_url_builder,
    web_search_builder,
)
from wahabot.ai.tools.whatsapp import (
    EscalationChannel,
    escalate,
    forward_message,
    react_to_message,
    read_chat,
    send_media,
    send_message,
    stay_silent,
)
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings

__all__ = ["build_default_tools"]


def build_default_tools(
    waha: WahaClient,
    settings: Settings | None = None,
    escalation_channel: EscalationChannel | None = None,
) -> list[BaseTool]:
    """Build all bundled tools.

    Tools resolve the current run's session/chat target and delivery
    latches through the run-scoped binding (``bind_target``), so one
    toolset safely serves concurrent runs across different chats.
    ``escalation_channel`` carries the per-agent operator target and
    cooldowns; a fresh one is created when the caller has none to share
    (tests, one-off agents).
    """
    if settings is None:
        from wahabot.settings import get_settings

        settings = get_settings()
    channel = escalation_channel or EscalationChannel()
    tools = [
        send_message(waha),
        stay_silent(),
        escalate(waha, channel, settings),
        react_to_message(waha),
        send_media(waha, settings),
        read_chat(waha),
        forward_message(waha),
        web_search_builder(settings),
        visit_url_builder(settings),
    ]
    if settings.shell_tool:
        tools.append(shell_builder(settings))
    return tools
