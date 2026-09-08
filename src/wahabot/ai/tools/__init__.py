"""Bundled tools for the function calling agent.

``whatsapp`` holds the WhatsApp tools (send/react/image/file/history/
metadata/search/forward/resolve), ``external`` the research and host tools
(web search, page fetch, stock prices, YouTube transcripts, shell),
``schemas`` the Pydantic parameter schemas for all of them, and
``envelope`` the unified JSON envelope (``ok`` / ``error``) every tool
returns.
"""

from llama_index.core.tools import BaseTool

from wahabot.ai.tools.external import (
    shell_builder,
    stock_price_builder,
    visit_url_builder,
    web_search_builder,
    youtube_transcript_builder,
)
from wahabot.ai.tools.whatsapp import (
    EscalationChannel,
    escalate,
    fetch_chat_messages,
    forward_message,
    get_chat,
    react_to_message,
    recent_chats,
    resolve_chat,
    search_messages,
    send_file,
    send_image,
    send_message,
    stay_silent,
)
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings

__all__ = ["build_default_tools"]


def build_default_tools(
    waha: WahaClient,
    target: dict[str, str],
    settings: Settings | None = None,
    escalation_channel: EscalationChannel | None = None,
) -> list[BaseTool]:
    """Build all bundled tools bound to the shared session/chat holder.

    ``escalation_channel`` carries the per-agent operator target and
    cooldowns; a fresh one is created when the caller has none to share
    (tests, one-off agents).
    """
    if settings is None:
        from wahabot.settings import get_settings

        settings = get_settings()
    channel = escalation_channel or EscalationChannel()
    tools = [
        send_message(waha, target),
        stay_silent(),
        escalate(waha, target, channel),
        react_to_message(waha, target),
        send_image(waha, target),
        send_file(waha, target, settings.max_file_bytes),
        fetch_chat_messages(waha, target),
        get_chat(waha, target),
        search_messages(waha, target),
        forward_message(waha, target),
        resolve_chat(waha, target),
        recent_chats(waha, target),
        web_search_builder(settings),
        stock_price_builder(),
        youtube_transcript_builder(),
        visit_url_builder(settings),
    ]
    if settings.shell_tool:
        tools.append(shell_builder(settings))
    return tools
