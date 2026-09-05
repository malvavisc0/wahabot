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
    fetch_chat_messages,
    forward_message,
    get_chat,
    react_to_message,
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
) -> list[BaseTool]:
    """Build all bundled tools bound to the shared session/chat holder."""
    if settings is None:
        from wahabot.settings import get_settings

        settings = get_settings()
    tools = [
        send_message(waha, target),
        stay_silent(),
        react_to_message(waha, target),
        send_image(waha, target),
        send_file(waha, target, settings.max_file_bytes),
        fetch_chat_messages(waha, target),
        get_chat(waha, target),
        search_messages(waha, target),
        forward_message(waha, target),
        resolve_chat(waha, target),
        web_search_builder(settings),
        stock_price_builder(),
        youtube_transcript_builder(),
        visit_url_builder(settings),
    ]
    if settings.shell_tool:
        tools.append(shell_builder(settings))
    return tools
