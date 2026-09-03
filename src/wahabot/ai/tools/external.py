"""External research and host tools for the function calling agent.

Builders for tools that reach outside WhatsApp: web search, page fetch,
stock prices, YouTube transcripts and the (opt-in) shell. Each binds the
shared settings and wraps a plain function from its own module, keeping
the "return a status string, never raise" contract.
"""

from llama_index.core.tools import BaseTool, FunctionTool

from wahabot.ai.finance import fetch_current_stock_price
from wahabot.ai.shell import shell_command
from wahabot.ai.tools.schemas import (
    FetchStockPriceSchema,
    GetYoutubeTranscriptSchema,
    ShellCommandSchema,
    VisitUrlSchema,
    WebSearchSchema,
)
from wahabot.ai.visit_url import visit_url
from wahabot.ai.web_search import web_search
from wahabot.ai.youtube import get_youtube_transcript
from wahabot.settings import Settings

__all__ = [
    "shell_builder",
    "stock_price_builder",
    "visit_url_builder",
    "web_search_builder",
    "youtube_transcript_builder",
]


def web_search_builder(settings: Settings) -> BaseTool:
    """Build the web search tool bound to settings."""

    def web_search_fn(query: str, max_results: int | None = None) -> str:
        """Search the web.

        Args:
            query: The search query text.
            max_results: Optional max results to return; defaults to the
                configured limit.
        """
        return web_search(settings, query, max_results=max_results)

    return FunctionTool.from_defaults(
        fn=web_search_fn,
        fn_schema=WebSearchSchema,
        name="web_search",
        description=(
            "Search the web via the webserp metasearch CLI and return "
            "normalised results (title, url, snippet, engine). Use to "
            "answer questions needing up-to-date or external information."
        ),
    )


def shell_builder(settings: Settings) -> BaseTool:
    """Build the shell execution tool bound to settings."""

    def shell_command_fn(command: str) -> str:
        return shell_command(settings, command)

    return FunctionTool.from_defaults(
        fn=shell_command_fn,
        fn_schema=ShellCommandSchema,
        name="run_shell_command",
        description=(
            "Run a shell command on the host and return its output (bounded by "
            "settings, capped timeout). Use for filesystem/process/system "
            "operations the other tools cannot do. Enable only via WAHABOT_SHELL_TOOL."
        ),
    )


def stock_price_builder() -> BaseTool:
    """Build the current-stock-price tool."""

    def stock_price_fn(ticker: str) -> str:
        return fetch_current_stock_price(ticker)

    return FunctionTool.from_defaults(
        fn=stock_price_fn,
        fn_schema=FetchStockPriceSchema,
        name="fetch_current_stock_price",
        description=(
            "Fetch the current price of a stock, ETF or crypto ticker "
            "(e.g. AAPL, BTC-USD). Use for price and day-change questions."
        ),
    )


def visit_url_builder(settings: Settings) -> BaseTool:
    """Build the website-fetching tool bound to settings."""

    def visit_url_fn(url: str) -> str:
        return visit_url(settings, url)

    return FunctionTool.from_defaults(
        fn=visit_url_fn,
        fn_schema=VisitUrlSchema,
        name="visit_url",
        description=(
            "Fetch a web page and return its visible text. Uses a real "
            "Chrome browser TLS fingerprint (curl_cffi) to avoid being "
            "blocked. Use to read the content of a specific URL."
        ),
    )


def youtube_transcript_builder() -> BaseTool:
    """Build the YouTube transcript tool."""

    def youtube_transcript_fn(url: str) -> str:
        return get_youtube_transcript(url)

    return FunctionTool.from_defaults(
        fn=youtube_transcript_fn,
        fn_schema=GetYoutubeTranscriptSchema,
        name="get_youtube_transcript",
        description=(
            "Fetch a YouTube video's captions/transcript as text. Use to "
            "extract the spoken content of a video for summarization. "
            "Only works when captions are available."
        ),
    )
