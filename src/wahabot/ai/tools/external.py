"""External research and host tools for the function calling agent.

Builders for tools that reach outside WhatsApp: web search, page fetch
and the (opt-in) shell. Each binds the shared settings and wraps a plain
function from its own module, keeping the "return the JSON envelope,
never raise" contract. The 7b merge (docs/bug-report-2c665d8.md) also
*dropped* two niche tools: stock prices now come through web_search
results like any other fact, and YouTube transcripts are inlined by
``visit_url`` via yt-dlp's caption tracks. Neither niche warranted its
own schema and tokens.
"""

from llama_index.core.tools import BaseTool, FunctionTool

from wahabot.ai.tools.schemas import (
    ShellCommandSchema,
    VisitUrlSchema,
    WebSearchSchema,
)
from wahabot.ai.tools.shell import shell_command
from wahabot.ai.tools.visit_url import visit_url
from wahabot.ai.tools.web_search import web_search
from wahabot.settings import Settings

__all__ = [
    "shell_builder",
    "visit_url_builder",
    "web_search_builder",
]


def web_search_builder(settings: Settings) -> BaseTool:
    """Build the web search tool bound to settings."""

    def web_search_fn(
        query: str, max_results: int | None = None, reason: str = ""
    ) -> str:
        return web_search(settings, query, max_results=max_results)

    return FunctionTool.from_defaults(
        fn=web_search_fn,
        fn_schema=WebSearchSchema,
        name="web_search",
        description=(
            "Search the web for up-to-date or external information. "
            "Returns `results`: title, url, snippet per hit. Read a "
            "promising hit with visit_url."
        ),
    )


def shell_builder(settings: Settings) -> BaseTool:
    """Build the shell execution tool bound to settings."""

    def shell_command_fn(command: str, reason: str = "") -> str:
        return shell_command(settings, command)

    return FunctionTool.from_defaults(
        fn=shell_command_fn,
        fn_schema=ShellCommandSchema,
        name="run_shell_command",
        description=(
            "Run a bash command on the host — for what the other tools "
            "cannot do: filesystem, processes, system state, running "
            "utilities. The system prompt's Host block lists the "
            "available binaries; use those, do not guess others. "
            "Returns exit_code, stdout, stderr (truncated "
            f"past {settings.shell_max_output} chars; killed after "
            f"{int(settings.shell_timeout)}s — keep commands quick and "
            "quiet)."
        ),
    )


def visit_url_builder(settings: Settings) -> BaseTool:
    """Build the website-fetching tool bound to settings."""

    def visit_url_fn(url: str, reason: str = "") -> str:
        return visit_url(settings, url)

    return FunctionTool.from_defaults(
        fn=visit_url_fn,
        fn_schema=VisitUrlSchema,
        name="visit_url",
        description=(
            "Read a web page's visible text. For Instagram/Facebook/"
            "TikTok/YouTube and similar video links you get the video's "
            "real metadata (title, description, uploader, duration, "
            "views). A YouTube link with captions also carries its "
            "`transcript` — the spoken content, so you can summarize or "
            "answer about the video itself; `transcript_truncated` true "
            "means only the first part fit. Say you have the video's "
            "info or captions, never that you watched the video."
        ),
    )
