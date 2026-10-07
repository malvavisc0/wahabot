"""External research and host tools for the function calling agent.

Builders for tools that reach outside WhatsApp: web search, page fetch
and the (opt-in) shell. Each binds settings and returns JSON-envelope
feedback. The workflow also wraps unexpected exceptions. Captions and
raw output files are best-effort, not guarantees of complete content.
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
    """Build bounded search with retrieval guidance matching shell availability."""
    deref = (
        "Read file.path via run_shell_command if needed."
        if settings.shell_tool
        else "Only previews are accessible to you; the operator can open file.path."
    )

    def web_search_fn(
        query: str, max_results: int | None = None, reason: str = ""
    ) -> str:
        return web_search(settings, query, max_results=max_results)

    return FunctionTool.from_defaults(
        fn=web_search_fn,
        fn_schema=WebSearchSchema,
        name="web_search",
        description=(
            "Search external information. Returns count and results with "
            "title, url, optional content snippet and engine. max_results "
            "caps total valid findings across engines. partial/failed_engines "
            "identify incomplete search; empty hits with failed engines is an error. "
            "content_truncated "
            "marks a cut snippet; full selected results attempt a file spill. "
            f"{deref} Snippets are leads, not verified page contents; "
            "open relevant hits with visit_url."
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
            f"Returns exit_code, stdout, stderr (decoded, trimmed previews capped "
            f"at {max(settings.shell_max_output, 200)} bytes per stream; "
            f"wall-time limit {max(settings.shell_timeout, 1):g}s plus bounded cleanup — "
            "keep commands quick). Check exit_code, not ok=true, for command success; "
            "timeout/start failures use ok=false without an exit_code. "
            "When a stream was cut, captured bytes attempt a "
            "spill file — `file.path` for stdout, `stderr_file.path` for "
            "stderr — read them in bounded parts. Missing file metadata "
            "means no full capture was provided; capture_errors marks lost output. "
            "No stdin or sandbox."
        ),
    )


def visit_url_builder(settings: Settings) -> BaseTool:
    """Build page/metadata/caption reads, not video viewing or browser automation."""
    deref = (
        "Read provided file.path/transcript_file.path with run_shell_command "
        + "in bounded parts."
        if settings.shell_tool
        else "Only previews are accessible to you; the operator can open provided files."
    )

    def visit_url_fn(url: str, reason: str = "") -> str:
        return visit_url(settings, url)

    return FunctionTool.from_defaults(
        fn=visit_url_fn,
        fn_schema=VisitUrlSchema,
        name="visit_url",
        description=(
            "Fetch HTTP(S) response text without browser rendering or available "
            "media-host metadata via yt-dlp. kind=post includes an item_count "
            "bounded to 100 inspected entries with item_count_truncated; video "
            "description previews flag description_truncated after 800 characters. "
            "ok=true means retrieval, not that "
            "a login wall is the requested content. Text previews cap at 4000 "
            "characters; cut bodies attempt file spills. YouTube transcript "
            "is best-effort fetched captions, possibly auto-generated or in "
            "another language, not verified speech. transcript_truncated "
            "marks a 6000-character preview with an optional transcript_file. "
            f"{deref} Missing captions are not proof none exist. "
            "Previews are not download-size caps. Metadata/captions do not "
            "establish what the video visually shows."
        ),
    )
