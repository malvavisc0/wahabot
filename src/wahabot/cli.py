"""CLI entrypoints and commands for wahabot."""

import hashlib
import hmac
import json
import platform
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from typing import Any

import httpx
import typer
import uvicorn
from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from wahabot.ai.context import render_system_prompt
from wahabot.commands import build_command_event, register_command_handler
from wahabot.core.access import load_session_config
from wahabot.core.waha import WahaClient
from wahabot.handlers import agent_lock, append_to_memory, register_agent_handler
from wahabot.reactions import register_reaction_handler
from wahabot.settings import Settings, get_settings, setup_logging
from wahabot.status import register_session_status_handler, seed_health

app = typer.Typer(
    name="wahabot",
    help="A WhatsApp bot that answers chats with an LLM agent.",
    no_args_is_help=True,
    add_completion=True,
)
sessions_app = typer.Typer(
    name="sessions",
    help="Manage per-session config files.",
    no_args_is_help=True,
)
app.add_typer(sessions_app, name="sessions")

console = Console()


@app.command()
def version() -> None:
    """Show the wahabot version."""
    console.print(get_version("wahabot"))


def _redact_key(key: str) -> bool:
    """True when a settings key holds a secret and must be redacted."""
    lowered = key.casefold()
    return any(part in lowered for part in ("key", "secret", "hmac"))


def _feature_flags(settings: Settings) -> list[str]:
    """Human-readable feature toggles derived from the loaded settings.

    Enabled features carry their key parameter; disabled ones are named
    ``no-<feature>`` so the banner answers "is X on?" at one glance.
    """
    flags = [
        "vision" if settings.vision else "no-vision",
        "shell" if settings.shell_tool else "no-shell",
        (
            f"transcribe({settings.transcribe_language})"
            if settings.transcribe_url
            else "no-transcribe"
        ),
        "langfuse"
        if settings.langfuse_public_key and settings.langfuse_secret_key
        else "no-langfuse",
    ]
    return flags


def _version_line() -> str:
    """A short line for the startup banner with the wahabot version."""
    try:
        version = get_version("wahabot")
    except PackageNotFoundError:
        version = "unknown"
    return f"wahabot {version} (Python {platform.python_version()})"


def _log_startup_banner(
    settings: Settings, host: str | None = None, port: int | None = None
) -> None:
    """Log the running configuration in one readable banner at startup."""
    logger.info("──────────────── wahabot startup ────────────────")
    logger.info("{line}", line=_version_line())
    logger.info("Session: {session}", session=settings.session)
    logger.info(
        "LLM: {model} @ {base}", model=settings.llm_model, base=settings.llm_api_base
    )
    logger.info("Memory: {tokens} token ceiling", tokens=settings.memory_token_limit)
    logger.info(
        "Features: {features}",
        features=", ".join(_feature_flags(settings)),
    )
    if settings.transcribe_url:
        logger.info("Transcription: {url}", url=settings.transcribe_url)
    logger.info(
        "Webhook: http://{host}:{port}/api/webhook/{session}",
        host=host or settings.host,
        port=port or settings.port,
        session=settings.session,
    )
    logger.info("──────────────────────────────────────────────")


def _kv_table(rows: Sequence[tuple[str, Text | str]]) -> Table:
    """A borderless key→value table — the CLI's one output style."""
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(style="bold cyan", justify="right")
    table.add_column(overflow="fold")
    for key, value in rows:
        table.add_row(key, value)
    return table


def _dash(value: str | None) -> Text:
    """A config value as escaped rich text, or a dim dash when unset."""
    if not value:
        return Text("-", style="dim")
    return Text(value)


def _format_list(values: set[str]) -> Text:
    """A config list as a comma-separated line, or a dim dash when empty."""
    return Text(", ".join(sorted(values))) if values else _dash(None)


def _ok(message: str) -> None:
    """A success line, styled like every other CLI confirmation."""
    console.print(f"[green]✓[/] {message}")


@app.command()
def config() -> None:
    """Show the loaded configuration (secrets redacted)."""
    settings = get_settings()
    rows: list[tuple[str, Text | str]] = []
    for key, value in settings.model_dump().items():
        shown = Text("***", style="dim") if _redact_key(key) else _dash(str(value))
        rows.append((key, shown))
    console.print(_kv_table(rows))


#: Template written by ``sessions init``; edit the placeholders after generating.
_SESSION_TEMPLATE: dict[str, object] = {
    "whitelist": [],
    "blacklist": [],
    "goal": "Be a helpful, concise, and friendly assistant in WhatsApp chats.",
    "system_prompt": (
        "You are {{bot_name}}, texting on WhatsApp. "
        "Today is {{date}}. Current time {{time}} ({{tz}}).\n"
        "\n"
        "Each message arrives as `[Sender] text`, plus bracketed "
        "metadata — never repeat it in replies. "
        '`[reaction X from Sender to your message: "..."]` means '
        "someone reacted to something you said: context only, never "
        "answer it directly.\n"
        "\n"
        "## Operator commands\n"
        "\n"
        "Turns starting with `[operator command]` come from the bot's "
        "operator, not from a chat participant. Rules change:\n"
        "\n"
        "- Do what the instruction says — that IS the task; the usual "
        "stay_silent restraint does not apply.\n"
        "- The run has no chat of its own: deliver results with "
        "`send_message(chat=…)` to the target the instruction names.\n"
        "- Resolve people/group names to JIDs with `resolve_chat`; if it "
        "returns several candidates, pick the closest and say which you "
        "picked.\n"
        "\n"
        "You can reach chats beyond the current one: resolve a person or "
        "group name with `resolve_chat`, then pass the JID as `chat` to "
        "`send_message`, `send_image`, `send_file` or `forward_message`. "
        "Documents (PDFs etc.) go via `send_file` — from a URL, or a "
        "local file you created."
    ),
    "bot_name": None,
    "bot_mention_regex": None,
    "group_participation": "mentioned",
}


@sessions_app.command("init")
def session_init(
    name: str = typer.Option(
        None, "--name", "-n", help="Session name. [default: WAHABOT_SESSION]"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite an existing config file."
    ),
) -> None:
    """Write a starter session config to <data>/sessions/<name>.json."""
    settings = get_settings()
    if name:
        settings.session = name
    path = settings.access_config
    if path.exists() and not force:
        raise typer.BadParameter(f"{path} already exists; pass --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_SESSION_TEMPLATE, indent=2) + "\n")
    _ok(f"Wrote [bold]{path}[/] — edit system_prompt (and goal/bot_name) before serving.")


@sessions_app.command("list")
def session_list() -> None:
    """List session config files found in <data>/sessions/."""
    settings = get_settings()
    sessions_dir = settings.access_config.parent
    configs = sorted(sessions_dir.glob("*.json")) if sessions_dir.is_dir() else []
    if not configs:
        console.print(f"[dim]No session configs in {sessions_dir}[/]")
        return
    rows = [(path.stem, _dash(load_session_config(path).bot_name)) for path in configs]
    console.print(_kv_table(rows))


@sessions_app.command("view")
def session_view(
    name: str = typer.Option(
        None, "--name", "-n", help="Session name. [default: WAHABOT_SESSION]"
    ),
    raw: bool = typer.Option(
        False, "--raw", help="Show the system prompt without variable expansion."
    ),
    plain: bool = typer.Option(
        False, "--plain", help="Print the prompt as plain text, not markdown."
    ),
) -> None:
    """Show a session's config, with the system prompt rendered."""
    settings = get_settings()
    if name:
        settings.session = name
    if not settings.access_config.exists():
        raise typer.BadParameter(f"Session config not found: {settings.access_config}")
    config = load_session_config(settings.access_config)
    prompt = config.system_prompt
    if not raw:
        prompt = render_system_prompt(
            prompt, settings.timezone, config.bot_name, config.goal
        )
    console.print(
        _kv_table(
            [
                ("session", settings.session),
                ("group_participation", config.group_participation),
                ("bot_name", _dash(config.bot_name)),
                ("bot_mention_regex", _dash(config.bot_mention_regex)),
                ("whitelist", _format_list(config.whitelist)),
                ("blacklist", _format_list(config.blacklist)),
                ("goal", _dash(config.goal)),
            ]
        )
    )
    title = "system_prompt" if raw else "system_prompt (rendered)"
    body: Any = prompt if plain else Markdown(prompt)
    console.print(Panel(body, title=title, expand=False))


def ensure_session_live(waha: WahaClient, settings: Settings) -> None:
    """Abort startup when the WAHA session is not reachable."""
    try:
        me = waha.get_me(settings.session)
    except httpx.HTTPError as exc:
        message = (
            f"WAHA session '{settings.session}' is not reachable at"
            f" {settings.waha_url}: {exc}"
        )
        raise typer.BadParameter(message) from exc
    logger.info(
        "WAHA session {session} is live as {push_name} ({id})",
        session=settings.session,
        push_name=me.get("pushName") or "?",
        id=me.get("id") or "?",
    )


@app.command()
def serve(
    host: str = typer.Option(
        None, "--host", "-h", help="Bind address. [default: WAHABOT_HOST]"
    ),
    port: int = typer.Option(
        None, "--port", "-p", help="Bind port. [default: WAHABOT_PORT]"
    ),
    session: str = typer.Option(
        None, "--session", "-s", help="WAHA session name. [default: WAHABOT_SESSION]"
    ),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload."),
) -> None:
    """Run the webhook server: WAHA events in, agent replies out."""
    settings = get_settings()
    if session:
        settings.session = session
    setup_logging(settings)
    host = host or settings.host
    port = port or settings.port
    _log_startup_banner(settings, host, port)
    if not settings.access_config.exists():
        message = (
            f"Session config not found: {settings.access_config}"
            " (create it with `wahabot sessions init`)"
        )
        raise typer.BadParameter(message)
    waha = WahaClient(base_url=settings.waha_url, api_key=settings.waha_api_key)
    ensure_session_live(waha, settings)
    seed_health(waha, settings.session)
    agent, _config_reloader = register_agent_handler(settings, waha=waha)
    register_reaction_handler(waha=waha, remember=append_to_memory)
    register_command_handler(
        settings,
        waha,
        agent,
        agent_lock,
    )
    register_session_status_handler(waha, settings.session)
    (settings.journal_dir / settings.session).mkdir(parents=True, exist_ok=True)
    uvicorn.run(
        "wahabot.webhook:app",
        host=host,
        port=port,
        reload=reload,
        log_config=None,
        http="wahabot.core.protocol:LoggingH11Protocol",
    )


@app.command()
def tell(
    text: str = typer.Argument(
        help='Instruction for the agent, e.g. "send a summary to the group Familia".'
    ),
    host: str = typer.Option(
        None, "--host", "-h", help="Webhook host. [default: WAHABOT_HOST]"
    ),
    port: int = typer.Option(
        None, "--port", "-p", help="Webhook port. [default: WAHABOT_PORT]"
    ),
    session: str = typer.Option(
        None, "--session", "-s", help="WAHA session name. [default: WAHABOT_SESSION]"
    ),
) -> None:
    """Send an operator command to the running agent.

    The agent runs the instruction with its full toolset on a fresh
    context — not a chat turn, so whitelist and group gating do not
    apply and no chat's memory is touched. Results are delivered to
    whatever chats the instruction names (people and groups by name,
    resolved via the bot's contact list). Fire-and-forget: the result
    lands in WhatsApp, not in this terminal.
    """
    settings = get_settings()
    if session:
        settings.session = session
    event = build_command_event(settings.session, text)
    body = json.dumps(event).encode()
    signature = hmac.new(
        settings.webhook_hmac_key.encode(), body, hashlib.sha512
    ).hexdigest()
    url = (
        f"http://{host or settings.host}:{port or settings.port}"
        f"/api/webhook/{settings.session}"
    )
    try:
        response = httpx.post(
            url,
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Hmac": signature,
            },
            timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise typer.BadParameter(f"Webhook at {url} rejected the command: {exc}") from exc
    _ok(f"Command delivered to session [bold]{settings.session}[/]")


def main() -> None:
    """Run the wahabot CLI."""
    setup_logging(get_settings())
    app()


if __name__ == "__main__":
    main()
