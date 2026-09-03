"""CLI entrypoints and commands for wahabot."""

import json
from importlib.metadata import version as get_version

import httpx
import typer
import uvicorn
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from wahabot.ai.context import render_system_prompt
from wahabot.core.access import load_session_config
from wahabot.core.waha import WahaClient
from wahabot.handlers import register_agent_handler
from wahabot.reactions import register_reaction_handler
from wahabot.settings import Settings, get_settings, setup_logging

app = typer.Typer(
    name="wahabot",
    help="wahabot command-line interface.",
    no_args_is_help=True,
    add_completion=True,
)
sessions_app = typer.Typer(
    name="sessions",
    help="Manage per-session config files.",
    no_args_is_help=True,
)
app.add_typer(sessions_app, name="sessions")


@app.command()
def version() -> None:
    """Show the wahabot version."""
    typer.echo(get_version("wahabot"))


@app.command()
def config() -> None:
    """Show the loaded configuration (secrets redacted)."""
    settings = get_settings()
    for key, value in settings.model_dump().items():
        shown = "***" if "key" in key else value
        typer.echo(f"{key}={shown}")


#: Template written by ``init-session``; edit the placeholders after generating.
_SESSION_TEMPLATE: dict[str, object] = {
    "whitelist": [],
    "blacklist": [],
    "goal": "",
    "system_prompt": (
        "You are {{bot_name}}, texting on WhatsApp. "
        "Today is {{date}}. Current time {{time}} ({{tz}})."
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
    typer.echo(f"Wrote {path} — edit system_prompt (and goal/bot_name) before serving.")


@sessions_app.command("list")
def session_list() -> None:
    """List session config files found in <data>/sessions/."""
    settings = get_settings()
    sessions_dir = settings.access_config.parent
    configs = sorted(sessions_dir.glob("*.json")) if sessions_dir.is_dir() else []
    if not configs:
        typer.echo(f"No session configs in {sessions_dir}")
        return
    for path in configs:
        typer.echo(path.stem)


@sessions_app.command("view")
def session_view(
    name: str = typer.Option(
        None, "--name", "-n", help="Session name. [default: WAHABOT_SESSION]"
    ),
    raw: bool = typer.Option(
        False, "--raw", help="Show the system prompt without variable expansion."
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
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(style="bold cyan", justify="right")
    table.add_column(overflow="fold")
    table.add_row("session", settings.session)
    table.add_row("group_participation", config.group_participation)
    table.add_row("bot_name", _dash(config.bot_name))
    table.add_row("bot_mention_regex", _dash(config.bot_mention_regex))
    table.add_row("whitelist", _format_list(config.whitelist))
    table.add_row("blacklist", _format_list(config.blacklist))
    table.add_row("goal", _dash(config.goal))
    Console().print(table)
    title = "system_prompt" if raw else "system_prompt (rendered)"
    Console().print(Panel(prompt, title=title, expand=False))


def _format_list(values: set[str]) -> Text:
    """Render a config list as a comma-separated line, or a dash when empty."""
    return Text(", ".join(sorted(values))) if values else _dash(None)


def _dash(value: str | None) -> Text:
    """A config value as escaped rich text, or a dim dash when unset."""
    if not value:
        return Text("-", style="dim")
    return Text(value)


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
    """Run the webhook server for WAHA events."""
    settings = get_settings()
    if session:
        settings.session = session
    setup_logging(settings)
    waha = WahaClient(base_url=settings.waha_url, api_key=settings.waha_api_key)
    ensure_session_live(waha, settings)
    register_agent_handler(settings, waha=waha)
    register_reaction_handler(waha=waha)
    (settings.journal_dir / settings.session).mkdir(parents=True, exist_ok=True)
    host = host or settings.host
    port = port or settings.port
    logger.info(
        "Starting webhook server on {host}:{port} for session {session}",
        host=host,
        port=port,
        session=settings.session,
    )
    logger.info(
        "Webhook URL: http://{host}:{port}/api/webhook/{session}",
        host=host,
        port=port,
        session=settings.session,
    )
    uvicorn.run(
        "wahabot.webhook:app",
        host=host,
        port=port,
        reload=reload,
        log_config=None,
        http="wahabot.core.protocol:LoggingH11Protocol",
    )


def main() -> None:
    """Run the wahabot CLI."""
    setup_logging(get_settings())
    app()


if __name__ == "__main__":
    main()
