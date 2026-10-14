"""CLI entrypoints and commands for wahabot."""

import datetime
import json
from importlib.metadata import version as get_version
from typing import Any

import typer
import uvicorn
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from wahabot.ai.context import render_system_prompt
from wahabot.commands import build_command_event, register_command_handler
from wahabot.core import cli as cli_helpers
from wahabot.core.access import load_session_config
from wahabot.core.audit import audit_dir
from wahabot.core.waha import WahaClient
from wahabot.handlers import register_agent_handler, register_forget_handler
from wahabot.reactions import register_reaction_handler
from wahabot.settings import get_settings, setup_logging
from wahabot.status import (
    register_session_status_handler,
    seed_health,
)

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


@app.command()
def config() -> None:
    """Show the loaded configuration (secrets redacted)."""
    settings = get_settings()
    rows: list[tuple[str, Text | str]] = []
    for key, value in settings.model_dump().items():
        shown = (
            Text("***", style="dim")
            if cli_helpers.redact_key(key)
            else cli_helpers.dash(str(value))
        )
        rows.append((key, shown))
    console.print(cli_helpers.kv_table(rows))


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
    path.write_text(json.dumps(cli_helpers.session_template(), indent=2) + "\n")
    cli_helpers.ok(
        console,
        f"Wrote [bold]{path}[/] — edit system_prompt (and goal/bot_name) before serving.",
    )


@sessions_app.command("list")
def session_list() -> None:
    """List session config files found in <data>/sessions/."""
    settings = get_settings()
    sessions_dir = settings.access_config.parent
    configs = sorted(sessions_dir.glob("*.json")) if sessions_dir.is_dir() else []
    if not configs:
        console.print(f"[dim]No session configs in {sessions_dir}[/]")
        return
    rows = [
        (path.stem, cli_helpers.dash(load_session_config(path).bot_name))
        for path in configs
    ]
    console.print(cli_helpers.kv_table(rows))


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
            prompt,
            settings.timezone,
            config.bot_name,
            config.goal,
            config.memory,
            operator_name=config.operator_name,
            **cli_helpers.identity_kwargs(settings),
        )
    console.print(
        cli_helpers.kv_table(
            [
                ("session", settings.session),
                ("group_participation", config.group_participation),
                ("bot_name", cli_helpers.dash(config.bot_name)),
                ("operator_name", cli_helpers.dash(config.operator_name)),
                ("bot_mention_regex", cli_helpers.dash(config.bot_mention_regex)),
                ("whitelist", cli_helpers.format_list(config.whitelist)),
                ("blacklist", cli_helpers.format_list(config.blacklist)),
                ("goal", cli_helpers.dash(config.goal)),
                ("memory", cli_helpers.dash(config.memory)),
            ]
        )
    )
    title = "system_prompt" if raw else "system_prompt (rendered)"
    body: Any = prompt if plain else Markdown(prompt)
    console.print(Panel(body, title=title, expand=False))


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
    cli_helpers.log_startup_banner(settings, host, port)
    if not settings.access_config.exists():
        message = (
            f"Session config not found: {settings.access_config}"
            " (create it with `wahabot sessions init`)"
        )
        raise typer.BadParameter(message)
    waha = WahaClient(base_url=settings.waha_url, api_key=settings.waha_api_key)
    try:
        cli_helpers.ensure_session_live(waha, settings)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    seed_health(waha, settings.session)
    agent, _config_reloader = register_agent_handler(settings, waha=waha)
    register_reaction_handler(waha, agent, settings)
    register_command_handler(settings, waha, agent)
    register_session_status_handler(waha, settings.session)
    register_forget_handler(settings)
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
        help='Instruction for the agent, e.g. "send a summary to the group Family".'
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
    try:
        cli_helpers.post_signed_event(
            build_command_event(settings.session, text), settings, host, port
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    cli_helpers.ok(console, f"Command delivered to session [bold]{settings.session}[/]")


@app.command()
def forget(
    chat_id: str = typer.Argument(help="The chat JID whose memory to wipe."),
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
    """Wipe one chat's memory in the running bot.

    Posts a signed ``forget`` event to the webhook; the server drops the
    live context and the memory file under that chat's run lock, so there is
    no resurrection window. The webhook must be reachable — with the bot
    stopped, delete the file directly instead:
    ``rm data/memory/<session>/<chat-id>.json``.
    """
    settings = get_settings()
    if session:
        settings.session = session
    try:
        cli_helpers.post_signed_event(
            cli_helpers.build_forget_event(settings.session, chat_id),
            settings,
            host,
            port,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    cli_helpers.ok(console, f"Forget delivered to session [bold]{settings.session}[/]")


@app.command()
def escalations(
    session: str = typer.Option(
        None, "--session", "-s", help="WAHA session name. [default: WAHABOT_SESSION]"
    ),
    days: int = typer.Option(
        7, "--days", "-d", help="How many days back to list (audit journal files)."
    ),
) -> None:
    """List the bot's escalations from the audit journal.

    Each confirmed ``escalate`` tool call is persisted with status
    "open"; this reads the audit journal's escalation entries, newest
    first. The durable record the chat notification scrolled away.
    """
    settings = get_settings()
    # Read-only view: never mutate the process-wide cached settings
    # (``--session`` picks the audit directory here, nothing else).
    session_name = session or settings.session
    # The journal's day files are named by the writer's UTC date, so
    # the day cutoff is computed in UTC too — a local-today boundary
    # would hide or show a file around midnight depending on the
    # host's timezone.
    since = datetime.datetime.now(tz=datetime.UTC).date() - datetime.timedelta(days=days)
    entries = cli_helpers.escalation_entries(
        audit_dir(settings.data_dir, session_name), since
    )
    if not entries:
        console.print(f"[dim]No escalations in the last {days} day(s).[/]")
        return
    console.print(cli_helpers.escalation_rows(entries))


@app.command()
def notebook(
    chat: str = typer.Option(
        "", "--chat", "-c", help="Show this chat's notes only (JID, or '@session')."
    ),
    session: str = typer.Option(
        None, "--session", "-s", help="WAHA session name. [default: WAHABOT_SESSION]"
    ),
) -> None:
    """List the bot's durable notebook notes.

    Read-only view of the notebooks the ``notebook`` tool and the
    wrap-up lessons write (``data/notebook.sqlite3``): without
    ``--chat`` every scope is listed with its note count; with
    ``--chat`` that scope's notes are shown in full, newest first.
    """
    from wahabot.core.notebook import Notebook

    settings = get_settings()
    # Read-only view: never mutate the process-wide cached settings.
    session_name = session or settings.session
    store = Notebook(settings.data_dir)
    if not store.path.exists():
        console.print(f"[dim]No notebook yet at {store.path}[/]")
        return
    if not chat:
        scopes = store.scopes(session_name)
        if not scopes:
            console.print(f"[dim]No notes in any scope for session {session_name}[/]")
            return
        table = Table(box=None, show_header=True, pad_edge=False)
        table.add_column("chat", overflow="fold")
        table.add_column("notes", justify="right")
        for chat_id, count in scopes:
            table.add_row(chat_id, str(count))
        console.print(table)
        return
    notes = store.read(session_name, chat)
    if not notes:
        console.print(f"[dim]No notes for {chat} in session {session_name}[/]")
        return
    for note in notes:
        console.print(
            Panel(
                note.text,
                title=note.key,
                border_style="dim",
                expand=False,
            )
        )


def main() -> None:
    """Run the wahabot CLI."""
    setup_logging(get_settings())
    app()


if __name__ == "__main__":
    main()
