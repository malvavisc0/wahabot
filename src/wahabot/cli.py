"""CLI entrypoints and commands for wahabot."""

import contextlib
import datetime
import json
import os
import sys
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

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
from wahabot.core.db import Database
from wahabot.core.notebook import Notebook
from wahabot.core.schedule.hours import effective_timezone
from wahabot.core.skills.drafts import SkillDraftService
from wahabot.core.skills.errors import SkillError
from wahabot.core.skills.loader import import_skill, load_skill
from wahabot.core.skills.secrets import SecretStore
from wahabot.core.skills.service import SkillService
from wahabot.core.skills.store import SkillStore
from wahabot.core.waha import WahaClient
from wahabot.handlers import register_agent_handler, register_forget_handler
from wahabot.reactions import register_reaction_handler
from wahabot.settings import Settings, get_settings, setup_logging
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
skills_app = typer.Typer(
    name="skills",
    help="List, import, test, enable and disable configured skills.",
    no_args_is_help=True,
)
app.add_typer(skills_app, name="skills")

console = Console()


def skill_runtime(
    settings: Settings,
) -> tuple[SkillStore, SkillDraftService]:
    """Build the CLI's skill stores and draft service against the data dir."""
    database = Database(settings.data_dir / "wahabot.db")
    database.migrate()
    store = SkillStore(database.path)
    secrets = SecretStore(database.path, settings.secret_key)
    waha = WahaClient(base_url=settings.waha_url, api_key=settings.waha_api_key)
    service = SkillService(
        store,
        secrets,
        settings.skills_dir,
        settings.session,
        waha,
        data_dir=settings.data_dir,
    )
    return store, SkillDraftService(store, secrets, settings.skills_dir, service)


def skill_host(base_urls: list[str]) -> str:
    """The base host only: never a credential-bearing URL."""
    if not base_urls:
        return "local SQL"
    return urlsplit(base_urls[0]).hostname or "unknown"


@skills_app.command("list")
def skills_list() -> None:
    """List configured skills without URLs or secrets."""
    settings = get_settings()
    store, drafts = skill_runtime(settings)
    skills = store.list()
    if not skills:
        console.print("[dim]No skills configured.[/]")
        return
    table = Table(box=None, show_header=True, pad_edge=False)
    for column in ("name", "lifecycle", "enabled", "test", "chats", "host"):
        table.add_column(column)
    for skill in skills:
        drafts.refresh_staleness(skill.name)
        skill = store.get(skill.name) or skill
        base_urls = skill.base_urls
        with contextlib.suppress(SkillError):
            base_urls = load_skill(skill.name, settings.skills_dir).effective_base_urls
        table.add_row(
            skill.name,
            skill.lifecycle,
            "yes" if skill.enabled else "no",
            skill.test_state,
            skill.customer_chats,
            skill_host(base_urls),
        )
    console.print(table)


@skills_app.command("import")
def skills_import(
    path: Annotated[
        Path, typer.Argument(help="Skill folder containing skill.yaml and SKILL.md.")
    ],
    enable: bool = typer.Option(
        False, "--enable", help="Run the safe check and enable when it passes."
    ),
) -> None:
    """Import a skill folder into <data>/skills/."""
    settings = get_settings()
    try:
        store, drafts = skill_runtime(settings)
        skill = import_skill(path, settings.skills_dir, store)
        if not enable:
            cli_helpers.ok(
                console,
                f"Imported [bold]{skill.name}[/]; "
                + f"run `wahabot skills test {skill.name}`.",
            )
            return
        result = drafts.run_test(skill.name)
        if not result.passed or not drafts.enable(skill.name):
            raise typer.BadParameter(f"{skill.name} stays inactive: {result.reason}")
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    cli_helpers.ok(console, f"Imported and enabled [bold]{skill.name}[/].")


@skills_app.command("test")
def skills_test(
    name: str = typer.Argument(help="Skill name."),
) -> None:
    """Run only the prepared safe check; never a write."""
    settings = get_settings()
    try:
        store, drafts = skill_runtime(settings)
        if store.get(name) is None:
            raise typer.BadParameter(f"unknown skill: {name}")
        result = drafts.run_test(name)
    except typer.BadParameter:
        raise
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not result.passed:
        console.print(f"[yellow]FAIL[/] {name}: {result.reason}")
        raise typer.Exit(1)
    console.print(f"[green]PASS[/] {name}: {result.reason}")


@skills_app.command("enable")
def skills_enable(
    name: str = typer.Argument(help="Skill name."),
) -> None:
    """Enable a tested, current skill."""
    settings = get_settings()
    try:
        store, drafts = skill_runtime(settings)
        skill = store.get(name)
        if skill is None:
            raise typer.BadParameter(f"unknown skill: {name}")
        if not drafts.enable(name):
            current = store.get(name) or skill
            reason = current.test_reason or current.test_state
            raise typer.BadParameter(f"{name} stays inactive: {reason}")
    except typer.BadParameter:
        raise
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    cli_helpers.ok(console, f"Enabled [bold]{name}[/].")


@skills_app.command("disable")
def skills_disable(
    name: str = typer.Argument(help="Skill name."),
) -> None:
    """Disable a skill without deleting its configuration."""
    settings = get_settings()
    try:
        store, drafts = skill_runtime(settings)
        if store.get(name) is None:
            raise typer.BadParameter(f"unknown skill: {name}")
        drafts.disable(name)
    except typer.BadParameter:
        raise
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    cli_helpers.ok(console, f"Disabled [bold]{name}[/].")


@app.command("hash-passphrase")
def hash_passphrase_command() -> None:
    """Read one passphrase from stdin and print its scrypt hash for .env."""
    from wahabot.web.auth import hash_passphrase

    raw = sys.stdin.readline()
    if not raw.strip():
        raise typer.BadParameter("a passphrase is required on stdin")
    typer.echo(hash_passphrase(raw.rstrip("\r\n")))


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
            effective_timezone(config.timezone, settings.timezone),
            config.bot_name,
            config.goal,
            config.memory,
            operator_name=config.operator_name,
            knowledge_only=config.knowledge_only,
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
        # The reload subprocess rebuilds Settings from the environment;
        # export the flag so the child serves the same session.
        os.environ["WAHABOT_SESSION"] = session
    setup_logging(settings)
    host = host or settings.host
    port = port or settings.port
    if port:
        settings.port = port
        os.environ["WAHABOT_PORT"] = str(port)
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
    registration = register_agent_handler(settings, waha=waha)
    register_reaction_handler(registration.transport, registration.agent, settings)
    register_command_handler(
        settings,
        registration.transport,
        registration.agent,
        registration.handlers.ticket_store,
    )
    register_session_status_handler(registration.transport, settings.session)
    register_forget_handler(settings, registration.agent)
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


def note_age(updated_at_ns: int) -> str:
    """A short relative age for a note's last update."""
    if not updated_at_ns:
        return "unknown"
    now = datetime.datetime.now().timestamp()
    seconds = max(0, int(now - updated_at_ns / 1_000_000_000))
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


@app.command()
def notebook(
    chat: str = typer.Option(
        "", "--chat", "-c", help="Show this chat's notes only (JID, or '@session')."
    ),
    session: str = typer.Option(
        None,
        "--session",
        "-s",
        help="WAHA session name. [default: all sessions; with --chat, WAHABOT_SESSION]",
    ),
    rules: bool = typer.Option(
        False, "--rules", help="With --chat, show only the chat's standing rules."
    ),
) -> None:
    """List the bot's durable notebook notes.

    Read-only view of the notebooks the ``notebook`` tool and the
    wrap-up lessons write (``data/notebook.sqlite3``). Chats are
    scoped by session — one server can run several — so the listing
    shows the session on every row: without flags, every session's
    scopes with note counts; with ``--session``, that bot's scopes;
    with ``--chat``, that scope's notes in full, newest first.
    """
    settings = get_settings()
    store = Notebook(settings.data_dir)
    if not store.path.exists():
        console.print(f"[dim]No notebook yet at {store.path}[/]")
        return
    if rules and not chat:
        console.print("[yellow]--rules needs --chat[/]")
        return
    if not chat:
        show_notebook_scopes(store, session)
        return
    show_chat_notes(store, session or settings.session, chat, rules)


def show_notebook_scopes(store: Notebook, session: str | None) -> None:
    """List the notebook's scopes and note counts."""
    scopes = store.scopes(session)
    if not scopes:
        scope_of = f"session {session}" if session else "any session"
        console.print(f"[dim]No notes in any scope for {scope_of}[/]")
        return
    table = Table(box=None, show_header=True, pad_edge=False)
    table.add_column("session", overflow="fold")
    table.add_column("chat", overflow="fold")
    table.add_column("notes", justify="right")
    for session_name, chat_id, count in scopes:
        table.add_row(session_name, chat_id, str(count))
    console.print(table)


def show_chat_notes(
    store: Notebook, session_name: str, chat: str, rules: bool = False
) -> None:
    """Print one scope's notes, newest first; *rules* shows only rules."""
    notes = store.read(session_name, chat, kind="rule" if rules else "")
    if not notes:
        label = "rules" if rules else "notes"
        console.print(f"[dim]No {label} for {chat} in session {session_name}[/]")
        return
    for note in notes:
        console.print(
            Panel(
                note.text,
                title=(
                    f"{session_name} · {note.key}\n"
                    f"{note.kind} · {note.author_kind} · {note_age(note.updated_at)}"
                ),
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
