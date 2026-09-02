"""CLI entrypoints and commands for whabot."""

from importlib.metadata import version as get_version

import typer
import uvicorn
from loguru import logger

from whabot.core.waha import WahaClient
from whabot.handlers import register_agent_handler
from whabot.reactions import register_reaction_handler
from whabot.settings import get_settings, setup_logging

app = typer.Typer(
    name="whabot",
    help="whabot command-line interface.",
    no_args_is_help=True,
    add_completion=True,
)


@app.command()
def version() -> None:
    """Show the whabot version."""
    typer.echo(get_version("whabot"))


@app.command()
def config() -> None:
    """Show the loaded configuration (secrets redacted)."""
    settings = get_settings()
    for key, value in settings.model_dump().items():
        shown = "***" if "key" in key else value
        typer.echo(f"{key}={shown}")


@app.command()
def serve(
    host: str = typer.Option(
        None, "--host", "-h", help="Bind address. [default: WHABOT_HOST]"
    ),
    port: int = typer.Option(
        None, "--port", "-p", help="Bind port. [default: WHABOT_PORT]"
    ),
    session: str = typer.Option(
        None, "--session", "-s", help="WAHA session name. [default: WHABOT_SESSION]"
    ),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload."),
) -> None:
    """Run the webhook server for WAHA events."""
    settings = get_settings()
    if session:
        settings.session = session
    setup_logging(settings)
    waha = WahaClient(base_url=settings.waha_url, api_key=settings.waha_api_key)
    register_agent_handler(settings, waha=waha)
    register_reaction_handler(settings, waha=waha)
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
        "whabot.webhook:app",
        host=host,
        port=port,
        reload=reload,
        http="whabot.core.connection_log:LoggingH11Protocol",
    )


def main() -> None:
    """Run the whabot CLI."""
    setup_logging(get_settings())
    app()


if __name__ == "__main__":
    main()
