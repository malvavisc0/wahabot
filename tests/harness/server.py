import contextlib
import hashlib
import hmac
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import uvicorn
from fastapi import FastAPI

import wahabot.webhook as webhook_module
from tests.harness.constants import SESSION
from wahabot.settings import Settings


@contextlib.contextmanager
def serving(app: FastAPI) -> Iterator[int]:
    """Run *app* on an ephemeral localhost port; yields the port."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    # Uvicorn binds port 0 → find the actual port from its bound socket.
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def smoke_settings(data_dir: Path, llm_base: str) -> Settings:
    """Settings for a run; `_env_file=None` never touches the real .env.

    The Langfuse fields are pinned empty because pydantic-settings
    still reads process env vars even with `_env_file=None` — a
    developer shell with real LANGFUSE_* keys would otherwise arm the
    OTel exporter and every test run would ship fake-LLM traces to the
    real Langfuse instance. ``timezone`` and ``tts_url`` are pinned for
    the same reason: they change wire-visible output (the pinned-state
    block, and whether a voice note ships), so a leaked env var must
    not reach the run under test.

    Burst assembly is pinned OFF so single-message wire tests keep the
    immediate-run semantics they assert; the burst tests opt in via
    ``bot.rebuild(burst_enabled=True, …)`` with short windows.
    """
    return Settings(
        webhook_hmac_key="smoke-hmac-key",
        waha_url="http://waha.invalid",
        waha_api_key="waha-key",
        llm_api_base=llm_base,
        llm_api_key="sk-smoke",
        llm_model="smoke-model",
        session=SESSION,
        data_dir=data_dir,
        transcribe_url="http://whisper.invalid",
        burst_enabled=False,
        triage_gate=False,
        timezone="UTC",
        tts_url="",
        langfuse_public_key="",
        langfuse_secret_key="",
        langfuse_tracing_environment="test",
        _env_file=None,
    )


def write_session_config(settings: Settings) -> None:
    """A minimal working session config, as `sessions init` would write."""
    path = settings.access_config
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "whitelist": [],
        "blacklist": [],
        "goal": "",
        "system_prompt": (
            "You are {{bot_name}} texting on WhatsApp.\n{{operator_tools}}"
        ),
        "bot_name": "kai",
        "bot_mention_regex": "@?kai",
        "group_participation": "mentioned",
    }
    path.write_text(json.dumps(config, indent=2))


def sign(body: bytes, key: str) -> str:
    """Real production HMAC: sha512 hex digest, as WAHA sends it."""
    return hmac.new(key.encode(), body, hashlib.sha512).hexdigest()


def reset_handlers() -> None:
    """Drop handlers from any previous register_* call in this process."""
    for registry in webhook_module._registries.values():  # pyright: ignore[reportPrivateUsage]
        registry.clear()
    webhook_module._shutdown_hooks.clear()  # pyright: ignore[reportPrivateUsage]
    if hasattr(webhook_module.app.state, "intake_writer"):
        delattr(webhook_module.app.state, "intake_writer")
