"""pytest fixtures for the wahabot test suite.

The old ``scripts/smoke_test.py`` booted one stack and mutated module
globals between scenarios. Here the boot order becomes fixtures:

- ``stack`` (session-scoped) boots the fake LLM and the real webhook
  app once on two ephemeral ports. Both are ports, not state — per-test
  cost is zero.
- ``bot`` (function-scoped) clears the process-global handler
  registries, builds a fresh ``RecordingWaha``, registers every handler,
  seeds health, and pins ``get_settings`` to the test's settings for the
  duration. Its ``post`` method collapses the old 39x "json.dumps ->
  patch -> httpx.post -> HMAC header -> assert 200" boilerplate into one
  call.
"""

import json
import tempfile
import unittest.mock
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

import wahabot.settings as settings_module
from tests.harness import (
    ME_JID,
    SESSION,
    FakeLlm,
    RecordingWaha,
    reset_handlers,
    serving,
    sign,
    smoke_settings,
    write_session_config,
)
from wahabot.ai.albums import reset as reset_albums
from wahabot.commands import register_command_handler
from wahabot.core.echoes import _echoes  # pyright: ignore[reportPrivateUsage]
from wahabot.core.runs import (
    _chat_lock_pending,  # pyright: ignore[reportPrivateUsage]
    _chat_locks,  # pyright: ignore[reportPrivateUsage]
)
from wahabot.core.runs import contexts as runs_contexts
from wahabot.handlers import (
    _seen_ids,  # pyright: ignore[reportPrivateUsage]
    register_agent_handler,
    register_forget_handler,
)
from wahabot.reactions import (
    _last_reaction_notes,  # pyright: ignore[reportPrivateUsage]
    register_reaction_handler,
)
from wahabot.status import (
    register_session_status_handler,
    seed_health,
    set_session_health,
)
from wahabot.status import (
    state as status_state,
)


def webhook_app() -> Any:
    """The real FastAPI webhook app (imported lazily to avoid cycles)."""
    import wahabot.webhook as webhook_module

    return webhook_module.app


class Stack:
    """The booted servers plus the per-test bot builder."""

    def __init__(self) -> None:
        self.llm = FakeLlm()
        self._servers: list[Any] = []
        self.start()

    def start(self) -> None:
        """Boot the fake LLM and webhook app on ephemeral ports."""
        llm_ctx = serving(self.llm.app())
        self._servers.append(llm_ctx)
        self.llm_port = llm_ctx.__enter__()
        hook_ctx = serving(webhook_app())
        self._servers.append(hook_ctx)
        self.hook_port = hook_ctx.__enter__()

    def stop(self) -> None:
        """Shut down the booted servers."""
        for ctx in reversed(self._servers):
            ctx.__exit__(None, None, None)
        self._servers.clear()

    def _settings(self, data_dir: Path) -> Any:
        return smoke_settings(data_dir, f"http://127.0.0.1:{self.llm_port}/v1")

    def register(self, data_dir: Path | None = None, **overrides: Any) -> Bot:
        """Clear registries and register a fresh agent/handlers set.

        ``overrides`` are keyword adjustments to the base settings (e.g.
        ``memory_persist=False``, ``video=False``).
        """
        reset_handlers()
        if data_dir is None:
            data_dir = Path(tempfile.mkdtemp(prefix="wahabot-bot-"))
        settings = self._settings(data_dir)
        for key, value in overrides.items():
            setattr(settings, key, value)
        write_session_config(settings)
        settings_module.get_settings.cache_clear()
        waha = RecordingWaha()
        agent, _reloader = register_agent_handler(settings, waha=waha)
        register_command_handler(settings, waha, agent)
        register_reaction_handler(waha, agent, settings)
        register_forget_handler(settings)
        seed_health(waha, SESSION)
        register_session_status_handler(waha, SESSION)
        return Bot(self, settings, waha, agent)


class Bot:
    """One test's isolated bot: settings, recording WAHA, and helpers."""

    def __init__(
        self, stack: Stack, settings: Any, waha: RecordingWaha, agent: Any
    ) -> None:
        self.stack = stack
        self.settings = settings
        self.waha = waha
        self.agent = agent

    # -- posting ---------------------------------------------------------

    def url(self) -> str:
        return f"http://127.0.0.1:{self.stack.hook_port}/api/webhook/{SESSION}"

    def post(self, event: dict[str, Any], *, expect: int = 200) -> httpx.Response:
        """Sign *event*, POST it to the webhook, assert the status code."""
        body = json.dumps(event).encode()
        response = httpx.post(
            self.url(),
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Hmac": sign(body, self.settings.webhook_hmac_key),
            },
            timeout=120,
        )
        assert response.status_code == expect, (
            f"webhook returns {expect} (got {response.status_code}: {response.text})"
        )
        return response

    def post_raw(
        self, body: bytes, *, headers: dict[str, str] | None = None, expect: int = 200
    ) -> httpx.Response:
        """POST raw bytes with exactly the given headers (no implicit signing)."""
        merged = {"Content-Type": "application/json"}
        if headers:
            merged.update(headers)
        response = httpx.post(self.url(), content=body, headers=merged, timeout=120)
        assert response.status_code == expect, (
            f"webhook returns {expect} (got {response.status_code}: {response.text})"
        )
        return response

    def rebuild(self, **overrides: Any) -> Bot:
        """Tear down and register a new bot with *overrides* applied."""
        return self.stack.register(**overrides)


@pytest.fixture(scope="session")
def stack() -> Iterator[Stack]:
    """Boot the fake LLM and webhook app once for the whole session."""
    s = Stack()
    try:
        yield s
    finally:
        s.stop()
        s.llm.clear()


@pytest.fixture()
def _reset_registries() -> Iterator[None]:
    """Wipe every process-global tracker before and after each test."""
    reset_handlers()
    _echoes.clear()
    _seen_ids.clear()
    _chat_locks.clear()
    _chat_lock_pending.clear()
    _last_reaction_notes.clear()
    runs_contexts.clear()
    reset_albums()
    set_session_health("WORKING")
    status_state.operator_jid = ME_JID
    status_state.operator_lid = "491555000000@lid"
    yield
    reset_handlers()
    _echoes.clear()
    _seen_ids.clear()
    _chat_locks.clear()
    _chat_lock_pending.clear()
    _last_reaction_notes.clear()
    runs_contexts.clear()
    reset_albums()
    status_state.operator_jid = ""
    status_state.operator_lid = ""


@pytest.fixture()
def bot(stack: Stack, _reset_registries: None) -> Iterator[Bot]:
    """A fresh isolated bot per test, with settings pinned."""
    bot = stack.register()
    stack.llm.clear()
    with unittest.mock.patch("wahabot.settings.get_settings", return_value=bot.settings):
        yield bot
