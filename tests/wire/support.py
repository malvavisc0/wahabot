import json
import shutil
import time
import unittest.mock
from collections.abc import Callable
from typing import Any, cast

import httpx
import pytest

from tests.conftest import Bot
from tests.harness import (
    CHAT_ID,
    tool_call_response,
    waha_event,
)

"""Wire scenarios ported from ``scripts/smoke_test.py``'s ``main``.

Each scenario posts a genuine event over HTTP (real HMAC signature)
against the booted stack and asserts the bot's behavior. The ``bot``
fixture provides an isolated agent + settings per test; ``bot.post``
handles signing and status assertion.
"""

POLL_TIMEOUT = 15


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


requires_ffmpeg = pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not installed")


def _wait(pred: Callable[[], bool], timeout: float = POLL_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


def _audit_entries(bot: Bot, kind: str, chat_id: str) -> list[dict[str, Any]]:
    """The bot's audit journal entries of *kind* in *chat_id*, as dicts."""
    import datetime as _dt

    from wahabot.core.audit import audit_dir

    directory = audit_dir(bot.settings.data_dir, bot.settings.session)
    day = _dt.datetime.now(tz=_dt.UTC).strftime("%Y-%m-%d")
    path = directory / f"{day}.jsonl"
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("kind") == kind and entry.get("chat_id") == chat_id:
            entries.append(entry)
    return entries


def user_turns_of(request: dict[str, Any]) -> list[str]:
    """The user-role turn texts of a captured LLM request."""
    return [
        str(m.get("content", ""))
        for m in request.get("messages", [])
        if m.get("role") == "user"
    ]


def _run_failing_delivery_case(
    bot: Bot,
    tool: str,
    waha_method: str,
    args: dict[str, Any],
    record: str | None,
) -> None:
    """One delivery tool's WAHA send fails; the run must degrade to text.

    The mock fails the tool's send on the first call only, then
    delegates to the recording method, so the model's fallback text
    lands for real and proves the delivery latch stayed open.
    """
    failing_bot = bot.rebuild()
    llm = failing_bot.stack.llm
    llm.clear()  # agent_round counts every request ever recorded
    llm.override = tool_call_response(
        tool,
        {"reason": "wire test", **args},
        call_id=f"call_{tool}_fail",
        response_id=f"chatcmpl-{tool}-fail",
    )
    calls = {"count": 0}
    real_method = getattr(type(failing_bot.waha), waha_method)

    def flaky_send(*call_args: Any, **call_kw: Any) -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("waha gone")
        return cast(str, real_method(failing_bot.waha, *call_args, **call_kw))

    with unittest.mock.patch.object(
        type(failing_bot.waha), waha_method, side_effect=flaky_send
    ):
        # send_media is one tool for all five media kinds, so the event
        # id must vary by kind or dedup would drop every case after the
        # first; the other tools already differ by name.
        mid = f"{tool.upper()}FAIL"
        if args.get("kind"):
            mid += f"-{args['kind']}"
        failing_bot.post(waha_event(mid))
    assert _wait(lambda: calls["count"] >= 1), tool  # the tool's send raised
    assert _wait(lambda: len(llm.requests) >= 2), tool  # the envelope looped back
    assert "delivery was not confirmed" in str(llm.requests[1]["messages"]), tool
    if record is not None:  # the tool's own send never landed
        assert not getattr(failing_bot.waha, record), (tool, record)
    if tool == "send_message":  # the failed tool text must not ride along
        assert all(entry[2] != args["text"] for entry in failing_bot.waha.sent), tool
    # Fail-soft: the run survives and the model's fallback text lands.
    assert _wait(lambda: len(failing_bot.waha.sent) >= 1), tool
    assert failing_bot.waha.sent[-1][2] == "smoke final answer", tool


def _judicious_bot(bot: Bot) -> Bot:
    """The same bot with ``group_participation: judicious``, hot-reloaded.

    The config file is rewritten in place (the reloader stats it per
    event), so no re-registration is needed.
    """
    path = bot.settings.access_config
    config = json.loads(path.read_text())
    config["group_participation"] = "judicious"
    path.write_text(json.dumps(config))
    return bot


def _group_event(mid: str, body: str, participant: str) -> dict[str, Any]:
    """A group text event that does not name the bot (unaddressed)."""
    event = waha_event(mid)
    event["payload"]["participant"] = participant
    event["payload"]["body"] = body
    event["payload"]["_data"] = dict(event["payload"]["_data"])
    event["payload"]["_data"]["id"] = {"_serialized": f"false_{CHAT_ID}_{mid}"}
    return event
