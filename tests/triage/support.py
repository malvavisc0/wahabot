import asyncio
from pathlib import Path
from typing import Any

import httpx

from wahabot.ai.triage import (
    TriageContext,
    TriageResult,
    TriageVerdict,
)
from wahabot.core.models import WahaEvent
from wahabot.settings import Settings

"""Triage gate: parsing, fail-soft behavior, policy, and run wiring."""

SESSION = "s1"

CHAT = "491555000001@c.us"

GROUP = "491555000002@g.us"


def gate_settings(tmp_path: Path) -> Settings:
    return Settings(
        webhook_hmac_key="k",
        waha_url="http://waha.invalid",
        waha_api_key="k",
        llm_api_base="http://llm.invalid",
        llm_api_key="k",
        langfuse_public_key="",
        langfuse_secret_key="",
        session=SESSION,
        data_dir=tmp_path,
        gate_api_base="http://gate.invalid/v1",
        gate_timeout=0.2,
        _env_file=None,
    )


def gate_transport(content: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    return httpx.MockTransport(handler)


class FakeWaha:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def send_text(self, session: str, chat_id: str, text: str) -> str:
        self.sent.append((session, chat_id, text))
        return "sent-id"


class FakeChannel:
    def __init__(self) -> None:
        self.stamped: list[str] = []
        self.operator_jid = "491555000009@c.us"

    def cooldown_refusal(self, chat_id: str) -> str | None:
        return None

    def stamp(self, chat_id: str) -> None:
        self.stamped.append(chat_id)


class FakeGate:
    timeout = 0.5

    def __init__(self, result: TriageResult | None = None, error: bool = False) -> None:
        self.result = result or TriageResult(TriageVerdict())
        self.error = error
        self.calls = 0

    async def classify_bounded(self, context: TriageContext) -> TriageResult:
        self.calls += 1
        await asyncio.sleep(0)
        if self.error:
            raise AssertionError("gate must not be called")
        return self.result

    def warn_once(self, kind: str, message: str, **fields: Any) -> None:
        pass

    def audit(self, session: str, chat_id: str, result: TriageResult) -> None:
        pass


class Sentinel(Exception):
    pass


def message_event(chat: str, body: str) -> WahaEvent:
    return WahaEvent(
        event="message",
        session=SESSION,
        payload={"id": "m1", "from": chat, "fromMe": False, "body": body},
    )
