"""Tool-side #m<n> resolution and read_chat records, cross-chat included."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from wahabot.ai.serials import SerialRegistry, registry_file
from wahabot.ai.tools.whatsapp import (
    RunTarget,
    bind_target,
    react_to_message,
    read_chat,
    reset_target,
    resolve_message_token,
    send_message,
)
from wahabot.settings import Settings

SESSION = "smoke"
CHAT = "491555000010@g.us"
OTHER = "491555000011@g.us"
TOKEN_CHAT = "self"


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        webhook_hmac_key="k",
        waha_url="http://waha.invalid",
        waha_api_key="k",
        llm_api_base="http://llm.invalid",
        llm_api_key="k",
        data_dir=tmp_path,
        session=SESSION,
        triage_gate=False,
        langfuse_public_key="",
        langfuse_secret_key="",
        _env_file=None,
    )


def _json(out: Any) -> dict[str, Any]:
    return json.loads(str(out.raw_output))


def _serial(tmp_path: Path, chat: str, message_id: str, serial: int) -> None:
    registry = SerialRegistry(registry_file(tmp_path, SESSION, chat))
    for number in range(1, serial):
        registry.next(message_id=f"filler-{number}")
    registry.next(message_id=message_id, jid="491555000101@c.us", name="Ada Sample")


class Waha:
    def __init__(self, messages: list[dict[str, Any]] | None = None) -> None:
        self.messages = messages or []
        self.calls: list[tuple[Any, ...]] = []

    def get_chat_overview(self, session: str, chat_id: str) -> None:
        return None

    def fetch_chat_messages(
        self, session: str, chat_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        self.calls.append(("fetch", chat_id, limit))
        return self.messages

    def send_text(
        self,
        session: str,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        mentions: list[str] | None = None,
    ) -> str:
        self.calls.append(("send", chat_id, text, reply_to, mentions))
        return "SENT"

    def send_reaction(self, session: str, message_id: str, reaction: str) -> None:
        self.calls.append(("react", message_id, reaction))

    def send_seen(self, session: str, chat_id: str) -> None:
        return None

    def start_typing(self, session: str, chat_id: str) -> None:
        return None

    def stop_typing(self, session: str, chat_id: str) -> None:
        return None


def test_send_reply_token_resolves_in_the_target_chat(
    settings: Settings, tmp_path: Path
) -> None:
    _serial(tmp_path, CHAT, "RAW7", 7)
    waha = Waha()
    token = bind_target(RunTarget(session=SESSION, chat_id=CHAT))
    try:
        out = _json(cast(Any, send_message(waha, settings))(text="hola", reply_to="#m7"))
    finally:
        reset_target(token)
    assert out["ok"] is True
    assert ("send", CHAT, "hola", "RAW7", None) in waha.calls


def test_send_reply_raw_id_passes_through(settings: Settings) -> None:
    waha = Waha()
    token = bind_target(RunTarget(session=SESSION, chat_id=CHAT))
    try:
        out = _json(cast(Any, send_message(waha, settings))(text="hola", reply_to="RAW7"))
    finally:
        reset_target(token)
    assert out["ok"] is True
    assert ("send", CHAT, "hola", "RAW7", None) in waha.calls


def test_unknown_token_fails_soft(settings: Settings) -> None:
    waha = Waha()
    token = bind_target(RunTarget(session=SESSION, chat_id=CHAT))
    try:
        out = _json(cast(Any, send_message(waha, settings))(text="hola", reply_to="#m99"))
    finally:
        reset_target(token)
    assert out["ok"] is False and "unknown message token" in out["error"]
    assert waha.calls == []


def test_operator_send_resolves_cross_chat(settings: Settings, tmp_path: Path) -> None:
    _serial(tmp_path, CHAT, "RAW7", 7)
    waha = Waha()
    target = RunTarget(session=SESSION, chat_id=TOKEN_CHAT, armed=True)
    target.allowed_chats.add(CHAT)
    token = bind_target(target)
    try:
        out = _json(
            cast(Any, send_message(waha, settings))(
                text="hola", chat=CHAT, reply_to="#m7"
            )
        )
    finally:
        reset_target(token)
    assert out["ok"] is True
    assert ("send", CHAT, "hola", "RAW7", None) in waha.calls


def test_operator_reaction_finds_a_unique_token(
    settings: Settings, tmp_path: Path
) -> None:
    _serial(tmp_path, CHAT, "RAW7", 7)
    waha = Waha()
    target = RunTarget(session=SESSION, chat_id=TOKEN_CHAT, armed=True)
    target.allowed_chats.add(CHAT)
    token = bind_target(target)
    try:
        out = _json(
            cast(Any, react_to_message(waha, settings))(message_id="#m7", reaction="👍")
        )
    finally:
        reset_target(token)
    assert out["ok"] is True
    assert ("react", "RAW7", "👍") in waha.calls


def test_ambiguous_operator_token_refuses_to_guess(
    settings: Settings, tmp_path: Path
) -> None:
    _serial(tmp_path, CHAT, "RAW7", 7)
    _serial(tmp_path, OTHER, "RAW7B", 7)
    waha = Waha()
    target = RunTarget(session=SESSION, chat_id=TOKEN_CHAT, armed=True)
    target.allowed_chats.add(CHAT)
    token = bind_target(target)
    try:
        out = _json(
            cast(Any, react_to_message(waha, settings))(message_id="#m7", reaction="👍")
        )
    finally:
        reset_target(token)
    assert out["ok"] is False and "unknown message token" in out["error"]
    assert waha.calls == []


def test_operator_can_resolve_a_chat_token(settings: Settings, tmp_path: Path) -> None:
    raw = f"false_{CHAT}_RAW7"
    _serial(tmp_path, CHAT, raw, 7)
    waha = Waha(
        [
            {
                "id": raw,
                "from": CHAT,
                "participant": "491555000101@c.us",
                "body": "hola",
                "_data": {"type": "text"},
            }
        ]
    )
    target = RunTarget(session=SESSION, chat_id=TOKEN_CHAT, armed=True)
    target.allowed_chats.add(CHAT)
    token = bind_target(target)
    try:
        out = _json(cast(Any, read_chat(waha, settings))(mode="list", chat="#m7"))
    finally:
        reset_target(token)
    assert out["ok"] is True
    assert any(call[0] == "fetch" and call[1] == CHAT for call in waha.calls)
    assert out["records"] and "#m7" in out["records"][0]


def test_read_chat_returns_records_with_tokens(
    settings: Settings, tmp_path: Path
) -> None:
    _serial(tmp_path, CHAT, "RAW7", 7)
    waha = Waha(
        [
            {
                "id": "RAW7",
                "from": CHAT,
                "participant": "491555000101@c.us",
                "body": "hola",
                "_data": {"type": "text", "notifyName": "Ada Sample"},
            }
        ]
    )
    token = bind_target(RunTarget(session=SESSION, chat_id=CHAT))
    try:
        out = _json(cast(Any, read_chat(waha, settings))(mode="list"))
    finally:
        reset_target(token)
    assert "messages" not in out
    assert out["records"] == ["[[13:00 Ada Sample #m7]] text :: hola"] or out["records"][
        0
    ].endswith("#m7]] text :: hola")


def test_explicit_name_mentions_resolve_through_the_roster(
    settings: Settings,
) -> None:
    waha = Waha(
        [
            {
                "id": "A",
                "from": CHAT,
                "participant": "491555000101@c.us",
                "body": "hi",
                "_data": {"type": "text", "notifyName": "Ada Sample"},
            }
        ]
    )
    token = bind_target(RunTarget(session=SESSION, chat_id=CHAT))
    try:
        out = _json(
            cast(Any, send_message(waha, settings))(text="hola", mentions=["Ada Sample"])
        )
    finally:
        reset_target(token)
    assert out["ok"] is True
    assert ("send", CHAT, "hola", None, ["491555000101@c.us"]) in waha.calls


def test_shared_name_tags_neither_and_warns(settings: Settings) -> None:
    waha = Waha(
        [
            {
                "id": "A",
                "from": CHAT,
                "participant": "491555000101@c.us",
                "body": "hi",
                "_data": {"type": "text", "notifyName": "Ada Sample"},
            },
            {
                "id": "B",
                "from": CHAT,
                "participant": "491555000102@c.us",
                "body": "hi",
                "_data": {"type": "text", "notifyName": "Ada Sample"},
            },
        ]
    )
    token = bind_target(RunTarget(session=SESSION, chat_id=CHAT))
    try:
        out = _json(
            cast(Any, send_message(waha, settings))(text="hola", mentions=["Ada Sample"])
        )
    finally:
        reset_target(token)
    assert out["ok"] is True
    assert "warning" in out and "Ada Sample" in out["warning"]
    assert ("send", CHAT, "hola", None, None) in waha.calls


def test_resolve_message_token_is_soft(settings: Settings) -> None:
    raw, error = resolve_message_token(
        "#m1", session=SESSION, chat_id=CHAT, settings=settings
    )
    assert raw == "" and error is not None
    raw, error = resolve_message_token(
        "RAW", session=SESSION, chat_id=CHAT, settings=settings
    )
    assert raw == "RAW" and error is None
