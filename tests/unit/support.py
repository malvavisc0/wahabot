import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from tests.harness import (
    CHAT_ID,
    FOREIGN_JID,
    OWN_LID,
    SESSION,
)
from wahabot.core.models import WahaEvent

"""Unit-level checks ported from ``scripts/smoke_test.py``'s ``check_units``.

Pure-function assertions (JID shapes, mimetypes, fences, video markers,
echo cache) plus the WAHA wire-shape block, split into plain test
functions. No servers needed except the wire block, which uses a mock
transport, not a port.
"""


def _addressed_event() -> WahaEvent:
    return WahaEvent(
        id="e-unaddressed",
        timestamp=1,
        event="message",
        session=SESSION,
        me={"id": "491555000000@c.us", "lid": "491555000000@lid"},
        payload={
            "from": CHAT_ID,
            "participant": "491555000001@c.us",
            "body": "hello",
        },
    )


def _chat_event() -> WahaEvent:
    return WahaEvent(
        id="e1",
        timestamp=1,
        event="message",
        session=SESSION,
        me={"id": "491555000000@c.us", "lid": "491555000000@lid"},
        payload={
            "from": CHAT_ID,
            "participant": "491555000001@c.us",
            "body": "hi",
            "_data": {"mentionedJidList": [{"_serialized": "491555000000@lid"}]},
        },
    )


def _fake_ydl(info: dict[str, Any] | None) -> Any:
    """A YoutubeDL stand-in whose extract_info returns *info*."""

    class FakeYDL:
        def __init__(self, opts: dict[str, Any]) -> None:
            self.opts = opts

        def __enter__(self) -> FakeYDL:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def extract_info(
            self, url: str, download: bool, process: bool
        ) -> dict[str, Any] | None:
            assert download is False
            assert process is False
            return info

    return FakeYDL


STAGE_MODULES = {
    "post_info": "video",
    "meta_fields": "video",
    "attach_caption_tracks": "video",
    "paragraph": "captions",
}


def _mock_httpx_client(wire: Callable[[httpx.Request], httpx.Response]) -> Any:
    """A patched httpx.Client factory that routes every request through *wire*.

    Built against the real class captured before patching, so the
    factory never recurses into its own patch.
    """
    real_client = httpx.Client

    def factory(**kwargs: Any) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(wire), **kwargs)

    return factory


class _PinWaha:
    """The chat-list/messages surface the resolver needs, faked.

    ``list_chats`` mimics the operator's conversation list (newest
    first); ``fetch_chat_messages`` returns canned slimmable messages.
    ``fetched`` records which chat was read, so tests can assert the
    resolver looked at the *named* chat, not the newest one.
    """

    def __init__(self, messages: list[dict[str, Any]] | None = None) -> None:
        self.messages = messages or []
        self.fetched: list[str] = []

    def list_chats(self, session: str, limit: int = 200) -> list[dict[str, Any]]:
        return [
            {"id": CHAT_ID, "name": "Bridge Club"},
            {"id": FOREIGN_JID, "name": "Nadia"},
        ]

    def list_contacts(self, session: str, limit: int = 500) -> list[dict[str, Any]]:
        return []

    def fetch_chat_messages(
        self, session: str, chat_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        self.fetched.append(chat_id)
        return self.messages


_PIN_MESSAGES = [
    {
        "id": f"false_{CHAT_ID}_LAST",
        "from": FOREIGN_JID,
        "body": "cual es la disponibilidad del salon para el viernes",
    },
    {
        "id": f"false_{CHAT_ID}_OLD1",
        "from": FOREIGN_JID,
        "body": "older message",
    },
]


def _journal_fixture_lines() -> list[dict[str, Any]]:
    """Three chat rows plus noise: member, operator (`app`), bot (`api`)."""
    return [
        {
            "event": "message.any",
            "session": SESSION,
            "payload": {
                "id": f"false_{CHAT_ID}_MEMBER_491555000001@c.us",
                "timestamp": 300,
                "from": CHAT_ID,
                "fromMe": False,
                "participant": "491555000001@c.us",
                "body": "member asking",
                "_data": {"notifyName": "Alex Rivers"},
            },
        },
        {
            "event": "message.any",
            "session": SESSION,
            "payload": {
                "id": f"true_{CHAT_ID}_OP_{OWN_LID}",
                "timestamp": 100,
                "from": OWN_LID,
                "fromMe": True,
                "to": CHAT_ID,
                "source": "app",
                "participant": OWN_LID,
                "body": "operator typing on the phone",
                "_data": {"notifyName": "Operator Human"},
            },
        },
        {
            "event": "message.any",
            "session": SESSION,
            "payload": {
                "id": f"true_{CHAT_ID}_BOT_{OWN_LID}",
                "timestamp": 200,
                "from": OWN_LID,
                "fromMe": True,
                "to": CHAT_ID,
                "source": "api",
                "participant": OWN_LID,
                # A body longer than the inline list cap stays whole in export.
                "body": "bot reply " + "x" * 260,
            },
        },
        {
            # Same id twice: redelivered backlog must collapse to one row.
            "event": "message.any",
            "session": SESSION,
            "payload": {
                "id": f"false_{CHAT_ID}_MEMBER_491555000001@c.us",
                "timestamp": 301,
                "from": CHAT_ID,
                "fromMe": False,
                "participant": "491555000001@c.us",
                "body": "member asking",
            },
        },
        {
            "event": "message.reaction",
            "session": SESSION,
            "payload": {"id": "react", "to": CHAT_ID, "fromMe": True, "timestamp": 250},
        },
        {
            "event": "message.any",
            "session": SESSION,
            "payload": {"id": "elsewhere", "from": "19999999999-1@g.us", "timestamp": 5},
        },
    ]


def _write_journal(day: Path, lines: list[dict[str, Any]], torn: bool = False) -> None:
    """Write *lines* as the session's daily journal file."""
    with (day / "2026-01-01.jsonl").open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")
        if torn:
            handle.write('{"event": "message.any", "payl')


def _group_waha() -> Any:
    """A stub WAHA with a bare LID roster, whose recent messages carry
    ``notifyName`` for the LID participants."""

    class StubWaha:
        def get_chat_overview(self, _session: str, chat_id: str) -> dict[str, Any]:
            return {
                "id": chat_id,
                "participants": [{"id": "491555000007@lid"}],
            }

        def fetch_chat_messages(
            self,
            _session: str,
            _chat_id: str,
            limit: int = 100,
        ) -> list[dict[str, Any]]:
            return [
                {
                    "participant": {"_serialized": "491555000007@lid"},
                    "_data": {"notifyName": "Milo Petrov"},
                },
                {
                    "participant": {"_serialized": "491555000009@lid"},
                    "_data": {"notifyName": "Tavo Ríos"},
                },
            ]

    return StubWaha()


def _conflicting_waha() -> Any:
    """A stub WAHA whose roster and messages disagree on one JID's name.

    Only the roster's entry may survive — roster names win.
    """

    class ConflictingWaha:
        def get_chat_overview(self, _session: str, chat_id: str) -> dict[str, Any]:
            return {
                "id": chat_id,
                "participants": [{"id": "491555000007@lid", "name": "Roster Name"}],
            }

        def fetch_chat_messages(
            self,
            _session: str,
            _chat_id: str,
            limit: int = 100,
        ) -> list[dict[str, Any]]:
            return [
                {
                    "participant": {"_serialized": "491555000007@lid"},
                    "_data": {"notifyName": "Message Name"},
                }
            ]

    return ConflictingWaha()


def _group_event(
    participant: Any = "491555000007@lid", name: str | None = None
) -> WahaEvent:
    payload: dict[str, Any] = {
        "from": CHAT_ID,
        "participant": participant,
        "body": "hola",
    }
    if name is not None:
        payload["_data"] = {"notifyName": name}
    return WahaEvent(
        id="tag1",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload=payload,
    )


_PURGE_SCRIPT = Path("scripts/purge_leaked_silence.py")


def member_event(**payload: Any) -> WahaEvent:
    base: dict[str, Any] = {
        "from": CHAT_ID,
        "participant": "491555000001@lid",
        "body": "",
    }
    base.update(payload)
    return WahaEvent(
        id="e",
        timestamp=1,
        event="message",
        session=SESSION,
        me={"id": "491555000000@c.us", "lid": OWN_LID},
        payload=base,
    )
