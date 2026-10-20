"""Tests for the per-chat serial registry (``wahabot.ai.serials``)."""

from __future__ import annotations

from pathlib import Path

from wahabot.ai.serials import (
    SerialEntry,
    SerialRegistry,
    format_serial_token,
)


def test_assigns_monotonic_serials(tmp_path: Path) -> None:
    registry = SerialRegistry(tmp_path / "chat.ids.jsonl")
    assert registry.next(message_id="a", name="Ana") == 1
    assert registry.next(message_id="b", name="Ben") == 2
    assert registry.max_serial() == 2


def test_redelivery_returns_the_same_serial(tmp_path: Path) -> None:
    registry = SerialRegistry(tmp_path / "chat.ids.jsonl")
    first = registry.next(message_id="a")
    assert registry.next(message_id="a") == first
    assert registry.max_serial() == 1


def test_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "chat.ids.jsonl"
    SerialRegistry(path).next(message_id="a", jid="1@lid", name="Ana")
    reopened = SerialRegistry(path)
    assert reopened.max_serial() == 1
    assert reopened.next(message_id="b") == 2
    entry = reopened.resolve("#m1")
    assert entry == SerialEntry(1, "a", "1@lid", "Ana")


def test_resolve_rejects_non_tokens_and_unknowns(tmp_path: Path) -> None:
    registry = SerialRegistry(tmp_path / "chat.ids.jsonl")
    registry.next(message_id="a")
    assert registry.resolve("not-a-token") is None
    assert registry.resolve("491555@lid") is None
    assert registry.resolve("#m99") is None
    assert registry.resolve(format_serial_token(1)) is not None


def test_torn_tail_line_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "chat.ids.jsonl"
    registry = SerialRegistry(path)
    registry.next(message_id="a")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"m": 2, "id": "b"')  # truncated JSON
    reopened = SerialRegistry(path)
    assert reopened.max_serial() == 1
    assert reopened.next(message_id="c") == 2


def test_compact_keeps_the_newest_entries(tmp_path: Path) -> None:
    path = tmp_path / "chat.ids.jsonl"
    registry = SerialRegistry(path)
    for index in range(5):
        registry.next(message_id=f"m{index}")
    registry.compact(keep=2)
    assert registry.max_serial() == 5
    assert registry.resolve("#m4") is not None
    assert registry.resolve("#m2") is None
    assert len(path.read_text().splitlines()) == 2
