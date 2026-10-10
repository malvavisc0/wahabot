"""Journal of raw webhook events, one JSON object per line per day."""

import datetime
import json
from pathlib import Path
from typing import Any

from loguru import logger


def save_event(events_dir: Path, session: str, body: bytes) -> None:
    """Append the raw event body to the session's daily journal file.

    Files land at `<events_dir>/<session>/<YYYY-MM-DD>.jsonl`, one JSON
    object per line — greppable and replayable as-is. The body is the
    exact bytes WAHA sent (valid UTF-8 by the protocol).
    """
    day = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
    session_dir = events_dir / session
    session_dir.mkdir(parents=True, exist_ok=True)
    with (session_dir / f"{day}.jsonl").open("ab") as journal:
        journal.write(body + b"\n")


def chat_messages(events_dir: Path, session: str, chat_id: str) -> list[dict[str, Any]]:
    """The chat's recorded ``message.any`` payloads, deduped, oldest first.

    A member's message has ``from`` = the chat; a message written on
    the shared account (bot or operator) has ``to`` = the chat with
    ``fromMe`` true — one direction filter loses the account, so both
    sides are collected. Re-delivered backlog collapses to one row per
    message id. A line torn by a crash is skipped with a debug log
    (the journal appends whole events; a partial tail line would
    otherwise kill every later read of the file). Only what the
    journal recorded is returned: gaps exist wherever the service was
    down, and WAHA's own window may differ.
    """
    session_dir = events_dir / session
    seen: set[str] = set()
    payloads: list[tuple[float, dict[str, Any]]] = []
    for path in sorted(session_dir.glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            try:
                event: dict[str, Any] = json.loads(line)
            except ValueError:
                logger.debug("skipping torn journal line in {path}", path=path)
                continue
            payload = chat_event_payload(event, chat_id)
            message_id = str(payload.get("id") or "") if payload else ""
            if payload is None or message_id in seen:
                continue
            seen.add(message_id)
            payloads.append((float(payload.get("timestamp") or 0), payload))
    return [payload for _, payload in sorted(payloads, key=lambda row: row[0])]


def chat_event_payload(event: dict[str, Any], chat_id: str) -> dict[str, Any] | None:
    """The event's message payload when it belongs to *chat_id*, else None.

    A member's message has ``from`` = the chat; a message written on
    the shared account has ``to`` = the chat with ``fromMe`` true.
    Reactions and non-``message.any`` events never qualify.
    """
    payload: dict[str, Any] = event.get("payload") or {}
    if event.get("event") != "message.any":
        return None
    in_chat = (payload.get("to") == chat_id and payload.get("fromMe")) or (
        payload.get("from") == chat_id and not payload.get("fromMe")
    )
    return payload if in_chat else None
