"""Helpers for parsing WAHA message payloads."""

from typing import Any


def chat_id_from_message_id(message_id: str) -> str:
    """Return the chat JID embedded in a serialized message id.

    Serialized ids have the form ``{fromMe}_{chat}_{message_id}[_{participant}]``
    and chat JIDs never contain underscores, so the chat is the second segment.
    """
    parts = message_id.split("_")
    return parts[1] if len(parts) > 1 else message_id


def message_preview(payload: dict[str, Any]) -> str:
    """Return a short human-readable preview of a message, for logs."""
    body = str(payload.get("body", "")).strip()
    if body:
        return body if len(body) <= 80 else body[:77] + "..."
    data = payload.get("_data", {})
    kind = data.get("type")
    return f"[{kind}]" if kind else "[media]"
