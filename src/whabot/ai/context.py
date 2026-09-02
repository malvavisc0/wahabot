"""Reply context rendering and the agent entrypoint."""

import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from llama_index.core.workflow import Context
from loguru import logger

from whabot.ai.messages import message_replies_to
from whabot.ai.workflow import FunctionCallingAgentWorkflow
from whabot.core.models import WahaEvent

__all__ = [
    "handle_message",
    "render_system_prompt",
    "reply_context",
    "reply_context_section",
    "reply_description",
]


def render_system_prompt(prompt: str, tz_name: str = "UTC") -> str:
    """Substitute date/time placeholders in the system prompt.

    Supported placeholders (all use the ``tz_name`` timezone):

    - ``{{now}}`` / ``{{datetime}}`` — full timestamp, e.g. ``2026-09-02 14:05 UTC``
    - ``{{date}}`` — date only, e.g. ``2026-09-02``
    - ``{{time}}`` — time only, e.g. ``14:05``
    - ``{{tz}}`` — the timezone name, e.g. ``UTC``

    Unknown/invalid timezone names fall back to UTC.
    """
    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError, ValueError, OSError:
        tz = datetime.UTC

    now = datetime.datetime.now(tz=tz)
    replacements = {
        "{{now}}": now.strftime("%Y-%m-%d %H:%M %Z"),
        "{{datetime}}": now.strftime("%Y-%m-%d %H:%M %Z"),
        "{{date}}": now.strftime("%Y-%m-%d"),
        "{{time}}": now.strftime("%H:%M"),
        "{{tz}}": tz_name,
    }
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def reply_context(message_reply: dict[str, Any] | None) -> str:
    """Render the quoted message as context for the agent.

    Empty/None input yields nothing; the caller decides whether to
    include a ``Reply context`` block in the prompt.
    """
    if not message_reply:
        return ""
    id_ = message_reply.get("id") or message_reply.get("stanzaId") or ""
    participant = (
        message_reply.get("participant")
        or message_reply.get("_data", {}).get("author")
        or ""
    )
    description = reply_description(message_reply)
    parts: list[str] = []
    if participant:
        parts.append(f"from: {participant}")
    if id_:
        parts.append(f"id: {id_}")
    if description:
        parts.append(description)
    return "; ".join(parts)


def reply_description(message_reply: dict[str, Any]) -> str:
    """Describe the quoted message body/media, truncated for the prompt."""
    media = message_reply.get("hasMedia") and message_reply.get("media")
    if media:
        mimetype = media.get("mimetype", "media")
        filename = media.get("filename") or ""
        description = f"{mimetype} {filename}".strip()
    else:
        description = str(message_reply.get("body", "") or "").strip()
    return description[:400]


def reply_context_section(message_reply: dict[str, Any] | None) -> str:
    """Message text with the quoted/replied-to message attached as context."""
    if not message_reply:
        return ""
    line = reply_context(message_reply)
    return f"\n[Message quoting] {line}" if line else ""


async def handle_message(
    event: WahaEvent,
    agent: FunctionCallingAgentWorkflow,
    ctx: Context | None = None,
) -> str:
    """Run the agent workflow over an incoming message event and return its reply."""
    chat_id = str(event.payload.get("from", ""))
    body = str(event.payload.get("body", ""))
    logger.info("Agent handling message from {chat_id}", chat_id=chat_id)
    user_msg = body + reply_context_section(message_replies_to(event))
    result = await agent.run(input=user_msg, ctx=ctx)
    content: str | None = result.message.content
    return content.strip() if content else ""
