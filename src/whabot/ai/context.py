"""Reply context rendering and the agent entrypoint."""

import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from llama_index.core.base.llms.types import ImageBlock
from llama_index.core.workflow import Context
from loguru import logger

from whabot.ai.messages import message_replies_to
from whabot.ai.url_images import fetch_url_images, image_urls
from whabot.ai.workflow import FunctionCallingAgentWorkflow
from whabot.core.models import WahaEvent
from whabot.settings import Settings

__all__ = [
    "handle_message",
    "render_system_prompt",
    "reply_context",
    "reply_context_section",
    "reply_description",
    "sender_tag",
]


def render_system_prompt(
    prompt: str, tz_name: str = "UTC", bot_name: str | None = None
) -> str:
    """Substitute date/time/name placeholders in the system prompt.

    Supported placeholders (all but ``{{bot_name}}`` use the ``tz_name``
    timezone):

    - ``{{now}}`` / ``{{datetime}}`` — full timestamp, e.g. ``2026-09-02 14:05 UTC``
    - ``{{date}}`` — date only, e.g. ``2026-09-02``
    - ``{{time}}`` — time only, e.g. ``14:05``
    - ``{{tz}}`` — the timezone name, e.g. ``UTC``
    - ``{{bot_name}}`` — the bot's display name, e.g. ``Kai``

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
        "{{bot_name}}": bot_name or "the bot",
    }
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def sender_tag(event: WahaEvent) -> str:
    """The sender's display identity for the agent prompt, e.g. ``[Ana]``.

    Prefers the WhatsApp display name (``_data.notifyName``, then a
    top-level ``notifyName`` for engines that hoist it); falls back to
    the participant/author id so a group turn is never anonymous.
    Returns an empty tag when nothing is known (should not happen).
    """
    data = event.payload.get("_data", {})
    name = str(data.get("notifyName") or event.payload.get("notifyName") or "").strip()
    if name:
        return f"[{name}]"
    participant = event.payload.get("participant") or data.get("author") or ""
    return f"[{participant}]" if participant else ""


def reply_context(message_reply: dict[str, Any] | None) -> str:
    """Render the quoted message as context for the agent.

    Empty/None input yields nothing; the caller decides whether to
    include a ``Reply context`` block in the prompt.
    """
    if not message_reply:
        return ""
    parts = [
        part
        for part in (
            quoted_participant(message_reply),
            reply_description(message_reply),
        )
        if part
    ]
    return "; ".join(parts)


def quoted_participant(message_reply: dict[str, Any]) -> str:
    """The quoted message's sender, prefixed ``from:``; empty when unknown.

    Prefers the quoted message's display name (``_data.notifyName``) over
    the raw participant/author id.
    """
    data = message_reply.get("_data", {})
    participant = (
        data.get("notifyName")
        or message_reply.get("participant")
        or data.get("author")
        or ""
    )
    return f"from: {participant}" if participant else ""


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
    image: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> str:
    """Run the agent workflow over an incoming message event and return its reply.

    ``image`` carries downloaded image bytes (``data`` + ``mimetype``);
    when present, it rides along as ``image_blocks`` on the run and is
    injected into the first LLM call only — memory stays text-only, so
    no megabyte payloads accumulate in the rolling buffer.

    With ``settings.vision`` enabled, image URLs sniffed from the
    message text are fetched too (see ``whabot.ai.url_images``), so a
    bare link like "look at this https://host/pic.png" shows the model
    the picture as well.
    """
    chat_id = str(event.payload.get("from", ""))
    body = str(event.payload.get("body", "")).strip()
    logger.info("Agent handling message from {chat_id}", chat_id=chat_id)
    tag = sender_tag(event)
    text = f"{tag} {body}".strip() if tag else body
    images = [image] if image is not None else []
    if settings is not None and settings.vision:
        urls = image_urls(text, settings.max_url_images)
        images.extend(fetch_url_images(settings, urls))
    if images and not body:
        text = f"{tag} (image)".strip()
    user_msg = text + reply_context_section(message_replies_to(event))
    image_blocks = [
        ImageBlock(
            image=img["data"],
            image_mimetype=img.get("mimetype") or "image/jpeg",
        )
        for img in images
    ]
    result = await agent.run(input=user_msg, image_blocks=image_blocks, ctx=ctx)
    content: str | None = result.message.content
    return content.strip() if content else ""
