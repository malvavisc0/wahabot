"""Reply context rendering and the agent entrypoint."""

import datetime
import json
import re
import time
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from llama_index.core.base.llms.types import ChatMessage, ImageBlock
from llama_index.core.workflow import Context
from loguru import logger

from wahabot.ai.messages import message_replies_to
from wahabot.ai.tools.url_images import fetch_url_images, image_urls
from wahabot.ai.workflow import FunctionCallingAgentWorkflow
from wahabot.core.models import WahaEvent
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings

__all__ = [
    "final_reply",
    "handle_message",
    "is_error_narration",
    "is_silence_narration",
    "participant_names",
    "render_system_prompt",
    "reply_context",
    "reply_context_section",
    "sender_tag",
]

#: Replies that narrate a chosen silence instead of being one. Small
#: models asked to "reply with an empty string to stay silent" often
#: answer with meta-commentary ("I'll stay silent here — ...", "No
#: response.") — matching it here keeps it out of the chat. The reply
#: must *be about* staying silent, not merely contain the word (so
#: "silence is golden, but I'll answer anyway" still goes through).
#: The same happens after a delivered reaction or reply: the model
#: pattern-completes "I already reacted to that message, so I'm done
#: here." instead of going quiet — the reaction/reply already went out
#: via the tool, so the narration is chatter, not an answer.
_SILENCE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^no response\b",
        r"^no reply\b",
        r"^nothing to (add|say)\b",
        r"^nothing (more|else) to (add|say|do)\b",
        r"^(i'?ll |i will |i'?m )?(stay|staying|remain|choosing to stay)[\s'-]*silent\b",
        r"^(i'?ll |i will )?(stay|keep) (quiet|out of (this|it|the conversation))\b",
        r"^silence[.!…]?$",
        r"^\(silence\)$",
        r"^not (addressed|directed) (to|at) me\b",
        r"^(i'?ll |i will )?say nothing\b",
        r"^i (already )?(reacted|replied|sent|answered)\b[^.]*" + r"(so )?i'?m done\b",
        r"^i (already )?(reacted|replied|sent|answered)\b[^.]*"
        + r"(so )?(there'?s?|there is) nothing (more|else|left) (to )?(add|say|do)",
        r"^i'?m done (here|with this)\b",
    )
)


def is_silence_narration(reply: str) -> bool:
    """True when *reply* narrates a silence instead of being one.

    Stripped of surrounding whitespace/quotes/parentheses and matched
    case-insensitively against the silence-meta patterns; anything the
    model actually wanted to say still goes through.
    """
    cleaned = reply.strip().strip("\"'`()").strip()
    return any(pattern.search(cleaned) for pattern in _SILENCE_PATTERNS)


def is_error_narration(reply: str) -> bool:
    """True when *reply* is an error payload, not a chat answer.

    Small models sometimes *write* an API error as their reply — e.g.
    a made-up ``{"error": {"message": "resource exhausted …", "type":
    "upstream_error", "code": "resource_exhausted"}}`` naming a provider
    the bot never used. Whatever the model's reason (pattern-
    completing text it has seen), the result must never reach the
    chat. Only near-JSON bodies whose top level is an ``error`` object
    match; genuine prose answers never do.
    """
    stripped = reply.strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError, ValueError:
        return False
    if not isinstance(data, dict):
        return False
    error = cast(dict[str, Any], data).get("error")
    return isinstance(error, dict) and any(
        key in error for key in ("message", "code", "type", "status")
    )


def render_system_prompt(
    prompt: str,
    tz_name: str = "UTC",
    bot_name: str | None = None,
    goal: str = "",
) -> str:
    """Substitute date/time/name placeholders in the system prompt.

    When ``goal`` is non-empty, it is prepended as a ``Goal:`` block so
    the model always starts from the bot's intended purpose.

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
        goal = goal.replace(key, value)
    goal = goal.strip()
    if goal:
        return f"Goal: {goal}\n\n{prompt}"
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


def message_id_note(event: WahaEvent) -> str:
    """The incoming message's serialized id as a prompt note.

    The id is what ``send_message(reply_to=…)`` and ``react_to_message``
    need to quote or react to *this* message — without it in the turn
    text the model can only quote ids it fetched itself.
    """
    data = event.payload.get("_data", {})
    message_id = str(data.get("id", {}).get("_serialized", "")) if data else ""
    if not message_id:
        message_id = str(event.payload.get("id", ""))
    return f"\n[message id: {message_id}]" if message_id else ""


def reply_context(
    message_reply: dict[str, Any] | None,
    participant_names: dict[str, str] | None = None,
) -> str:
    """Render the quoted message as context for the agent.

    Empty/None input yields nothing; the caller decides whether to
    include a ``Reply context`` block in the prompt. *participant_names*
    maps chat JIDs to display names so the quoted sender renders as a
    name, not a raw ``@lid`` JID.
    """
    if not message_reply:
        return ""
    sender = quoted_participant(message_reply, participant_names)
    description = reply_description(message_reply)
    if sender and description:
        return f'{sender}: "{description}"'
    return sender or description


def quoted_participant(
    message_reply: dict[str, Any],
    participant_names: dict[str, str] | None = None,
) -> str:
    """The quoted message's sender display name; empty when unknown.

    Prefers the quoted message's own ``_data.notifyName``, then the
    chat's participant roster (*participant_names*), then the raw
    participant/author id stripped of its ``@…`` domain — a bare number
    reads like an id, a full JID reads like noise.
    """
    data = message_reply.get("_data", {})
    jid = str(message_reply.get("participant") or data.get("author") or "")
    name = str(data.get("notifyName") or "").strip()
    if not name and jid and participant_names:
        name = participant_names.get(jid, "")
    if name:
        return name
    return jid.split("@", 1)[0] if jid else ""


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


def reply_context_section(
    message_reply: dict[str, Any] | None,
    participant_names: dict[str, str] | None = None,
) -> str:
    """Message text with the quoted/replied-to message attached as context."""
    if not message_reply:
        return ""
    line = reply_context(message_reply, participant_names)
    return f"\n[quoting] {line}" if line else ""


#: Per-chat participant roster cache (JID → display name), refreshed
#: at most once per TTL. Quoted messages carry no sender name (WAHA's
#: ``replyTo._data`` has only type/kind/body), so the roster is the only
#: way to render "Ada" instead of "000000000000000@lid".
ROSTER_TTL_S = 3600
roster_cache: dict[tuple[str, str], tuple[float, dict[str, str]]] = {}


def participant_names(
    waha: WahaClient | None, session: str, chat_id: str
) -> dict[str, str]:
    """JID → display name for a chat's participants, cached per chat.

    Fails soft: any WAHA error yields an empty map and quoted senders
    fall back to their bare id. Non-group chats skip the lookup — a DM
    partner's name already rides the sender tag.
    """
    if waha is None or not chat_id.endswith("@g.us"):
        return {}
    now = time.monotonic()
    cached = roster_cache.get((session, chat_id))
    if cached and now - cached[0] < ROSTER_TTL_S:
        return cached[1]
    try:
        overview = waha.get_chat_overview(session, chat_id)
    except Exception as exc:
        logger.debug(
            "Participant roster fetch failed for {chat}: {exc}", chat=chat_id, exc=exc
        )
        return {}
    names = roster_names(overview)
    roster_cache[(session, chat_id)] = (now, names)
    return names


def roster_names(overview: dict[str, Any]) -> dict[str, str]:
    """Extract JID → name from a chat overview's participant list."""
    pairs = (roster_entry(entry) for entry in roster_entries(overview))
    return {jid: name for jid, name in pairs if jid and name}


def roster_entries(overview: dict[str, Any]) -> list[Any]:
    """The participant list, whether top-level or nested under ``_chat``."""
    participants: Any = overview.get("participants")
    if not isinstance(participants, list):
        blob: Any = overview.get("_chat")
        nested = cast(dict[str, Any], blob) if isinstance(blob, dict) else {}
        participants = nested.get("participants")
    return participants if isinstance(participants, list) else []


def roster_entry(entry: Any) -> tuple[str, str]:
    """One participant's ``(jid, display_name)``; empty strings when absent."""
    if not isinstance(entry, dict):
        return "", ""
    item: dict[str, Any] = entry
    jid = str(item.get("id") or "")
    name = str(
        item.get("name") or item.get("pushname") or item.get("notifyName") or ""
    ).strip()
    return jid, name


async def handle_message(
    event: WahaEvent,
    agent: FunctionCallingAgentWorkflow,
    ctx: Context | None = None,
    image: dict[str, Any] | None = None,
    settings: Settings | None = None,
    waha: WahaClient | None = None,
) -> str:
    """Run the agent workflow over an incoming message event and return its reply.

    ``image`` carries downloaded image bytes (``data`` + ``mimetype``);
    when present, it rides along as ``image_blocks`` on the run and is
    injected into the first LLM call only — memory stays text-only, so
    no megabyte payloads accumulate in the rolling buffer.

    With ``settings.vision`` enabled, image URLs sniffed from the
    message text are fetched too (see ``wahabot.ai.tools.url_images``), so a
    bare link like "look at this https://host/pic.png" shows the model
    the picture as well.
    """
    chat_id = str(event.payload.get("from", ""))
    body = str(event.payload.get("body", "")).strip()
    logger.info("Agent handling message from {chat_id}", chat_id=chat_id)
    tag = sender_tag(event)
    text = f"{tag} {body}".strip() if tag else body
    images = collect_images(image, settings, text)
    if images and not body:
        text = f"{tag} (image)".strip()
    names = participant_names(waha, event.session, chat_id)
    user_msg = text + message_id_note(event)
    user_msg += reply_context_section(message_replies_to(event), names)
    image_blocks = [
        ImageBlock(
            image=img["data"],
            image_mimetype=img.get("mimetype") or "image/jpeg",
        )
        for img in images
    ]
    result = await agent.run(input=user_msg, image_blocks=image_blocks, ctx=ctx)
    return final_reply(result)


def collect_images(
    image: dict[str, Any] | None,
    settings: Settings | None,
    text: str,
) -> list[dict[str, Any]]:
    """The message's images: the attached one plus sniffed URL images."""
    images = [image] if image is not None else []
    if settings is not None and settings.vision:
        urls = image_urls(text, settings.max_url_images)
        images.extend(fetch_url_images(settings, urls))
    return images


def final_reply(result: Any) -> str:
    """The run's reply text, emptied when the model narrated instead of answered.

    Silence narration ("I'll stay silent here — ...") and invented
    error payloads (``{"error": {...}}``) are chatter, not answers:
    both are dropped so they never reach the chat.
    """
    message = cast(ChatMessage, result.message)
    content = message.content
    reply = content.strip() if isinstance(content, str) else ""
    if is_silence_narration(reply):
        logger.debug("Filtering silence narration: {reply!r}", reply=reply)
        return ""
    if is_error_narration(reply):
        logger.warning(
            "Dropping invented error payload as final reply: {reply!r}", reply=reply[:200]
        )
        return ""
    return reply
