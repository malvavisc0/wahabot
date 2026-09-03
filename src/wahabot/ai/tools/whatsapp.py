"""WhatsApp tools for the function calling agent.

Each builder takes the shared mutable ``target`` holder refreshed by the
handler before every agent run with the current ``session`` and default
``chat_id``, so the shared agent's tools always speak for the message
being handled. Tools calling a WAHA endpoint raise on HTTP errors; the
tool functions here return a status string instead, so a failure feeds
back to the model rather than crashing the workflow.
"""

import mimetypes
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from llama_index.core.tools import BaseTool, FunctionTool

from wahabot.ai.tools.schemas import (
    FetchChatMessagesSchema,
    ForwardMessageSchema,
    GetChatSchema,
    ReactToMessageSchema,
    SearchMessagesSchema,
    SendImageSchema,
    SendMessageSchema,
)
from wahabot.core.waha import WahaClient, message_media

__all__ = [
    "fetch_chat_messages",
    "forward_message",
    "get_chat",
    "react_to_message",
    "search_messages",
    "send_image",
    "send_message",
]


#: Extension → MIME mapping for image URLs. Required because naive
#: ``image/{ext}`` synthesis (or hardcoding image/jpeg) sends unregistered
#: types such as ``image/jpg`` and ``image/tif`` which WAHA/WhatsApp reject.
_IMAGE_MIME_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".svg": "image/svg+xml",
    ".avif": "image/avif",
}


def send_message(waha: WahaClient, target: dict[str, str]) -> BaseTool:
    """Build a tool that sends a WhatsApp text message.

    Sends to the current chat by default; the model may pass an explicit
    ``chat`` (a group or a person's JID) to reach a different target.
    """

    def send_message_fn(chat: str | None = None, text: str = "") -> str:
        """Send a WhatsApp text message.

        Args:
            chat: Optional chat id (group or person JID, e.g.
                `1234567890@g.us` or `9876543210@c.us`). Omit to reply
                in the current conversation.
            text: The text to send.
        """
        if not text.strip():
            return "Error: empty message text."
        session = target.get("session", "")
        chat_id = chat or target.get("chat_id", "")
        if not session or not chat_id:
            return "Error: no active conversation context."
        waha.send_text(session, chat_id, text)
        target["sent"] = chat_id
        return f"Sent '{text}' to {chat_id}."

    return FunctionTool.from_defaults(
        fn=send_message_fn,
        fn_schema=SendMessageSchema,
        name="send_message",
        description=(
            "Send a WhatsApp text message. Use this to reply in the "
            "current chat (omit chat) or to message another group or "
            "person (pass chat)."
        ),
    )


def react_to_message(waha: WahaClient, target: dict[str, str]) -> BaseTool:
    """Build a tool that reacts to a WhatsApp message."""

    def react_to_message_fn(message_id: str, reaction: str = "") -> str:
        """React to a message.

        Args:
            message_id: The serialized id of the message to react to
                (e.g. `false_12132132130@c.us_AAAAAAAAAAAAAAAAAAAA`).
            reaction: The emoji to react with, or empty string to remove
                an existing reaction.
        """
        session = target.get("session", "")
        if not session:
            return "Error: no active conversation context."
        if not message_id:
            return "Error: message_id is required."
        waha.send_reaction(session, message_id, reaction)
        return (
            f"Reacted to {message_id} with '{reaction}'."
            if reaction
            else f"Removed reaction from {message_id}."
        )

    return FunctionTool.from_defaults(
        fn=react_to_message_fn,
        fn_schema=ReactToMessageSchema,
        name="react_to_message",
        description=(
            "React with an emoji to a WhatsApp message. Provide the "
            "message's serialized id (use fetch_chat_messages to find "
            "ids). Pass an empty reaction to remove the bot's reaction."
        ),
    )


def send_image(waha: WahaClient, target: dict[str, str]) -> BaseTool:
    """Build a tool that sends an image to a chat."""

    def send_image_fn(
        url: str | None = None,
        caption: str = "",
        chat: str | None = None,
    ) -> str:
        """Send an image.

        Args:
            url: Public URL of the image to send.
            caption: Optional caption text.
            chat: Optional chat id. Omit to send to the current chat.
        """
        session = target.get("session", "")
        chat_id = chat or target.get("chat_id", "")
        if not session or not chat_id:
            return "Error: no active conversation context."
        if not url:
            return "Error: url is required."
        waha.send_image(
            session,
            chat_id,
            file={"mimetype": infer_image_mimetype(url), "url": url},
            caption=caption,
        )
        caption_text = f" with caption: {caption}" if caption else ""
        return f"Sent image{caption_text} to {chat_id}."

    return FunctionTool.from_defaults(
        fn=send_image_fn,
        fn_schema=SendImageSchema,
        name="send_image",
        description=(
            "Send an image to a WhatsApp chat from a public URL. "
            "Pass chat to reach another group/person, else sends to the "
            "current chat. caption is optional."
        ),
    )


def infer_image_mimetype(url: str) -> str:
    """Best-effort image mimetype from a URL's path extension.

    Prefers a curated extension→MIME map (avoids unregistered types such
    as ``image/jpg`` and ``image/tif``), then ``mimetypes``, then falls
    back to ``image/jpeg`` when nothing is known.
    """
    path = PurePosixPath(urlsplit(url).path)
    ext = path.suffix.lower()
    mapped = _IMAGE_MIME_BY_EXT.get(ext)
    if mapped:
        return mapped
    guessed = mimetypes.guess_type(path.name or "")[0]
    return guessed if guessed and guessed.startswith("image/") else "image/jpeg"


def fetch_chat_messages(waha: WahaClient, target: dict[str, str]) -> BaseTool:
    """Build a tool that fetches recent messages from a chat."""

    def fetch_chat_messages_fn(chat: str | None = None, limit: int = 20) -> str:
        """Fetch recent messages from a chat.

        Args:
            chat: Optional chat id. Omit to fetch from the current chat.
            limit: Max messages to return (default 20).
        """
        session = target.get("session", "")
        chat_id = chat or target.get("chat_id", "")
        if not session or not chat_id:
            return "Error: no active conversation context."
        messages = waha.fetch_chat_messages(session, chat_id, limit=limit)
        lines: list[str] = []
        for m in messages[:limit]:
            who = (
                "me"
                if m.get("fromMe")
                else str(m.get("participant") or m.get("from", ""))
            )
            body = str(m.get("body", "") or "").strip()[:200]
            message_id = str(m.get("id", ""))
            if not body:
                media = message_media(m)
                mimetype = str(media.get("mimetype", "media") or "media")
                filename = media.get("filename") or ""
                body = f"[{mimetype}] {filename}".strip()
            lines.append(f"[id:{message_id}] {who}: {body}")
        return "\n".join(lines) if lines else "(no messages found)"

    return FunctionTool.from_defaults(
        fn=fetch_chat_messages_fn,
        fn_schema=FetchChatMessagesSchema,
        name="fetch_chat_messages",
        description=(
            "Fetch the most recent messages of a chat as text lines, each "
            "prefixed with its `[id:...]` serialized message id. Use to "
            "read what has been said in the current or another chat; the "
            "ids let you forward or react to a message. limit caps lines."
        ),
    )


def get_chat(waha: WahaClient, target: dict[str, str]) -> BaseTool:
    """Build a tool that returns metadata about a chat."""

    def get_chat_fn(chat: str | None = None) -> str:
        """Get metadata about a chat (name, participants count, ...).

        Args:
            chat: Optional chat id. Omit for the current chat.
        """
        session = target.get("session", "")
        chat_id = chat or target.get("chat_id", "")
        if not session or not chat_id:
            return "Error: no active conversation context."
        overview = waha.get_chat_overview(session, chat_id)
        if not overview:
            return f"No metadata found for {chat_id}."
        return summarize_chat(chat_id, overview)

    return FunctionTool.from_defaults(
        fn=get_chat_fn,
        fn_schema=GetChatSchema,
        name="get_chat",
        description=(
            "Get metadata (name, participants count, group flags, unread "
            "count) about a WhatsApp chat. Omit chat for the current chat."
        ),
    )


def summarize_chat(chat_id: str, overview: dict[str, Any]) -> str:
    """Render a compact, model-friendly summary of a chat overview dict.

    Extracts the stable scalar fields and the participant JIDs, skipping
    nested blobs like ``lastMessage`` and ``picture`` that carry no
    useful metadata for the model.
    """
    scalar = _chat_scalars(overview)
    _add_participant_summary(scalar, overview)
    if _needs_raw_fallback(scalar):
        scalar["chat_id"] = chat_id
        scalar["_raw"] = str(overview)[:1000]
    return "\n".join(f"{key}: {value}" for key, value in scalar.items())


def _chat_scalars(overview: dict[str, Any]) -> dict[str, Any]:
    """Stable scalar chat fields, skipping empty/absent values."""
    return {
        key: overview[key]
        for key in (
            "id",
            "name",
            "isReadOnly",
            "isGroup",
            "muted",
            "archived",
            "pinned",
            "unreadCount",
        )
        if overview.get(key) not in (None, "", False)
    }


def _add_participant_summary(scalar: dict[str, Any], overview: dict[str, Any]) -> None:
    """Add the participant count and (when small) the JIDs to *scalar*.

    WAHA returns participants either at top level or under ``_chat``, as
    plain JID strings or as ``{"id": ...}`` objects depending on engine.
    """
    participants: Any = overview.get("participants")
    if not isinstance(participants, list):
        chat_blob: Any = overview.get("_chat")
        if isinstance(chat_blob, dict):
            blob: dict[str, Any] = chat_blob
            participants = blob.get("participants")
        else:
            participants = None
    if not isinstance(participants, list):
        return
    scalar["participants"] = len(participants)
    jids = [_participant_jid(p) for p in participants]
    jids = [j for j in jids if j]
    if 0 < len(jids) <= 20:
        scalar["participant_jids"] = ", ".join(jids)


def _participant_jid(participant: Any) -> str:
    """The JID of one participant entry (string or ``{"id": ...}`` object)."""
    if isinstance(participant, dict):
        entry: dict[str, Any] = participant
        return str(entry.get("id") or "")
    return str(participant or "")


def _needs_raw_fallback(scalar: dict[str, Any]) -> bool:
    """True when the overview yielded almost nothing useful."""
    return not scalar or all(key == "id" for key in scalar)


def search_messages(waha: WahaClient, target: dict[str, str]) -> BaseTool:
    """Build a tool that searches recent messages for text."""

    def search_messages_fn(
        query: str,
        chat: str | None = None,
        limit: int = 20,
    ) -> str:
        """Search a chat's recent messages containing a text substring.

        Searches body text, media filenames and mimetypes.

        Args:
            query: The text to look for.
            chat: Optional chat id to scope the search. Omit to search
                the current chat.
            limit: Max matches to return (default 20).
        """
        session = target.get("session", "")
        chat_id = chat or target.get("chat_id", "")
        if not session or not chat_id:
            return "Error: no active conversation context."
        if not query.strip():
            return "Error: query is required."
        messages = waha.search_messages(session, query, chat_id, limit=limit)
        return _format_matches(messages[:limit])

    return FunctionTool.from_defaults(
        fn=search_messages_fn,
        fn_schema=SearchMessagesSchema,
        name="search_messages",
        description=(
            "Search a chat's recent messages for a text substring in body, "
            "media filename or mimetype. Returns matching message lines. "
            "Pass chat to search another chat, else the current one."
        ),
    )


def _format_matches(messages: list[dict[str, Any]]) -> str:
    """Render search hits as ``chat_id: body`` lines for the model."""
    lines: list[str] = []
    for m in messages:
        chat_id = str(m.get("chatId") or _remote_chat_id(m))
        body = str(m.get("body", "") or "").strip()[:200]
        if not body:
            media = message_media(m)
            mimetype = str(media.get("mimetype", "media") or "media")
            filename = media.get("filename") or ""
            body = f"[{mimetype}] {filename}".strip()
        lines.append(f"{chat_id}: {body}")
    return "\n".join(lines) if lines else "(no matches)"


def _remote_chat_id(message: dict[str, Any]) -> str:
    """The chat id buried at ``_data.id.remote``, when present."""
    data = message.get("_data")
    if not isinstance(data, dict):
        return ""
    data_dict: dict[str, Any] = data
    data_id = data_dict.get("id")
    if isinstance(data_id, dict):
        remote: dict[str, Any] = data_id
        return str(remote.get("remote", ""))
    return ""


def forward_message(waha: WahaClient, target: dict[str, str]) -> BaseTool:
    """Build a tool that forwards a message to a chat."""

    def forward_message_fn(message_id: str, chat: str | None = None) -> str:
        """Forward a message to a chat.

        Args:
            message_id: The serialized id of the message to forward.
            chat: Optional chat id to forward into. Defaults to current.
        """
        session = target.get("session", "")
        chat_id = chat or target.get("chat_id", "")
        if not session or not chat_id:
            return "Error: no active conversation context."
        if not message_id:
            return "Error: message_id is required."
        waha.forward_message(session, chat_id, message_id)
        return f"Forwarded {message_id} to {chat_id}."

    return FunctionTool.from_defaults(
        fn=forward_message_fn,
        fn_schema=ForwardMessageSchema,
        name="forward_message",
        description=(
            "Forward an existing WhatsApp message (by its serialized id) "
            "to a chat. Pass chat to choose the destination, else current."
        ),
    )
