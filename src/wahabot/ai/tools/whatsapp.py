"""WhatsApp tools for the function calling agent.

Each builder takes the shared mutable ``target`` holder refreshed by the
handler before every agent run with the current ``session`` and default
``chat_id``, so the shared agent's tools always speak for the message
being handled. Tools calling a WAHA endpoint raise on HTTP errors; the
tool functions here return the shared JSON envelope instead, so a
failure feeds back to the model rather than crashing the workflow.
"""

import json
import mimetypes
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from llama_index.core.tools import BaseTool, FunctionTool
from loguru import logger

from wahabot.ai.messages import jid_string
from wahabot.ai.tools.envelope import error, ok
from wahabot.ai.tools.schemas import (
    FetchChatMessagesSchema,
    ForwardMessageSchema,
    GetChatSchema,
    ReactToMessageSchema,
    ResolveChatSchema,
    SearchMessagesSchema,
    SendImageSchema,
    SendMessageSchema,
    StaySilentSchema,
)
from wahabot.core.waha import WahaClient

__all__ = [
    "chat_jid",
    "fetch_chat_messages",
    "forward_message",
    "get_chat",
    "participant_jid",
    "react_to_message",
    "resolve_chat",
    "roster_entries",
    "search_matches",
    "search_messages",
    "send_image",
    "send_message",
    "sender_names",
    "slim_message",
    "stay_silent",
    "summarize_chat",
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


def chat_jid(chat: str | None, target: dict[str, str]) -> str:
    """The chat JID to act on: *chat*, else the current conversation.

    Models sometimes pass a serialized message id (``false_<jid>_…``,
    scraped from a ``[message id: …]`` annotation) instead of a bare
    JID — strip the ``false_``/``true_`` sender prefix and anything
    after the JID so the call still lands in the right chat.
    """
    value = chat or target.get("chat_id", "")
    for prefix in ("false_", "true_"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    return value.split("_")[0] if "@" in value.split("_")[0] else value


def send_message(waha: WahaClient, target: dict[str, str]) -> BaseTool:
    """Build a tool that sends a WhatsApp text message.

    Sends to the current chat by default; the model may pass an explicit
    ``chat`` (a group or a person's JID) to reach a different target.

    One message per run: once a send succeeds, further calls fail with
    an error envelope instead of sending again. A looping model (the
    same tool call repeated dozens of times) can therefore deliver at
    most one message per incoming event.
    """

    def send_message_fn(
        chat: str | None = None,
        text: str = "",
        reply_to: str | None = None,
        mentions: list[str] | None = None,
    ) -> str:
        """Send a WhatsApp text message.

        Args:
            chat: Optional chat id (group or person JID, e.g.
                `1234567890@g.us` or `9876543210@c.us`). Omit to reply
                in the current conversation.
            text: The text to send.
            reply_to: Optional serialized message id to quote — the
                text goes out as a native quote-reply with that message
                attached.
            mentions: Optional JIDs to @-mention; each mentioned
                person's display name must appear in text as `@<name>`.
        """
        if not text.strip():
            return error("empty message text")
        if target.get("sent"):
            return error(
                f"message already sent this run (to {target['sent']}); do not send again"
            )
        session = target.get("session", "")
        chat_id = chat_jid(chat, target)
        if not session or not chat_id:
            return error("no active conversation context")
        waha.send_text(session, chat_id, text, reply_to=reply_to, mentions=mentions)
        target["sent"] = chat_id
        fields: dict[str, Any] = {
            "chat": chat_id,
            "text": text,
            "mentions": mentions or [],
        }
        if mentions and "@" not in text:
            fields["warning"] = (
                "no `@name` in the text — WhatsApp pairs each mention JID "
                "with an `@<name>` token, so nobody was notified"
            )
        return ok(**fields)

    return FunctionTool.from_defaults(
        fn=send_message_fn,
        fn_schema=SendMessageSchema,
        name="send_message",
        description=(
            "Send a WhatsApp text message. Use this to reply in the "
            "current chat (omit chat) or to message another group or "
            "person (pass chat). To answer a specific message, pass its "
            "id as reply_to — the incoming message's own id rides the "
            "turn as [message id: …], others come from "
            "fetch_chat_messages. To @-mention someone (real highlight "
            "+ notification), pass their JID in mentions and write "
            "@<their name> in the text. Send at most once per run."
        ),
    )


def stay_silent() -> BaseTool:
    """Build the explicit silence tool: choose to not reply at all.

    Models follow a tool call far more reliably than the "reply with
    an empty string" instruction — without this, small models narrate
    their silence ("I'll stay silent here — ...") and the narration is
    sent to the chat as a normal reply.
    """

    def stay_silent_fn() -> str:
        """Stay silent in this conversation (send nothing)."""
        return ok()

    return FunctionTool.from_defaults(
        fn=stay_silent_fn,
        fn_schema=StaySilentSchema,
        name="stay_silent",
        description=(
            "Stay silent: say nothing in this chat. Call this instead of "
            "replying when the message needs no answer (not addressed to "
            "you, nothing useful to add). Never combine with send_message."
        ),
    )


def react_to_message(waha: WahaClient, target: dict[str, str]) -> BaseTool:
    """Build a tool that reacts to a WhatsApp message.

    One reaction per run: like the send tools, a successful reaction
    latches the holder and further calls fail with an error envelope —
    a looping model (the same react call repeated) cannot spam emoji.
    """

    def react_to_message_fn(message_id: str, reaction: str = "") -> str:
        """React to a message.

        Args:
            message_id: The serialized id of the message to react to
                (e.g. `false_12132132130@c.us_AAAAAAAAAAAAAAAAAAAA`).
            reaction: The emoji to react with, or empty string to remove
                an existing reaction.
        """
        if target.get("reacted"):
            return error("already reacted this run; do not react again")
        session = target.get("session", "")
        if not session:
            return error("no active conversation context")
        if not message_id:
            return error("message_id is required")
        waha.send_reaction(session, message_id, reaction)
        target["reacted"] = message_id
        return ok(message_id=message_id, reaction=reaction, removed=not reaction)

    return FunctionTool.from_defaults(
        fn=react_to_message_fn,
        fn_schema=ReactToMessageSchema,
        name="react_to_message",
        description=(
            "React with an emoji to a WhatsApp message. Provide the "
            "message's serialized id (use fetch_chat_messages to find "
            "ids). Pass an empty reaction to remove the bot's reaction. "
            "React at most once per run."
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
        if target.get("sent"):
            return error(
                f"message already sent this run (to {target['sent']}); do not send again"
            )
        session = target.get("session", "")
        chat_id = chat_jid(chat, target)
        if not session or not chat_id:
            return error("no active conversation context")
        if not url:
            return error("url is required")
        mimetype = infer_image_mimetype(url)
        waha.send_image(
            session,
            chat_id,
            file={"mimetype": mimetype, "url": url},
            caption=caption,
        )
        target["sent"] = chat_id
        return ok(chat=chat_id, url=url, mimetype=mimetype, caption=caption)

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


def slim_message(message: dict[str, Any], max_body: int = 200) -> dict[str, Any]:
    """The model-relevant fields of a WAHA message, without the noise.

    WAHA messages carry a raw ``_data`` blob (messageSecret,
    reportingToken, engine flags — ~90% of the payload) that is useless
    to the model and inflates every tool result. Slimmed messages keep
    valid JSON and stay small enough for the memory budget. Message
    bodies are capped at *max_body* chars — a huge paste cannot push
    one message past the whole result budget.
    """
    keys = ("id", "timestamp", "from", "fromMe", "participant", "body", "hasMedia", "ack")
    slimmed = {key: message[key] for key in keys if message.get(key) is not None}
    body = slimmed.get("body")
    if isinstance(body, str) and len(body) > max_body:
        slimmed["body"] = body[:max_body] + "…"
        slimmed["body_truncated"] = True
    return slimmed


#: Whole-message budget for list-tool envelopes, in serialized chars.
#: Kept under ``MAX_TOOL_RESULT_TOKENS`` (2000) so the workflow-level
#: hard cap never mangles the envelope: list results are trimmed to
#: whole messages *before* serialization and stay parseable JSON.
_LIST_ENVELOPE_BUDGET = 1800


def fit_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """The most recent whole messages that fit the envelope budget.

    Newest messages are the most useful, so the list keeps the tail.
    ``count`` stays the total fetched (the model should know how much
    exists); ``returned`` is what actually fits, and ``truncated``
    flags the cut. The envelope is always valid JSON.
    """
    kept: list[dict[str, Any]] = []
    used = 0
    for message in reversed(messages):
        item = json.dumps(message, ensure_ascii=False)
        if kept and used + len(item) > _LIST_ENVELOPE_BUDGET:
            break
        kept.append(message)
        used += len(item)
    kept.reverse()
    return {
        "messages": kept,
        "count": len(messages),
        "returned": len(kept),
        "truncated": len(kept) < len(messages),
    }


def fetch_chat_messages(waha: WahaClient, target: dict[str, str]) -> BaseTool:
    """Build a tool that fetches recent messages from a chat."""

    def fetch_chat_messages_fn(chat: str | None = None, limit: int = 20) -> str:
        """Fetch recent messages from a chat.

        Args:
            chat: Optional chat id. Omit to fetch from the current chat.
            limit: Max messages to return (default 20).
        """
        session = target.get("session", "")
        chat_id = chat_jid(chat, target)
        if not session or not chat_id:
            return error("no active conversation context")
        messages = waha.fetch_chat_messages(session, chat_id, limit=limit)
        return ok(chat=chat_id, **fit_messages([slim_message(m) for m in messages]))

    return FunctionTool.from_defaults(
        fn=fetch_chat_messages_fn,
        fn_schema=FetchChatMessagesSchema,
        name="fetch_chat_messages",
        description=(
            "Fetch the most recent messages of a chat. Returns a JSON "
            "envelope with `messages` (each carrying its serialized `id`, "
            "`body`, sender and media info): `count` is how many were "
            "found, `returned` how many fit (oldest are dropped when "
            "`truncated` is true — raise limit to look further back). "
            "Use to read the current or another chat; the ids let you "
            "forward or react to a message. limit caps the number of "
            "messages fetched."
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
        chat_id = chat_jid(chat, target)
        if not session or not chat_id:
            return error("no active conversation context")
        overview = waha.get_chat_overview(session, chat_id)
        if not overview:
            return error(f"no metadata found for {chat_id}")
        names = sender_names(waha, session, chat_id)
        return ok(**summarize_chat(chat_id, overview, names))

    return FunctionTool.from_defaults(
        fn=get_chat_fn,
        fn_schema=GetChatSchema,
        name="get_chat",
        description=(
            "Get metadata (name, participant count, group flags, unread "
            "count) about a WhatsApp chat as a JSON envelope. For small "
            "chats it includes `participant_list`: the JID of each "
            "member, with `name` where known — use those JIDs for the "
            "`mentions` parameter of send_message. Omit chat for the "
            "current chat."
        ),
    )


def summarize_chat(
    chat_id: str,
    overview: dict[str, Any],
    names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a compact, model-friendly summary of a chat overview dict.

    Extracts the stable scalar fields and the participant roster,
    skipping nested blobs like ``lastMessage`` and ``picture`` that
    carry no useful metadata for the model. *names* (JID → display
    name, see :func:`sender_names`) enriches roster entries: the
    roster in LID groups holds bare JIDs, and names are the only way
    the model can pair a JID with a person for mentions. Returns a
    dict ready for the envelope.
    """
    scalar = chat_scalars(overview)
    add_participant_summary(scalar, overview, names or {})
    if not scalar or all(key == "id" for key in scalar):
        scalar["chat_id"] = chat_id
        scalar["_raw"] = str(overview)[:1000]
    return scalar


def sender_names(
    waha: WahaClient, session: str, chat_id: str, limit: int = 100
) -> dict[str, str]:
    """JID → display name for a chat's recent senders.

    The group roster carries only JIDs and admin flags; display names
    ride the messages themselves (``_data.notifyName`` per sender).
    Fails soft — an unreadable chat yields no names and the summary
    falls back to bare JIDs.
    """
    try:
        messages = waha.fetch_chat_messages(session, chat_id, limit=limit)
    except Exception as exc:
        logger.debug(
            "sender names unavailable for {chat_id}: {exc}", chat_id=chat_id, exc=exc
        )
        return {}
    names: dict[str, str] = {}
    for message in messages:
        jid = jid_string(message.get("participant"))
        name = str(message.get("_data", {}).get("notifyName") or "").strip()
        if jid and name and jid not in names:
            names[jid] = name
    return names


def chat_scalars(overview: dict[str, Any]) -> dict[str, Any]:
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


def add_participant_summary(
    scalar: dict[str, Any],
    overview: dict[str, Any],
    names: dict[str, str],
) -> None:
    """Add the participant count and (when small) id/name pairs.

    Names come from *names* (recent senders); roster entries without a
    known name keep the bare JID so the model can still mention by id.
    """
    participants = roster_entries(overview)
    if not isinstance(participants, list):
        return
    scalar["participants"] = len(participants)
    pairs = [
        {"id": jid, "name": names[jid]} if jid in names else {"id": jid}
        for jid in (participant_jid(p) for p in participants)
        if jid
    ]
    if 0 < len(pairs) <= 20:
        scalar["participant_list"] = pairs


def roster_entries(overview: dict[str, Any]) -> list[Any] | None:
    """The participant list, wherever WAHA put it.

    Engines differ: top level, under ``_chat``, or (LID groups) nested
    inside ``_chat.groupMetadata.participants``; entries are plain JID
    strings or ``{"id": {...}}`` objects.
    """
    blob = overview.get("_chat")
    chat_blob: dict[str, Any] = blob if isinstance(blob, dict) else {}
    for candidates in (
        overview.get("participants"),
        chat_blob.get("participants"),
        chat_blob.get("groupMetadata", {}).get("participants"),
    ):
        if isinstance(candidates, list):
            return candidates
    return None


def participant_jid(participant: Any) -> str:
    """The JID of one participant entry (string, ``{"id": ...}``, or JID object)."""
    if isinstance(participant, dict):
        entry: dict[str, Any] = participant
        return jid_string(entry.get("id"))
    return jid_string(participant)


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
        chat_id = chat_jid(chat, target)
        if not session or not chat_id:
            return error("no active conversation context")
        if not query.strip():
            return error("query is required")
        messages = waha.search_messages(session, query, chat_id, limit=limit)
        return ok(
            chat=chat_id,
            query=query,
            **fit_messages([slim_message(m) for m in messages]),
        )

    return FunctionTool.from_defaults(
        fn=search_messages_fn,
        fn_schema=SearchMessagesSchema,
        name="search_messages",
        description=(
            "Search a chat's recent messages for a text substring in "
            "body, media filename or mimetype. Returns a JSON envelope "
            "with matching `messages`: `count` is how many matched, "
            "`returned` how many fit (oldest are dropped when "
            "`truncated` is true). Pass chat to search another chat, "
            "else the current one."
        ),
    )


def resolve_chat(waha: WahaClient, target: dict[str, str]) -> BaseTool:
    """Build a tool that resolves a person/group name to chat JIDs.

    The model knows chats by name ("send it to the group Familia") but
    WAHA speaks JIDs. Fetches the chat list (contacts only when no chat
    matches), matches case-insensitively — exact name first, then
    substring — and returns up to ``_RESOLVE_CHAT_CANDIDATES`` matches
    for the model to pick from.
    """
    session = target.get("session", "")

    def resolve_chat_fn(name: str = "") -> str:
        """Resolve a person or group name to chat JIDs.

        Args:
            name: The person or group name to look up.
        """
        if not session:
            return error("no active conversation context")
        if not name.strip():
            return error("name is required")
        matches: list[dict[str, Any]] = []
        try:
            chats = waha.list_chats(session)
            matches = search_matches(chats, name)
            if not matches:
                contacts = waha.list_contacts(session)
                matches = search_matches(contacts, name)
        except Exception as exc:
            return error(f"could not search chats: {exc}")
        if not matches:
            return error(f"no chat or contact named like {name!r}")
        return ok(name=name, matches=matches)

    return FunctionTool.from_defaults(
        fn=resolve_chat_fn,
        fn_schema=ResolveChatSchema,
        name="resolve_chat",
        description=(
            "Resolve a person or group NAME to WhatsApp chat JIDs. "
            "Returns a JSON envelope with `matches` (up to 5, each "
            "`{id, name}`): exact names rank first, then substring "
            "matches. Pick the right JID and pass it as `chat` to "
            "send_message/send_image/forward_message. When several "
            "match, choose the closest and mention which you picked."
        ),
    )


#: How many name candidates the tool returns, to keep the envelope small.
_RESOLVE_CHAT_CANDIDATES = 5


def search_matches(entries: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """Chat/contact entries whose name matches *name*, best first.

    Exact (case-insensitive) name matches come first, then substring
    matches; each keeps only its ``id`` and ``name``. The list is capped
    at ``_RESOLVE_CHAT_CANDIDATES``.
    """
    needle = name.casefold()
    pairs = [
        {"id": str(e.get("id", "")), "name": str(e.get("name", ""))}
        for e in entries
        if e.get("id")
    ]
    exact = [p for p in pairs if p["name"].casefold() == needle]
    partial = [p for p in pairs if needle in p["name"].casefold() and p not in exact]
    return (exact + partial)[:_RESOLVE_CHAT_CANDIDATES]


def forward_message(waha: WahaClient, target: dict[str, str]) -> BaseTool:
    """Build a tool that forwards a message to a chat."""

    def forward_message_fn(message_id: str, chat: str | None = None) -> str:
        """Forward a message to a chat.

        Args:
            message_id: The serialized id of the message to forward.
            chat: Optional chat id to forward into. Defaults to current.
        """
        if target.get("sent"):
            return error(
                f"message already sent this run (to {target['sent']}); do not send again"
            )
        session = target.get("session", "")
        chat_id = chat_jid(chat, target)
        if not session or not chat_id:
            return error("no active conversation context")
        if not message_id:
            return error("message_id is required")
        waha.forward_message(session, chat_id, message_id)
        target["sent"] = chat_id
        return ok(message_id=message_id, chat=chat_id)

    return FunctionTool.from_defaults(
        fn=forward_message_fn,
        fn_schema=ForwardMessageSchema,
        name="forward_message",
        description=(
            "Forward an existing WhatsApp message (by its serialized id) "
            "to a chat. Pass chat to choose the destination, else current."
        ),
    )
