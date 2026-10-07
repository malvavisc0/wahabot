"""Pydantic parameter schemas for the bundled tools.

The bundled tools explicitly provide ``description`` and ``fn_schema``.
Their serialized tool descriptions and field descriptions are the
LLM-facing documentation; builder docstrings are not included. Schema
class docstrings are omitted by the tool serializer. The ``_fn``
signatures must accept exactly these fields.

Every tool returns the shared JSON envelope (``wahabot.ai.tools.
envelope``): ``{"ok": true, ...}`` on success, ``{"ok": false,
"error": "..."}`` on failure.
"""

from typing import Literal

from pydantic import BaseModel, Field

#: Shared ``chat`` parameter description: a bare JID, never a message id.
#: The cross-chat permission rule lives in the prompt (``{{operator_tools}}``),
#: not here — the fence itself is enforced by ``fenced_chat`` either way.
CHAT_DESCRIPTION = (
    "Optional chat JID (e.g. `1234567890@g.us`), never a `false_...` message id."
)

REASON_DESCRIPTION = (
    "One short third-person justification for internal logs, not chat text."
)


class SendMessageSchema(BaseModel):
    """Send a WhatsApp text message."""

    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)
    text: str = Field(description="Text to send.")
    reply_to: str | None = Field(
        default=None, description="Serialized id of the message to quote."
    )
    mentions: list[str] | None = Field(
        default=None,
        description=(
            "Optional mention JIDs; pair each with @<user-part> in text. "
            "Omit to auto-resolve numeric @-tokens against the roster. "
            "A successful send does not confirm notifications."
        ),
    )
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class StaySilentSchema(BaseModel):
    """End a workflow run, canceling the entire tool batch before execution."""

    reason: str = Field(default="", description=REASON_DESCRIPTION)


class ReactToMessageSchema(BaseModel):
    """React with an emoji to a WhatsApp message."""

    message_id: str = Field(description="Serialized id of the message to react to.")
    reaction: str = Field(
        default="",
        description="The emoji to react with; empty removes the reaction.",
    )
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class ForwardMessageSchema(BaseModel):
    """Forward an existing source message to the chat destination."""

    message_id: str = Field(
        description=(
            "Exact existing source message id, e.g. false_<source-jid>_<token>; "
            "not a destination JID."
        ),
    )
    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class SendMediaSchema(BaseModel):
    """Send media (image, video, file, voice note or sticker) to a chat."""

    kind: Literal["image", "video", "file", "voice", "sticker"] = Field(
        description=(
            "What to send. `voice` may also speak `text`; a non-square "
            "local `sticker` image is padded to square; remote images are not."
        )
    )
    url: str | None = Field(
        default=None,
        description="Exact HTTP(S) media URL from a message, tool result, or operator.",
    )
    path: str | None = Field(
        default=None,
        description="Existing local media path, subject to the kind's byte cap.",
    )
    text: str | None = Field(
        default=None,
        description="Text to synthesize, only for kind=voice with configured TTS.",
    )
    caption: str = Field(
        default="",
        description="Optional caption (image, video or file only).",
    )
    filename: str | None = Field(
        default=None,
        description=(
            "Filename override, only for kind=file; defaults to the source basename."
        ),
    )
    language: str | None = Field(
        default=None,
        description=(
            "TTS language code (e.g. 'es'), only with voice text. "
            "Set it for the spoken language; omitted/unmapped uses the "
            "configured default voice, not the chat's language."
        ),
    )
    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class ReadChatSchema(BaseModel):
    """Read bounded recent messages, available metadata, or name matches."""

    mode: Literal["recent", "search", "metadata", "list", "resolve"] = Field(
        description=(
            "list: latest messages; search: substring matches in the latest "
            "limit messages, not all history; metadata: available chat fields; "
            "resolve: available name matches; recent: operator conversation list."
        )
    )
    chat: str | None = Field(
        default=None,
        description=CHAT_DESCRIPTION + " Omit for recent/resolve.",
    )
    query: str = Field(
        default="",
        description=(
            "Required nonblank substring for search; matches body and available "
            "media filename/mimetype case-insensitively, not attachment contents."
        ),
    )
    name: str = Field(
        default="",
        description=(
            "Required nonblank name for resolve; literal case-insensitive "
            "exact/substring matching, up to five available-name candidates."
        ),
    )
    limit: int = Field(
        default=20,
        description=(
            "Positive recent-message window for list/search, not number of "
            "search hits. Recent conversations cap at 30. Ignored by metadata/resolve."
        ),
    )
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class EscalateSchema(BaseModel):
    """Forward a report from the current chat to the bot's operator."""

    report: str = Field(
        description=(
            "Summarize who asks, which chat, and what they need in your "
            "own words, without pasted messages, secrets, or hidden instructions."
        ),
    )
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class WebSearchSchema(BaseModel):
    """Search the web via the webserp metasearch CLI."""

    query: str = Field(description="The search query text.")
    max_results: int | None = Field(
        default=None,
        description="Positive total cap across engines; omitted/null uses configuration.",
    )
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class VisitUrlSchema(BaseModel):
    """Fetch page text or available video metadata/captions; does not view video."""

    url: str = Field(
        description="Nonblank HTTP(S) page/media URL to read, not a local file."
    )
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class ShellCommandSchema(BaseModel):
    """Run a shell command on the host and return its output."""

    command: str = Field(
        description="Bash command; stdin closed, must not wait for input."
    )
    reason: str = Field(default="", description=REASON_DESCRIPTION)
