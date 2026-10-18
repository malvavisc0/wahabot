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

from wahabot.core.notebook import MAX_NOTE_KEY_LENGTH, MAX_NOTE_LENGTH, MAX_READ_NOTES

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

    mode: Literal["recent", "search", "metadata", "list", "resolve", "export"] = Field(
        description=(
            "Which mode: list, search (needs query), metadata, resolve (needs "
            "name), recent (operator-only conversation list), or export (the "
            "chat's deep recorded history)."
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
    limit: int | None = Field(
        default=None,
        description=(
            "Positive recent-message window for list/search (default 20), "
            "not number of search hits. export: row cap keeping the newest; "
            "omit for everything recorded. Recent conversations cap at 30. "
            "Ignored by metadata/resolve."
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


class NotebookSchema(BaseModel):
    """Read, create/replace, or delete private durable conversation notes."""

    mode: Literal["read", "write", "delete"] = Field(
        description=(
            "read searches notes, write creates/replaces a key, delete removes it."
        )
    )
    scope: Literal["chat", "session"] = Field(
        default="chat",
        description=(
            "chat: private conversation notes. session: operator-only behavior "
            "policies shared with every chat; never put private chat data there."
        ),
    )
    key: str = Field(
        default="",
        max_length=MAX_NOTE_KEY_LENGTH,
        description="Meaningful note key; required for write/delete, optional for read.",
    )
    text: str = Field(
        default="",
        max_length=MAX_NOTE_LENGTH,
        description="Concise note body for write (at most 512 tokens); not a transcript.",
    )
    kind: Literal["fact", "lesson", "rule"] = Field(
        default="fact",
        description=(
            "fact: a durable chat fact or preference. lesson: an operational "
            "workaround. rule: a standing instruction for this chat "
            "(operator-only; a member's rule request is recorded as a fact)."
        ),
    )
    query: str = Field(
        default="",
        description="Optional literal case-insensitive search in note keys and text.",
    )
    limit: int = Field(
        default=MAX_READ_NOTES,
        ge=1,
        le=MAX_READ_NOTES,
        description="Maximum read matches, newest first; a token budget also applies.",
    )
    chat: str | None = Field(
        default=None,
        description=(
            CHAT_DESCRIPTION
            + " Omit for current-chat notes or the private operator notebook."
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


class ImageSchema(BaseModel):
    """Generate an image from a full scene caption."""

    prompt: str = Field(
        description=(
            "A rich, full-scene caption — subject/action, setting, lighting, "
            "composition, style, colours; the service does not expand prompts."
        )
    )
    size: str = Field(
        default="",
        description="Output size WxH; empty uses the configured default (1024x1024).",
    )
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class EditImageSchema(BaseModel):
    """Repaint an existing image to match a prompt."""

    prompt: str = Field(
        description=(
            "A rich caption of the finished scene — what changes and what stays "
            "the same; the service does not expand prompts."
        )
    )
    source: str = Field(
        default="",
        description=(
            "Local path or URL of the image to edit; empty uses the image the "
            "sender just sent this turn."
        ),
    )
    strength: float | None = Field(
        default=None,
        description="How much to repaint, 0-1; empty uses the service default (0.9).",
    )
    reason: str = Field(default=REASON_DESCRIPTION)
