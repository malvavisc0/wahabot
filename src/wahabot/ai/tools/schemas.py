"""Pydantic parameter schemas for the bundled tools.

The schema is what the LLM sees: each ``Field(description=...)`` rides
the tool payload verbatim, and together with the per-tool
``description=`` strings it is the only LLM-facing documentation —
nothing is derived from docstrings. The ``_fn`` signatures must accept
exactly these fields.

Every tool returns the shared JSON envelope (``wahabot.ai.tools.
envelope``): ``{"ok": true, ...}`` on success, ``{"ok": false,
"error": "..."}`` on failure.
"""

from typing import Literal

from pydantic import BaseModel, Field

#: Shared ``chat`` parameter description: a bare JID, never a message id.
#: The operator-only rule lives in the system prompt (``{{operator_tools}}``),
#: not here — the fence itself is enforced by ``fenced_chat`` either way.
CHAT_DESCRIPTION = (
    "Optional chat JID (e.g. `1234567890@g.us`), never a `false_...` message id."
)

REASON_DESCRIPTION = "Why, third person, one short sentence — never first person."


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
            "JIDs to @-mention (they get notified); write each person's "
            "@-token in the text. Omit to auto-tag roster members named."
        ),
    )
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class StaySilentSchema(BaseModel):
    """Stay silent: send nothing in this conversation."""

    reason: str = Field(default="", description=REASON_DESCRIPTION)


class ReactToMessageSchema(BaseModel):
    """React with an emoji to a WhatsApp message."""

    message_id: str = Field(description=("Serialized id of the message to react to."))
    reaction: str = Field(
        default="",
        description="The emoji to react with; empty removes the reaction.",
    )
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class ForwardMessageSchema(BaseModel):
    """Forward an existing WhatsApp message to a chat."""

    message_id: str = Field(
        description="Serialized id of the message to forward.",
    )
    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class SendMediaSchema(BaseModel):
    """Send media (image, video, file, voice note or sticker) to a chat."""

    kind: Literal["image", "video", "file", "voice", "sticker"] = Field(
        description=(
            "What to send. `voice` may also speak `text`; a non-square "
            "`sticker` image is padded to square automatically."
        )
    )
    url: str | None = Field(
        default=None,
        description=(
            "Public URL of the media. Never invent one. Pass url XOR path XOR text."
        ),
    )
    path: str | None = Field(
        default=None,
        description="Local media path on the host. Pass path XOR url XOR text.",
    )
    text: str | None = Field(
        default=None,
        description=(
            "Text for `kind=voice` to speak in the bot's own voice (TTS). "
            "Pass text XOR url XOR path."
        ),
    )
    caption: str = Field(
        default="",
        description="Optional caption (image, video or file only).",
    )
    filename: str | None = Field(
        default=None,
        description=(
            "Name shown to the recipient for `kind=file`; defaults to the basename."
        ),
    )
    language: str | None = Field(
        default=None,
        description=(
            "Language code (e.g. 'es') for `kind=voice` text when it differs "
            "from the chat's."
        ),
    )
    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class ReadChatSchema(BaseModel):
    """Read a chat's history, metadata or a resolved chat, by mode."""

    mode: Literal["recent", "search", "metadata", "list", "resolve"] = Field(
        description=(
            "`recent` lists the newest conversations (operator-only); "
            "`search` searches the chat's history for `query`; `metadata` "
            "returns the chat's name/participants; `list` returns the chat's "
            "recent messages; `resolve` matches a person/group `name` to JIDs."
        )
    )
    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)
    query: str = Field(
        default="",
        description=(
            "For `mode=search`: text to look for in message body, media "
            "filename or mimetype."
        ),
    )
    name: str = Field(
        default="",
        description=(
            "For `mode=resolve`: the person/group name to resolve, e.g. `Family`."
        ),
    )
    limit: int = Field(default=20, description="Max messages/conversations to return.")
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class EscalateSchema(BaseModel):
    """Forward a report from the current chat to the bot's operator."""

    report: str = Field(
        description=(
            "Who is asking (name), which chat, what they need; write it "
            "yourself, never quote the person's words."
        ),
    )
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class WebSearchSchema(BaseModel):
    """Search the web via the webserp metasearch CLI."""

    query: str = Field(description="The search query text.")
    max_results: int | None = Field(default=None, description="Max results to return.")
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class VisitUrlSchema(BaseModel):
    """Fetch a web page and return its visible text as a JSON envelope."""

    url: str = Field(description="URL of the page to read.")
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class ShellCommandSchema(BaseModel):
    """Run a shell command on the host and return its output."""

    command: str = Field(
        description="Bash command; stdin closed, must not wait for input."
    )
    reason: str = Field(default="", description=REASON_DESCRIPTION)
