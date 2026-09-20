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


class SendImageSchema(BaseModel):
    """Send an image to a WhatsApp chat from a public URL."""

    url: str | None = Field(
        default=None,
        description="Public URL of the image. Never invent one.",
    )
    caption: str = Field(default="", description="Optional caption.")
    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class FetchChatMessagesSchema(BaseModel):
    """Fetch the most recent messages of a chat as a JSON envelope."""

    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)
    limit: int = Field(default=20, description="Max messages to return.")
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class GetChatSchema(BaseModel):
    """Get metadata (name, participants, ...) about a WhatsApp chat."""

    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class SearchMessagesSchema(BaseModel):
    """Search a chat's recent messages for a text substring."""

    query: str = Field(
        description="Text to look for in message body, media filename or mimetype."
    )
    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)
    limit: int = Field(
        default=20,
        description="Max matches to return.",
    )
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class ForwardMessageSchema(BaseModel):
    """Forward an existing WhatsApp message to a chat."""

    message_id: str = Field(
        description="Serialized id of the message to forward.",
    )
    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class ResolveChatSchema(BaseModel):
    """Resolve a person/group name to WhatsApp chat JIDs."""

    name: str = Field(description="The person/group name to resolve, e.g. `Family`.")
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class RecentChatsSchema(BaseModel):
    """List the most recent WhatsApp conversations."""

    limit: int = Field(default=10, description="How many conversations to return.")
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


class SendFileSchema(BaseModel):
    """Send a document (PDF, etc.) to a WhatsApp chat."""

    url: str | None = Field(
        default=None, description="Public URL of the document. Pass url XOR path."
    )
    path: str | None = Field(
        default=None, description="Local file path on the host. Pass path XOR url."
    )
    caption: str = Field(default="", description="Optional caption.")
    filename: str | None = Field(
        default=None,
        description="Name shown to the recipient; defaults to the basename.",
    )
    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class SendVideoSchema(BaseModel):
    """Send a video to a WhatsApp chat."""

    url: str | None = Field(
        default=None, description="Public URL of the video. Pass url XOR path."
    )
    path: str | None = Field(
        default=None, description="Local video path on the host. Pass path XOR url."
    )
    caption: str = Field(default="", description="Optional caption.")
    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class SendVoiceSchema(BaseModel):
    """Send a voice note to a WhatsApp chat."""

    text: str | None = Field(default=None, description="Text the bot speaks aloud.")
    url: str | None = Field(
        default=None, description="Public URL of audio to relay. One of text/url/path."
    )
    path: str | None = Field(
        default=None,
        description="Local audio path on the host. One of text/url/path.",
    )
    language: str | None = Field(
        default=None,
        description="Language code (e.g. 'es') when it differs from the chat's.",
    )
    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class SendStickerSchema(BaseModel):
    """Send a sticker (WebP image) to a WhatsApp chat."""

    url: str | None = Field(
        default=None, description="Public URL of the WebP sticker. url XOR path."
    )
    path: str | None = Field(
        default=None, description="Local WebP path on the host. path XOR url."
    )
    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class WebSearchSchema(BaseModel):
    """Search the web via the webserp metasearch CLI."""

    query: str = Field(description="The search query text.")
    max_results: int | None = Field(default=None, description="Max results to return.")
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class FetchStockPriceSchema(BaseModel):
    """Fetch the current price for a stock, ETF, or crypto ticker."""

    ticker: str = Field(description="Ticker, e.g. `AAPL`, `MSFT`, `BTC-USD`.")
    reason: str = Field(default="", description=REASON_DESCRIPTION)


class GetYoutubeTranscriptSchema(BaseModel):
    """Fetch and format a YouTube video's captions/transcript."""

    url: str = Field(description="YouTube video URL.")
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
