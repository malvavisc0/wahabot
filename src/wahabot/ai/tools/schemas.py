"""Explicit Pydantic schemas for the bundled WhatsApp tools.

llama-index's auto-schema generation does not reliably extract
parameter descriptions from docstrings, so every tool parameter here
carries a ``Field(description=...)`` for the LLM to read. Passing
``fn_schema=`` to ``FunctionTool.from_defaults`` substitutes the
docstring-derived schema with these definitions.

``chat`` is always optional: omit it to target the current conversation,
or pass a group/person JID (e.g. ``1234567890@g.us`` or ``9876543210@c.us``)
to reach another chat.

Every tool returns the shared JSON envelope (see
``wahabot.ai.tools.envelope``): ``{"ok": true, ...payload}`` on success,
``{"ok": false, "error": "..."}`` on failure.
"""

from pydantic import BaseModel, Field

#: Shared ``chat`` parameter description: a bare JID, never a message id.
CHAT_DESCRIPTION = (
    "Optional chat id — a bare JID like `1234567890@g.us` or "
    "`9876543210@c.us`, never a `false_...` message id. Omit for the "
    "current chat."
)


class SendMessageSchema(BaseModel):
    """Send a WhatsApp text message."""

    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)
    text: str = Field(
        default="",
        description="The text to send. Must be non-empty.",
    )
    reply_to: str | None = Field(
        default=None,
        description=(
            "Optional serialized id of a message to quote (e.g. "
            "`false_11111111111@c.us_AAAAAAAAAAAAAAAAAAAA`). The text is sent "
            "as a native quote-reply with that message attached. Use ids from "
            "fetch_chat_messages."
        ),
    )
    mentions: list[str] | None = Field(
        default=None,
        description=(
            "Optional JIDs of people to truly @-mention (highlighted, they get "
            "notified). For every JID here, that person's display name must "
            "appear in `text` as `@<name>` — WhatsApp pairs the two. Get JIDs "
            "and names from `get_chat` (participants) or the `participant` "
            "field of messages in fetch_chat_messages. Typing `@name` alone "
            "does NOT notify anyone."
        ),
    )


class StaySilentSchema(BaseModel):
    """Stay silent: send nothing in this conversation."""

    model_config = {"extra": "forbid"}


class ReactToMessageSchema(BaseModel):
    """React with an emoji to a WhatsApp message."""

    message_id: str = Field(
        description=(
            "The serialized id of the message to react to (e.g. "
            "`false_12132132130@c.us_AAAAAAAAAAAAAAAAAAAA`)."
        )
    )
    reaction: str = Field(
        default="",
        description=(
            "The emoji to react with, or empty string to remove an existing reaction."
        ),
    )


class SendImageSchema(BaseModel):
    """Send an image to a WhatsApp chat from a public URL."""

    url: str | None = Field(
        default=None,
        description="Public URL of the image to send. Required.",
    )
    caption: str = Field(
        default="",
        description="Optional caption text.",
    )
    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)


class FetchChatMessagesSchema(BaseModel):
    """Fetch the most recent messages of a chat as a JSON envelope."""

    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)
    limit: int = Field(
        default=20,
        description="Max messages to return (default 20).",
    )


class GetChatSchema(BaseModel):
    """Get metadata (name, participants, ...) about a WhatsApp chat."""

    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)


class SearchMessagesSchema(BaseModel):
    """Search a chat's recent messages for a text substring."""

    query: str = Field(
        description="The text to look for in message body, media filename or mimetype.",
    )
    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)
    limit: int = Field(
        default=20,
        description="Max matches to return (default 20).",
    )


class ForwardMessageSchema(BaseModel):
    """Forward an existing WhatsApp message to a chat."""

    message_id: str = Field(
        description="The serialized id of the message to forward.",
    )
    chat: str | None = Field(default=None, description=CHAT_DESCRIPTION)


class ResolveChatSchema(BaseModel):
    """Resolve a person/group name to WhatsApp chat JIDs."""

    name: str = Field(
        description=(
            "The person or group name to resolve, e.g. `Familia` or `Ana`. "
            "Matched case-insensitively against chat and contact names."
        ),
    )


class WebSearchSchema(BaseModel):
    """Search the web via the webserp metasearch CLI."""

    query: str = Field(
        description="The search query text.",
    )
    max_results: int | None = Field(
        default=None,
        description=(
            "Optional max results to return. Defaults to the configured limit if omitted."
        ),
    )


class FetchStockPriceSchema(BaseModel):
    """Fetch the current price for a stock, ETF, or crypto ticker."""

    ticker: str = Field(
        description=("A stock, ETF or crypto symbol, e.g. `AAPL`, `MSFT`, or `BTC-USD`."),
    )


class GetYoutubeTranscriptSchema(BaseModel):
    """Fetch and format a YouTube video's captions/transcript."""

    url: str = Field(
        description="A YouTube video URL, e.g. `https://youtube.com/watch?v=...`.",
    )


class VisitUrlSchema(BaseModel):
    """Fetch a web page and return its visible text as a JSON envelope."""

    url: str = Field(
        description="The web page URL to visit.",
    )


class ShellCommandSchema(BaseModel):
    """Run a shell command on the host and return its output."""

    command: str = Field(
        description=(
            "The shell command line to execute (e.g. `ls -la`). Runs through "
            "bash, so pipes, redirection and the usual shell features work. "
            "stdin is closed — the command must not wait for input."
        ),
    )
