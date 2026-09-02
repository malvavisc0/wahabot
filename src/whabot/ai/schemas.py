"""Explicit Pydantic schemas for the bundled WhatsApp tools.

llama-index's auto-schema generation does not reliably extract
parameter descriptions from docstrings, so every tool parameter here
carries a ``Field(description=...)`` for the LLM to read. Passing
``fn_schema=`` to ``FunctionTool.from_defaults`` substitutes the
docstring-derived schema with these definitions.

``chat`` is always optional: omit it to target the current conversation,
or pass a group/person JID (e.g. ``1234567890@g.us`` or ``9876543210@c.us``)
to reach another chat.
"""

from pydantic import BaseModel, Field


class SendMessageSchema(BaseModel):
    """Send a WhatsApp text message."""

    chat: str | None = Field(
        default=None,
        description=(
            "Optional chat id (group or person JID, e.g. `1234567890@g.us` or "
            "`9876543210@c.us`). Omit to reply in the current conversation."
        ),
    )
    text: str = Field(
        default="",
        description="The text to send. Must be non-empty.",
    )


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


class SendSeenSchema(BaseModel):
    """Mark a WhatsApp chat as read/seen."""

    chat: str | None = Field(
        default=None,
        description="Optional chat id. Omit to mark the current chat.",
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
    chat: str | None = Field(
        default=None,
        description="Optional chat id. Omit to send to the current chat.",
    )


class FetchChatMessagesSchema(BaseModel):
    """Fetch the most recent messages of a chat as text lines."""

    chat: str | None = Field(
        default=None,
        description="Optional chat id. Omit to fetch from the current chat.",
    )
    limit: int = Field(
        default=20,
        description="Max messages to return (default 20).",
    )


class GetChatSchema(BaseModel):
    """Get metadata (name, participants, ...) about a WhatsApp chat."""

    chat: str | None = Field(
        default=None,
        description="Optional chat id. Omit for the current chat.",
    )


class GetContactSchema(BaseModel):
    """Get details about a WhatsApp contact by its id."""

    contact_id: str = Field(
        description="A contact's JID (e.g. `9876543210@c.us`).",
    )


class SearchMessagesSchema(BaseModel):
    """Search a chat's recent messages for a text substring."""

    query: str = Field(
        description="The text to look for in message body, media filename or mimetype.",
    )
    chat: str | None = Field(
        default=None,
        description=(
            "Optional chat id to scope the search. Omit to search the current chat."
        ),
    )
    limit: int = Field(
        default=20,
        description="Max matches to return (default 20).",
    )


class ForwardMessageSchema(BaseModel):
    """Forward an existing WhatsApp message to a chat."""

    message_id: str = Field(
        description="The serialized id of the message to forward.",
    )
    chat: str | None = Field(
        default=None,
        description="Optional chat id to forward into. Defaults to current.",
    )


class SetTypingSchema(BaseModel):
    """Show or hide the typing indicator in a WhatsApp chat."""

    typing: bool = Field(
        description="True to start typing, False to stop.",
    )
    chat: str | None = Field(
        default=None,
        description="Optional chat id. Omit for the current chat.",
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


class TickerSchema(BaseModel):
    """Base schema for ticker-keyed tools."""

    ticker: str = Field(
        description=("A stock, ETF or crypto symbol, e.g. `AAPL`, `MSFT`, or `BTC-USD`."),
    )


class FetchStockPriceSchema(TickerSchema):
    """Fetch the current price for a stock, ETF, or crypto ticker."""


class FetchCompanyInfoSchema(TickerSchema):
    """Fetch company fundamentals and metadata for a ticker."""


class FetchTickerNewsSchema(TickerSchema):
    """Fetch recent news for a stock or crypto ticker."""

    max_articles: int = Field(
        default=10,
        description="Maximum number of articles to return.",
    )


class GetYoutubeTranscriptSchema(BaseModel):
    """Fetch and format a YouTube video's captions/transcript."""

    url: str = Field(
        description="A YouTube video URL, e.g. `https://youtube.com/watch?v=...`.",
    )


class VisitUrlSchema(BaseModel):
    """Fetch a web page and return its visible text."""

    url: str = Field(
        description="The web page URL to visit.",
    )
