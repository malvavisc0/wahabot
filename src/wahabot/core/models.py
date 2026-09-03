"""WAHA webhook event payload models."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WahaEvent(BaseModel):
    """Payload WAHA sends to webhook endpoints.

    See https://waha.devlike.pro/docs/how-to/events/#event-payload
    """

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    timestamp: int | None = None
    event: str
    session: str
    me: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] | None = None
    engine: str | None = None
