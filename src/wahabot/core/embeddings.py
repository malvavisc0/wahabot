"""OpenAI-compatible embeddings client for the LAN service.

Dense recall for the per-chat thread index uses an OpenAI-compatible
``/v1/embeddings`` endpoint (EmbeddingGemma-2 on ``:9195``); this is a
tiny :class:`BaseEmbedding` subclass rather than the
``llama-index-embeddings-openai`` package, which would add a dependency
for one HTTP call.
"""

from __future__ import annotations

from typing import Any, cast, override

import httpx
from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field


class LanEmbedding(BaseEmbedding):
    """An OpenAI-compatible ``/v1/embeddings`` client for one LAN model."""

    api_base: str = Field(default="")
    api_key: str = Field(default="", repr=False, exclude=True)
    timeout: float = Field(default=10.0)

    def url(self) -> str:
        """The endpoint URL; *api_base* includes the provider's ``/v1``."""
        return f"{self.api_base.rstrip('/')}/embeddings"

    def headers(self) -> dict[str, str]:
        """Bearer auth when a key is configured."""
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def body(self, texts: list[str]) -> dict[str, Any]:
        """The OpenAI-compatible request payload."""
        return {"model": self.model_name, "input": texts}

    def post(self, texts: list[str]) -> list[list[float]]:
        """One synchronous embeddings request."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self.url(), json=self.body(texts), headers=self.headers()
            )
            response.raise_for_status()
        return vectors(response.json())

    async def apost(self, texts: list[str]) -> list[list[float]]:
        """One asynchronous embeddings request."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.url(), json=self.body(texts), headers=self.headers()
            )
            response.raise_for_status()
        return vectors(response.json())

    @override
    def _get_query_embedding(self, query: str) -> list[float]:
        return self.post([query])[0]

    @override
    async def _aget_query_embedding(self, query: str) -> list[float]:
        return (await self.apost([query]))[0]

    @override
    def _get_text_embedding(self, text: str) -> list[float]:
        return self.post([text])[0]

    @override
    async def _aget_text_embedding(self, text: str) -> list[float]:
        return (await self.apost([text]))[0]

    @override
    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self.post(texts)

    @override
    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await self.apost(texts)


def vectors(payload: Any) -> list[list[float]]:
    """The ``data[].embedding`` rows, in the service's index order."""
    data = cast(dict[str, Any], payload)["data"]
    rows = sorted(cast(list[dict[str, Any]], data), key=lambda row: row.get("index", 0))
    return [cast(list[float], row["embedding"]) for row in rows]
