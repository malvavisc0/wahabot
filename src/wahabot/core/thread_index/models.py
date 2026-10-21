import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from wahabot.ai.history import INBOUND_ID_RE, narration_kind
from wahabot.ai.messages import WRAP_UP_NOTE_KWARG


class EmbeddingModel(Protocol):
    """The two embedding calls the indexer needs, sync-free."""

    async def aget_query_embedding(self, query: str) -> list[float]:
        """Embed one retrieval query."""
        ...

    async def aget_text_embedding_batch(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float]]:
        """Embed a batch of turn texts."""
        ...


@dataclass(frozen=True, slots=True)
class ThreadTurn:
    """One indexable turn: a user or assistant message, never a tool."""

    role: str
    text: str

    @property
    def node_id(self) -> str:
        """Stable content hash: the same turn is always one node."""
        return thread_node_id(self.role, self.text)


def thread_node_id(role: str, text: str) -> str:
    """SHA-256 of ``role + NUL + text`` — the idempotent node key."""
    digest = hashlib.sha256()
    digest.update(role.encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def index_file(data_dir: Path, session: str, chat_id: str) -> Path:
    """Path of a chat's FTS/turn index, next to its memory file."""
    return data_dir / "memory" / session / f"{chat_id}.index.sqlite3"


def vector_file(data_dir: Path, session: str, chat_id: str) -> Path:
    """Path of a chat's dense vector store, next to its memory file."""
    return data_dir / "memory" / session / f"{chat_id}.vectors.json"


def turn_from_message(message: ChatMessage) -> ThreadTurn | None:
    """The indexable turn a message carries, or ``None``.

    Only user/assistant text is indexed. Tool results are raw JSON,
    narration (silence/emoji/error self-reports) is not conversation, and
    a private wrap-up note was never sent to the chat.
    """
    if message.role not in (MessageRole.USER, MessageRole.ASSISTANT):
        return None
    text = INBOUND_ID_RE.sub("", str(message.content or "")).strip()
    if not text or message.additional_kwargs.get(WRAP_UP_NOTE_KWARG):
        return None
    if message.role == MessageRole.ASSISTANT and narration_kind(text):
        return None
    return ThreadTurn(message.role.value, text)
