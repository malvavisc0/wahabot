"""Thread recall: per-chat index, trim hook, hybrid retrieval.

The acceptance story is the promise fixture: a 30-turn support thread
whose turn-5 promise is evicted by the FIFO trim; at turn 30 the
assembled prompt must still carry the verbatim promise slice. The rest
of the file covers the invariants that make the index safe on WhatsApp:
hash-keyed idempotency, the user/assistant role filter, the forget wipe,
the kill switches, embeddings-down fail-soft, and the rule that indexing
never restores trimmed turns into the live buffer.
"""

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.workflow import Context

from wahabot.ai.messages import WRAP_UP_NOTE_KWARG
from wahabot.ai.pinned_state import render_pinned_block
from wahabot.ai.tools.whatsapp import RunTarget, bind_target, reset_target
from wahabot.ai.workflow import FunctionCallingAgentWorkflow
from wahabot.core.notebook import Note
from wahabot.core.persistence import load_memory, save_memory
from wahabot.core.thread_index import (
    EMBED_TEXT_CHARS,
    EMBED_TEXT_CHARS_RETRY,
    ThreadIndexer,
    forget_thread_index,
    index_file,
    vector_file,
)
from wahabot.core.thread_index import indexer as thread_index_module
from wahabot.evals.fixtures import support_threads
from wahabot.settings import Settings

SESSION = "s1"
CHAT = "491555000001@c.us"


class FakeEmbedding:
    """A deterministic embedding model: PDF texts land near one axis."""

    def __init__(self, *, down: bool = False) -> None:
        self.down = down
        self.batch_calls = 0
        self.query_calls = 0
        self.queries: list[str] = []
        self.texts: list[str] = []

    async def aget_text_embedding_batch(
        self, texts: list[str], **_: Any
    ) -> list[list[float]]:
        self.batch_calls += 1
        if self.down:
            raise RuntimeError("embeddings service down")
        self.texts.extend(texts)
        return [self.vector(text) for text in texts]

    async def aget_query_embedding(self, query: str) -> list[float]:
        self.query_calls += 1
        if self.down:
            raise RuntimeError("embeddings service down")
        self.queries.append(query)
        return self.vector(query)

    @staticmethod
    def vector(text: str) -> list[float]:
        lower = text.lower()
        related = ("pdf", "mañana", "morgen", "dokument")
        return [1.0, 0.0] if any(word in lower for word in related) else [0.0, 1.0]


@pytest.fixture
def recall_settings(tmp_path: Path) -> Settings:
    return Settings(
        webhook_hmac_key="k",
        waha_url="http://waha.invalid",
        waha_api_key="k",
        llm_api_base="http://llm.invalid",
        llm_api_key="k",
        langfuse_public_key="",
        langfuse_secret_key="",
        session=SESSION,
        data_dir=tmp_path,
        _env_file=None,
    )


def recall_workflow(
    settings: Settings, indexer: ThreadIndexer
) -> FunctionCallingAgentWorkflow:
    settings.learned_memory = False
    llm = Mock(spec=FunctionCallingLLM)
    llm.metadata = LLMMetadata(is_function_calling_model=True)
    return FunctionCallingAgentWorkflow(
        llm=llm,
        system_prompt="Support agent.",
        settings=settings,
        memory_token_limit=160,
        thread_index=indexer,
    )


async def recall_context(
    agent: FunctionCallingAgentWorkflow, messages: list[ChatMessage]
) -> Context:
    ctx = Context(agent)
    memory = ChatMemoryBuffer(
        token_limit=agent.memory_token_limit, tokenizer_fn=str.split
    )
    await memory.aset(messages)
    await ctx.store.set("memory", memory)
    return ctx


def fixture_messages(thread: Any) -> list[ChatMessage]:
    return [
        ChatMessage(
            role=MessageRole.USER if turn.role == "user" else MessageRole.ASSISTANT,
            content=turn.text,
        )
        for turn in thread.turns
    ]


def index_rows(data_dir: Path) -> list[tuple[str, str]]:
    with closing(sqlite3.connect(index_file(data_dir, SESSION, CHAT))) as connection:
        rows = connection.execute(
            "SELECT role, text FROM turns ORDER BY ordinal"
        ).fetchall()
    return [(str(row[0]), str(row[1])) for row in rows]


def embedded_count(data_dir: Path) -> int:
    with closing(sqlite3.connect(index_file(data_dir, SESSION, CHAT))) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM turns WHERE embedded = 1"
        ).fetchone()
    return int(row[0])


def test_trimmed_promise_slice_reaches_the_turn_30_prompt(
    recall_settings: Settings,
) -> None:
    """The phase-5 acceptance test: the turn-5 promise is in the turn-30 prompt."""
    thread = support_threads()[0]
    promise = thread.turns[4].text
    assert promise == "Revisaré la solicitud y te enviaré el PDF mañana."
    indexer = ThreadIndexer(recall_settings.data_dir, FakeEmbedding())
    agent = recall_workflow(recall_settings, indexer)
    target = RunTarget(session=SESSION, chat_id=CHAT)

    async def scenario() -> tuple[list[ChatMessage], list[ChatMessage]]:
        token = bind_target(target)
        try:
            ctx = await recall_context(agent, fixture_messages(thread))
            await agent.chat_history(ctx)
            memory = await ctx.store.get("memory")
            trimmed = await memory.aget_all()
            await agent.merge_learned()
            fresh = await recall_context(
                agent,
                [ChatMessage(role=MessageRole.USER, content=thread.turns[-1].text)],
            )
            return await agent.chat_history(fresh), trimmed
        finally:
            reset_target(token)

    history, trimmed = asyncio.run(scenario())
    system = str(history[0].content)

    assert len(trimmed) < len(thread.turns)
    assert promise not in [str(message.content) for message in trimmed]
    assert "## Recalled from earlier in this thread" in system
    assert "Leads, verify before relying on them" in system
    assert promise in system
    assert thread.turns[3].text in system


def test_the_same_turn_indexed_twice_is_one_node(recall_settings: Settings) -> None:
    indexer = ThreadIndexer(recall_settings.data_dir, FakeEmbedding())
    turn = ChatMessage(role=MessageRole.USER, content="my reference is REF-7781")

    async def scenario() -> None:
        indexer.schedule(SESSION, CHAT, [turn])
        await indexer.drain()
        indexer.schedule(SESSION, CHAT, [turn])
        await indexer.drain()

    asyncio.run(scenario())

    assert index_rows(recall_settings.data_dir) == [("user", "my reference is REF-7781")]
    store = SimpleVectorStore.from_persist_path(
        str(vector_file(recall_settings.data_dir, SESSION, CHAT))
    )
    assert len(store.data.embedding_dict) == 1


def test_message_id_metadata_does_not_change_the_content_hash(
    recall_settings: Settings,
) -> None:
    indexer = ThreadIndexer(recall_settings.data_dir, FakeEmbedding())
    first = ChatMessage(
        role=MessageRole.USER, content="my reference is REF-7781 [message id: A]"
    )
    redelivered = ChatMessage(
        role=MessageRole.USER, content="my reference is REF-7781 [message id: B]"
    )

    async def scenario() -> None:
        indexer.schedule(SESSION, CHAT, [first])
        await indexer.drain()
        indexer.schedule(SESSION, CHAT, [redelivered])
        await indexer.drain()

    asyncio.run(scenario())

    assert index_rows(recall_settings.data_dir) == [("user", "my reference is REF-7781")]
    store = SimpleVectorStore.from_persist_path(
        str(vector_file(recall_settings.data_dir, SESSION, CHAT))
    )
    assert len(store.data.embedding_dict) == 1


def test_indexing_keeps_only_visible_user_and_assistant_turns(
    recall_settings: Settings,
) -> None:
    indexer = ThreadIndexer(recall_settings.data_dir, FakeEmbedding())
    messages = [
        ChatMessage(role=MessageRole.USER, content="hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="hi"),
        ChatMessage(role=MessageRole.TOOL, content='{"ok": true}'),
        ChatMessage(role=MessageRole.ASSISTANT, content="No response."),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="private lesson",
            additional_kwargs={WRAP_UP_NOTE_KWARG: True},
        ),
        ChatMessage(role=MessageRole.SYSTEM, content="system prompt"),
        ChatMessage(role=MessageRole.USER, content="   "),
    ]

    async def scenario() -> None:
        indexer.schedule(SESSION, CHAT, messages)
        await indexer.drain()

    asyncio.run(scenario())

    assert index_rows(recall_settings.data_dir) == [
        ("user", "hello"),
        ("assistant", "hi"),
    ]


def test_forget_wipes_the_thread_index(recall_settings: Settings) -> None:
    indexer = ThreadIndexer(recall_settings.data_dir, FakeEmbedding())
    turn = ChatMessage(role=MessageRole.USER, content="remember REF-7781")

    async def scenario() -> None:
        indexer.schedule(SESSION, CHAT, [turn])
        await indexer.drain()

    asyncio.run(scenario())
    assert index_file(recall_settings.data_dir, SESSION, CHAT).exists()
    assert vector_file(recall_settings.data_dir, SESSION, CHAT).exists()

    assert forget_thread_index(recall_settings.data_dir, SESSION, CHAT) is True

    assert not index_file(recall_settings.data_dir, SESSION, CHAT).exists()
    assert not vector_file(recall_settings.data_dir, SESSION, CHAT).exists()


def test_forget_clears_pending_index_work(recall_settings: Settings) -> None:
    indexer = ThreadIndexer(recall_settings.data_dir, FakeEmbedding())
    turn = ChatMessage(role=MessageRole.USER, content="remember REF-7781")

    indexer.schedule(SESSION, CHAT, [turn])
    indexer.forget(SESSION, CHAT)
    asyncio.run(indexer.drain())

    assert not index_file(recall_settings.data_dir, SESSION, CHAT).exists()


def test_a_transient_index_write_failure_keeps_the_batch(
    recall_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexer = ThreadIndexer(recall_settings.data_dir, FakeEmbedding())
    turn = ChatMessage(role=MessageRole.USER, content="remember REF-7781")
    original = thread_index_module.insert_turns
    calls = 0

    def flaky_insert(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise sqlite3.OperationalError("database is locked")
        original(*args, **kwargs)

    monkeypatch.setattr(thread_index_module, "insert_turns", flaky_insert)
    indexer.schedule(SESSION, CHAT, [turn])
    asyncio.run(indexer.drain())
    assert calls == 1
    assert not index_file(recall_settings.data_dir, SESSION, CHAT).exists()

    asyncio.run(indexer.drain())

    assert calls == 2
    assert index_rows(recall_settings.data_dir) == [("user", "remember REF-7781")]


def test_large_batches_drain_across_persist_points(
    recall_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(thread_index_module, "EMBED_BATCH", 2)
    indexer = ThreadIndexer(recall_settings.data_dir, FakeEmbedding())
    messages = [
        ChatMessage(role=MessageRole.USER, content=f"turn {index}") for index in range(3)
    ]

    async def scenario() -> tuple[int, int]:
        indexer.schedule(SESSION, CHAT, messages)
        await indexer.drain()
        first = embedded_count(recall_settings.data_dir)
        await indexer.drain()
        return first, embedded_count(recall_settings.data_dir)

    first, second = asyncio.run(scenario())

    assert first == 2
    assert second == 3


def test_thread_recall_kill_switch_disables_the_index(
    recall_settings: Settings,
) -> None:
    recall_settings.thread_recall = False
    llm = Mock(spec=FunctionCallingLLM)
    llm.metadata = LLMMetadata(is_function_calling_model=True)

    agent = FunctionCallingAgentWorkflow(llm=llm, settings=recall_settings)

    assert agent.thread_index is None


def test_memory_persist_off_disables_the_index(recall_settings: Settings) -> None:
    recall_settings.memory_persist = False
    llm = Mock(spec=FunctionCallingLLM)
    llm.metadata = LLMMetadata(is_function_calling_model=True)

    agent = FunctionCallingAgentWorkflow(llm=llm, settings=recall_settings)

    assert agent.thread_index is None


def test_dense_embedding_caps_long_turns(recall_settings: Settings) -> None:
    fake = FakeEmbedding()
    indexer = ThreadIndexer(recall_settings.data_dir, fake)
    long_text = "x" * 5000
    turn = ChatMessage(role=MessageRole.USER, content=long_text)

    async def scenario() -> None:
        indexer.schedule(SESSION, CHAT, [turn])
        await indexer.drain()

    asyncio.run(scenario())

    assert fake.texts
    assert max(len(text) for text in fake.texts) <= EMBED_TEXT_CHARS + len("[user] ")
    assert index_rows(recall_settings.data_dir) == [("user", long_text)]


class FlakyEmbedding(FakeEmbedding):
    """A model that rejects the first batch and accepts the shorter retry."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    async def aget_text_embedding_batch(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float]]:
        if not self.failed:
            self.failed = True
            self.batch_calls += 1
            raise RuntimeError("input is too large")
        return await super().aget_text_embedding_batch(texts, **kwargs)


def test_long_embedding_batches_retry_shorter(recall_settings: Settings) -> None:
    fake = FlakyEmbedding()
    indexer = ThreadIndexer(recall_settings.data_dir, fake)
    turn = ChatMessage(role=MessageRole.USER, content="y" * 5000)

    async def scenario() -> None:
        indexer.schedule(SESSION, CHAT, [turn])
        await indexer.drain()

    asyncio.run(scenario())

    assert fake.failed is True
    assert fake.batch_calls == 2
    assert fake.texts
    assert max(len(text) for text in fake.texts) <= (
        EMBED_TEXT_CHARS_RETRY + len("[user] ")
    )
    assert embedded_count(recall_settings.data_dir) == 1


def test_embeddings_down_falls_back_to_fts(recall_settings: Settings) -> None:
    fake = FakeEmbedding(down=True)
    indexer = ThreadIndexer(recall_settings.data_dir, fake)
    promise = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="Revisaré la solicitud y te enviaré el PDF mañana.",
    )

    async def scenario() -> list[str]:
        indexer.schedule(SESSION, CHAT, [promise])
        await indexer.drain()
        return await indexer.recall(SESSION, CHAT, "¿me enviaste el PDF?")

    slices = asyncio.run(scenario())

    assert any("PDF" in item for item in slices)
    assert not vector_file(recall_settings.data_dir, SESSION, CHAT).exists()
    # The batch and its shorter retry both fail, then the breaker skips
    # the down service entirely.
    assert fake.batch_calls == 2
    assert fake.query_calls == 0


def test_fts_keeps_the_exact_reference_token_floor(
    recall_settings: Settings,
) -> None:
    indexer = ThreadIndexer(recall_settings.data_dir)
    reference = ChatMessage(
        role=MessageRole.USER, content="My reference is REF-7781; please keep it."
    )

    async def scenario() -> list[str]:
        indexer.schedule(SESSION, CHAT, [reference])
        await indexer.drain()
        return await indexer.recall(SESSION, CHAT, "what about REF-7781?")

    slices = asyncio.run(scenario())

    assert any("REF-7781" in item for item in slices)


def test_dense_recall_covers_a_cross_lingual_question(
    recall_settings: Settings,
) -> None:
    indexer = ThreadIndexer(recall_settings.data_dir, FakeEmbedding())
    promise = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="Revisaré la solicitud y te enviaré el PDF mañana.",
    )
    question = "Hast du mir das Dokument morgen geschickt?"

    async def scenario() -> list[str]:
        indexer.schedule(SESSION, CHAT, [promise])
        await indexer.drain()
        return await indexer.recall(SESSION, CHAT, question)

    slices = asyncio.run(scenario())

    assert any("te enviaré el PDF mañana" in item for item in slices)


def test_indexing_never_restores_trimmed_turns_to_the_buffer(
    recall_settings: Settings,
) -> None:
    thread = support_threads()[0]
    promise = thread.turns[4].text
    indexer = ThreadIndexer(recall_settings.data_dir, FakeEmbedding())
    agent = recall_workflow(recall_settings, indexer)
    target = RunTarget(session=SESSION, chat_id=CHAT)

    async def scenario() -> tuple[list[ChatMessage], list[ChatMessage]]:
        token = bind_target(target)
        try:
            ctx = await recall_context(agent, fixture_messages(thread))
            await agent.chat_history(ctx)
            memory = await ctx.store.get("memory")
            trimmed = await memory.aget_all()
            await agent.merge_learned()
            after = await memory.aget_all()
            save_memory(recall_settings.data_dir, SESSION, CHAT, memory)
            restored = load_memory(recall_settings.data_dir, SESSION, CHAT)
            assert restored is not None
            return trimmed, after
        finally:
            reset_target(token)

    trimmed, after = asyncio.run(scenario())

    assert [str(message.content) for message in trimmed] == [
        str(message.content) for message in after
    ]
    assert promise not in [str(message.content) for message in after]
    restored = load_memory(recall_settings.data_dir, SESSION, CHAT)
    assert restored is not None
    assert promise not in [str(message.content) for message in restored.get_all()]
    assert promise in [text for _, text in index_rows(recall_settings.data_dir)]


def test_recall_is_dropped_before_pinned_state_when_the_budget_is_tight() -> None:
    block = render_pinned_block(
        policies=(Note("reply-style", "Be concise."),),
        recall=["user: " + "promise " * 200],
        token_budget=100,
        prompt_prefix="prompt ",
        prompt_budget=60,
        tokenizer=str.split,
    )

    assert "- reply-style: Be concise." in block
    assert "## Recalled from earlier in this thread" not in block


def test_recall_only_block_is_empty_when_it_cannot_fit() -> None:
    block = render_pinned_block(
        recall=["promise " * 200],
        token_budget=100,
        prompt_prefix="prompt ",
        prompt_budget=5,
        tokenizer=str.split,
    )

    assert block == ""


def test_recall_yields_to_the_recap_when_both_cannot_fit() -> None:
    block = render_pinned_block(
        recap="- promised the PDF tomorrow.",
        recall=["user: " + "lead " * 200],
        token_budget=100,
        prompt_prefix="prompt ",
        prompt_budget=60,
        tokenizer=str.split,
    )

    assert "## Conversation recap" in block
    assert "## Recalled from earlier in this thread" not in block


def test_recall_section_is_marked_as_leads() -> None:
    block = render_pinned_block(recall=["user: REF-7781"], token_budget=100)

    assert "## Recalled from earlier in this thread" in block
    assert "Leads, verify before relying on them" in block
    assert "- user: REF-7781" in block
