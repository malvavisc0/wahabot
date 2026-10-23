import asyncio
import datetime
import json
import unittest.mock
from pathlib import Path
from typing import Any

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import ToolSelection

from tests.harness import (
    CHAT_ID,
    FOREIGN_JID,
    SESSION,
)
from wahabot.ai.tools.runcontext import FENCE_ERROR as FENCE_ERROR_TEXT
from wahabot.ai.tools.whatsapp import (
    OPERATOR_ARMED,
    OPERATOR_KEY,
    chat_jid,
    fenced_chat,
    fenced_message_id,
)
from wahabot.core.models import WahaEvent


def test_chat_jid_resolution() -> None:
    holder = {"chat_id": CHAT_ID}
    assert chat_jid(None, holder) == CHAT_ID
    assert chat_jid(f"false_{CHAT_ID}_ABCDEF", holder) == CHAT_ID
    assert chat_jid(CHAT_ID, holder) == CHAT_ID


def test_fenced_chat() -> None:
    holder = {"chat_id": CHAT_ID}
    jid, err = fenced_chat(None, holder)
    assert jid == CHAT_ID and err is None
    jid, err = fenced_chat(CHAT_ID, holder)
    assert jid == CHAT_ID and err is None
    jid, err = fenced_chat(f"false_{CHAT_ID}_ABC", holder)
    assert jid == CHAT_ID and err is None
    jid, err = fenced_chat(FOREIGN_JID, holder)
    assert jid is None and err == FENCE_ERROR_TEXT
    jid, err = fenced_chat("1234", holder)
    assert jid is None and err is not None and "not a valid chat id" in err
    op_holder = {**holder, OPERATOR_KEY: OPERATOR_ARMED}
    jid, err = fenced_chat(FOREIGN_JID, op_holder)
    assert jid == FOREIGN_JID and err is None


def test_fenced_message_id() -> None:
    holder = {"chat_id": CHAT_ID}
    op_holder = {**holder, OPERATOR_KEY: OPERATOR_ARMED}
    mid, _ = fenced_message_id(f"false_{CHAT_ID}_XYZ", holder)
    assert mid == f"false_{CHAT_ID}_XYZ"
    mid, _ = fenced_message_id(f"false_{FOREIGN_JID}_XYZ", holder)
    assert mid is None
    mid, _ = fenced_message_id("odd-shape-no-at", holder)
    assert mid == "odd-shape-no-at"
    mid, _ = fenced_message_id(f"false_{FOREIGN_JID}_XYZ", op_holder)
    assert mid == f"false_{FOREIGN_JID}_XYZ"


def test_album_reassembly() -> None:
    from wahabot.ai.albums import Albums, is_album_container

    completed: list[list[str]] = []

    async def on_complete(buffer: object) -> None:
        completed.append(list(buffer.image_ids))  # type: ignore[attr-defined]

    albums = Albums(on_complete)
    container = WahaEvent(
        id="e3",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={
            "id": f"false_{CHAT_ID}_ALBUM",
            "from": CHAT_ID,
            "body": "",
            "_data": {"type": "album", "expectedImageCount": 2},
        },
    )
    assert is_album_container(container)
    assert albums.buffer(container) == f"false_{CHAT_ID}_ALBUM"
    image_event = WahaEvent(
        id="e4",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={
            "id": f"false_{CHAT_ID}_IMG1",
            "from": CHAT_ID,
            "body": "",
            "_data": {"type": "image"},
        },
    )
    assert albums.buffer(image_event) == f"false_{CHAT_ID}_ALBUM"
    assert albums.buffer(image_event) == f"false_{CHAT_ID}_ALBUM"
    assert albums.buffer(image_event) is None


def test_burst_buffer_state_machine() -> None:
    """Burst buffering: hold, extend, isolate senders, cap, complete."""

    from wahabot.ai import bursts

    def burst_event(
        mid: str, body: str, participant: str | None = None, chat: str = CHAT_ID
    ) -> WahaEvent:
        payload: dict[str, Any] = {
            "id": mid,
            "from": chat,
            "body": body,
            "_data": {"type": "chat"},
        }
        if participant is not None:
            payload["participant"] = participant
            payload["fromMe"] = False
        return WahaEvent(
            id=f"evt-{mid}",
            timestamp=1,
            event="message",
            session=SESSION,
            me={},
            payload=payload,
        )

    async def scenario() -> None:
        completed: list[str] = []

        async def record(buffer: bursts.BurstBuffer) -> None:
            completed.append(":".join(buffer.ids()))

        buffers = bursts.Bursts(0.3, 0.6, record)

        key = bursts.burst_key(burst_event("m1", "leak", participant="4915…@c.us"))
        assert key == (SESSION, CHAT_ID, "4915…@c.us")

        assert buffers.add("m1", burst_event("m1", "leak", participant="4915…@c.us"))
        other = burst_event("m2", "I also say something", participant="4916…@c.us")
        assert bursts.burst_key(other) == (SESSION, CHAT_ID, "4916…@c.us")
        assert buffers.add("m2", other)
        assert buffers.add("m3", burst_event("m3", "flat 3B", participant="4915…@c.us"))
        dm = burst_event("dm1", "hello", chat="4915…@c.us")
        assert bursts.burst_key(dm) == (SESSION, "4915…@c.us", "4915…@c.us")
        assert buffers.add("dm1", dm)

        await asyncio.sleep(0.5)
        assert sorted(completed) == ["dm1", "m1:m3", "m2"]
        assert not buffers.pending(CHAT_ID)
        buffers.reset()

    asyncio.run(scenario())


def test_burst_hold_cap() -> None:
    """The cap fires the flush mid-burst even while the sender keeps typing."""

    from wahabot.ai import bursts

    def burst_event(mid: str) -> WahaEvent:
        return WahaEvent(
            id=f"evt-{mid}",
            timestamp=1,
            event="message",
            session=SESSION,
            me={},
            payload={
                "id": mid,
                "from": CHAT_ID,
                "body": "more",
                "participant": "4915…@c.us",
                "fromMe": False,
                "_data": {"type": "chat"},
            },
        )

    async def scenario() -> None:
        flushed: list[int] = []

        async def record(buffer: bursts.BurstBuffer) -> None:
            flushed.append(len(buffer.message_ids))

        buffers = bursts.Bursts(5.0, 0.25, record)
        assert buffers.add("m1", burst_event("m1"))
        await asyncio.sleep(0.15)
        assert buffers.add("m2", burst_event("m2"))
        await asyncio.sleep(0.2)
        assert flushed and flushed[0] == 2
        assert len(flushed) == 1
        buffers.reset()

    asyncio.run(scenario())


def test_burst_refuses_unresolvable_sender() -> None:
    """A group message with no participant/author never buffers."""
    from wahabot.ai import bursts

    async def complete(_buffer: bursts.BurstBuffer) -> None:
        return None

    buffers = bursts.Bursts(0.2, 5.0, complete)
    event = WahaEvent(
        id="e-no-sender",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={"id": "m1", "from": CHAT_ID, "body": "leak"},
    )
    assert buffers.add("m1", event) is False
    assert not buffers.pending(CHAT_ID)
    buffers.reset()


def test_audit_save_action_and_caps(tmp_path: Path) -> None:
    """save_action appends JSONL under data/audit/<session>/ and caps fields."""
    from wahabot.core.audit import save_action

    save_action(
        tmp_path, SESSION, "reply", chat_id=CHAT_ID, reply="x" * 500, message_id="m1"
    )
    day = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
    path = tmp_path / "audit" / SESSION / f"{day}.jsonl"
    entry = json.loads(path.read_text(encoding="utf-8").strip())
    assert entry["kind"] == "reply"
    assert entry["chat_id"] == CHAT_ID
    assert entry["message_id"] == "m1"
    # The 500-char reply was capped to the 300-char audit budget.
    assert len(entry["reply"]) == 300
    assert entry["at"]


def test_tool_outcome_detects_enveloped_failures() -> None:
    """The audit ok flag reads the tool envelope, not the wrapper prefix.

    Tools report failures as ``{"ok": false, ...}`` envelopes — a
    refused escalation, a failed send, a fence refusal — which the
    old "Encountered error" prefix check journaled as successes.
    """
    from wahabot.ai.workflow import tool_outcome, tool_outcome_ok

    assert tool_outcome('{"ok": true, "chat": "c"}') == "completed"
    assert tool_outcome_ok('{"ok": true, "chat": "c"}')
    assert tool_outcome('{"ok": false, "error": "cooldown"}') == "failed"
    assert not tool_outcome_ok('{"ok": false, "error": "cooldown"}')
    assert tool_outcome("Tool nope does not exist") == "unknown"
    assert not tool_outcome_ok("Tool nope does not exist")
    # A non-JSON success payload (defensive: some tools return prose)
    # reads as completed, never crashes the audit.
    assert tool_outcome("done") == "completed"


@pytest.mark.parametrize(
    ("content", "outcome", "expected_ok"),
    [
        ('{"ok": true}', "completed", True),
        ('{"ok": false, "error": "refused"}', "failed", False),
        ("Tool missing does not exist", "unknown", False),
    ],
)
def test_tool_audit_preserves_outcome_fields(
    content: str, outcome: str, expected_ok: bool
) -> None:
    from wahabot.ai.workflow import FunctionCallingAgentWorkflow

    llm = unittest.mock.Mock()
    llm.metadata.is_function_calling_model = True
    agent = FunctionCallingAgentWorkflow(llm=llm)
    selection = ToolSelection(
        tool_id="tc", tool_name="sample", tool_kwargs={"query": "hello"}
    )
    message = ChatMessage(role="tool", content=content)
    with (
        unittest.mock.patch(
            "wahabot.ai.workflow.step_mixin.run_tool_call",
            new=unittest.mock.AsyncMock(return_value=message),
        ),
        unittest.mock.patch.object(agent, "audit") as audit,
    ):
        assert asyncio.run(agent.run_and_audit_tool_call({}, selection)) is message
    audit.assert_called_once_with(
        "tool_call",
        tool="sample",
        args={"query": "hello"},
        ok=expected_ok,
        outcome=outcome,
    )


def test_correction_rewrite_replaces_a_blame_reply() -> None:
    from llama_index.core.base.llms.types import ChatResponse
    from llama_index.core.workflow import Context

    from wahabot.ai.workflow import FunctionCallingAgentWorkflow

    llm = unittest.mock.Mock()
    llm.metadata.is_function_calling_model = True
    llm.achat = unittest.mock.AsyncMock(
        return_value=ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Tienes razón, me equivoqué al leer el mensaje.",
            )
        )
    )
    agent = FunctionCallingAgentWorkflow(llm=llm)
    context = Context(agent)
    memory = ChatMemoryBuffer.from_defaults()
    memory.put(ChatMessage(role=MessageRole.USER, content="read again: not you"))
    response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Te has equivocado, no era eso.",
        )
    )

    async def scenario() -> ChatResponse:
        await context.store.set("memory", memory)
        return await agent.rewrite_blame(context, response)

    rewritten = asyncio.run(scenario())

    assert "te has equivocado" not in str(rewritten.message.content)
    assert llm.achat.await_count == 1


def test_correction_rewrite_silences_a_still_blame_reply() -> None:
    from llama_index.core.base.llms.types import ChatResponse
    from llama_index.core.workflow import Context

    from wahabot.ai.workflow import FunctionCallingAgentWorkflow

    llm = unittest.mock.Mock()
    llm.metadata.is_function_calling_model = True
    llm.achat = unittest.mock.AsyncMock(
        return_value=ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Sí, te has equivocado otra vez.",
            )
        )
    )
    agent = FunctionCallingAgentWorkflow(llm=llm)
    context = Context(agent)
    memory = ChatMemoryBuffer.from_defaults()
    memory.put(ChatMessage(role=MessageRole.USER, content="read again: not you"))
    response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Te has equivocado, no era eso.",
        )
    )

    async def scenario() -> ChatResponse:
        await context.store.set("memory", memory)
        return await agent.rewrite_blame(context, response)

    rewritten = asyncio.run(scenario())

    assert str(rewritten.message.content) == ""


def test_correction_rewrite_keeps_a_non_blame_reply() -> None:
    from llama_index.core.base.llms.types import ChatResponse
    from llama_index.core.workflow import Context

    from wahabot.ai.workflow import FunctionCallingAgentWorkflow

    llm = unittest.mock.Mock()
    llm.metadata.is_function_calling_model = True
    llm.achat = unittest.mock.AsyncMock()
    agent = FunctionCallingAgentWorkflow(llm=llm)
    context = Context(agent)
    response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Tienes razón, me equivoqué al leer el mensaje.",
        )
    )

    rewritten = asyncio.run(agent.rewrite_blame(context, response))

    assert rewritten is response
    llm.achat.assert_not_awaited()
