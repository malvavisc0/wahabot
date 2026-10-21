import asyncio
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Context
from workflows.context.serializers import JsonSerializer

from tests.harness import CHAT_ID, chat_event
from tests.harness import SESSION as HARNESS_SESSION
from tests.triage.support import (
    CHAT,
    GROUP,
    SESSION,
    FakeGate,
    Sentinel,
    message_event,
)
from tests.wire.support import _wait
from wahabot.ai.context import handle_message
from wahabot.ai.messages import TURN_HANDLED_KWARG
from wahabot.ai.triage import (
    TriageResult,
    TriageVerdict,
    parse_verdict,
    result_from_dict,
    result_to_dict,
)
from wahabot.ai.workflow import FunctionCallingAgentWorkflow
from wahabot.core.notebook import Notebook
from wahabot.core.persistence import load_memory
from wahabot.evals.fixtures import GateLabel, load_gate_labels


def test_hard_muted_chat_never_answers_but_still_runs_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `chat-hard-mute` rule mutes the chat even when the gate would answer.

    The gate still runs (an adversarial read or a human request can
    still escalate), the turn is recorded stamped handled, and the
    agent is never started.
    """
    store = Notebook(tmp_path)
    store.write(
        SESSION,
        CHAT,
        "chat-hard-mute",
        "muted until the dust settles",
        kind="rule",
        author_kind="operator",
    )
    gate = FakeGate(result=TriageResult(TriageVerdict(participate=True)))
    agent = FunctionCallingAgentWorkflow(
        llm=Mock(spec=FunctionCallingLLM),
        system_prompt="x",
        notebook=store,
        triage=cast(Any, gate),
    )
    run = AsyncMock(side_effect=AssertionError("agent must not run"))
    monkeypatch.setattr(agent, "run", run)

    async def scenario() -> tuple[str, Any, list[ChatMessage]]:
        ctx = Context(agent)
        memory = ChatMemoryBuffer(token_limit=100, tokenizer_fn=str.split)
        await ctx.store.set("memory", memory)
        reply, target = await handle_message(
            message_event(CHAT, "hello?"),
            agent,
            ctx=ctx,
            gate_first=True,
        )
        return reply, target, await memory.aget_all()

    reply, target, stored = asyncio.run(scenario())

    assert reply == ""
    assert target.chat_id == CHAT
    assert gate.calls == 1
    run.assert_not_awaited()
    assert any(
        "hello?" in str(message.content)
        and message.additional_kwargs.get(TURN_HANDLED_KWARG)
        for message in stored
    )


def test_blocked_group_does_not_start_the_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = FakeGate(
        result=TriageResult(TriageVerdict(participate=False, gist="joked about the chat"))
    )
    agent = FunctionCallingAgentWorkflow(
        llm=Mock(spec=FunctionCallingLLM),
        system_prompt="x",
        triage=cast(Any, gate),
    )
    run = AsyncMock(side_effect=AssertionError("agent must not run"))
    monkeypatch.setattr(agent, "run", run)

    async def scenario() -> tuple[str, Any, list[ChatMessage], Context]:
        ctx = Context(agent)
        memory = ChatMemoryBuffer(token_limit=100, tokenizer_fn=str.split)
        await ctx.store.set("memory", memory)
        reply, target = await handle_message(
            message_event(GROUP, "lol did you see that"),
            agent,
            ctx=ctx,
            addressed=False,
            gate_first=True,
        )
        return reply, target, await memory.aget_all(), ctx

    reply, target, stored, ctx = asyncio.run(scenario())

    assert reply == ""
    assert target.chat_id == GROUP
    assert gate.calls == 1
    assert any("lol did you see that" in str(message.content) for message in stored)
    # Regression: everything the gate leaves in the context must survive
    # the workflow's JSON state serialization.
    ctx.store.to_dict(JsonSerializer())
    run.assert_not_awaited()


def test_gate_outage_keeps_an_unaddressed_group_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block mode fails closed: an unavailable gate does not wake the agent.

    Only a verdict that grants participation starts an unaddressed group
    run; the failed verdict is stored/audited and the turn is kept as a
    mechanical ledger line so the next prompt still knows it happened.
    """
    gate = FakeGate(result=TriageResult.fallback(failed=True))
    agent = FunctionCallingAgentWorkflow(
        llm=Mock(spec=FunctionCallingLLM),
        system_prompt="x",
        triage=cast(Any, gate),
    )
    run = AsyncMock(
        return_value=ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="saw it")
        )
    )
    monkeypatch.setattr(agent, "run", run)

    async def scenario() -> tuple[str, Any, Any]:
        ctx = Context(agent)
        reply, target = await handle_message(
            message_event(GROUP, "did you see that"),
            agent,
            ctx=ctx,
            addressed=False,
            gate_first=True,
        )
        stored = await ctx.store.get("triage_result", default=None)
        return reply, target, stored

    reply, target, stored = asyncio.run(scenario())

    assert reply == ""
    assert target.chat_id == GROUP
    assert gate.calls == 1
    run.assert_not_awaited()
    assert isinstance(stored, dict) and stored["failed"] is True


def test_dm_starts_the_gate_before_the_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = FakeGate()
    agent = FunctionCallingAgentWorkflow(
        llm=Mock(spec=FunctionCallingLLM),
        system_prompt="x",
        triage=cast(Any, gate),
    )

    async def fail_run(**kwargs: Any) -> None:
        await asyncio.sleep(0)
        raise Sentinel

    monkeypatch.setattr(agent, "run", fail_run)

    async def scenario() -> None:
        ctx = Context(agent)
        await handle_message(message_event(CHAT, "hello"), agent, ctx=ctx)

    with pytest.raises(Sentinel):
        asyncio.run(scenario())

    assert gate.calls == 1


def test_race_gate_state_is_serializable(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = FakeGate()
    agent = FunctionCallingAgentWorkflow(
        llm=Mock(spec=FunctionCallingLLM),
        system_prompt="x",
        triage=cast(Any, gate),
    )

    async def serialize_then_fail(**kwargs: Any) -> None:
        await asyncio.sleep(0)
        ctx = kwargs["ctx"]
        ctx.store.to_dict(JsonSerializer())
        raise Sentinel

    monkeypatch.setattr(agent, "run", serialize_then_fail)

    async def scenario() -> None:
        ctx = Context(agent)
        await handle_message(message_event(CHAT, "hello"), agent, ctx=ctx)

    with pytest.raises(Sentinel):
        asyncio.run(scenario())

    assert gate.calls == 1


def test_operator_run_never_calls_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = FakeGate(error=True)
    agent = FunctionCallingAgentWorkflow(
        llm=Mock(spec=FunctionCallingLLM),
        system_prompt="x",
        triage=cast(Any, gate),
    )

    async def fail_run(**kwargs: Any) -> None:
        await asyncio.sleep(0)
        raise Sentinel

    monkeypatch.setattr(agent, "run", fail_run)

    async def scenario() -> None:
        ctx = Context(agent)
        await handle_message(message_event("operator", "do the thing"), agent, ctx=ctx)

    with pytest.raises(Sentinel):
        asyncio.run(scenario())

    assert gate.calls == 0


def test_gate_blocks_an_unaddressed_group_on_the_wire(bot: Any) -> None:
    config_path = bot.settings.access_config
    config = json.loads(config_path.read_text())
    config["group_participation"] = "judicious"
    config_path.write_text(json.dumps(config))
    gate = FakeGate(
        result=TriageResult(TriageVerdict(participate=False, gist="joked about the chat"))
    )
    bot.agent.triage = cast(Any, gate)
    before = len(bot.stack.llm.requests)

    bot.post(chat_event(body="lol did you see that", mid="GATE-BLOCK"))

    assert gate.calls == 1
    assert len(bot.stack.llm.requests) == before
    assert bot.waha.sent == []
    assert (HARNESS_SESSION, CHAT_ID) in bot.waha.seen_chats
    memory = load_memory(bot.settings.data_dir, HARNESS_SESSION, CHAT_ID)
    assert memory is not None
    contents = [str(message.content) for message in memory.get_all()]
    assert any("joked about the chat" in text and "— #m" in text for text in contents)
    assert all("lol did you see that" not in text for text in contents)


def test_parse_verdict_clamps_the_ledger_gist() -> None:
    raw = json.dumps(
        {
            "adversarial": False,
            "escalate": False,
            "participate": False,
            "why": "banter",
            "gist": '"' + " ".join(f"word{i}" for i in range(20)) + '"',
        }
    )
    verdict = parse_verdict(raw)
    assert len(verdict.gist.split()) == 16
    assert verdict.gist.endswith("…")
    assert '"' not in verdict.gist


def test_gist_roundtrips_through_context_state() -> None:
    result = TriageResult(TriageVerdict(gist="asked about the invoice"))
    assert result_from_dict(result_to_dict(result)).verdict.gist == (
        "asked about the invoice"
    )


def test_gate_silenced_turn_survives_into_the_next_run(bot: Any) -> None:
    """A gate-silenced group message rides the next run's history.

    The gated turn is stored stamped ``turn_handled`` (the gate's
    decision concluded it, exactly like a silent run's stamp). Without
    the stamp the next run's ``repair_memory`` dropped the unstamped
    trailing user message as run scaffolding: the silenced message
    vanished from the buffer, the recap and the thread index alike,
    and replies came out "factually correct but out of context".
    """
    config_path = bot.settings.access_config
    config = json.loads(config_path.read_text())
    config["group_participation"] = "judicious"
    config_path.write_text(json.dumps(config))
    gate = FakeGate(
        result=TriageResult(TriageVerdict(participate=False, gist="joked about the chat"))
    )
    bot.agent.triage = cast(Any, gate)

    # Message 1: unaddressed banter; the gate keeps the bot silent.
    bot.post(chat_event(body="lol did you see that", mid="GATE-BLOCK"))
    assert gate.calls == 1
    assert bot.waha.sent == []

    # Message 2: addressed to the bot, so the agent runs and must see
    # message 1 in its history.
    gate.result = TriageResult(TriageVerdict(participate=True))
    llm = bot.stack.llm
    llm.clear()
    bot.post(chat_event(body="kai what did you think of it?", mid="FOLLOW-UP"))
    assert _wait(lambda: len(llm.requests) >= 1)
    turns = [
        str(m.get("content", ""))
        for m in llm.requests[0]["messages"]
        if m.get("role") == "user"
    ]
    assert any("joked about the chat" in turn and "— #m" in turn for turn in turns), (
        "the gate-silenced message must ride the next run as a ledger line"
    )
    assert all("lol did you see that" not in turn for turn in turns)
    assert any("viste" in turn or "did you think" in turn for turn in turns), (
        "the new message itself must ride its own run"
    )


def test_load_gate_labels_reads_recorded_verdicts(tmp_path: Path) -> None:
    path = tmp_path / "gate.json"
    path.write_text(
        json.dumps(
            {
                "case-a": {"adversarial": True, "participate": False},
                "case-b": {"escalate": True},
            }
        )
    )

    labels = load_gate_labels(path)

    assert labels["case-a"] == GateLabel(adversarial=True, participate=False)
    assert labels["case-b"] == GateLabel(escalate=True)
