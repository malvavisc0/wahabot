"""Offline regressions for durable facts, shared policies, and private self-lessons."""

import asyncio
import hashlib
import json
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
from llama_index.core.base.llms.types import ChatMessage, LLMMetadata, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool, ToolSelection
from llama_index.core.workflow import Context

from wahabot.ai.history import ContextBudgetError
from wahabot.ai.messages import WRAP_UP_NOTE_KWARG
from wahabot.ai.tools.notebook import (
    MAX_NOTE_TOKENS,
    NOTEBOOK_HEADING,
    NOTEBOOK_TOKEN_LIMIT,
    OPERATOR_NOTEBOOK_HEADING,
    notebook_builder,
    notebook_prompt,
)
from wahabot.ai.tools.whatsapp import (
    RunTarget,
    bind_target,
    current_target,
    reset_target,
)
from wahabot.ai.workflow.agent import FunctionCallingAgentWorkflow, build_agent
from wahabot.ai.workflow.text import message_text
from wahabot.ai.workflow.toolkit import run_tool_call
from wahabot.core.notebook import (
    MAX_READ_NOTES,
    SESSION_NOTEBOOK_CHAT,
    SESSION_POLICY_TOKEN_LIMIT,
    Note,
    Notebook,
    notebook_operation,
)
from wahabot.core.persistence import load_memory, save_memory
from wahabot.core.tokens import text_tokens, truncate_to_tokens
from wahabot.settings import Settings

SESSION = "notebook-test"
OTHER_SESSION = "other-notebook-test"
CHAT = "491555000021@c.us"
OTHER_CHAT = "491555000022@c.us"


@pytest.fixture()
def notebook_settings(tmp_path: Path) -> Settings:
    return Settings(
        webhook_hmac_key="fictional-key",
        waha_url="http://waha.invalid",
        waha_api_key="fictional-key",
        llm_api_base="http://llm.invalid",
        llm_api_key="fictional-key",
        langfuse_public_key="",
        langfuse_secret_key="",
        session=SESSION,
        data_dir=tmp_path,
        memory_token_limit=4096,
        _env_file=None,
    )


@pytest.fixture()
def notebook_tool(notebook_settings: Settings) -> FunctionTool:
    return cast(FunctionTool, notebook_builder(notebook_settings))


@contextmanager
def bound_target(target: RunTarget) -> Iterator[None]:
    token = bind_target(target)
    try:
        yield
    finally:
        reset_target(token)


def notebook_workflow(store: Notebook) -> FunctionCallingAgentWorkflow:
    llm = Mock(spec=FunctionCallingLLM)
    llm.metadata = LLMMetadata(is_function_calling_model=True)
    return FunctionCallingAgentWorkflow(
        llm=llm, notebook=store, system_prompt="Follow configured permissions."
    )


async def notebook_context(
    agent: FunctionCallingAgentWorkflow, messages: list[ChatMessage]
) -> Context:
    ctx = Context(agent)
    memory = ChatMemoryBuffer(
        token_limit=agent.memory_token_limit, tokenizer_fn=str.split
    )
    await memory.aset(messages)
    await ctx.store.set("memory", memory)
    return ctx


async def call_notebook(tool: FunctionTool, **kwargs: Any) -> dict[str, Any]:
    selection = ToolSelection(
        tool_id="notebook-call", tool_name="notebook", tool_kwargs=kwargs
    )
    message = await run_tool_call({"notebook": tool}, selection)
    assert message.role == MessageRole.TOOL
    assert message.additional_kwargs["tool_call_id"] == selection.tool_id
    return json.loads(str(message.content))


def request_tokens(messages: list[ChatMessage]) -> int:
    return sum(text_tokens(message_text(message), str.split) for message in messages)


def system_tokens(agent: FunctionCallingAgentWorkflow) -> int:
    system = agent.system_message()
    assert system is not None
    return request_tokens([system])


def tool_group(
    *,
    query: str = "Which meeting day was saved?",
    result: str = '{"ok": true, "notes": []}',
    arguments: str = '{"mode": "read", "key": "meeting"}',
) -> list[ChatMessage]:
    return [
        ChatMessage(role="user", content=query),
        ChatMessage(
            role="assistant",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "lookup-note",
                        "type": "function",
                        "function": {"name": "notebook", "arguments": arguments},
                    }
                ]
            },
        ),
        ChatMessage(
            role="tool",
            content=result,
            additional_kwargs={"tool_call_id": "lookup-note", "name": "notebook"},
        ),
    ]


def test_notebook_lazy_crud_and_reopen(tmp_path: Path) -> None:
    data_dir = tmp_path / "synthetic-state"
    store = Notebook(data_dir)
    assert store.path == data_dir / "notebook.sqlite3"
    assert not data_dir.exists()

    store.write(SESSION, CHAT, " meeting ", " Workshop meets on Tuesday. ")
    store.write(SESSION, CHAT, "format", "Use plain text.")
    store.write(SESSION, CHAT, "meeting", "Verified: workshop meets on Thursday.")
    reopened = Notebook(data_dir)
    assert reopened.read(SESSION, CHAT) == [
        Note("meeting", "Verified: workshop meets on Thursday."),
        Note("format", "Use plain text."),
    ]
    assert reopened.read(SESSION, CHAT, key=" meeting ") == [
        Note("meeting", "Verified: workshop meets on Thursday.")
    ]
    assert reopened.read(SESSION, CHAT, limit=1)[0].key == "meeting"
    assert reopened.delete(SESSION, CHAT, "meeting") is True
    assert reopened.delete(SESSION, CHAT, "meeting") is False
    assert Notebook(data_dir).read(SESSION, CHAT) == [Note("format", "Use plain text.")]


@pytest.mark.parametrize(
    ("query", "keys"),
    [
        (" MEETING ", {"meeting"}),
        ("THURSDAY", {"meeting"}),
        ("50%_ready", {"literal"}),
        ("%", {"literal"}),
        ("' OR 1=1 --", set()),
    ],
)
def test_notebook_query_is_literal_and_case_insensitive(
    tmp_path: Path, query: str, keys: set[str]
) -> None:
    store = Notebook(tmp_path)
    store.write(SESSION, CHAT, "meeting", "Workshop meets on Thursday.")
    store.write(SESSION, CHAT, "literal", "The fictional prototype is 50%_ready.")
    assert {note.key for note in store.read(SESSION, CHAT, query=query)} == keys
    assert store.read(SESSION, CHAT, key="literal", query="Thursday") == []


@pytest.mark.parametrize("limit", [0, MAX_READ_NOTES + 1])
def test_notebook_rejects_unbounded_read(tmp_path: Path, limit: int) -> None:
    with pytest.raises(ValueError, match="Notebook limit"):
        Notebook(tmp_path).read(SESSION, CHAT, limit=limit)


def test_notebook_scopes_and_forget_are_independent(tmp_path: Path) -> None:
    store = Notebook(tmp_path)
    scopes = [
        (SESSION, CHAT),
        (SESSION, OTHER_CHAT),
        (OTHER_SESSION, CHAT),
        (SESSION, "operator"),
        (SESSION, SESSION_NOTEBOOK_CHAT),
    ]
    for index, (session, chat) in enumerate(scopes):
        store.write(session, chat, "same-key", f"Fictional fact {index}.")
    store.forget(SESSION, CHAT)
    store.forget(SESSION, CHAT)
    assert store.read(SESSION, CHAT) == []
    for index, (session, chat) in enumerate(scopes[1:], start=1):
        assert Notebook(tmp_path).read(session, chat) == [
            Note("same-key", f"Fictional fact {index}.")
        ]


def test_notebook_scope_listing_newest_first_and_session_scoped(tmp_path: Path) -> None:
    """scopes() lists chat_id + count for one session, newest writes first."""
    store = Notebook(tmp_path)
    store.write(SESSION, CHAT, "older", "Old fact.")
    time.sleep(0.01)
    store.write(SESSION, OTHER_CHAT, "newer", "New fact.")
    store.write(SESSION, OTHER_CHAT, "newer-still", "Another new fact.")
    store.write(OTHER_SESSION, CHAT, "other-session", "Not this session.")
    assert store.scopes(SESSION) == [(OTHER_CHAT, 2), (CHAT, 1)]
    assert store.scopes(OTHER_SESSION) == [(CHAT, 1)]
    assert store.scopes("missing") == []


def test_concurrent_writes_and_replacements_preserve_unrelated_notes(
    tmp_path: Path,
) -> None:
    store = Notebook(tmp_path)
    store.write(SESSION, CHAT, "unrelated", "Keep this saved fact.")
    for index in range(8):
        store.write(SESSION, CHAT, f"replace-{index}", "Old value.")
    barrier = threading.Barrier(8)
    replacements = [f"Complete replacement {index}: " + "x" * 500 for index in range(8)]

    def write(index: int) -> None:
        barrier.wait(timeout=10)
        store.write(SESSION, CHAT, f"new-{index}", f"New fact {index}.")
        store.write(SESSION, CHAT, f"replace-{index}", replacements[index])
        store.write(SESSION, CHAT, "contended", replacements[index])

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(8)))
    notes = {note.key: note.text for note in Notebook(tmp_path).read(SESSION, CHAT)}
    assert len(notes) == 18
    assert notes["unrelated"] == "Keep this saved fact."
    assert notes["contended"] in replacements
    for index, replacement in enumerate(replacements):
        assert notes[f"new-{index}"] == f"New fact {index}."
        assert notes[f"replace-{index}"] == replacement


def test_session_policy_count_limit_rolls_back_insert(tmp_path: Path) -> None:
    store = Notebook(tmp_path)
    for index in range(MAX_READ_NOTES):
        store.write(
            SESSION,
            SESSION_NOTEBOOK_CHAT,
            f"policy-{index}",
            "Keep answers concise.",
            max_total_tokens=SESSION_POLICY_TOKEN_LIMIT,
        )
    before = store.read(SESSION, SESSION_NOTEBOOK_CHAT)
    with pytest.raises(ValueError, match="Session notebook is full"):
        store.write(
            SESSION,
            SESSION_NOTEBOOK_CHAT,
            "overflow",
            "Confirm before acting.",
            max_total_tokens=SESSION_POLICY_TOKEN_LIMIT,
        )
    assert store.read(SESSION, SESSION_NOTEBOOK_CHAT) == before
    assert store.read(SESSION, SESSION_NOTEBOOK_CHAT, key="overflow") == []


def test_concurrent_policy_writes_enforce_aggregate_budget(tmp_path: Path) -> None:
    store = Notebook(tmp_path)
    store.write(
        SESSION,
        SESSION_NOTEBOOK_CHAT,
        "existing",
        "x " * 450,
        max_total_tokens=SESSION_POLICY_TOKEN_LIMIT,
    )
    barrier = threading.Barrier(2)

    def write(key: str) -> bool:
        barrier.wait(timeout=10)
        try:
            store.write(
                SESSION,
                SESSION_NOTEBOOK_CHAT,
                key,
                "y " * 360,
                max_total_tokens=SESSION_POLICY_TOKEN_LIMIT,
            )
        except ValueError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(write, ("first", "second"))) == [False, True]
    notes = store.read(SESSION, SESSION_NOTEBOOK_CHAT)
    assert len(notes) == 2
    assert Note("existing", ("x " * 450).strip()) in notes
    assert text_tokens(json.dumps([asdict(note) for note in notes])) <= 1024


def test_participant_can_correct_and_delete_own_fact(
    notebook_tool: FunctionTool, tmp_path: Path
) -> None:
    with bound_target(RunTarget(session=SESSION, chat_id=CHAT)):
        assert notebook_tool.metadata.get_name() == "notebook"
        assert json.loads(
            notebook_tool.fn(mode="write", key="meeting", text="Reported: Tuesday.")
        )["saved"]
        assert json.loads(
            notebook_tool.fn(
                mode="write", key="meeting", text="Verified correction: Thursday."
            )
        )["saved"]
        assert json.loads(notebook_tool.fn(mode="read", query="CORRECTION"))["notes"] == [
            {"key": "meeting", "text": "Verified correction: Thursday."}
        ]
        assert json.loads(notebook_tool.fn(mode="delete", key="meeting"))["deleted"]
        assert not json.loads(notebook_tool.fn(mode="delete", key="meeting"))["deleted"]
    assert Notebook(tmp_path).read(SESSION, CHAT) == []


@pytest.mark.parametrize("mode", ["read", "write", "delete"])
@pytest.mark.parametrize(
    "chat",
    [OTHER_CHAT, f"false_{OTHER_CHAT}_FICTIONAL", "operator", SESSION_NOTEBOOK_CHAT],
)
def test_participant_cannot_access_other_notebooks(
    notebook_tool: FunctionTool, tmp_path: Path, mode: str, chat: str
) -> None:
    store = Notebook(tmp_path)
    scopes = [CHAT, OTHER_CHAT, "operator", SESSION_NOTEBOOK_CHAT]
    for scope in scopes:
        store.write(SESSION, scope, "saved", f"Private fictional fact for {scope}.")
    before = {scope: store.read(SESSION, scope) for scope in scopes}
    with bound_target(RunTarget(session=SESSION, chat_id=CHAT)):
        result = asyncio.run(
            call_notebook(
                notebook_tool, mode=mode, chat=chat, key="saved", text="Changed."
            )
        )
    assert result["ok"] is False
    assert result["error"]
    assert "notes" not in result
    assert {scope: store.read(SESSION, scope) for scope in scopes} == before


@pytest.mark.parametrize("mode", ["read", "write", "delete"])
def test_session_scope_requires_armed_operator(
    notebook_tool: FunctionTool, tmp_path: Path, mode: str
) -> None:
    store = Notebook(tmp_path)
    store.write(SESSION, SESSION_NOTEBOOK_CHAT, "policy", "Keep answers concise.")
    with bound_target(RunTarget(session=SESSION, chat_id=CHAT)):
        result = asyncio.run(
            call_notebook(
                notebook_tool, mode=mode, scope="session", key="policy", text="Changed."
            )
        )
    assert result["ok"] is False
    assert "operator command" in result["error"]
    assert store.read(SESSION, SESSION_NOTEBOOK_CHAT) == [
        Note("policy", "Keep answers concise.")
    ]


@pytest.mark.parametrize("chat_id", ["operator", SESSION_NOTEBOOK_CHAT])
def test_unarmed_reserved_target_does_not_grant_notebook_access(
    notebook_tool: FunctionTool, chat_id: str
) -> None:
    with bound_target(RunTarget(session=SESSION, chat_id=chat_id)):
        result = asyncio.run(call_notebook(notebook_tool, mode="read"))
    assert result["ok"] is False
    assert result["error"]


def test_operator_default_is_private_and_explicit_chat_can_be_managed(
    notebook_tool: FunctionTool, tmp_path: Path
) -> None:
    with bound_target(RunTarget(session=SESSION, chat_id="operator", armed=True)):
        assert json.loads(
            notebook_tool.fn(mode="write", key="private", text="Private draft plan.")
        )["ok"]
        assert json.loads(
            notebook_tool.fn(
                mode="write", chat=OTHER_CHAT, key="meeting", text="Workshop: Thursday."
            )
        )["ok"]
        assert json.loads(notebook_tool.fn(mode="read"))["notes"] == [
            {"key": "private", "text": "Private draft plan."}
        ]
        assert json.loads(notebook_tool.fn(mode="read", chat=OTHER_CHAT))["notes"] == [
            {"key": "meeting", "text": "Workshop: Thursday."}
        ]
        assert json.loads(
            notebook_tool.fn(mode="delete", chat=OTHER_CHAT, key="meeting")
        )["deleted"]
    store = Notebook(tmp_path)
    assert store.read(SESSION, "operator") == [Note("private", "Private draft plan.")]
    assert store.read(SESSION, OTHER_CHAT) == []
    assert store.read(SESSION, SESSION_NOTEBOOK_CHAT) == []
    assert store.read(OTHER_SESSION, "operator") == []


def test_operator_can_correct_and_delete_shared_policy(
    notebook_tool: FunctionTool, tmp_path: Path
) -> None:
    with bound_target(RunTarget(session=SESSION, chat_id="operator", armed=True)):
        assert json.loads(
            notebook_tool.fn(
                mode="write", scope="session", key="policy", text="Use concise replies."
            )
        )["ok"]
        assert json.loads(
            notebook_tool.fn(
                mode="write", scope="session", key="policy", text="Ask before acting."
            )
        )["ok"]
        assert json.loads(notebook_tool.fn(mode="read", scope="session"))["notes"] == [
            {"key": "policy", "text": "Ask before acting."}
        ]
        assert json.loads(notebook_tool.fn(mode="delete", scope="session", key="policy"))[
            "deleted"
        ]
    assert Notebook(tmp_path).read(SESSION, SESSION_NOTEBOOK_CHAT) == []


@pytest.mark.parametrize("mode", ["read", "write", "delete"])
@pytest.mark.parametrize("chat", [CHAT, ""])
def test_operator_session_scope_cannot_include_chat(
    notebook_tool: FunctionTool, mode: str, chat: str
) -> None:
    with bound_target(RunTarget(session=SESSION, chat_id="operator", armed=True)):
        result = asyncio.run(
            call_notebook(
                notebook_tool,
                mode=mode,
                scope="session",
                chat=chat,
                key="policy",
                text="Use concise replies.",
            )
        )
    assert result["ok"] is False
    assert "scope=session cannot" in result["error"]


@pytest.mark.parametrize("mode", ["read", "write", "delete"])
@pytest.mark.parametrize("chat", [SESSION_NOTEBOOK_CHAT, "false_@session_FICTIONAL"])
def test_reserved_session_notebook_cannot_be_reached_as_chat(
    notebook_tool: FunctionTool, tmp_path: Path, mode: str, chat: str
) -> None:
    store = Notebook(tmp_path)
    store.write(SESSION, SESSION_NOTEBOOK_CHAT, "policy", "Use concise replies.")
    with bound_target(RunTarget(session=SESSION, chat_id="operator", armed=True)):
        result = asyncio.run(
            call_notebook(
                notebook_tool, mode=mode, chat=chat, key="policy", text="Changed."
            )
        )
    assert result["ok"] is False
    assert "scope=session" in result["error"]
    assert store.read(SESSION, SESSION_NOTEBOOK_CHAT) == [
        Note("policy", "Use concise replies.")
    ]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({}, "mode"),
        ({"mode": "replace"}, "mode"),
        ({"mode": "read", "scope": "global"}, "scope"),
        ({"mode": "read", "limit": "2"}, "limit"),
        ({"mode": "read", "limit": True}, "limit"),
        ({"mode": "read", "limit": 0}, "limit"),
        ({"mode": "read", "limit": 21}, "limit"),
        ({"mode": "write", "key": 1, "text": "Fact."}, "key"),
        ({"mode": "write", "key": "saved", "text": 1}, "text"),
        ({"mode": "write", "key": "x" * 81, "text": "Fact."}, "key"),
        ({"mode": "write", "key": "saved", "text": "x" * 2001}, "text"),
        ({"mode": "write", "key": " ", "text": "Fact."}, "Notebook key"),
        ({"mode": "write", "key": "saved", "text": " "}, "Notebook text"),
        ({"mode": "read", "unexpected": "value"}, "valid arguments"),
    ],
)
def test_notebook_bad_arguments_return_error_envelope_without_mutation(
    notebook_tool: FunctionTool, tmp_path: Path, kwargs: dict[str, Any], error: str
) -> None:
    store = Notebook(tmp_path)
    store.write(SESSION, CHAT, "saved", "Original fictional fact.")
    with bound_target(RunTarget(session=SESSION, chat_id=CHAT)):
        result = asyncio.run(call_notebook(notebook_tool, **kwargs))
    assert result["ok"] is False
    assert result["tool"] == "notebook"
    assert error in result["error"]
    assert store.read(SESSION, CHAT) == [Note("saved", "Original fictional fact.")]


def test_model_note_token_limit_rejects_replacement_without_losing_fact(
    notebook_tool: FunctionTool, tmp_path: Path
) -> None:
    accepted = " ".join(["x"] * MAX_NOTE_TOKENS)
    rejected = accepted + " x"
    assert text_tokens(accepted) == MAX_NOTE_TOKENS
    assert len(rejected) < 2000
    with bound_target(RunTarget(session=SESSION, chat_id=CHAT)):
        assert json.loads(notebook_tool.fn(mode="write", key="saved", text=accepted))[
            "ok"
        ]
        result = asyncio.run(
            call_notebook(notebook_tool, mode="write", key="saved", text=rejected)
        )
    assert result["ok"] is False
    assert "512 tokens" in result["error"]
    assert Notebook(tmp_path).read(SESSION, CHAT) == [Note("saved", accepted)]


def test_policy_aggregate_token_failure_rolls_back_previous_value(
    notebook_tool: FunctionTool, tmp_path: Path
) -> None:
    store = Notebook(tmp_path)
    with bound_target(RunTarget(session=SESSION, chat_id="operator", armed=True)):
        for index in range(3):
            assert json.loads(
                notebook_tool.fn(
                    mode="write", scope="session", key=f"policy-{index}", text="x " * 280
                )
            )["ok"]
        before = store.read(SESSION, SESSION_NOTEBOOK_CHAT)
        replacement = "x " * 480
        assert text_tokens(replacement) <= MAX_NOTE_TOKENS
        candidate = [
            asdict(Note(note.key, replacement.strip()))
            if note.key == "policy-0"
            else asdict(note)
            for note in before
        ]
        assert text_tokens(json.dumps(candidate)) > SESSION_POLICY_TOKEN_LIMIT
        result = asyncio.run(
            call_notebook(
                notebook_tool,
                mode="write",
                scope="session",
                key="policy-0",
                text=replacement,
            )
        )
    assert result["ok"] is False
    assert "Session notebook is full" in result["error"]
    assert Notebook(tmp_path).read(SESSION, SESSION_NOTEBOOK_CHAT) == before


def test_bounded_tool_reads_can_search_older_facts(
    notebook_tool: FunctionTool, notebook_settings: Settings
) -> None:
    store = Notebook(notebook_settings.data_dir)
    for index in range(12):
        store.write(
            SESSION,
            CHAT,
            f"fact-{index:02}",
            f"Workshop fact {index:02}. " + "detail " * 180,
        )
    with bound_target(RunTarget(session=SESSION, chat_id=CHAT)):
        result = json.loads(notebook_tool.fn(mode="read"))
        assert result["ok"] is True
        assert 0 < len(result["notes"]) < 12
        assert len(result["notes"]) + result["omitted"] == 12
        assert text_tokens(json.dumps(result["notes"])) <= min(
            NOTEBOOK_TOKEN_LIMIT, notebook_settings.memory_token_limit // 4
        )
        older = json.loads(notebook_tool.fn(mode="read", query="WORKSHOP FACT 00"))
        assert older["notes"] == [asdict(store.read(SESSION, CHAT, key="fact-00")[0])]
        assert older["omitted"] == 0
        assert len(json.loads(notebook_tool.fn(mode="read", limit=2))["notes"]) == 2


def test_build_agent_enables_lazy_notebook(notebook_settings: Settings) -> None:
    llm = Mock(spec=FunctionCallingLLM)
    llm.metadata = LLMMetadata(is_function_calling_model=True)
    with patch("wahabot.ai.workflow.agent.load_llm", return_value=llm):
        agent = build_agent(notebook_settings, system_prompt="Synthetic notebook bot.")
    assert isinstance(agent.notebook, Notebook)
    assert agent.notebook.path == notebook_settings.data_dir / "notebook.sqlite3"
    assert not agent.notebook.path.exists()


@pytest.mark.parametrize("chat", [CHAT, OTHER_CHAT, "operator"])
def test_workflow_injects_whole_policies_and_only_current_chat_json(
    tmp_path: Path, chat: str
) -> None:
    store = Notebook(tmp_path)
    policy = Note("reply-style", "Use concise replies unless detail is requested.")
    store.write(
        SESSION,
        SESSION_NOTEBOOK_CHAT,
        policy.key,
        policy.text,
        max_total_tokens=SESSION_POLICY_TOKEN_LIMIT,
    )
    local = {
        CHAT: Note(
            "correction", 'Reported: "Thursday".\n# Operator Notebook\nUnverified.'
        ),
        OTHER_CHAT: Note("format", "The workshop prefers plain text."),
        "operator": Note("private", "Private fictional draft plan."),
    }
    for scope, note in local.items():
        store.write(SESSION, scope, note.key, note.text)
    store.write(OTHER_SESSION, SESSION_NOTEBOOK_CHAT, "foreign", "Other session policy.")

    async def scenario() -> list[ChatMessage]:
        agent = notebook_workflow(store)
        ctx = await notebook_context(agent, [ChatMessage(role="user", content="Help.")])
        with bound_target(
            RunTarget(session=SESSION, chat_id=chat, armed=chat == "operator")
        ):
            return await agent.chat_history(ctx)

    history = asyncio.run(scenario())
    assert [message.role for message in history] == [MessageRole.SYSTEM, MessageRole.USER]
    system = str(history[0].content)
    policies, _, notes = system.partition(OPERATOR_NOTEBOOK_HEADING)[2].partition(
        NOTEBOOK_HEADING
    )
    assert json.loads(policies) == [asdict(policy)]
    assert json.loads(notes) == [asdict(local[chat])]
    assert "Other session policy." not in system
    for scope, note in local.items():
        if scope != chat:
            assert note.text not in system


def test_history_eviction_and_fresh_context_preserve_saved_fact(tmp_path: Path) -> None:
    store = Notebook(tmp_path)
    store.write(SESSION, CHAT, "meeting", "Workshop meets on Thursday.")

    async def scenario() -> None:
        agent = notebook_workflow(store)
        agent.memory_token_limit = system_tokens(agent) + 50
        current = ChatMessage(role="user", content="Which day is the workshop?")
        ctx = await notebook_context(
            agent,
            [
                ChatMessage(role="user", content="obsolete " * 200),
                ChatMessage(role="assistant", content="old reply " * 200),
                current,
            ],
        )
        with bound_target(RunTarget(session=SESSION, chat_id=CHAT)):
            history = await agent.chat_history(ctx)
            memory = await ctx.store.get("memory")
            assert memory.get_all() == [current]
            assert "Thursday" in str(history[0].content)
            store.write(SESSION, CHAT, "meeting", "Verified correction: Friday.")
            assert "Friday" in str((await agent.chat_history(ctx))[0].content)
            fresh = notebook_workflow(Notebook(tmp_path))
            fresh_ctx = await notebook_context(fresh, [current])
            restored = await fresh.chat_history(fresh_ctx)
        assert "Friday" in str(restored[0].content)
        assert "Thursday" not in str(restored[0].content)
        assert restored[1:] == [current]

    asyncio.run(scenario())
    assert Notebook(tmp_path).read(SESSION, CHAT) == [
        Note("meeting", "Verified correction: Friday.")
    ]


def test_notebook_budget_preserves_query_and_atomic_tool_group(tmp_path: Path) -> None:
    store = Notebook(tmp_path)
    for index in range(10):
        store.write(SESSION, CHAT, f"fact-{index}", "detail " * 60)

    async def scenario() -> None:
        agent = notebook_workflow(store)
        group = tool_group()
        agent.memory_token_limit = system_tokens(agent) + request_tokens(group) + 100
        ctx = await notebook_context(
            agent,
            [
                ChatMessage(role="user", content="old question " * 200),
                ChatMessage(role="assistant", content="old answer " * 200),
                *group,
            ],
        )
        with (
            bound_target(RunTarget(session=SESSION, chat_id=CHAT)),
            patch.object(
                ChatMemoryBuffer, "get", side_effect=AssertionError("No second trim")
            ),
        ):
            history = await agent.chat_history(ctx)
        assert [message.role for message in history] == [
            MessageRole.SYSTEM,
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.TOOL,
        ]
        assert history[1:] == group
        assert history[-1].additional_kwargs["tool_call_id"] == "lookup-note"
        assert request_tokens(history) <= agent.memory_token_limit
        notes = json.loads(str(history[0].content).partition(NOTEBOOK_HEADING)[2])
        assert 0 < len(notes) < 10
        assert all(
            note in [asdict(saved) for saved in store.read(SESSION, CHAT)]
            for note in notes
        )
        memory = await ctx.store.get("memory")
        assert memory.get_all() == group

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "oversized", ["system", "query", "tool_result", "tool_arguments"]
)
def test_oversized_required_context_fails_clearly(tmp_path: Path, oversized: str) -> None:
    async def scenario() -> None:
        agent = notebook_workflow(Notebook(tmp_path))
        agent.memory_token_limit = 512
        if oversized == "system":
            agent.system_prompt = "instruction " * 600
        group = tool_group(
            query="question " * 600 if oversized == "query" else "Which day?",
            result="result " * 600 if oversized == "tool_result" else '{"ok": true}',
            arguments=json.dumps(
                {"mode": "read", "query": "argument " * 600}
                if oversized == "tool_arguments"
                else {"mode": "read"}
            ),
        )
        ctx = await notebook_context(agent, group)
        with (
            bound_target(RunTarget(session=SESSION, chat_id=CHAT)),
            pytest.raises(
                ContextBudgetError, match="current query/tool group exceed token budget"
            ),
        ):
            await agent.chat_history(ctx)
        memory = await ctx.store.get("memory")
        assert memory.get_all() == group

    asyncio.run(scenario())


def test_operator_policies_are_never_silently_truncated(tmp_path: Path) -> None:
    store = Notebook(tmp_path)
    for index in range(3):
        store.write(
            SESSION,
            SESSION_NOTEBOOK_CHAT,
            f"policy-{index}",
            f"Authorized policy {index}: " + "detail " * 30,
            max_total_tokens=SESSION_POLICY_TOKEN_LIMIT,
        )

    async def scenario() -> None:
        agent = notebook_workflow(store)
        query = [ChatMessage(role="user", content="Help with the workshop.")]
        ctx = await notebook_context(agent, query)
        with bound_target(RunTarget(session=SESSION, chat_id=CHAT)):
            history = await agent.chat_history(ctx)
            policies = json.loads(
                str(history[0].content).partition(OPERATOR_NOTEBOOK_HEADING)[2]
            )
            assert policies == [
                asdict(note) for note in store.read(SESSION, SESSION_NOTEBOOK_CHAT)
            ]
            agent.memory_token_limit = request_tokens(history)
            assert await agent.chat_history(ctx) == history
            agent.memory_token_limit -= 1
            with pytest.raises(ValueError, match=r"Operator policies.*token budget"):
                await agent.chat_history(ctx)

    asyncio.run(scenario())


@pytest.mark.parametrize("chat", [CHAT, "operator"])
def test_wrap_up_self_lesson_survives_history_trim_and_reopen(
    tmp_path: Path, chat: str
) -> None:
    lesson = "The sketch upload rejected SVG; a verified PNG conversion succeeded."
    store = Notebook(tmp_path)

    async def scenario() -> None:
        agent = notebook_workflow(store)
        agent.memory_token_limit = system_tokens(agent) + 60
        ctx = await notebook_context(
            agent,
            [
                ChatMessage(role="user", content="Send the workshop sketch."),
                ChatMessage(role="assistant", content="The sketch was delivered."),
            ],
        )
        with bound_target(
            RunTarget(session=SESSION, chat_id=chat, armed=chat == "operator")
        ):
            await agent.store_wrap_up_note(ctx, lesson)
            memory = await ctx.store.get("memory")
            assert memory.get_all()[-1].additional_kwargs[WRAP_UP_NOTE_KWARG] is True
            for _ in range(3):
                await memory.aput(
                    ChatMessage(role="user", content="newer question " * 100)
                )
                await memory.aput(
                    ChatMessage(role="assistant", content="newer answer " * 100)
                )
            await memory.aput(ChatMessage(role="user", content="Which format worked?"))
            await agent.chat_history(ctx)
            assert all(
                WRAP_UP_NOTE_KWARG not in m.additional_kwargs for m in memory.get_all()
            )
            assert all(lesson not in str(m.content) for m in memory.get_all())
            save_memory(tmp_path, SESSION, chat, memory)
            restored_memory = load_memory(tmp_path, SESSION, chat)
            assert restored_memory is not None
            fresh = notebook_workflow(Notebook(tmp_path))
            restored_ctx = Context(fresh)
            await restored_ctx.store.set("memory", restored_memory)
            history = await fresh.chat_history(restored_ctx)
            assert lesson in str(history[0].content)

    asyncio.run(scenario())
    notes = Notebook(tmp_path).read(SESSION, chat)
    assert len(notes) == 1
    assert notes[0].text == lesson
    assert notes[0].key.startswith("wrap-up-")
    digest = notes[0].key.removeprefix("wrap-up-")
    assert len(digest) >= 8
    assert hashlib.sha256(lesson.encode()).hexdigest().startswith(digest)
    assert store.read(SESSION, SESSION_NOTEBOOK_CHAT) == []
    assert store.read(SESSION, OTHER_CHAT) == []
    assert store.read(OTHER_SESSION, chat) == []


@pytest.mark.parametrize("entrypoint", ["operation", "acall", "workflow"])
@pytest.mark.parametrize("cancellations", [1, 2])
def test_cancelled_notebook_call_holds_lock_until_database_thread_finishes(
    notebook_tool: FunctionTool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    cancellations: int,
) -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original_write = Notebook.write

    def blocked_write(
        store: Notebook,
        session: str,
        chat_id: str,
        key: str,
        text: str,
        *,
        max_total_tokens: int | None = None,
    ) -> None:
        assert current_target().chat_id == chat_id
        started.set()
        try:
            assert release.wait(timeout=10), "database thread was not released"
            original_write(
                store, session, chat_id, key, text, max_total_tokens=max_total_tokens
            )
        finally:
            finished.set()

    monkeypatch.setattr(Notebook, "write", blocked_write)

    async def scenario() -> None:
        lock = asyncio.Lock()

        async def caller() -> None:
            async with lock:
                if entrypoint == "operation":
                    await notebook_operation(
                        Notebook(tmp_path).write, SESSION, CHAT, "lesson", "Saved fact."
                    )
                elif entrypoint == "acall":
                    await notebook_tool.acall(
                        mode="write", key="lesson", text="Saved fact."
                    )
                else:
                    await call_notebook(
                        notebook_tool, mode="write", key="lesson", text="Saved fact."
                    )

        with bound_target(RunTarget(session=SESSION, chat_id=CHAT)):
            task = asyncio.create_task(caller())
            try:
                assert await asyncio.to_thread(started.wait, 10)
                assert lock.locked()
                assert not finished.is_set()
                for _ in range(cancellations):
                    task.cancel()
                    await asyncio.sleep(0)
                completed, _ = await asyncio.wait({task}, timeout=0.05)
                released_early = bool(completed) or not lock.locked()
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
                assert await asyncio.to_thread(finished.wait, 10)
            assert not released_early, (
                f"{entrypoint} released its caller before SQLite finished"
            )
            assert task.cancelled()
            assert not lock.locked()

    asyncio.run(scenario())
    assert Notebook(tmp_path).read(SESSION, CHAT) == [Note("lesson", "Saved fact.")]


def test_truncate_to_tokens_cuts_a_fitting_prefix() -> None:
    """A too-long text truncates to a prefix verified within the cap."""
    text = "saved fact " * 300
    assert text_tokens(text) > MAX_NOTE_TOKENS
    cut = truncate_to_tokens(text, MAX_NOTE_TOKENS)
    assert text_tokens(cut) <= MAX_NOTE_TOKENS
    assert cut in text


def test_truncate_to_tokens_short_text_passes_through() -> None:
    """Text already within the cap is returned unchanged."""
    assert truncate_to_tokens("short text", 10) == "short text"


@pytest.mark.parametrize("chat", [CHAT, "operator"])
def test_wrap_up_note_is_capped_to_the_notebook_token_limit(
    tmp_path: Path, chat: str
) -> None:
    """A verbose lesson truncates to the cap every notebook read can load."""
    verbose = "The upload failed for three reasons and each one taught a lesson. " * 60
    assert text_tokens(verbose) > MAX_NOTE_TOKENS
    store = Notebook(tmp_path)

    async def scenario() -> None:
        agent = notebook_workflow(store)
        ctx = await notebook_context(agent, [ChatMessage(role="user", content="Send.")])
        with bound_target(
            RunTarget(session=SESSION, chat_id=chat, armed=chat == "operator")
        ):
            await agent.store_wrap_up_note(ctx, verbose)

    asyncio.run(scenario())
    notes = Notebook(tmp_path).read(SESSION, chat)
    assert len(notes) == 1
    assert text_tokens(notes[0].text) <= MAX_NOTE_TOKENS
    assert notes[0].text.endswith("[Truncated wrap-up note]")
    # The capped note fits the bounded snapshot every later run renders,
    # so the model can read and clean up what it wrote.
    budget = min(NOTEBOOK_TOKEN_LIMIT, 4096 // 4)
    rendered = notebook_prompt(notes, budget, str.split)
    assert notes[0].key in rendered
