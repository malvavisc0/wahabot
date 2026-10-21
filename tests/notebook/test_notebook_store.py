import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool

from tests.notebook.support import (
    CHAT,
    OTHER_CHAT,
    OTHER_SESSION,
    SESSION,
    _note_pairs,
    bound_target,
    call_notebook,
)
from wahabot.ai.history import ContextBudgetError
from wahabot.ai.pinned_state import (
    PINNED_STATE_TOKEN_LIMIT,
    note_line,
    render_pinned_block,
)
from wahabot.ai.tools.notebook import (
    MAX_NOTE_TOKENS,
    NOTEBOOK_TOKEN_LIMIT,
)
from wahabot.ai.tools.whatsapp import (
    RunTarget,
)
from wahabot.ai.workflow.agent import build_agent
from wahabot.core.notebook import (
    MAX_READ_NOTES,
    SESSION_NOTEBOOK_CHAT,
    SESSION_POLICY_TOKEN_LIMIT,
    Note,
    Notebook,
)
from wahabot.core.tokens import text_tokens
from wahabot.settings import Settings


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
    """scopes() lists session + chat_id + count, newest first.

    Chats are scoped by session — one server can run several — so the
    listing carries the session on every row; without a session filter
    every bot's scopes are listed, and the chat JID never spans them.
    """
    store = Notebook(tmp_path)
    store.write(SESSION, CHAT, "older", "Old fact.")
    time.sleep(0.01)
    store.write(SESSION, OTHER_CHAT, "newer", "New fact.")
    store.write(SESSION, OTHER_CHAT, "newer-still", "Another new fact.")
    store.write(OTHER_SESSION, CHAT, "other-session", "Not this session.")
    assert store.scopes(SESSION) == [
        (SESSION, OTHER_CHAT, 2),
        (SESSION, CHAT, 1),
    ]
    assert store.scopes(OTHER_SESSION) == [(OTHER_SESSION, CHAT, 1)]
    assert store.scopes("missing") == []
    # No filter: newest write first across every session — the
    # OTHER_SESSION note is the last write, so its scope leads.
    assert store.scopes() == [
        (OTHER_SESSION, CHAT, 1),
        (SESSION, OTHER_CHAT, 2),
        (SESSION, CHAT, 1),
    ]


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
        assert _note_pairs(notebook_tool.fn(mode="read", query="CORRECTION")) == [
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
        assert _note_pairs(notebook_tool.fn(mode="read")) == [
            {"key": "private", "text": "Private draft plan."}
        ]
        assert _note_pairs(notebook_tool.fn(mode="read", chat=OTHER_CHAT)) == [
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
        assert _note_pairs(notebook_tool.fn(mode="read", scope="session")) == [
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


def test_pinned_block_renders_every_field() -> None:
    """Policies, notes, and rules all render in place."""
    block = render_pinned_block(
        policies=(Note("reply-style", "Use concise replies."),),
        notes=(Note("customer-email", "marisol@example.com"),),
        rules="Support hours 09:00-18:00 Europe/Berlin.",
        token_budget=PINNED_STATE_TOKEN_LIMIT,
    )
    assert block.startswith("# Pinned State\n")
    for fragment in (
        "## Operator policies\n\n- reply-style: Use concise replies.",
        "## Notebook facts\n\n- customer-email: marisol@example.com",
        "## Session rules\n\nSupport hours 09:00-18:00 Europe/Berlin.",
    ):
        assert fragment in block, fragment
    for heading in ("## Contact", "## Open tickets", "## Latest events"):
        assert heading not in block


def test_pinned_block_never_renders_support_sections() -> None:
    """Support state is not a pinned section; the block omits it entirely."""
    block = render_pinned_block(
        policies=(Note("reply-style", "Use concise replies."),),
        notes=(Note("meeting", "Workshop meets on Thursday."),),
        rules="Timezone: Europe/Berlin.",
        token_budget=PINNED_STATE_TOKEN_LIMIT,
    )
    for heading in ("## Contact", "## Open tickets", "## Latest events"):
        assert heading not in block
    assert "## Operator policies" in block
    assert "## Notebook facts" in block
    assert "## Session rules" in block
    # An entirely empty block is "" — no headings for absent sources.
    assert render_pinned_block(token_budget=PINNED_STATE_TOKEN_LIMIT) == ""


def test_pinned_block_with_empty_notebook_renders_only_what_exists() -> None:
    """Empty sections are omitted; a section renders only its fields."""
    block = render_pinned_block(
        notes=(Note("customer-email", "ana@example.com"),),
        token_budget=PINNED_STATE_TOKEN_LIMIT,
    )
    assert "## Notebook facts\n\n- customer-email: ana@example.com" in block
    assert "## Operator policies" not in block
    assert "## Session rules" not in block
    for heading in ("## Contact", "## Open tickets", "## Latest events"):
        assert heading not in block


def test_pinned_block_bounds_notes_within_the_token_cap() -> None:
    """Whole notes, newest-first order preserved, within the token cap."""
    newest_first = [
        Note(f"fact-{index:02}", f"Workshop fact {index:02}. " + "detail " * 80)
        for index in range(12)
    ]
    block = render_pinned_block(notes=newest_first, token_budget=300, tokenizer=str.split)
    section = block.partition("## Notebook facts\n\n")[2]
    lines = [line for line in section.splitlines() if line.strip()]
    assert 0 < len(lines) < 12
    assert lines == [note_line(note) for note in newest_first[: len(lines)]]
    assert text_tokens("## Notebook facts\n\n" + section, str.split) <= 300


def test_pinned_block_raises_when_authoritative_state_cannot_fit() -> None:
    """Policies that cannot fit the reserved budget raise, never trim."""
    with pytest.raises(ContextBudgetError, match=r"Pinned state.*token budget"):
        render_pinned_block(
            policies=(Note("policy", "Keep answers concise. " * 100),),
            token_budget=PINNED_STATE_TOKEN_LIMIT,
            prompt_prefix="system prompt ",
            prompt_budget=50,
            tokenizer=str.split,
        )
