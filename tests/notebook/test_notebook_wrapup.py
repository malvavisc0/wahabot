import asyncio
import json
from pathlib import Path

import pytest
from llama_index.core.tools import FunctionTool

from tests.notebook.support import (
    CHAT,
    SESSION,
    bound_target,
    notebook_workflow,
)
from wahabot.ai.tools.whatsapp import (
    RunTarget,
)
from wahabot.core.notebook import (
    Notebook,
)
from wahabot.core.tokens import text_tokens


def test_operator_reserved_key_is_always_stored_as_a_rule(
    notebook_tool: FunctionTool, tmp_path: Path
) -> None:
    """A reserved rule key is stored as a rule whatever kind was written.

    The deterministic seams (`chat_rules`, the triage mute) read
    ``kind=rule``; an operator writing the key as a plain fact would
    silently never apply it, so the tool coerces the kind.
    """
    store = Notebook(tmp_path)
    with bound_target(RunTarget(session=SESSION, chat_id="operator", armed=True)):
        written = json.loads(
            notebook_tool.fn(
                mode="write",
                chat=CHAT,
                key="chat-hard-mute",
                text="muted until the dust settles",
                kind="fact",
            )
        )
    assert written["ok"]
    note = store.read(SESSION, CHAT, key="chat-hard-mute")[0]
    assert (note.kind, note.author_kind) == ("rule", "operator")


def test_chat_rules_are_capped_at_the_rule_token_budget(tmp_path: Path) -> None:
    """A write that pushes a chat's rules past the cap is refused, atomically."""
    from wahabot.core.notebook import CHAT_RULE_TOKEN_LIMIT

    store = Notebook(tmp_path)
    unit = "always answer in formal Spanish without emojis or slang "
    # The store writes the stripped text, so count the stripped tokens.
    per_rule = text_tokens(unit.strip())
    count = CHAT_RULE_TOKEN_LIMIT // per_rule + 1
    for index in range(count):
        if index == count - 1:
            with pytest.raises(ValueError, match="chat rules exceed"):
                store.write(
                    SESSION,
                    CHAT,
                    f"rule-{index}",
                    unit,
                    kind="rule",
                    author_kind="operator",
                )
        else:
            store.write(
                SESSION,
                CHAT,
                f"rule-{index}",
                unit,
                kind="rule",
                author_kind="operator",
            )
    # The refused write rolled back: the overflowing rule is not stored.
    assert store.read(SESSION, CHAT, key=f"rule-{count - 1}") == []
    # Facts are not rules: the same text as a fact writes fine.
    store.write(SESSION, CHAT, "big-fact", unit)
    assert store.read(SESSION, CHAT, key="big-fact")


def test_read_filters_notes_by_kind(tmp_path: Path) -> None:
    """`kind=` selects one kind; without it every kind reads."""
    store = Notebook(tmp_path)
    store.write(SESSION, CHAT, "tone", "formal", kind="rule", author_kind="operator")
    store.write(SESSION, CHAT, "meeting", "Thursdays.")
    store.write(
        SESSION, CHAT, "svg", "convert first", kind="lesson", author_kind="system"
    )
    assert [note.key for note in store.read(SESSION, CHAT, kind="rule")] == ["tone"]
    assert [note.key for note in store.read(SESSION, CHAT, kind="lesson")] == ["svg"]
    assert {note.key for note in store.read(SESSION, CHAT)} == {
        "tone",
        "meeting",
        "svg",
    }
    with pytest.raises(ValueError, match="kind"):
        store.read(SESSION, CHAT, kind="nope")


def test_topic_only_wrap_up_note_stores_nothing(tmp_path: Path) -> None:
    """A wrap-up reply that is only a `topic:` line stores no lesson."""
    store = Notebook(tmp_path)
    agent = notebook_workflow(store)
    with bound_target(RunTarget(session=SESSION, chat_id=CHAT)):
        asyncio.run(agent.store_wrap_up_note("topic: svg-upload"))
    assert store.read(SESSION, CHAT) == []
