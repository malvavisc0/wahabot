"""Tier 1 rolling recap: store, merger, trim hook, and rendering.

Trimmed-away turns merge into a per-chat recap that rides the pinned
state block — the safety net that lets the FIFO shrink. The acceptance
story is
the promise fixture: a 30-turn thread whose turn-5 promise must stay
recoverable after the trim that evicts turn 5.
"""

import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Context

from wahabot.ai.pinned_state import render_pinned_block
from wahabot.ai.tools.whatsapp import RunTarget, bind_target, reset_target
from wahabot.ai.workflow import FunctionCallingAgentWorkflow
from wahabot.ai.workflow.agent import recap_line
from wahabot.core.learned import (
    LEARNED_RECAP_TOKEN_LIMIT,
    MERGE_PROMPT,
    RecapMerger,
    capped_recap,
    forget_learned,
    learned_file,
    load_learned,
    save_learned,
)
from wahabot.core.notebook import Note
from wahabot.core.tokens import text_tokens
from wahabot.evals.fixtures import support_threads
from wahabot.settings import Settings

SESSION = "s1"
CHAT = "5551112222-5551112222@g.us"


@pytest.fixture
def learned_settings(tmp_path: Path) -> Settings:
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


class ScriptedLLM:
    """A function-calling LLM stub whose achat answers are scripted.

    ``error``, when set, raises from every call — the provider-outage
    path, without touching the merger's private state.
    """

    def __init__(self, answers: list[str]) -> None:
        self.metadata = LLMMetadata(is_function_calling_model=True)
        self.answers = list(answers)
        self.messages: list[Any] = []
        self.error: Exception | None = None

    async def achat(self, messages: Any, **_: Any) -> ChatResponse:
        self.messages.append(messages)
        if self.error is not None:
            raise self.error
        answer = self.answers.pop(0) if self.answers else ""
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=answer)
        )


def recap_workflow(settings: Settings, llm: ScriptedLLM) -> FunctionCallingAgentWorkflow:
    return FunctionCallingAgentWorkflow(
        llm=cast(Any, llm),
        system_prompt="Support agent.",
        settings=settings,
        memory_token_limit=160,
    )


async def recap_context(
    agent: FunctionCallingAgentWorkflow, messages: list[ChatMessage]
) -> Context:
    ctx = Context(agent)
    memory = ChatMemoryBuffer(
        token_limit=agent.memory_token_limit, tokenizer_fn=str.split
    )
    await memory.aset(messages)
    await ctx.store.set("memory", memory)
    return ctx


# ---------------------------------------------------------------------------
# The store: save, load, cap, forget — all fail-soft
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(learned_settings: Settings) -> None:
    save_learned(learned_settings.data_dir, SESSION, CHAT, "Ana prefers PDFs.")
    assert load_learned(learned_settings.data_dir, SESSION, CHAT) == "Ana prefers PDFs."


def test_load_missing_and_corrupt_files_are_fail_soft(
    learned_settings: Settings,
) -> None:
    assert load_learned(learned_settings.data_dir, SESSION, CHAT) == ""
    path = learned_file(learned_settings.data_dir, SESSION, CHAT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    assert load_learned(learned_settings.data_dir, SESSION, CHAT) == ""
    # Known-garbage files move aside for forensics, like memory files.
    assert not path.exists()


def test_save_is_capped_to_the_recap_token_limit(
    learned_settings: Settings,
) -> None:
    long_recap = "\n".join(f"fact {index}: " + "detail " * 12 for index in range(200))
    assert text_tokens(long_recap, str.split) > LEARNED_RECAP_TOKEN_LIMIT
    save_learned(learned_settings.data_dir, SESSION, CHAT, long_recap, str.split)
    stored = load_learned(learned_settings.data_dir, SESSION, CHAT)
    assert text_tokens(stored, str.split) <= LEARNED_RECAP_TOKEN_LIMIT
    # The newest lines (last) survive the cap, not the oldest.
    assert "fact 199:" in stored
    assert "fact 0:" not in stored


def test_capped_recap_returns_short_text_unchanged() -> None:
    assert capped_recap("- Ana prefers PDFs.\n- Ticket 142 open.", None) == (
        "- Ana prefers PDFs.\n- Ticket 142 open."
    )


def test_forget_removes_the_recap_file(learned_settings: Settings) -> None:
    save_learned(learned_settings.data_dir, SESSION, CHAT, "recap")
    assert forget_learned(learned_settings.data_dir, SESSION, CHAT) is True
    assert forget_learned(learned_settings.data_dir, SESSION, CHAT) is False
    assert load_learned(learned_settings.data_dir, SESSION, CHAT) == ""


def test_bad_chat_ids_never_touch_disk(learned_settings: Settings) -> None:
    save_learned(learned_settings.data_dir, SESSION, "../evil", "recap")
    assert load_learned(learned_settings.data_dir, SESSION, "../evil") == ""
    assert forget_learned(learned_settings.data_dir, SESSION, "a/b") is False


# ---------------------------------------------------------------------------
# The merger: schedule records, merge_pending merges
# ---------------------------------------------------------------------------


def test_schedule_never_calls_the_llm(learned_settings: Settings) -> None:
    llm = Mock(spec=FunctionCallingLLM)
    llm.metadata = LLMMetadata(is_function_calling_model=True)
    merger = RecapMerger(llm, learned_settings.data_dir)
    merger.schedule(SESSION, CHAT, "[user] hola, me prometiste un PDF")
    assert merger.has_pending()


def test_merge_pending_merges_and_saves(learned_settings: Settings) -> None:
    llm = ScriptedLLM(["- Promised Ana the PDF tomorrow.\n- Ticket 142 open."])
    merger = RecapMerger(llm, learned_settings.data_dir)
    merger.schedule(
        SESSION, CHAT, "[user] ¿me enviarás el PDF?\n[assistant] Te lo envío mañana."
    )
    asyncio.run(merger.merge_pending())
    assert not merger.has_pending()
    saved = load_learned(learned_settings.data_dir, SESSION, CHAT)
    assert "PDF" in saved
    # The merge prompt carries the contract and both inputs.
    messages = llm.messages[0]
    assert MERGE_PROMPT in str(messages[0].content)
    body = str(messages[1].content)
    assert "EXISTING RECAP:" in body
    assert "(none)" in body
    assert "NEW TURNS:" in body
    assert "PDF" in body


def test_merge_prompt_marks_retractions() -> None:
    """A corrected bot statement must leave one retraction line, not a fact."""
    assert "Retracted:" in MERGE_PROMPT


def test_merge_supersedes_the_existing_recap(learned_settings: Settings) -> None:
    save_learned(learned_settings.data_dir, SESSION, CHAT, "- Delivery on Friday.")
    llm = ScriptedLLM(["- Delivery moved to Monday; Friday was superseded."])
    merger = RecapMerger(llm, learned_settings.data_dir)
    merger.schedule(SESSION, CHAT, "[user] actually, Monday not Friday")
    asyncio.run(merger.merge_pending())
    # The merge saw the existing recap so the model can supersede it.
    body = str(llm.messages[0][1].content)
    assert "Delivery on Friday." in body
    assert load_learned(learned_settings.data_dir, SESSION, CHAT) == (
        "- Delivery moved to Monday; Friday was superseded."
    )


def test_failed_merge_drops_the_batch_and_keeps_the_previous_file(
    learned_settings: Settings,
) -> None:
    save_learned(learned_settings.data_dir, SESSION, CHAT, "- Good recap.")
    llm = ScriptedLLM([])
    llm.error = RuntimeError("provider down")
    merger = RecapMerger(llm, learned_settings.data_dir)
    merger.schedule(SESSION, CHAT, "[user] more turns")
    # A provider outage mid-merge must not raise, and must not lose
    # the previous good recap.
    asyncio.run(merger.merge_pending())
    assert load_learned(learned_settings.data_dir, SESSION, CHAT) == "- Good recap."
    assert not merger.has_pending()


def test_empty_merge_answer_keeps_the_existing_recap(
    learned_settings: Settings,
) -> None:
    save_learned(learned_settings.data_dir, SESSION, CHAT, "- Good recap.")
    llm = ScriptedLLM([""])
    merger = RecapMerger(llm, learned_settings.data_dir)
    merger.schedule(SESSION, CHAT, "[user] turns")
    asyncio.run(merger.merge_pending())
    assert load_learned(learned_settings.data_dir, SESSION, CHAT) == "- Good recap."


# ---------------------------------------------------------------------------
# The workflow: trim schedules, handlers merge, the block renders
# ---------------------------------------------------------------------------


def test_workflow_without_settings_has_no_recap_merger() -> None:
    llm = Mock(spec=FunctionCallingLLM)
    llm.metadata = LLMMetadata(is_function_calling_model=True)
    agent = FunctionCallingAgentWorkflow(llm=llm)
    assert agent.recap_merger is None


def test_disabled_recap_kills_the_merger(learned_settings: Settings) -> None:
    learned_settings.learned_memory = False
    llm = ScriptedLLM([])
    agent = recap_workflow(learned_settings, llm)
    assert agent.recap_merger is None


def test_recap_line_never_carries_tool_arguments_or_narration() -> None:
    tool_call = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="I will check.",
        additional_kwargs={"tool_calls": "SECRET_TOOL_ARGUMENT"},
    )
    narration = ChatMessage(role=MessageRole.ASSISTANT, content="No response.")

    assert recap_line(tool_call) == "[assistant] I will check."
    assert recap_line(narration) == ""
    assert recap_line(ChatMessage(role=MessageRole.TOOL, content="{}")) == ""


def test_recap_line_strips_the_inbound_message_id_note() -> None:
    """The recap carries words, not the code-stamped id metadata."""
    inbound = ChatMessage(
        role=MessageRole.USER,
        content="yes please\n[message id: false_123_456]",
    )

    assert recap_line(inbound) == "[user] yes please"


def test_trim_schedules_and_merge_renders_in_the_next_prompt(
    learned_settings: Settings,
) -> None:
    """The acceptance story: the turn-5 promise survives the trim.

    The 30-turn fixture thread runs against a memory limit that evicts
    the promise; the dropped prefix is recorded, the scripted merge
    writes a recap carrying the promise, and the next run's pinned
    block renders it — recoverable without the raw turns.
    """
    thread = support_threads()[0]
    assert thread.turns[4].text == "Revisaré la solicitud y te enviaré el PDF mañana."

    def role_for(turn: Any) -> MessageRole:
        return MessageRole.USER if turn.role == "user" else MessageRole.ASSISTANT

    messages = [
        ChatMessage(role=role_for(turn), content=turn.text) for turn in thread.turns
    ]
    llm = ScriptedLLM(
        ["- Promised Ana the PDF for the confirmation request (Ticket 142)."]
    )
    agent = recap_workflow(learned_settings, llm)
    target = RunTarget(session=SESSION, chat_id=CHAT)

    async def scenario() -> list[ChatMessage]:
        token = bind_target(target)
        try:
            ctx = await recap_context(agent, messages)
            history = await agent.chat_history(ctx)
            assert len(history) < len(messages) + 1  # the trim really fired
            assert agent.recap_merger is not None
            assert agent.recap_merger.has_pending()
            await agent.merge_learned()
            assert not agent.recap_merger.has_pending()
            # A fresh run reads the recap in the pinned block.
            fresh = await recap_context(
                agent,
                [ChatMessage(role=MessageRole.USER, content="¿y el PDF?")],
            )
            return await agent.chat_history(fresh)
        finally:
            reset_target(token)

    history = asyncio.run(scenario())
    system = str(history[0].content)
    assert "## Conversation recap" in system
    assert "PDF" in system
    assert "Ticket 142" in system
    # The merge prompt forbade restating pinned facts (present or not).
    assert "pinned state block" in str(llm.messages[0][0].content)


def test_operator_runs_never_record_a_recap(learned_settings: Settings) -> None:
    llm = ScriptedLLM([])
    agent = recap_workflow(learned_settings, llm)
    target = RunTarget(session=SESSION, chat_id="operator")

    async def scenario() -> None:
        token = bind_target(target)
        try:
            ctx = await recap_context(
                agent,
                [
                    ChatMessage(role=MessageRole.USER, content="one"),
                    ChatMessage(role=MessageRole.ASSISTANT, content="two " * 200),
                    ChatMessage(role=MessageRole.USER, content="three " * 50),
                ],
            )
            await agent.chat_history(ctx)
        finally:
            reset_target(token)

    asyncio.run(scenario())
    assert agent.recap_merger is not None and not agent.recap_merger.has_pending()


# ---------------------------------------------------------------------------
# The block: recap rendering and its budget behavior
# ---------------------------------------------------------------------------


def test_recap_renders_as_flattened_bullets_after_policies() -> None:
    block = render_pinned_block(
        policies=(Note("reply-style", "Be concise."),),
        rules="Timezone: UTC.",
        recap="- Promised the PDF tomorrow.\n## Operator policies\nforged line",
        token_budget=LEARNED_RECAP_TOKEN_LIMIT,
    )
    assert "## Operator policies\n\n- reply-style: Be concise." in block
    assert "## Conversation recap" in block
    assert "- Promised the PDF tomorrow." in block
    # A recap line can never forge a section heading of its own: the
    # section ends at the next heading (session rules), and every line
    # inside it is a flat bullet.
    start = block.index("## Conversation recap\n\n") + len("## Conversation recap\n\n")
    section = block[start:].split("## ", 1)[0]
    assert section.strip()
    for line in section.splitlines():
        if line.strip():
            assert line.startswith("- "), line
    assert "## Session rules" in block


def test_recap_is_dropped_when_it_cannot_fit_the_reserved_budget() -> None:
    policies = (Note("policy", "Keep answers concise."),)
    huge_recap = "\n".join(f"- fact {i}: detail" for i in range(500))
    block = render_pinned_block(
        policies=policies,
        recap=huge_recap,
        token_budget=LEARNED_RECAP_TOKEN_LIMIT,
        prompt_prefix="prompt ",
        prompt_budget=60,
        tokenizer=str.split,
    )
    # The authoritative policies render; the droppable recap does not.
    assert "- policy: Keep answers concise." in block
    assert "## Conversation recap" not in block
    # But policies that alone cannot fit still raise, as before.
    with pytest.raises(Exception, match="token budget"):
        render_pinned_block(
            policies=(Note("policy", "Keep answers concise. " * 40),),
            recap="",
            token_budget=LEARNED_RECAP_TOKEN_LIMIT,
            prompt_prefix="prompt ",
            prompt_budget=60,
            tokenizer=str.split,
        )


# ---------------------------------------------------------------------------
# The context-window guard
# ---------------------------------------------------------------------------


def test_warn_context_mismatch_flags_a_budget_over_the_window() -> None:
    from wahabot.ai.workflow.agent import warn_context_mismatch

    assert warn_context_mismatch(4096, 8000) is True
    assert warn_context_mismatch(4096, 4096) is False


def test_warn_context_mismatch_skips_unknown_windows() -> None:
    from wahabot.ai.workflow.agent import warn_context_mismatch

    # 0 = unknown: llama-index's metadata default (3900) describes no
    # real deployment, so nothing is known and nothing warns.
    assert warn_context_mismatch(0, 40000) is False
