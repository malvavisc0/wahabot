"""Header-owned policy, recall dedupe, recap cap, prompt budget."""

from __future__ import annotations

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from wahabot.ai.history import (
    ADDRESSED_POLICY,
    DM_POLICY,
    PARTICIPATE_POLICY,
    UNADDRESSED_POLICY,
    turn_header,
)
from wahabot.ai.pinned_state import dedupe_recall
from wahabot.ai.records import format_record
from wahabot.core.cli import session_template
from wahabot.core.learned import RECAP_MAX_LINES, capped_recap
from wahabot.core.tokens import text_tokens


def _message() -> list[ChatMessage]:
    record = format_record(
        time_label="18:00", speaker="Ada", serial=1, kind="text", body="hello"
    )
    return [ChatMessage(role=MessageRole.USER, content=record)]


def test_header_owns_the_turn_policy() -> None:
    addressed = turn_header(_message(), group=True, addressed=True, speak=True)[0]
    assert ADDRESSED_POLICY in addressed.content
    participated = turn_header(_message(), group=True, addressed=False, speak=True)[0]
    assert PARTICIPATE_POLICY in participated.content
    silent = turn_header(_message(), group=True, addressed=False, speak=False)[0]
    assert UNADDRESSED_POLICY in silent.content
    dm = turn_header(_message(), group=False, addressed=True, speak=True, direct=True)[0]
    assert DM_POLICY in dm.content


def test_constant_prompt_has_no_group_policy() -> None:
    prompt = str(session_template()["system_prompt"])
    assert "In groups, default to silence" not in prompt
    assert "header's policy line" in prompt


def test_constant_prompt_stays_within_budget() -> None:
    prompt = str(session_template()["system_prompt"])
    assert text_tokens(prompt) <= 1300


def test_recall_drops_live_window_duplicates() -> None:
    live = ["[[18:00 Ada #m12]] text :: hello", "a reply"]
    slices = [
        "user: [[18:00 Ada #m12]] text :: hello | assistant: a reply",
        "a reply",
        "user: something from an older turn",
    ]
    assert dedupe_recall(slices, live) == ["user: something from an older turn"]


def test_recall_keeps_a_slice_that_a_live_record_only_references() -> None:
    """A bare ``quote=#m<n>`` reference is a pointer, not a verbatim copy."""
    live = ["[[18:00 Ada #m12]] text quote=#m11 :: what did you mean"]
    slices = ["user: [[18:00 Ada #m11]] text :: the only full text of it"]
    assert dedupe_recall(slices, live) == slices


def test_recap_keeps_only_the_newest_lines() -> None:
    recap = "\n".join(f"line {number}" for number in range(20))
    kept = capped_recap(recap).splitlines()
    assert len(kept) == RECAP_MAX_LINES
    assert kept[-1] == "line 19"
    assert kept[0] == "line 10"
