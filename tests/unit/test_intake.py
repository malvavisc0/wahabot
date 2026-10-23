import asyncio
import json
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.harness import RecordingWaha
from wahabot.ai.intake import (
    InMemoryIntakeStore,
    IntakeLedger,
    max_age_for_run_timeout,
)
from wahabot.ai.tools.whatsapp import (
    RunTarget,
    bind_target,
    clear_delivery_refusal,
    delivery_guard,
    forward_message,
    react_to_message,
    reset_target,
    send_media,
    send_message,
)
from wahabot.handlers.base import HandlerBase
from wahabot.settings import Settings


def test_ledger_reports_newer_pending_ids_outside_the_turn() -> None:
    ledger = IntakeLedger()
    now = time.monotonic()
    ledger.record("s", "chat", "A", "[[09:00 Ana #m1]] text :: first", at=now - 2)
    ledger.record("s", "chat", "B", "[[09:02 Ana #m2]] text :: newest", at=now - 1)

    newer = ledger.pending("s", "chat", after=now - 2, exclude={"A"})

    assert [row.message_id for row in newer] == ["B"]
    assert newer[0].record.endswith("newest")


def test_ledger_claim_hides_a_row_until_the_run_finishes() -> None:
    ledger = IntakeLedger()
    ledger.record("s", "chat", "A", "first")
    ledger.record("s", "chat", "B", "newest")

    claimed = ledger.claim("s", "chat", ["B"])

    assert [row.message_id for row in claimed] == ["B"]
    assert [row.message_id for row in ledger.pending("s", "chat")] == ["A"]
    assert not ledger.handled("s", "chat", ["B"])

    ledger.finish("s", "chat", ["B"], delivered=True)

    assert ledger.handled("s", "chat", ["B"])


def test_ledger_finish_turn_releases_claims_when_the_run_was_silent() -> None:
    ledger = IntakeLedger()
    ledger.record("s", "chat", "A", "first")
    ledger.record("s", "chat", "B", "newest")
    ledger.claim("s", "chat", ["B"])

    ledger.finish_turn("s", "chat", ["A"], ["B"], delivered=False)

    assert not ledger.handled("s", "chat", ["B"])
    assert [row.message_id for row in ledger.pending("s", "chat")] == ["A", "B"]


def test_ledger_finish_turn_marks_turn_and_claims_delivered() -> None:
    ledger = IntakeLedger()
    ledger.record("s", "chat", "A", "first")
    ledger.record("s", "chat", "B", "newest")
    ledger.claim("s", "chat", ["B"])

    ledger.finish_turn("s", "chat", ["A"], ["B"], delivered=True)

    assert ledger.handled("s", "chat", ["A", "B"])
    assert ledger.pending("s", "chat") == []


def test_claimed_reports_an_active_claim_until_it_settles() -> None:
    ledger = IntakeLedger()
    ledger.record("s", "chat", "A", "first")
    ledger.record("s", "chat", "B", "newest")

    assert not ledger.claimed("s", "chat", ["A", "B"])
    ledger.claim("s", "chat", ["B"])

    assert ledger.claimed("s", "chat", ["A", "B"])
    assert not ledger.claimed("s", "chat", ["A"])

    ledger.finish_turn("s", "chat", ["A"], ["B"], delivered=True)

    assert not ledger.claimed("s", "chat", ["A", "B"])
    assert ledger.handled("s", "chat", ["B"])


def test_release_never_flips_a_row_another_run_settled() -> None:
    ledger = IntakeLedger()
    ledger.record("s", "chat", "A", "first")
    ledger.record("s", "chat", "B", "newest")
    ledger.finish("s", "chat", ["B"], delivered=True)

    ledger.release("s", "chat", ["A", "B"])

    assert [row.message_id for row in ledger.pending("s", "chat")] == ["A"]
    assert ledger.handled("s", "chat", ["B"])


def test_ledger_expires_rows_past_the_age_bound() -> None:
    ledger = IntakeLedger(max_age_s=60)
    ledger.record("s", "chat", "old", "old", at=time.monotonic() - 61)
    ledger.record("s", "chat", "fresh", "fresh")

    assert [row.message_id for row in ledger.pending("s", "chat")] == ["fresh"]


def test_ledger_bounds_messages_per_chat_independently() -> None:
    ledger = IntakeLedger(max_messages=2)
    ledger.record("s", "one", "a", "a")
    ledger.record("s", "one", "b", "b")
    ledger.record("s", "one", "c", "c")
    ledger.record("s", "two", "d", "d")

    assert [row.message_id for row in ledger.pending("s", "one")] == ["b", "c"]
    assert [row.message_id for row in ledger.pending("s", "two")] == ["d"]


def test_ledger_storage_is_injectable() -> None:
    store = InMemoryIntakeStore()
    first = IntakeLedger(store=store)
    first.record("s", "chat", "A", "shared")

    second = IntakeLedger(store=store)

    assert [row.message_id for row in second.pending("s", "chat")] == ["A"]


def test_delivery_guard_refuses_an_older_reply_and_carries_the_newest_record() -> None:
    ledger = IntakeLedger()
    started = time.monotonic()
    ledger.record("s", "chat", "A", "[[09:00 Ana #m1]] text :: first", at=started)
    ledger.record("s", "chat", "B", "[[09:02 Ana #m2]] text :: newest", at=started + 1)
    target = RunTarget(
        session="s",
        chat_id="chat",
        intake=ledger,
        turn_ids=["A"],
        turn_started_at=started,
    )

    refusal = delivery_guard(target)

    assert refusal is not None
    assert "newest" in refusal and "#m2" in refusal
    assert "Do not send this reply. Read the newest record and answer it once." in refusal
    assert target.claimed_ids == ["B"]


def test_delivery_guard_allows_the_claimed_send_and_leaves_settling_to_the_run() -> None:
    ledger = IntakeLedger()
    started = time.monotonic()
    ledger.record("s", "chat", "A", "first", at=started)
    ledger.record("s", "chat", "B", "newest", at=started + 1)
    target = RunTarget(
        session="s",
        chat_id="chat",
        intake=ledger,
        turn_ids=["A"],
        turn_started_at=started,
    )
    assert delivery_guard(target) is not None
    assert target.anchor_id == "B"

    clear_delivery_refusal(target)
    assert delivery_guard(target) is None
    assert not ledger.handled("s", "chat", ["B"])

    ledger.finish_turn("s", "chat", ["A"], target.claimed_ids, delivered=True)

    assert ledger.handled("s", "chat", ["B"])


def test_failed_send_releases_the_claim_for_the_newer_message() -> None:
    ledger = IntakeLedger()
    started = time.monotonic()
    ledger.record("s", "chat", "A", "first", at=started)
    ledger.record("s", "chat", "B", "newest", at=started + 1)
    target = RunTarget(
        session="s",
        chat_id="chat",
        intake=ledger,
        turn_ids=["A"],
        turn_started_at=started,
    )
    assert delivery_guard(target) is not None
    clear_delivery_refusal(target)
    assert delivery_guard(target) is None

    ledger.finish_turn("s", "chat", ["A"], target.claimed_ids, delivered=False)

    assert not ledger.handled("s", "chat", ["B"])
    assert [row.message_id for row in ledger.pending("s", "chat")] == ["A", "B"]


def test_delivery_guard_exempts_operator_runs_and_turns_without_ids() -> None:
    ledger = IntakeLedger()
    started = time.monotonic()
    ledger.record("s", "chat", "A", "first", at=started)
    ledger.record("s", "chat", "B", "newest", at=started + 1)

    operator = RunTarget(
        session="s",
        chat_id="chat",
        intake=ledger,
        turn_ids=["A"],
        turn_started_at=started,
        armed=True,
    )
    bare = RunTarget(session="s", chat_id="chat", intake=ledger)

    assert delivery_guard(operator) is None
    assert delivery_guard(bare) is None


def test_delivery_guard_ignores_older_pending_rows() -> None:
    ledger = IntakeLedger()
    started = time.monotonic()
    ledger.record("s", "chat", "old", "old", at=started - 10)
    target = RunTarget(
        session="s",
        chat_id="chat",
        intake=ledger,
        turn_ids=["A"],
        turn_started_at=started,
    )

    assert delivery_guard(target) is None


def test_delivery_guard_ignores_newer_records_that_would_not_wake_a_run() -> None:
    ledger = IntakeLedger()
    started = time.monotonic()
    ledger.record("s", "chat", "A", "first", at=started)
    ledger.record("s", "chat", "B", "banter", at=started + 1, addressed=False)
    target = RunTarget(
        session="s",
        chat_id="chat",
        intake=ledger,
        turn_ids=["A"],
        turn_started_at=started,
    )

    assert delivery_guard(target) is None
    assert target.claimed_ids == []


@pytest.mark.parametrize(
    ("builder", "kwargs"),
    [
        (send_message, {"text": "old reply"}),
        (send_media, {"kind": "image"}),
        (forward_message, {"message_id": "false_chat_OLD"}),
        (react_to_message, {"message_id": "false_chat_OLD", "reaction": "👍"}),
    ],
)
def test_every_delivery_tool_refuses_a_reply_superseded_by_newer_intake(
    builder: Any, kwargs: dict[str, Any], unit_settings: Settings
) -> None:
    ledger = IntakeLedger()
    started = time.monotonic()
    ledger.record("s", "chat", "A", "first", at=started)
    ledger.record("s", "chat", "B", "newest", at=started + 1)
    target = RunTarget(
        session="s",
        chat_id="chat",
        intake=ledger,
        turn_ids=["A"],
        turn_started_at=started,
    )
    waha = RecordingWaha()
    tool = builder(waha) if builder is forward_message else builder(waha, unit_settings)
    token = bind_target(target)
    try:
        result = str(cast(Any, tool).fn(**kwargs))
    finally:
        reset_target(token)

    envelope = json.loads(result)
    assert envelope["ok"] is False
    assert "Read the newest record" in envelope["error"]
    assert waha.sent == [] and waha.sent_images == [] and waha.reactions == []


def test_delivery_guard_blocks_a_second_delivery_in_the_same_round() -> None:
    ledger = IntakeLedger()
    started = time.monotonic()
    ledger.record("s", "chat", "A", "first", at=started)
    ledger.record("s", "chat", "B", "newest", at=started + 1)
    target = RunTarget(
        session="s",
        chat_id="chat",
        intake=ledger,
        turn_ids=["A"],
        turn_started_at=started,
    )

    assert delivery_guard(target) is not None
    second = delivery_guard(target)

    assert second is not None and "this tool round" in second
    assert target.claimed_ids == ["B"]


def test_delivery_guard_claims_and_shows_every_newer_record() -> None:
    ledger = IntakeLedger()
    started = time.monotonic()
    ledger.record("s", "chat", "A", "first", at=started)
    for i in range(5):
        ledger.record("s", "chat", f"N{i}", f"record-{i}", at=started + i + 1)
    target = RunTarget(
        session="s",
        chat_id="chat",
        intake=ledger,
        turn_ids=["A"],
        turn_started_at=started,
    )

    refusal = delivery_guard(target)

    assert refusal is not None
    for i in range(5):
        assert f"record-{i}" in refusal
    assert target.claimed_ids == [f"N{i}" for i in range(5)]
    assert "left for their own run" not in refusal


def test_handled_addressed_only_ignores_unaddressed_album_rows() -> None:
    ledger = IntakeLedger()
    ledger.record("s", "chat", "container", "album", addressed=True)
    ledger.record("s", "chat", "image", "image", addressed=False)
    ledger.finish("s", "chat", ["container"], delivered=True)

    assert not ledger.handled("s", "chat", ["container", "image"])
    assert ledger.handled("s", "chat", ["container", "image"], addressed_only=True)


def test_intake_window_outlives_a_custom_run_timeout() -> None:
    assert max_age_for_run_timeout(3600) > 3600
    assert max_age_for_run_timeout(1) >= 1800


class SuppressionHandler(HandlerBase):
    """Only the state ``turn_suppressed`` needs."""

    def __init__(self, intake: IntakeLedger, run_timeout: float) -> None:
        self.intake = intake
        self.settings = SimpleNamespace(run_timeout=run_timeout)


def test_turn_suppressed_ignores_unaddressed_album_rows() -> None:
    ledger = IntakeLedger()
    ledger.record("s", "chat", "container", "album", addressed=True)
    ledger.record("s", "chat", "image", "image", addressed=False)
    ledger.finish("s", "chat", ["container"], delivered=True)
    handler = SuppressionHandler(ledger, run_timeout=1.0)

    suppressed = asyncio.run(handler.turn_suppressed("s", "chat", ["container", "image"]))

    assert suppressed is True


def test_delivery_guard_ignores_newer_rows_settled_silent_or_failed() -> None:
    ledger = IntakeLedger()
    started = time.monotonic()
    ledger.record("s", "chat", "A", "first", at=started)
    ledger.record("s", "chat", "silent", "stale", at=started + 1)
    ledger.finish(
        "s", "chat", ["silent"], delivered=False, outcome="silent", reason="stale"
    )
    ledger.record("s", "chat", "failed", "retrying", at=started + 2)
    ledger.fail("s", "chat", ["failed"], error="crash", next_attempt_at=started + 60)
    target = RunTarget(
        session="s",
        chat_id="chat",
        intake=ledger,
        turn_ids=["A"],
        turn_started_at=started,
        owner="run-a",
    )

    assert delivery_guard(target) is None
    assert target.claimed_ids == []


def test_claim_records_the_owner_and_release_is_owner_scoped() -> None:
    ledger = IntakeLedger()
    ledger.record("s", "chat", "B", "newest")

    claimed = ledger.claim("s", "chat", ["B"], owner="run-a")

    assert [row.owner for row in claimed] == ["run-a"]
    assert ledger.row("s", "chat", "B").owner == "run-a"

    ledger.release("s", "chat", ["B"], owner="another-run")
    assert ledger.row("s", "chat", "B").state == "processing"

    ledger.release("s", "chat", ["B"], owner="run-a")
    assert ledger.row("s", "chat", "B").state == "pending"
