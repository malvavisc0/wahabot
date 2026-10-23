import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from wahabot.ai.intake import (
    IntakeLedger,
    claim_lease_for_run_timeout,
    max_age_for_run_timeout,
)
from wahabot.core.db import Database
from wahabot.core.intake.sqlite import SqliteIntakeStore


def durable_ledger(tmp_path: Path) -> IntakeLedger:
    db = Database(tmp_path / "wahabot.db")
    db.migrate()
    return IntakeLedger(store=SqliteIntakeStore(db.path))


def test_record_is_idempotent_on_message_id_across_reopen(tmp_path: Path) -> None:
    ledger = durable_ledger(tmp_path)
    first = ledger.record(
        "s", "chat", "m1", "[[09:00 Ana #m1]] text :: original", burst_key="m1"
    )
    duplicate = ledger.record("s", "chat", "m1", "different record")

    reopened = durable_ledger(tmp_path)
    stored = reopened.row("s", "chat", "m1")

    assert duplicate.record == first.record == "[[09:00 Ana #m1]] text :: original"
    assert stored is not None and stored.record == first.record
    assert len(reopened.rows("s", "chat")) == 1
    assert stored.burst_key == "m1"


def test_record_stores_the_raw_event_and_reconstructs_it(tmp_path: Path) -> None:
    ledger = durable_ledger(tmp_path)
    raw = {"id": "evt", "event": "message", "session": "s", "payload": {"id": "m1"}}
    ledger.record(
        "s",
        "chat",
        "m1",
        "record",
        raw_json=json.dumps(raw),
        sender="491555000001@c.us",
        received_at=1_700_000_000.0,
    )

    stored = ledger.row("s", "chat", "m1")

    assert stored.event.payload["id"] == "m1"
    assert stored.sender == "491555000001@c.us"
    assert stored.received_at == 1_700_000_000.0


def test_claim_is_atomic_across_threads(tmp_path: Path) -> None:
    ledger = durable_ledger(tmp_path)
    ledger.record("s", "chat", "m1", "one")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda owner: ledger.claim("s", "chat", ["m1"], owner=owner),
                ("a", "b"),
            )
        )

    claimed = [row for result in results for row in result]
    assert [row.message_id for row in claimed] == ["m1"]
    assert claimed[0].state == "processing"
    assert claimed[0].owner in {"a", "b"}


def test_pending_applies_the_ledger_time_and_size_bounds(tmp_path: Path) -> None:
    fresh = durable_ledger(tmp_path / "fresh")
    fresh.record("s", "c", "old", "old", at=time.monotonic() - 61)
    fresh.record("s", "c", "new", "new")
    bounded = IntakeLedger(store=fresh.store, max_age_s=60, max_messages=1)

    assert [row.message_id for row in bounded.pending("s", "c")] == ["new"]
    assert [row.message_id for row in bounded.pending("s", "other")] == []


def test_release_returns_processing_rows_to_pending(tmp_path: Path) -> None:
    ledger = durable_ledger(tmp_path)
    ledger.record("s", "chat", "m1", "one")
    ledger.claim("s", "chat", ["m1"])

    ledger.release("s", "chat", ["m1"])

    assert ledger.row("s", "chat", "m1").state == "pending"
    assert ledger.pending("s", "chat")


def test_finish_and_finish_turn_record_outcomes(tmp_path: Path) -> None:
    ledger = durable_ledger(tmp_path)
    ledger.record("s", "chat", "a", "a")
    ledger.record("s", "chat", "b", "b")
    ledger.claim("s", "chat", ["b"])

    ledger.finish("s", "chat", ["a"], delivered=True, outcome="delivered")
    ledger.finish_turn(
        "s", "chat", ["b"], ["a"], delivered=False, outcome="silent", reason="quiet"
    )

    first = ledger.row("s", "chat", "a")
    second = ledger.row("s", "chat", "b")
    assert first.state == "done" and first.outcome == "delivered"
    assert second.state == "done" and second.outcome == "silent"
    assert second.reason == "quiet"


def test_due_and_stale_processing_rows_are_queryable(tmp_path: Path) -> None:
    ledger = durable_ledger(tmp_path)
    ledger.claim_lease_s = 0
    ledger.record("s", "chat", "pending", "pending")
    ledger.record("s", "chat", "failed", "failed")
    ledger.fail(
        "s",
        "chat",
        ["failed"],
        error="waha down",
        next_attempt_at=time.time() - 1,
    )
    ledger.record("s", "chat", "stale", "stale")
    ledger.claim("s", "chat", ["stale"], owner="dead-run")

    due = {row.message_id for row in ledger.due(time.time())}
    stale = {row.message_id for row in ledger.stale_processing(time.time() + 1)}

    assert due == {"pending", "failed"}
    assert "stale" in stale
    assert ledger.stats()["unanswered"] >= 1


def test_unanswered_backlog_age_is_queryable(tmp_path: Path) -> None:
    ledger = durable_ledger(tmp_path)
    ledger.record("s", "chat", "old", "old", received_at=time.time() - 120)

    stats = ledger.stats()

    assert stats["unanswered"] == 1
    assert stats["oldest_age_s"] >= 119


def test_intake_window_outlives_a_custom_run_timeout() -> None:
    assert max_age_for_run_timeout(3600) > 3600
    assert max_age_for_run_timeout(1) >= 1800


def test_claim_lease_outlives_a_custom_run_timeout() -> None:
    """Recovery's stale sweep must not release a live run's claims."""
    assert claim_lease_for_run_timeout(3600) > 3600
    assert claim_lease_for_run_timeout(1) >= 1800
    # The lease may never undercut the intake window: a shorter lease
    # would let recovery release a live run's claims mid-run.
    for run_timeout in (1, 60, 1800, 3600, 7200):
        assert claim_lease_for_run_timeout(run_timeout) >= max_age_for_run_timeout(
            run_timeout
        )


def test_claim_takes_back_a_failed_row_for_its_re_run(tmp_path: Path) -> None:
    """Claims cover failed rows too: a recovery re-run owns its retry.

    Without this the recovery tick would still see a failed row as due
    work while a slower re-run answered it.
    """
    ledger = durable_ledger(tmp_path)
    ledger.record("s", "chat", "m1", "one")
    ledger.fail("s", "chat", ["m1"], error="waha down", next_attempt_at=time.time() - 1)

    claimed = ledger.claim("s", "chat", ["m1"], owner="re-run")

    assert [row.message_id for row in claimed] == ["m1"]
    row = ledger.row("s", "chat", "m1")
    assert row.state == "processing"
    assert row.owner == "re-run"
    assert ledger.due(time.time()) == []


def test_retention_purges_long_finished_rows_only(tmp_path: Path) -> None:
    ledger = durable_ledger(tmp_path)
    ledger.record("s", "chat", "finished", "old", received_at=time.time() - 40 * 86400)
    ledger.finish("s", "chat", ["finished"], delivered=True)
    ledger.record("s", "chat", "pending", "still waiting")

    removed = ledger.purge(30)

    assert removed == 1
    assert ledger.row("s", "chat", "finished") is None
    assert ledger.row("s", "chat", "pending") is not None
