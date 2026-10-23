import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.harness import RecordingWaha
from wahabot.core.db import Database
from wahabot.core.delivery.intents import IntentStore
from wahabot.core.transport.waha import WahaTransport


class SpyWaha(RecordingWaha):
    """A recording client that can fail one send with a chosen error."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_with: Exception | None = None
        self.observed_state: list[str] = []

    def send_text(
        self,
        session: str,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        mentions: list[str] | None = None,
    ) -> str:
        if self.fail_with is not None:
            error = self.fail_with
            self.fail_with = None
            raise error
        return super().send_text(session, chat_id, text, reply_to, mentions)


def stores(tmp_path: Path) -> tuple[IntentStore, SpyWaha, WahaTransport]:
    db = Database(tmp_path / "wahabot.db")
    db.migrate()
    intents = IntentStore(db.path)
    client = SpyWaha()
    return intents, client, WahaTransport(client, intents)


def test_intent_is_prepared_and_sending_before_the_transport_call(
    tmp_path: Path,
) -> None:
    intents, client, transport = stores(tmp_path)
    states: list[str] = []

    def probe(*args, **kwargs) -> str:
        states.append(intents.recent(1)[0].state)
        return "sent_1"

    client.send_text = probe  # type: ignore[method-assign]

    transport.send_text("s", "chat", "hello")

    assert states == ["sending"]
    intent = intents.recent(1)[0]
    assert intent.state == "sent"
    assert intent.transport_message_id == "sent_1"
    assert intent.chat_id == "chat"
    assert intent.kind == "text"


def test_confirmed_send_finishes_the_associated_intake_rows(tmp_path: Path) -> None:
    intents, _client, transport = stores(tmp_path)
    from wahabot.ai.intake import IntakeLedger
    from wahabot.core.intake.sqlite import SqliteIntakeStore

    ledger = IntakeLedger(store=SqliteIntakeStore(intents.path))
    ledger.record("s", "chat", "m1", "record")
    from wahabot.core.transport.context import bind_intake_ids, reset_intake_ids

    token = bind_intake_ids(("m1",))
    try:
        transport.send_text("s", "chat", "hello")
    finally:
        reset_intake_ids(token)

    row = ledger.row("s", "chat", "m1")
    assert intents.recent(1)[0].intake_ids == ["m1"]
    assert row.state == "done" and row.outcome == "delivered"


def test_connection_failure_is_a_retryable_definite_failure(tmp_path: Path) -> None:
    intents, client, transport = stores(tmp_path)
    client.fail_with = httpx.ConnectError("refused")

    with pytest.raises(httpx.ConnectError):
        transport.send_text("s", "chat", "hello")

    intent = intents.recent(1)[0]
    assert intent.state == "failed"
    assert intent.next_attempt_at is not None
    assert intent.attempts == 1


def test_server_error_before_acceptance_is_retryable(tmp_path: Path) -> None:
    intents, client, transport = stores(tmp_path)
    request = httpx.Request("POST", "http://waha.invalid/api/sendText")
    client.fail_with = httpx.HTTPStatusError(
        "server error", request=request, response=httpx.Response(503, request=request)
    )

    with pytest.raises(httpx.HTTPStatusError):
        transport.send_text("s", "chat", "hello")

    intent = intents.recent(1)[0]
    assert intent.state == "failed"
    assert intent.next_attempt_at is not None


def test_client_error_is_not_retried_and_goes_to_the_operator(tmp_path: Path) -> None:
    intents, client, transport = stores(tmp_path)
    request = httpx.Request("POST", "http://waha.invalid/api/sendText")
    client.fail_with = httpx.HTTPStatusError(
        "bad request", request=request, response=httpx.Response(400, request=request)
    )

    with pytest.raises(httpx.HTTPStatusError):
        transport.send_text("s", "chat", "hello")

    intent = intents.recent(1)[0]
    assert intent.state == "failed"
    assert intent.next_attempt_at is None
    assert intent.operator_required


def test_uncertain_timeout_is_unknown_and_never_retried(tmp_path: Path) -> None:
    intents, client, transport = stores(tmp_path)
    client.fail_with = httpx.ReadTimeout("response lost")

    with pytest.raises(httpx.ReadTimeout):
        transport.send_text("s", "chat", "hello")

    intent = intents.recent(1)[0]
    assert intent.state == "unknown"
    assert intent.next_attempt_at is None
    assert intent.operator_required
    assert intents.recover_due(now=10**12) == []


def test_recovery_resumes_a_prepared_intent_exactly_once(tmp_path: Path) -> None:
    intents, client, transport = stores(tmp_path)
    intent = intents.prepare(
        session="s", chat_id="chat", kind="text", payload={"text": "prepared"}
    )

    replayed = transport.recover_intents(now=10**12)

    assert replayed == [intent.id]
    assert intents.get(intent.id).state == "sent"
    assert len(client.sent) == 1


def test_recovery_marks_a_sending_intent_unknown_without_resending(
    tmp_path: Path,
) -> None:
    intents, client, transport = stores(tmp_path)
    intent = intents.prepare(
        session="s", chat_id="chat", kind="text", payload={"text": "in flight"}
    )
    intents.sending(intent.id)

    replayed = transport.recover_intents(now=10**12)

    assert replayed == []
    assert intents.get(intent.id).state == "unknown"
    assert intents.get(intent.id).operator_required
    assert client.sent == []


def test_recovery_never_repeats_a_confirmed_send(tmp_path: Path) -> None:
    intents, client, transport = stores(tmp_path)
    transport.send_text("s", "chat", "confirmed")

    transport.recover_intents(now=10**12)

    assert len(client.sent) == 1
    assert intents.recent(1)[0].state == "sent"


def test_recovery_retries_a_due_definite_failure_with_bounded_attempts(
    tmp_path: Path,
) -> None:
    intents, client, transport = stores(tmp_path)
    client.fail_with = httpx.ConnectError("refused")
    with pytest.raises(httpx.ConnectError):
        transport.send_text("s", "chat", "retry me")
    intent = intents.recent(1)[0]
    assert intent.next_attempt_at is not None

    transport.recover_intents(now=10**12)
    assert intents.get(intent.id).state == "sent"
    assert len(client.sent) == 1


def test_unknown_sends_stay_unknown_even_when_recovery_runs(tmp_path: Path) -> None:
    intents, client, transport = stores(tmp_path)
    client.fail_with = httpx.ReadTimeout("lost")
    with pytest.raises(httpx.ReadTimeout):
        transport.send_text("s", "chat", "maybe")

    transport.recover_intents(now=10**12)
    transport.recover_intents(now=10**12)

    assert intents.recent(1)[0].state == "unknown"
    assert client.sent == []


def _record(bot_intents: IntentStore, message_id: str) -> Any:
    from wahabot.ai.intake import IntakeLedger
    from wahabot.core.intake.sqlite import SqliteIntakeStore

    ledger = IntakeLedger(store=SqliteIntakeStore(bot_intents.path))
    ledger.record("s", "chat", message_id, "record")
    return ledger


def test_retryable_failure_holds_the_row_and_a_fallback_cancels_it(
    tmp_path: Path,
) -> None:
    from wahabot.core.transport.context import bind_intake_ids, reset_intake_ids

    intents, client, transport = stores(tmp_path)
    ledger = _record(intents, "m1")
    client.fail_with = httpx.ConnectError("refused")

    token = bind_intake_ids(("m1",))
    try:
        with pytest.raises(httpx.ConnectError):
            transport.send_text("s", "chat", "first")
        failed = intents.recent(1)[0]
        assert failed.state == "failed"
        assert failed.next_attempt_at is not None
        assert ledger.row("s", "chat", "m1").state == "delivering"

        transport.send_text("s", "chat", "fallback")
    finally:
        reset_intake_ids(token)

    assert ledger.row("s", "chat", "m1").outcome == "delivered"
    assert intents.get(failed.id).state == "cancelled"
    assert len(client.sent) == 1

    transport.recover_intents(now=10**12)
    assert len(client.sent) == 1


def test_unknown_send_settles_the_intake_row_operator_required(
    tmp_path: Path,
) -> None:
    from wahabot.core.transport.context import bind_intake_ids, reset_intake_ids

    intents, client, transport = stores(tmp_path)
    ledger = _record(intents, "m1")
    client.fail_with = httpx.ReadTimeout("lost")

    token = bind_intake_ids(("m1",))
    try:
        with pytest.raises(httpx.ReadTimeout):
            transport.send_text("s", "chat", "maybe")
    finally:
        reset_intake_ids(token)

    row = ledger.row("s", "chat", "m1")
    assert row.state == "done" and row.outcome == "operator_required"
    assert intents.recent(1)[0].state == "unknown"


def test_remote_accepted_but_unrecorded_local_send_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wahabot.core.transport.context import bind_intake_ids, reset_intake_ids

    intents, client, transport = stores(tmp_path)
    ledger = _record(intents, "m1")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("sqlite write failed")

    monkeypatch.setattr(intents, "sent", boom)

    token = bind_intake_ids(("m1",))
    try:
        with pytest.raises(RuntimeError):
            transport.send_text("s", "chat", "accepted remotely")
    finally:
        reset_intake_ids(token)

    assert len(client.sent) == 1
    assert intents.recent(1)[0].state == "unknown"
    row = ledger.row("s", "chat", "m1")
    assert row.state == "done" and row.outcome == "operator_required"

    transport.recover_intents(now=10**12)
    assert len(client.sent) == 1


def test_exhausted_retry_is_terminal_and_not_relisted(tmp_path: Path) -> None:
    from wahabot.core.transport.context import bind_intake_ids, reset_intake_ids

    db = Database(tmp_path / "wahabot.db")
    db.migrate()
    intents = IntentStore(db.path, max_attempts=1)
    client = SpyWaha()
    transport = WahaTransport(client, intents)
    ledger = _record(intents, "m1")
    client.fail_with = httpx.ConnectError("refused")

    token = bind_intake_ids(("m1",))
    try:
        with pytest.raises(httpx.ConnectError):
            transport.send_text("s", "chat", "one attempt")
    finally:
        reset_intake_ids(token)

    intent = intents.recent(1)[0]
    assert intent.state == "failed"
    assert intent.next_attempt_at is None
    assert intent.operator_required
    assert intents.recoverable(10**12) == []
    row = ledger.row("s", "chat", "m1")
    assert row.state == "done" and row.outcome == "operator_required"
    transport.recover_intents(now=10**12)
    assert client.sent == []


def test_reaction_intent_records_the_chat(tmp_path: Path) -> None:
    intents, client, transport = stores(tmp_path)

    transport.send_reaction("s", "chat", "m1", "ok")

    intent = intents.recent(1)[0]
    assert intent.kind == "reaction"
    assert intent.chat_id == "chat"
    assert client.reactions == [("m1", "ok")]


def test_purge_removes_settled_intents_but_keeps_pending_ones(
    tmp_path: Path,
) -> None:
    intents, _client, _transport = stores(tmp_path)
    sent = intents.prepare(
        session="s", chat_id="chat", kind="text", payload={"text": "done"}
    )
    intents.sending(sent.id)
    intents.sent(sent.id, "wamid")
    pending = intents.prepare(
        session="s", chat_id="chat", kind="text", payload={"text": "pending"}
    )

    assert intents.purge(time.time() + 10) == 1
    with pytest.raises(KeyError):
        intents.get(sent.id)
    assert intents.get(pending.id).state == "prepared"


def test_ambiguous_operator_notice_is_retried(tmp_path: Path) -> None:
    intents, client, transport = stores(tmp_path)
    notice = intents.prepare(
        session="s",
        chat_id="operator",
        kind="operator_notice",
        payload={"text": "wahabot: notice"},
    )
    client.fail_with = httpx.ReadTimeout("lost")

    transport.recover_intent(notice, now=time.time())

    failed = intents.get(notice.id)
    assert failed.state == "failed"
    assert failed.next_attempt_at is not None

    client.fail_with = None
    transport.recover_intents(now=time.time() + 10**6)

    assert intents.get(notice.id).state == "sent"
    assert len(client.sent) == 1
