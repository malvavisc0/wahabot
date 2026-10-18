"""Offline tests for the chat-engine evaluation harness.

The fixtures carry a naive baseline reply; these tests prove the scorers
both accept a corrected reply and detect the baseline gaps, that the gate
metrics compute from recorded verdicts, and that exported traces load and
scan without any service.
"""

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from wahabot.ai.pinned_state import PINNED_STATE_TOKEN_LIMIT, render_pinned_block
from wahabot.evals import (
    Check,
    GateLabel,
    SupportThread,
    baseline_gate,
    canaries,
    check_canary,
    check_support,
    evaluate,
    gate_cases,
    load_trace,
    scan_trace,
    scan_traces,
    score_gate,
    support_threads,
)
from wahabot.evals.__main__ import main
from wahabot.evals.fixtures import RecordedRun

PROMISE_FIXTURE = "support_promise_30_turns"
REASK_FIXTURE = "support_no_reask_reference"


def _thread(fixture_id: str) -> SupportThread:
    return next(thread for thread in support_threads() if thread.fixture_id == fixture_id)


def _kinds(checks: Sequence[Check], *, passed: bool | None = None) -> set[str]:
    return {check.kind for check in checks if passed is None or check.passed is passed}


def test_default_fixtures_build_in_production_shape() -> None:
    promise = _thread(PROMISE_FIXTURE)
    assert len(promise.turns) == 30
    assert promise.pinned_state  # note, rules
    roles = {turn.role for turn in promise.turns}
    assert roles == {"user", "assistant"}
    # The promise sits at turn 5, the question at turn 30.
    assert "PDF" in promise.turns[4].text
    assert promise.turns[-1].role == "user"
    assert {case.category for case in gate_cases()} >= {
        "injection",
        "cross_sender",
        "angry_customer",
        "escalate",
        "benign",
    }


def test_support_checks_accept_a_correct_reply() -> None:
    corrected = {
        PROMISE_FIXTURE: (
            "Te enviaré el PDF mañana a ana@example.com; el Ticket 142 sigue abierto."
        ),
        REASK_FIXTURE: (
            "Sí, tengo tu REF-7781 anotada en el Ticket 155; la revisión sigue en curso."
        ),
    }
    for fixture_id, reply in corrected.items():
        checks = check_support(replace(_thread(fixture_id), reply=reply))
        assert all(check.passed for check in checks), (fixture_id, checks)


def test_support_facts_map_onto_the_rendered_pinned_block() -> None:
    """Every fixture fact appears verbatim in the block a run renders.

    The ``pinned_state`` facts are the block's contract: each value is
    a fragment the renderer actually emits, so a renderer change that
    drops one breaks here before the eval silently stops testing it.
    """
    for thread in support_threads():
        block = render_pinned_block(
            notes=thread.notes,
            rules=thread.rules,
            token_budget=PINNED_STATE_TOKEN_LIMIT,
        )
        assert block
        for fact in thread.pinned_state:
            assert fact.value in block, (thread.fixture_id, fact.label)


def test_support_checks_detect_the_baseline_gaps() -> None:
    promise_checks = check_support(_thread(PROMISE_FIXTURE))
    assert _kinds(promise_checks, passed=False) >= {
        "pinned_state",
        "promise_honored",
        "no_reask",
    }
    reask_checks = check_support(_thread(REASK_FIXTURE))
    assert _kinds(reask_checks, passed=False) >= {"pinned_state", "no_reask"}


def test_support_check_reports_what_failed() -> None:
    checks = check_support(_thread(PROMISE_FIXTURE))
    pinned = next(check for check in checks if check.kind == "pinned_state")
    assert "ana@example.com" in pinned.detail
    reask = next(check for check in checks if check.kind == "no_reask")
    assert "correo" in reask.detail


def test_canaries_flag_the_baseline_and_accept_corrections() -> None:
    by_id = {canary.fixture_id: canary for canary in canaries()}
    injection = by_id["canary_injection_operator_note"]
    cross = by_id["canary_cross_sender_email"]
    muted = by_id["canary_muted_member"]
    assert not check_canary(injection)[0].passed
    assert not check_canary(cross)[0].passed
    assert not check_canary(muted)[0].passed

    corrected = {
        injection.fixture_id: "No puedo compartir notas internas.",
        cross.fixture_id: "No puedo compartir datos de otras personas.",
        muted.fixture_id: "",
    }
    for fixture_id, reply in corrected.items():
        checks = check_canary(replace(by_id[fixture_id], reply=reply))
        assert all(check.passed for check in checks), (fixture_id, checks)
    # A leaked silence token is production-silent, so it must pass here too.
    silent_token = replace(muted, reply="I'll stay silent here.")
    assert all(check.passed for check in check_canary(silent_token))


def test_gate_perfect_labeler_scores_exactly() -> None:
    cases = gate_cases()
    labels = {case.fixture_id: case.expected for case in cases}
    report = score_gate(cases, labels)
    assert report.exact()
    assert report.angry_customer_flagged == 0
    assert report.angry_customer_total == 3


def test_gate_baseline_reports_the_recall_gap() -> None:
    cases = gate_cases()
    report = score_gate(cases, baseline_gate(cases))
    adversarial = report.score("adversarial")
    assert adversarial.recall() == 0.0
    # A do-nothing gate makes no false-positive predictions.
    assert adversarial.precision() == 1.0
    assert report.angry_customer_flagged == 0


def test_gate_counts_an_angry_customer_mislabelled_adversarial() -> None:
    cases = gate_cases()
    labels = baseline_gate(cases)
    for case in cases:
        if case.category == "angry_customer":
            labels[case.fixture_id] = GateLabel(adversarial=True)
    report = score_gate(cases, labels)
    assert report.angry_customer_flagged == 3
    assert report.score("adversarial").precision() < 1.0


def test_fixture_pinned_state_is_renderable_by_the_engine_block() -> None:
    """Every declared pinned fact is what the engine's block actually renders.

    The fixtures' declared facts are backed by ``notes``/``rules``
    rendered through the real ``render_pinned_block``: if the engine's
    block format changes, this fails instead of the fixtures silently
    drifting.
    """
    from wahabot.ai.pinned_state import (
        PINNED_STATE_TOKEN_LIMIT,
        render_pinned_block,
    )

    for thread in support_threads():
        block = render_pinned_block(
            notes=thread.notes,
            rules=thread.rules,
            token_budget=PINNED_STATE_TOKEN_LIMIT,
        )
        for fact in thread.pinned_state:
            assert fact.value in block, (thread.fixture_id, fact.label)
        for value in thread.pinned_state_used:
            assert value in block, (thread.fixture_id, value)


def test_evaluate_scores_builtins_and_serializes() -> None:
    report = evaluate()
    assert report.checks
    assert report.failures()  # the baseline replies are expected to fail
    payload = json.loads(json.dumps(report.as_dict()))
    assert payload["gate"]["angry_customer_flagged"] == 0
    assert "checks:" in report.summary()


def _write_trace(path: Path, *, reply: str) -> None:
    export = {
        "observations": [
            {
                "type": "GENERATION",
                "input": json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "system prompt"},
                            {"role": "user", "content": "hola"},
                            {"role": "assistant", "content": "hola, ¿en qué ayudo?"},
                            {"role": "user", "content": "no llega la confirmación"},
                        ]
                    }
                ),
                "output": json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "blocks": [
                                {"block_type": "thinking", "text": "hmm"},
                                {"block_type": "text", "text": reply},
                            ],
                        }
                    }
                ),
            }
        ]
    }
    path.write_text(json.dumps(export), encoding="utf-8")


def test_load_trace_reads_turns_and_reply(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    _write_trace(path, reply="Lo reviso ahora.")
    run = load_trace(path)
    assert [turn.role for turn in run.turns] == ["user", "assistant", "user"]
    assert [turn.text for turn in run.turns][1] == "hola, ¿en qué ayudo?"
    assert run.reply == "Lo reviso ahora."


def test_load_trace_skips_the_wrap_up_round_on_a_delivered_run(tmp_path: Path) -> None:
    """The reply is the delivered text, never the private wrap-up note.

    A delivered run's last text-bearing generation is the wrap-up round:
    its output is the private lesson the
    chat never saw. Scoring that as the reply would test the wrong text
    and never see the delivered message.
    """
    wrap_up_request = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {
                "role": "user",
                "content": "You already delivered your reply. "
                "Do not send more replies or use any tools. Review this run "
                "for a verified, reusable lesson, not an activity summary.",
            },
        ]
    }
    export = {
        "observations": [
            {
                "type": "GENERATION",
                "input": json.dumps({"messages": [{"role": "user", "content": "hola"}]}),
                "output": json.dumps(
                    {"message": {"role": "assistant", "content": "Te envié el PDF."}}
                ),
            },
            {
                "type": "GENERATION",
                "input": json.dumps(wrap_up_request),
                "output": json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Sent the PDF and confirmed receipt.",
                        }
                    }
                ),
            },
        ]
    }
    path = tmp_path / "delivered.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    run = load_trace(path)
    assert run.reply == "Te envié el PDF."


def test_load_trace_rejects_a_malformed_export(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"observations": []}', encoding="utf-8")
    try:
        load_trace(path)
    except ValueError as error:
        assert "GENERATION" in str(error)
    else:  # pragma: no cover - the guard above is the assertion
        raise AssertionError("expected ValueError for an export without a generation")


def test_load_trace_accepts_the_api_shape_and_skips_tool_only_tails(
    tmp_path: Path,
) -> None:
    request = {"messages": [{"role": "user", "content": "hola"}]}
    tool_only = {
        "message": {
            "role": "assistant",
            "blocks": [{"block_type": "tool_call", "tool_name": "send"}],
        }
    }
    export = {
        "data": [
            {
                "type": "GENERATION",
                "input": json.dumps(request),
                "output": json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "blocks": [{"block_type": "text", "text": "Hola."}],
                        }
                    }
                ),
            },
            {"type": "GENERATION", "input": None, "output": json.dumps(tool_only)},
        ],
        "meta": {},
    }
    path = tmp_path / "api.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    run = load_trace(path)
    assert run.reply == "Hola."
    assert [turn.role for turn in run.turns] == ["user"]


def test_scan_trace_flags_a_leaked_note() -> None:
    clean = RecordedRun(turns=(), reply="Lo reviso ahora.")
    assert scan_trace(clean, "clean").passed
    leaked = RecordedRun(
        turns=(),
        reply=(
            "Lo reviso ahora.\nSent a confirmation that the request to stop was honored."
        ),
    )
    check = scan_trace(leaked, "leaked")
    assert not check.passed
    assert "trailing note" in check.detail


def test_scan_traces_reports_bad_files_without_stopping(tmp_path: Path) -> None:
    _write_trace(tmp_path / "good.json", reply="Todo listo.")
    (tmp_path / "broken.json").write_text("{ truncated", encoding="utf-8")
    checks = {check.fixture_id: check for check in scan_traces(tmp_path)}
    assert checks["good"].passed
    assert not checks["broken"].passed


def test_cli_defaults_and_require_pass(tmp_path: Path) -> None:
    assert main([]) == 0
    assert main(["--require-pass"]) == 1  # the baseline replies fail
    _write_trace(tmp_path / "good.json", reply="Todo listo.")
    assert main(["--traces", str(tmp_path)]) == 0


def test_cli_json_counts_trace_checks(tmp_path: Path, capsys) -> None:
    _write_trace(tmp_path / "good.json", reply="Todo listo.")
    (tmp_path / "leaked.json").write_text("{}", encoding="utf-8")
    assert main(["--json", "--traces", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["traces"]) == 2
    assert sum(1 for check in payload["traces"] if not check["passed"]) == 1
