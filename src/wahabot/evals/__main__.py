"""``python -m wahabot.evals`` — run the offline evaluation harness.

Reports the scored fixtures and the gate's calibration; ``--traces DIR``
scans exported Langfuse traces too. ``--require-pass`` turns any failing
check into a nonzero exit, for use as a regression gate.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from wahabot.evals.fixtures import load_gate_labels
from wahabot.evals.runner import Report, evaluate, scan_traces
from wahabot.evals.scorers import Check


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    labels = load_gate_labels(args.gate_verdicts) if args.gate_verdicts else None
    report = evaluate(labels=labels)
    trace_checks = scan_traces(args.traces) if args.traces is not None else ()
    if args.json:
        _print_json(report, trace_checks)
    else:
        _print_text(report, trace_checks)
    failed = bool(report.failures()) or any(not check.passed for check in trace_checks)
    return 1 if args.require_pass and failed else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wahabot.evals",
        description="Score the chat engine's offline evaluation fixtures.",
    )
    parser.add_argument(
        "--traces", type=Path, help="directory of exported Langfuse traces to scan"
    )
    parser.add_argument(
        "--gate-verdicts",
        type=Path,
        help="JSON file of recorded gate verdicts keyed by fixture id",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--require-pass", action="store_true", help="exit nonzero when any check fails"
    )
    return parser


def _print_text(report: Report, trace_checks: tuple[Check, ...]) -> None:
    print(report.summary())
    for check in (*report.checks, *trace_checks):
        mark = "ok  " if check.passed else "FAIL"
        print(f"  {mark} {check.fixture_id} {check.kind}: {check.detail}")
    for score in report.gate.scores:
        rates = f"precision={score.precision():.2f} recall={score.recall():.2f}"
        print(f"  gate {score.label}: {rates}")


def _print_json(report: Report, trace_checks: tuple[Check, ...]) -> None:
    payload = report.as_dict()
    payload["traces"] = [asdict(check) for check in trace_checks]
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
