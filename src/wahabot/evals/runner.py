"""Run the evaluation fixtures and assemble one report.

The harness takes the built-in fixtures by default and accepts replacements,
so a phase can score a new reply set or a new gate labeler without touching
the fixtures. Exported Langfuse traces are scanned separately: their replies
must be exactly what the chat saw.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from wahabot.evals.fixtures import (
    Canary,
    GateCase,
    GateLabel,
    SupportThread,
    canaries,
    gate_cases,
    load_trace,
    support_threads,
)
from wahabot.evals.scorers import (
    Check,
    GateReport,
    baseline_gate,
    check_canary,
    check_support,
    scan_trace,
    score_gate,
)


@dataclass(frozen=True)
class Report:
    """The scored fixtures and the gate's calibration."""

    checks: tuple[Check, ...]
    gate: GateReport

    def failures(self) -> tuple[Check, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def summary(self) -> str:
        passed = sum(1 for check in self.checks if check.passed)
        return (
            f"{len(self.checks)} checks: {passed} passed, "
            f"{len(self.checks) - passed} failed; angry-customer mislabelled: "
            f"{self.gate.angry_customer_flagged}/{self.gate.angry_customer_total}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "checks": [asdict(check) for check in self.checks],
            "gate": {
                "scores": [
                    {
                        **asdict(score),
                        "precision": score.precision(),
                        "recall": score.recall(),
                    }
                    for score in self.gate.scores
                ],
                "angry_customer_flagged": self.gate.angry_customer_flagged,
                "angry_customer_total": self.gate.angry_customer_total,
            },
        }


def evaluate(
    *,
    support: Sequence[SupportThread] | None = None,
    groups: Sequence[Canary] | None = None,
    gate: Sequence[GateCase] | None = None,
    labels: Mapping[str, GateLabel] | None = None,
) -> Report:
    """Score the fixtures (built-ins by default) and return one report."""
    support = support_threads() if support is None else support
    groups = canaries() if groups is None else groups
    gate = gate_cases() if gate is None else gate
    labels = baseline_gate(gate) if labels is None else labels
    checks: list[Check] = []
    for thread in support:
        checks.extend(check_support(thread))
    for canary in groups:
        checks.extend(check_canary(canary))
    return Report(checks=tuple(checks), gate=score_gate(gate, labels))


def scan_traces(directory: Path) -> tuple[Check, ...]:
    """Scan every ``*.json`` Langfuse export in *directory*.

    A file that does not parse becomes a failed check rather than an
    exception: one bad export must not hide the rest of the run.
    """
    checks: list[Check] = []
    for path in sorted(directory.glob("*.json")):
        try:
            run = load_trace(path)
        except (ValueError, json.JSONDecodeError) as error:
            checks.append(Check(path.stem, "trace_visible", False, str(error)))
            continue
        checks.append(scan_trace(run, path.stem))
    return tuple(checks)
