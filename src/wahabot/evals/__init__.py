"""Offline evaluation harness for the chat engine.

Fixtures are production-shaped: DM support threads with the pinned state a
reply depends on, debate-group canaries for leaks, injection and mute
handling, and labelled adversarial DMs for the triage gate. The harness
scores recorded replies and gate verdicts offline — no services, no model —
so every later phase is measured against the same set.

Run it with ``python -m wahabot.evals``; pass ``--traces DIR`` to scan
exported Langfuse traces as well.
"""

from wahabot.evals.fixtures import (
    Canary,
    GateCase,
    GateLabel,
    PinnedFact,
    RecordedRun,
    SupportThread,
    Turn,
    canaries,
    gate_cases,
    load_trace,
    support_threads,
)
from wahabot.evals.runner import Report, evaluate, scan_traces
from wahabot.evals.scorers import (
    Check,
    GateReport,
    LabelScore,
    baseline_gate,
    check_canary,
    check_support,
    scan_trace,
    score_gate,
)

__all__ = [
    "Canary",
    "Check",
    "GateCase",
    "GateLabel",
    "GateReport",
    "LabelScore",
    "PinnedFact",
    "RecordedRun",
    "Report",
    "SupportThread",
    "Turn",
    "baseline_gate",
    "canaries",
    "check_canary",
    "check_support",
    "evaluate",
    "gate_cases",
    "load_trace",
    "scan_trace",
    "scan_traces",
    "score_gate",
    "support_threads",
]
