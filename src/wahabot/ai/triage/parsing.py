import json
from collections.abc import Mapping
from typing import Any

from wahabot.ai.triage.models import TriageContext, TriageResult, TriageVerdict
from wahabot.ai.triage.prompts import (
    TRIAGE_GIST_CHARS,
    TRIAGE_GIST_WORDS,
    TRIAGE_NEED_LENGTH,
    TRIAGE_NEED_LIMIT,
    TRIAGE_WHY_LENGTH,
)


def result_to_dict(result: TriageResult) -> dict[str, Any]:
    """A JSON-safe verdict for ``Context.store`` (which serializes state)."""
    return {
        "adversarial": result.verdict.adversarial,
        "escalate": result.verdict.escalate,
        "participate": result.verdict.participate,
        "why": result.verdict.why,
        "gist": result.verdict.gist,
        "needs": list(result.verdict.needs),
        "latency_ms": result.latency_ms,
        "timed_out": result.timed_out,
        "failed": result.failed,
    }


def result_from_dict(payload: Mapping[str, Any]) -> TriageResult:
    """Rebuild a verdict stored by :func:`result_to_dict`."""
    needs = payload.get("needs") or ()
    return TriageResult(
        TriageVerdict(
            adversarial=bool(payload.get("adversarial", False)),
            escalate=bool(payload.get("escalate", False)),
            participate=bool(payload.get("participate", False)),
            why=str(payload.get("why") or ""),
            gist=normalize_gist(str(payload.get("gist") or "")),
            needs=tuple(str(item) for item in needs if isinstance(item, str)),
        ),
        latency_ms=int(payload.get("latency_ms") or 0),
        timed_out=bool(payload.get("timed_out", False)),
        failed=bool(payload.get("failed", False)),
    )


def render_context(context: TriageContext) -> str:
    """Render the gate's bounded input: message, last turns, mute, rules."""
    quote = "none"
    if context.quoted_bot:
        quote = "the assistant"
    elif context.quoted_operator:
        quote = "the operator"
    elif context.quoted_other:
        quote = "another member"
    lines = [
        f"Chat type: {'group' if context.group else 'DM'}",
        f"Addressed to the assistant: {'yes' if context.addressed else 'no'}",
        f"Quote: {quote}",
        f"Side thread: {'yes' if context.side_thread else 'no'}",
        f"Session rules: {context.rules or 'none'}",
        f"Mute preferences: {context.mute or 'none'}",
        "Recent turns:",
    ]
    lines.extend(context.turns or ("(none)",))
    lines.extend(("Incoming message:", context.message))
    return "\n".join(lines)


def normalize_gist(gist: str) -> str:
    """Clamp the ledger gist: no newlines/quotes, ≤16 words, ≤120 chars."""
    text = " ".join(gist.replace('"', "").replace("'", "").split())
    words = text.split()
    if len(words) > TRIAGE_GIST_WORDS:
        text = " ".join(words[:TRIAGE_GIST_WORDS]) + "…"
    return text[:TRIAGE_GIST_CHARS]


def parse_verdict(raw: str) -> TriageVerdict:
    """Parse the gate's JSON object; raise when it is not usable."""
    payload = extract_json_object(raw)
    why = str(payload.get("why") or "")[:TRIAGE_WHY_LENGTH].strip()
    if why == "web" or why.startswith(("note:", "history:", "tool:")):
        why = ""
    return TriageVerdict(
        adversarial=as_bool(payload.get("adversarial"), default=False),
        escalate=as_bool(payload.get("escalate"), default=False),
        participate=as_bool(payload.get("participate"), default=False),
        why=why,
        gist=normalize_gist(str(payload.get("gist") or "")),
        needs=parse_needs(payload.get("needs")),
    )


def extract_json_object(raw: str) -> dict[str, Any]:
    """The first complete JSON object in *raw*, tolerating prose around it."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("triage response has no JSON object")
    payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("triage response JSON is not an object")
    return payload


def as_bool(value: Any, *, default: bool) -> bool:
    """Coerce a JSON-ish label to a bool, falling back on nonsense."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def parse_needs(value: Any) -> tuple[str, ...]:
    """Keep up to five short, unique hint strings."""
    if not isinstance(value, list):
        return ()
    needs: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        need = item.strip()[:TRIAGE_NEED_LENGTH]
        if need and need not in needs:
            needs.append(need)
    return tuple(needs[:TRIAGE_NEED_LIMIT])
