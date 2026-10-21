from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.tools import ToolSelection

from wahabot.ai.history import INBOUND_ID_RE
from wahabot.ai.triage.prompts import (
    TRIAGE_CONTEXT_TURNS,
    TRIAGE_RECALL_QUERY_CHARS,
    TRIAGE_TURN_CHARS,
    TRIAGE_WHY_LENGTH,
)

if TYPE_CHECKING:
    from wahabot.ai.triage.models import TriageVerdict


HARDENED_ALLOWED_TOOLS = frozenset(
    {
        "read_chat",
        "web_search",
        "visit_url",
        "escalate",
        "send_message",
        "stay_silent",
    }
)

ADVERSARIAL_MARKERS = (
    "ignore your",
    "ignore all",
    "system prompt",
    "developer mode",
    "other customer",
    "another customer",
    "phone number",
    "customer data",
    "account data",
    "password",
    "api key",
    "secret",
    "credit card",
    "list every",
)

ANGER_MARKERS = ("insult", "offensive", "angry", "frustrat", "profan", "rude")

CROSS_SENDER_MARKERS = (
    "other customer",
    "another customer",
    "other client",
    "another client",
)

CROSS_SENDER_WHY = (
    "cross-customer",
    "cross customer",
    "data probe",
    "exfiltration",
    "other customer",
    "another customer",
)


def calm_angry_customer(verdict: TriageVerdict, message: str) -> TriageVerdict:
    """Undo an adversarial read that is really an angry support request.

    The model is allowed to be wrong toward a normal run; an angry
    customer is the one false positive the product cannot afford. Only
    the model's own anger reason is overridden, and only when the
    message carries none of the hard adversarial markers.
    """
    if not verdict.adversarial:
        return verdict
    why = verdict.why.casefold()
    if not any(marker in why for marker in ANGER_MARKERS):
        return verdict
    lower = message.casefold()
    if any(marker in lower for marker in ADVERSARIAL_MARKERS):
        return verdict
    return replace(
        verdict,
        adversarial=False,
        why=f"not adversarial: {verdict.why}"[:TRIAGE_WHY_LENGTH],
    )


def restore_cross_sender_attack(verdict: TriageVerdict, message: str) -> TriageVerdict:
    """Restore a cross-customer probe the model described but did not flag."""
    if verdict.adversarial:
        return verdict
    lower = message.casefold()
    if not any(marker in lower for marker in CROSS_SENDER_MARKERS):
        return verdict
    why = verdict.why.casefold()
    if not any(marker in why for marker in CROSS_SENDER_WHY):
        return verdict
    return replace(verdict, adversarial=True)


def format_turns(
    messages: Sequence[ChatMessage], limit: int = TRIAGE_CONTEXT_TURNS
) -> tuple[str, ...]:
    """The last *limit* visible user/assistant turns, flattened."""
    turns: list[str] = []
    for message in messages:
        if message.role not in (MessageRole.USER, MessageRole.ASSISTANT):
            continue
        text = INBOUND_ID_RE.sub("", str(message.content or "")).strip()
        if not text:
            continue
        turns.append(f"{message.role.value}: {text[:TRIAGE_TURN_CHARS]}")
    return tuple(turns[-limit:])


def enrich_recall_query(query: str, needs: Sequence[str]) -> str:
    """Append ``history:`` topics to the recall query, capped like the index."""
    topics = [
        topic
        for need in needs
        if need.startswith("history:")
        and ":" in need
        and (topic := need.split(":", 1)[1].strip())
    ]
    if not topics:
        return query
    return (query + " " + " ".join(topics)).strip()[:TRIAGE_RECALL_QUERY_CHARS]


def is_tool_allowed(tool_call: ToolSelection, *, hardened: bool = True) -> bool:
    """Whether *tool_call* survives hardened mode's allowlist."""
    if not hardened:
        return True
    if tool_call.tool_name == "notebook":
        return str(tool_call.tool_kwargs.get("mode", "read")) == "read"
    return tool_call.tool_name in HARDENED_ALLOWED_TOOLS
