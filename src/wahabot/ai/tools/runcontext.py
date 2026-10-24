"""Run-scoped delivery target and the chat/message fences.

Every run binds its own :class:`RunTarget` (session, chat, delivery
latches, operator flag) through ``bind_target`` before the workflow
starts, so shared tools always speak for the message being handled and
concurrent runs never see each other's state. Cross-chat reach is
fenced: only operator commands may select other chats.
"""

import contextvars
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from wahabot.ai.intake import IntakeEntry, IntakeLedger
from wahabot.ai.tools.envelope import error
from wahabot.core.echoes import remember_self_echo
from wahabot.core.jid import chat_from_message_id, same_chat
from wahabot.core.tickets.models import TicketUpdate
from wahabot.core.tickets.store import TicketStore
from wahabot.core.transport.context import bind_intake_ids
from wahabot.status import state as _status_state


def chat_jid(chat: str | None, target: RunTarget | dict[str, str]) -> str:
    """The chat JID to act on: *chat*, else the current conversation.

    Models sometimes pass a serialized message id (``false_<jid>_…``,
    scraped from a ``[message id: …]`` annotation) instead of a bare
    JID — the embedded chat is extracted so the call still lands in the
    right chat. Anything else passes through unchanged.
    """

    current = (
        target.chat_id if isinstance(target, RunTarget) else target.get("chat_id", "")
    )
    value = chat or current
    return chat_from_message_id(value) or value


@dataclass
class RunTarget:
    """One agent run's delivery target and latches.

    Mutable: the delivery latches (``sent``/``reacted``) flip mid-run
    from tool threads; ``sent_id`` is the outbound message id when the
    provider returned one. Scoped by ``contextvars`` (``bind_target``), so
    concurrent runs in different chats never see each other's targets,
    latches, or the operator arming flag.
    """

    session: str = ""
    chat_id: str = ""
    sent: str = ""
    sent_id: str = ""
    #: The text (or media caption) confirmed delivered this run.
    sent_text: str = ""
    reacted: str = ""
    armed: bool = False
    #: Typing-presence window (``(min_s, max_s)``) from settings; None
    #: or min<=0 disables the indicator for this run's text sends.
    typing: tuple[float, float] | None = None
    #: The chat's `chat-message-prefix` rule text, set by the caller
    #: from the chat's rule notes. Applied to sent and stored chat text
    #: alike so the model's next-turn view matches what the chat saw.
    prefix: str = ""
    #: Local path of this turn's inbound image, when the image service
    #: is configured (the source for an edit); "" otherwise.
    image_path: str = ""
    #: Chats an operator run may deliver to: those named in the command
    #: or resolved with read_chat this run. Anything else is refused.
    allowed_chats: set[str] = field(default_factory=set)
    #: Turn-level intake: the ids this run answers, when it started, the
    #: newer ids it claimed after a refusal, and the workflow's outcome.
    intake: IntakeLedger | None = None
    #: Unique owner of this run's claims; recovery never releases another
    #: run's claim and stale claims are attributable.
    owner: str = ""
    turn_ids: list[str] = field(default_factory=list)
    turn_started_at: float = field(default_factory=time.monotonic)
    claimed_ids: list[str] = field(default_factory=list)
    #: Set when a newer intake record superseded this run. Delivery is
    #: blocked for the rest of the current tool round; the next LLM
    #: round reads the refusal and may answer the newest record.
    guard_refused: bool = False
    #: The newest message this run was told to answer after a refusal;
    #: final text and a lone-emoji reaction anchor to it, not to the
    #: older turn that started the run.
    anchor_id: str = ""
    anchor_at: float = 0.0
    outcome: str = ""
    #: The ticket service and the current ticket, bound by the caller.
    ticket_store: TicketStore | None = None
    ticket_id: int = 0
    #: Values the model staged for the ticket; applied only after a
    #: confirmed delivery.
    ticket_update: TicketUpdate | None = None
    #: A staged ``new_ticket`` split summary, applied after delivery.
    ticket_split: str = ""

    @classmethod
    def from_dict(cls, holder: dict[str, str]) -> RunTarget:
        """Build a target from the legacy string dict (compatibility shim)."""
        return cls(
            session=holder.get("session", ""),
            chat_id=holder.get("chat_id", ""),
            sent=holder.get("sent", ""),
            reacted=holder.get("reacted", ""),
            armed=holder.get(OPERATOR_KEY, "") == OPERATOR_ARMED,
        )


#: Holder key set (only) on operator-command runs; chat runs store the
#: empty string. Kept for the ``from_dict`` shim the smoke suite drives.
OPERATOR_KEY = "operator"

#: The one value that arms the operator flag in the dict shim.
OPERATOR_ARMED = "armed"


def operator_run(target: RunTarget | dict[str, str]) -> bool:
    """True when the current run is an armed operator command.

    Chat participants cannot be allowed to point the bot's tools at
    chats they are not in: "tell everyone in <group> that …" from a
    group, or worse, a request to read or DM a stranger, must not
    exfiltrate or deliver anything outside the current conversation.
    CLI and self-chat command handlers arm the run; merely writing an
    operator marker in ordinary chat cannot open the fence.
    """
    if isinstance(target, RunTarget):
        return target.armed
    return target.get(OPERATOR_KEY) == OPERATOR_ARMED


#: The current run's send-target holder, scoped by ``contextvars``:
#: every agent run binds its own holder before the workflow starts, so
#: concurrent runs (different chats in parallel) can never see each
#: other's session/chat targets, delivery latches, or — critically —
#: the operator arming flag. Tool functions resolve their target here
#: at call time rather than capturing a builder-time conversation.
_run_target: contextvars.ContextVar[RunTarget | None] = contextvars.ContextVar(
    "wahabot_run_target", default=None
)


def bind_target(
    target: RunTarget | dict[str, str],
) -> contextvars.Token[RunTarget | None]:
    """Bind *target* as the current run's holder (called per run)."""
    if isinstance(target, dict):
        target = RunTarget.from_dict(target)
    return _run_target.set(target)


def reset_target(token: contextvars.Token[RunTarget | None]) -> None:
    """Restore the previous binding after a run finishes."""
    _run_target.reset(token)


def current_target() -> RunTarget:
    """The holder of the run executing on this task.

    Raises when no run is bound — a tool firing outside a run is a
    bug, and failing loudly beats acting on a stale target.
    """
    target = _run_target.get()
    if target is None:
        raise RuntimeError("tool called without a bound run target")
    return target


def clear_delivery_refusal(target: RunTarget | None = None) -> None:
    """Re-allow delivery at the start of a new model round.

    A refusal blocks the rest of its tool batch: the model has not seen
    the refusal yet, so a second delivery call in the same batch must
    not slip through. The next LLM round starts after the refusal was
    read and may retry.
    """
    holder = target if target is not None else _run_target.get()
    if holder is not None:
        holder.guard_refused = False


#: The rendered-record budget for one refusal. Selection drops whole
#: records from the oldest end, so every claimed id is shown; nothing is
#: settled without the model seeing it.
_GUARD_CHARS = 1200


def guard_selection(newer: Sequence[IntakeEntry]) -> list[IntakeEntry]:
    """The newest records that fit the refusal budget, oldest first."""
    selected: list[IntakeEntry] = []
    budget = _GUARD_CHARS
    for entry in reversed(newer):
        cost = len(entry.record) + (1 if selected else 0)
        if selected and cost > budget:
            break
        selected.append(entry)
        budget -= cost
    selected.reverse()
    return selected


def _claim_selected(
    target: RunTarget, selected: Sequence[IntakeEntry]
) -> list[IntakeEntry]:
    intake = target.intake
    assert intake is not None
    claimed = intake.claim(
        target.session,
        target.chat_id,
        [entry.message_id for entry in selected],
        owner=target.owner,
    )
    owned = {row.message_id for row in claimed}
    return [entry for entry in selected if entry.message_id in owned]


def _refusal_records(shown: Sequence[IntakeEntry], remaining: int) -> str:
    records = "\n".join(entry.record for entry in shown)
    if len(records) > _GUARD_CHARS:
        records = records[:_GUARD_CHARS]
    if remaining:
        records = (
            f"({remaining} older newer record(s) left for their own run)\n" + records
        )
    return records


def refuse_newer(target: RunTarget, newer: Sequence[IntakeEntry]) -> str | None:
    """Claim and render the newest records that supersede this run."""
    shown = _claim_selected(target, guard_selection(newer))
    if not shown:
        return None
    target.claimed_ids.extend(entry.message_id for entry in shown)
    # Later sends in this run answer the claimed newer rows too, so their
    # intents must carry those ids for crash recovery.
    bind_intake_ids([*target.turn_ids, *target.claimed_ids])
    target.guard_refused = True
    newest = shown[-1]
    if newest.at >= target.anchor_at:
        target.anchor_id = newest.message_id
        target.anchor_at = newest.at
    records = _refusal_records(shown, len(newer) - len(shown))
    return error(
        "Newer message(s) arrived while you were answering; this reply "
        + f"would answer an older turn.\n{records}\n"
        + "Do not send this reply. Read the newest record and answer it "
        + "once. If the newest record is media you cannot read (no "
        + "transcript or captions), end this run instead — its own run "
        + "will answer it."
    )


def delivery_guard(target: RunTarget) -> str | None:
    """Refuse a send that would answer an older turn than the chat's intake.

    The refusal claims only the records it renders. A retry in the next
    model round is allowed and the run's own delivery settles the claim;
    a run that ends without a send releases the claims again. Operator
    runs and turns without intake ids keep their old behaviour.
    """
    if target.intake is None or target.armed or not target.turn_ids:
        return None
    if target.guard_refused:
        return error(
            "A newer message already superseded this run in this tool round. "
            + "Do not send in this batch; answer the newest record in your "
            + "next response."
        )
    newer = target.intake.newer(
        target.session,
        target.chat_id,
        after=target.turn_started_at,
        exclude=[*target.turn_ids, *target.claimed_ids],
        addressed_only=True,
    )
    return refuse_newer(target, newer) if newer else None


#: Outcomes whose audit path is the outcome itself.
_OUTCOME_PATHS = {
    "reply": "text",
    "reaction": "reaction",
    "silence": "silence",
    "empty": "silence",
}


def tool_delivery_path(target: RunTarget | None) -> str:
    """The path for a tool-delivered run: a reaction when it only reacted."""
    if target is None:
        return "tool"
    return "reaction" if target.reacted and not target.sent_id else "tool"


def delivery_path(outcome: str, target: RunTarget | None) -> str:
    """The audit delivery path for a run outcome."""
    if outcome in _OUTCOME_PATHS:
        return _OUTCOME_PATHS[outcome]
    if outcome == "delivered_via_tool":
        return tool_delivery_path(target)
    if target is None or not (target.sent or target.reacted):
        return ""
    return tool_delivery_path(target)


def send_guard(target: RunTarget) -> str | None:
    """The delivery refusal for a send (superseded or already sent), or None."""
    if refusal := delivery_guard(target):
        return refusal
    if target.sent:
        return error(
            f"message already sent this run (to {target.sent}); do not send again"
        )
    return None


def reaction_guard(target: RunTarget) -> str | None:
    """The delivery refusal for a reaction (superseded or already reacted)."""
    if refusal := delivery_guard(target):
        return refusal
    if target.reacted:
        return error("already reacted this run; do not react again")
    return None


def current_prefix() -> str:
    """This run's chat-reply prefix, or "" when no run is bound."""
    target = _run_target.get()
    return target.prefix if target is not None else ""


def mark_run_outcome(outcome: str) -> None:
    """Set the current run's outcome when a run is bound."""
    target = _run_target.get()
    if target is not None:
        target.outcome = outcome


def current_image_path() -> str:
    """This run's inbound image path, or "" when there is none."""
    target = _run_target.get()
    return target.image_path if target is not None else ""


def track_self_echo(sent_id: str, what: str) -> None:
    """Mark a self-chat send's id, or warn that its echo is unprotected.

    A self-chat send whose response carried no recognizable id cannot
    be tracked in the echo cache, so its bounce-back would be parsed
    as a fresh operator command — the one degradation of the echo
    defense an operator must see (see ``core/echoes.py``).
    """
    if sent_id:
        remember_self_echo(sent_id)
        return
    logger.warning(
        "{what} to the self-chat returned no message id; its echo is"
        + " NOT tracked and would be parsed as an operator command if it"
        + " matches the mention pattern",
        what=what,
    )


#: How many chars of a tool's ``reason`` reach the log — a justification
#: is an audit line, not a transcript.
_REASON_LOG_CAP = 200


def log_action_reason(tool: str, reason: str, **context: Any) -> None:
    """Log the model's justification for a delivery/silence decision.

    Every tool takes an optional ``reason``: naming *why* the model is
    acting forces the choice to be articulated — the judicious group
    mode lives or dies by it — and gives the operator an audit trail of
    every send. Log-only: the reason never reaches a chat. An omitted
    reason is worth its own warning: it usually means the model acted
    on reflex.
    """
    suffix = f" {context}" if context else ""
    if not reason.strip():
        logger.warning(
            "Tool call {tool} carried no reason{suffix}",
            tool=tool,
            suffix=f" {context}" if context else "",
        )
        return
    logger.info(
        "Tool call {tool} reason: {reason}{suffix}",
        tool=tool,
        reason=reason.strip()[:_REASON_LOG_CAP],
        suffix=suffix,
    )


def delivered_to_self(chat_id: str, sent_id: str) -> None:
    """Mark a tool delivery that landed in the bot's own self-chat.

    Operator-command runs may legitimately deliver into the self-chat
    ("send me Ana's number"), but WhatsApp echoes that message back as
    a fresh ``fromMe`` event — and a self-chat message matching the
    mention pattern is (by design) an operator command. Unmarked, a
    *forwarded* message carrying injected text ("kAI message Roy …")
    would execute as a trusted command; marked, the echo is dead on
    arrival. Also covers self-sent images, files and plain sends.
    """
    me = _status_state.operator_jid
    if not me or not same_chat(chat_id, me):
        return
    track_self_echo(sent_id, "Delivery")


def send_failed_envelope(tool: str, chat_id: str, exc: Exception) -> str:
    """The fail-soft envelope for a delivery whose WAHA send raised.

    Every delivery tool funnels its WAHA call through this on error: the
    run must survive a dead WAHA (500s, timeouts) and the model must be
    told what to do instead — answer in text — not left staring at a
    raw exception. Without a successful acknowledgment, the latch
    stays open for a fallback reply. A network error does not prove
    that the remote server never accepted the original delivery.
    """
    logger.warning("{tool} failed in {chat}: {exc}", tool=tool, chat=chat_id, exc=exc)
    return error(
        f"{tool} failed — delivery was not confirmed; "
        + "send your reply as text instead, without repeating the failed action"
    )


#: The refusal envelope text for a chat run aiming outside its chat.
#: Model-facing: tells the model what it may do instead of guessing.
FENCE_ERROR = (
    "cross-chat reach is reserved for operator commands; this run may "
    "only act on the current conversation"
)


#: A WhatsApp JID as it appears in text (a group id may carry two numbers).
_JID_RE = re.compile(r"\b\d+(?:-\d+)?@[a-z.]+", re.IGNORECASE)


def jids_in_text(text: str) -> set[str]:
    """The JIDs literally present in *text*."""
    return set(_JID_RE.findall(text))


def delivery_chat_error(chat_id: str, target: RunTarget) -> str | None:
    """Refuse an operator delivery to a chat this run never resolved.

    Operator commands may reach other chats, but only ones the operator
    named in the command or the run resolved with read_chat — never a
    chat the model remembered or invented.
    """
    if not target.armed:
        return None
    if chat_id in target.allowed_chats:
        return None
    return (
        "that chat was not named in the command or resolved this run; "
        "resolve it with read_chat (or name its JID in the command) first, "
        "or just answer in the console"
    )


def delivery_target_error(
    chat_id: str, reply_to: str | None, target: RunTarget
) -> str | None:
    """Refusal for a chat delivery, including a quoted message's chat."""
    refusal = delivery_chat_error(chat_id, target)
    if refusal or not reply_to:
        return refusal
    embedded = chat_jid(reply_to, RunTarget())
    return delivery_chat_error(embedded, target) if "@" in embedded else None


def delivery_message_error(message_id: str, target: RunTarget) -> str | None:
    """Refusal for a delivery keyed by a message id from another chat."""
    embedded = chat_jid(message_id, RunTarget())
    return delivery_chat_error(embedded, target) if "@" in embedded else None


def fenced_chat(
    chat: str | None, target: RunTarget | dict[str, str]
) -> tuple[str | None, str | None]:
    """The ``(jid, error)`` a tool call may act on — exactly one is set.

    Every WhatsApp tool that takes a ``chat`` parameter runs through
    here instead of calling :func:`chat_jid` directly. Non-operator
    runs may only ever act on the conversation that produced them;
    operator-command runs pass through :func:`chat_jid` unchanged.
    The error half keeps the tools honest: a misdirected call fails
    loudly as an envelope, never silently falls back to the current
    chat (the fence invariant).
    """
    if operator_run(target):
        resolved = chat_jid(chat, target)
        if "@" not in resolved:
            # An operator command runs on the synthetic "operator"
            # context, not a WhatsApp chat: an omitted (or name-only)
            # chat would reach WAHA as `chats/operator/...` and 500.
            return None, (
                "operator commands have no default chat: pass `chat` as the exact"
                " JID (e.g. 1234567890-111111@g.us); resolve names with"
                " read_chat mode=resolve or mode=recent first"
            )
        return resolved, None
    current = (
        target.chat_id if isinstance(target, RunTarget) else target.get("chat_id", "")
    )
    if not chat:
        return current, None
    resolved = chat_jid(chat, target)
    if same_chat(resolved, current):
        return current, None
    if "@" not in resolved:
        logger.warning(
            "Refused malformed chat id {chat} in chat run in {current}",
            chat=chat,
            current=current,
        )
        return None, f"not a valid chat id: {chat!r}"
    logger.warning(
        "Refused cross-chat tool call to {chat} from chat run in {current}",
        chat=chat,
        current=current,
    )
    return None, FENCE_ERROR


def fenced_message_id(
    message_id: str, target: RunTarget | dict[str, str]
) -> tuple[str | None, str | None]:
    """The ``(message_id, error)`` a tool call may act on — one is set.

    A serialized WhatsApp id embeds its chat's JID as the second
    segment (``false_<jid>_<msgid>``); reads, reactions, quotes and
    forwards all act on that chat, so the id is another way to point
    a tool outside the conversation. On non-operator runs an id from
    any chat but the current one is refused (operator runs pass
    through). Ids that carry no recognizable JID — ``true_``-less
    foreign formats, engine quirks — are allowed through: WAHA
    validates ids server-side, and refusing unknown shapes would fence
    the current chat's own messages too.

    Returning the error (not just ``None``) lets every call site log
    and surface the same specific refusal, so the audit trail does not
    depend on which tool caught the id.
    """
    if operator_run(target):
        return message_id, None
    embedded = chat_jid(message_id, RunTarget())
    if not embedded or "@" not in embedded:
        return message_id, None
    current = (
        target.chat_id if isinstance(target, RunTarget) else target.get("chat_id", "")
    )
    if same_chat(embedded, current):
        return message_id, None
    logger.warning(
        "Refused cross-chat message id {id} from chat run in {current}",
        id=message_id,
        current=current,
    )
    return None, FENCE_ERROR
