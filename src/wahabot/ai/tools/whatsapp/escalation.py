import time

from llama_index.core.tools import BaseTool, FunctionTool
from loguru import logger

from wahabot.ai.tools.envelope import error, ok
from wahabot.ai.tools.runcontext import (
    current_target,
    operator_run,
    track_self_echo,
)
from wahabot.ai.tools.schemas import (
    EscalateSchema,
)
from wahabot.core.audit import save_action
from wahabot.core.transport import Transport
from wahabot.core.transport.context import bind_intake_ids, reset_intake_ids
from wahabot.settings import Settings
from wahabot.status import state as _status_state

_ESCALATE_COOLDOWN_S = 3600


class EscalationChannel:
    """Per-agent escalation state: the cooldowns.

    Instance-scoped (one per ``build_default_tools`` call) rather than
    module-global: two agents built in one process — the smoke suite
    does exactly this — must never share a cooldown window. The
    operator JID is *not* cached here: it is read from
    ``status.state`` at call time, so a startup WAHA hiccup or a
    re-linked account can never wedge the lifeline (or aim it at a
    stale identity) for the process lifetime.
    """

    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    @property
    def operator_jid(self) -> str:
        """The current operator JID (single-sourced on ``status.state``)."""
        return _status_state.operator_jid

    def cooldown_refusal(self, chat_id: str) -> str | None:
        """The cooldown error text for *chat_id*, or None when clear."""
        last = self._last.get(chat_id)
        if last is None:
            return None
        elapsed = time.monotonic() - last
        if elapsed >= _ESCALATE_COOLDOWN_S:
            return None
        remaining = int(_ESCALATE_COOLDOWN_S - elapsed)
        message = f"already escalated from this chat; cooldown {remaining}s left"
        return (
            f"{message} — an earlier report from this chat was forwarded; "
            "this new report was not forwarded; do not claim it was"
        )

    def stamp(self, chat_id: str) -> None:
        """Record a successful escalation from *chat_id*."""
        self._last[chat_id] = time.monotonic()


def forward_escalation(
    waha: Transport,
    channel: EscalationChannel,
    settings: Settings,
    session: str,
    chat_id: str,
    report: str,
) -> str:
    """Forward a code- or model-written report from *chat_id* to the operator.

    Shared by the ``escalate`` tool and the triage gate. The cooldown is
    per chat, so a gate ping and an agent escalation in the same run can
    never both reach the operator. The report text is passed through;
    callers that do not have member text (the gate) write their own.
    """
    if not report.strip():
        return error("empty report text")
    operator_jid = channel.operator_jid
    if not operator_jid:
        return error("operator contact unknown; escalation unavailable")
    refusal = channel.cooldown_refusal(chat_id)
    if refusal is not None:
        return error(refusal)
    try:
        # An operator report does not answer the customer turn, so its
        # intent must not carry the turn's intake ids: a failure here must
        # neither hold those rows nor be cancelled by the customer reply.
        token = bind_intake_ids(())
        try:
            sent_id = waha.send_text(
                session,
                operator_jid,
                f"wahabot escalation from {chat_id}:\n{report.strip()}",
            )
        finally:
            reset_intake_ids(token)
    except Exception as exc:
        # No confirmed acknowledgment: leave the cooldown unstamped
        # and never claim the operator was notified.
        logger.warning(
            "Escalation from {chat_id} failed to send: {exc}",
            chat_id=chat_id,
            exc=exc,
        )
        return error(
            f"could not confirm forwarding to the operator: {exc}"
            + " — delivery was not confirmed; do not claim it went through "
            + "or repeat the report automatically"
        )
    track_self_echo(sent_id, "Escalation")
    channel.stamp(chat_id)
    # Durable record after the confirmed send — the audit journal's
    # fail-soft write never breaks the escalation.
    save_action(
        settings.data_dir,
        session,
        "escalation",
        chat_id=chat_id,
        report=report,
        sent_id=sent_id,
        status="open",
    )
    logger.info(
        "Escalated from {chat_id} to operator: {report}",
        chat_id=chat_id,
        report=report.strip()[:200],
    )
    return ok(chat=chat_id, escalated=True, sent_id=sent_id)


def escalate(waha: Transport, channel: EscalationChannel, settings: Settings) -> BaseTool:
    """Build a tool that forwards a report from the chat to the operator.

    The one sanctioned way a chat run reaches the operator: the target
    JID is not a parameter but the bot's own self-chat (the same one
    the operator reads up/down notifications in), carried by *channel*.
    Only the supplied report and current chat id are sent. Report text
    is passed through, not sanitized: summarizing without pasted messages
    or secrets is a model instruction, not an enforced tool guarantee.
    This send does not consume the normal reply allowance.

    In-memory cooldown per chat: a second escalate after a successful
    send inside the window fails with an error naming remaining seconds.
    A new channel (including a process restart) starts with no cooldowns.
    The send itself is fail-soft — an unreachable session must not
    crash the run, and the cooldown is stamped only after a confirmed
    delivery: a send error says forwarding was not confirmed. A
    cooldown error distinguishes the earlier delivery from this
    request, which was not forwarded.

    After a confirmed send, the tool attempts a fail-soft audit write
    (``kind: "escalation"``). Audit failure does not undo the delivery.
    """

    def escalate_fn(report: str, reason: str = "") -> str:
        if not report.strip():
            return error("empty report text")
        target = current_target()
        chat_id = target.chat_id
        if not chat_id or not target.session:
            return error("no active conversation context")
        if operator_run(target):
            # An operator command already talks to the operator; the tool
            # would only loop the report back to its author.
            refusal = "this run is already an operator command"
            return error(f"{refusal} — report directly in the instruction instead")
        if target.ticket_id:
            # A ticket turn has the richer hand_to_operator path (contact,
            # language, attachments and a durable ticket state). Escalating
            # here would bypass it and lose the ticket.
            return error(
                "ticket chats use hand_to_operator, not escalate: it records the "
                "ticket, copies attachments and alerts the operator"
            )
        return forward_escalation(
            waha, channel, settings, target.session, chat_id, report
        )

    return FunctionTool.from_defaults(
        fn=escalate_fn,
        fn_schema=EscalateSchema,
        name="escalate",
        description=(
            "Before replying, forward a participant's report to the fixed human operator "
            "(not in ticket chats: use hand_to_operator there). "
            "for human help, bot problems, complaints, or refusals. "
            "Not for operator commands. Does not consume the reply allowance. "
            "One successful forward per chat per hour. "
            "On success, confirm this report was forwarded. On "
            "cooldown, say an earlier report from this chat was "
            "forwarded but this new report was not. On send failure, "
            "say forwarding was not confirmed; do not automatically retry. "
            "Other errors mean the tool refused the forward. Do not "
            "claim it was read or promise a response time."
        ),
    )
