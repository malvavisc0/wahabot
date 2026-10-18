"""Run-scoped delivery target and the chat/message fences.

Every run binds its own :class:`RunTarget` (session, chat, delivery
latches, operator flag) through ``bind_target`` before the workflow
starts, so shared tools always speak for the message being handled and
concurrent runs never see each other's state. Cross-chat reach is
fenced: only operator commands may select other chats.
"""

import contextvars
from dataclasses import dataclass
from typing import Any

from loguru import logger

from wahabot.ai.tools.envelope import error
from wahabot.core.echoes import remember_self_echo
from wahabot.core.jid import chat_from_message_id, same_chat
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
    from tool threads. Scoped by ``contextvars`` (``bind_target``), so
    concurrent runs in different chats never see each other's targets,
    latches, or the operator arming flag.
    """

    session: str = ""
    chat_id: str = ""
    sent: str = ""
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


def current_prefix() -> str:
    """This run's chat-reply prefix, or "" when no run is bound."""
    target = _run_target.get()
    return target.prefix if target is not None else ""


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
