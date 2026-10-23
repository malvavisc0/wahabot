from dataclasses import dataclass

from loguru import logger

from wahabot.core.transport import Transport

HEALTHY_STATUS = "WORKING"


@dataclass
class SessionState:
    """Runtime health of the WAHA session and the operator alert target."""

    healthy: bool = True
    operator_jid: str = ""
    operator_lid: str = ""
    #: LLM endpoint reachability: False while every run fails with
    #: ``APIConnectionError``. Once-per-transition notify, like the
    #: session flag — an outage must not spam the operator per message.
    llm_healthy: bool = True
    #: Timeout notify latch: True while the operator has already been
    #: told about LLM timeouts (one alert per slow-stretch; the first
    #: successful run resets it). Separate from ``llm_healthy`` — a
    #: timeout does not mean the endpoint is unreachable, so it must
    #: never flip the outage flag or its "replies paused" narrative.
    llm_timeout_notified: bool = False


state = SessionState()


def session_healthy() -> bool:
    """True while the WAHA session is WORKING."""
    return state.healthy


def llm_healthy() -> bool:
    """True while the LLM endpoint is believed reachable."""
    return state.llm_healthy


def set_session_health(status: str) -> None:
    """Update the health flag from a session status string."""
    state.healthy = status == HEALTHY_STATUS


def own_identity(waha: Transport, session: str) -> dict[str, str]:
    """The account's own ids (``own_jid``/``own_lid``) from WAHA, or empty.

    One fetch shared by every consumer of the account identity: the
    operator-alert capture at startup/recovery and the CLI's rendered
    prompt view. A failed fetch returns an empty dict — callers decide
    what an unknown identity means (status keeps its last value; the
    prompt render drops the identity lines).
    """
    try:
        me = waha.get_me(session)
    except Exception as exc:
        logger.warning("Could not fetch own identity: {exc}", exc=exc)
        return {}
    own = str(me.get("id") or "")
    if not own:
        return {}
    return {"own_jid": own, "own_lid": str(me.get("lid") or "")}


def capture_operator_target(waha: Transport, session: str) -> None:
    """Remember the bot's own JID as the operator-notification target.

    Called at startup and on every session recovery: the first capture
    may fail against a restarting WAHA, and a re-linked account changes
    the JID — without the retry the escalation lifeline (and the echo
    tracking keyed on the same JID) would stay wedged for the process
    lifetime.
    """
    identity = own_identity(waha, session)
    if not identity:
        return
    state.operator_jid = identity["own_jid"]
    state.operator_lid = identity["own_lid"]
    logger.info(
        "Operator identity: jid={jid} lid={lid}",
        jid=state.operator_jid,
        lid=state.operator_lid or "unknown",
    )
