"""Session health: gate agent runs on the WAHA session being WORKING.

``wahabot`` is only as alive as its WAHA session: when the session
leaves ``WORKING`` every send would fail, yet the agent would still
burn LLM tokens, tool calls and transcriptions producing replies that
can never be delivered. This module tracks session health (seeded from
``GET /api/sessions/{session}`` at startup, updated by every
``session.status`` event), mutes message/command handling while the
session is down, and notifies the operator once per transition.

Recovery is ACTIVE, not only event-driven: WAHA's engines do not
reliably detect their own reconnection — a half-dead socket can leave
the session muted long after WhatsApp is back, until something (an
operator restarting the internet) forces WAHA to notice. So while
muted, a background poller asks WAHA directly whether the session is
``WORKING`` again (and every incoming message gets a cheap probe too:
webhooks being delivered is itself evidence the session is back). The
recovery notification is retried, not single-shot — a notify that
races WhatsApp's own reconnect settles instead of vanishing.

The same pattern guards the LLM endpoint: an unreachable provider
fails every run, so the first outage-class failure (connection error,
or a 5xx from a dead proxy in front of the provider) flips the health
flag and notifies the operator once (not once per dropped message),
and the first successful reply flips it back.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

import openai
from loguru import logger

from wahabot.core.echoes import remember_self_echo
from wahabot.core.waha import WahaClient
from wahabot.webhook import on_session_status

#: statuses reported by ``WASessionStatusBody`` that keep the bot muted;
#: ``WORKING`` (and nothing else) is healthy.
HEALTHY_STATUS = "WORKING"

#: Seconds between session-status probes while muted. WAHA answers in
#: milliseconds when alive; the poller only runs while unhealthy, and
#: only one run is outstanding at a time.
RECOVERY_POLL_S = 15.0

#: Retry budget for the recovery notification: the send races
#: WhatsApp's own reconnection, so a failed notify backs off and
#: retries instead of vanishing (the "back online" message the
#: operator never received).
_NOTIFY_ATTEMPTS = 3
_NOTIFY_BACKOFF_S = 2.0

#: The background poller's task handle, once armed (None while
#: healthy — nothing runs in the steady state).
_recovery_task: asyncio.Task[None] | None = None
#: Serializes probes: a poll and a message-triggered probe must not
#: both flip the flag and double-notify.
_probe_lock = asyncio.Lock()


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
    #: told about LLM timeouts (one 🟠 per slow-stretch; the first
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


def seed_health(waha: WahaClient, session: str) -> str:
    """Seed the health flag and operator target from WAHA.

    Called once at startup: the transition tracker is only as good as
    its starting point — a session that is already dead when wahabot
    boots may never emit a ``session.status`` event (nothing changes),
    and without this fetch the bot would happily "work" against it.
    Also captures the bot's own JID as the operator-notification
    target (read it in the "Message yourself" chat). Returns the
    status string; failures keep the optimistic default (healthy) so a
    flaky WAHA restart cannot wedge the bot muted.
    """
    try:
        info = waha.get_session(session)
    except Exception as exc:
        logger.warning(
            "Could not fetch session status at startup: {exc}; assuming WORKING", exc=exc
        )
        return HEALTHY_STATUS
    status = str(info.get("status") or HEALTHY_STATUS)
    set_session_health(status)
    if not state.healthy:
        logger.warning(
            "WAHA session {session} starts in {status} — bot muted",
            session=session,
            status=status,
        )
        # Startup is just another way to learn the session is down: the
        # matching WORKING event may never arrive (nothing "changed"
        # from WAHA's view), so the active poller starts here too.
        try:
            arm_recovery_poller(waha, session)
        except RuntimeError:
            # No running loop (a CLI command seeding outside serve):
            # the event path and the message probes still cover it.
            logger.debug("No event loop to arm the recovery poller at startup")
    capture_operator_target(waha, session)
    return status


def own_identity(waha: WahaClient, session: str) -> dict[str, str]:
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


def capture_operator_target(waha: WahaClient, session: str) -> None:
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


def register_session_status_handler(waha: WahaClient, session: str) -> None:
    """Watch ``session.status`` events: mute/unmute, notify on transitions.

    Going down also arms the active recovery poller
    (:func:`arm_recovery_poller`) — WAHA's engines do not reliably
    emit the matching ``WORKING`` event on their own reconnect, so the
    flag must not depend on the event arriving.
    """

    @on_session_status
    async def watch_session(event: Any) -> None:
        status = str(event.payload.get("status", ""))
        if not status:
            return
        if status == HEALTHY_STATUS:
            # The event path's recovery report: session_recovered owns
            # the flag flip (it early-returns when already healthy, so
            # a redundant WORKING event is a no-op).
            await session_recovered(waha, session)
            return
        was = session_healthy()
        set_session_health(status)
        if was:
            logger.warning("WAHA session status: {status}", status=status)
            await notify_operator(
                waha, session, session_down_message(status), kind="down"
            )
            arm_recovery_poller(waha, session)


def session_down_message(status: str) -> str:
    """The operator-facing text for a session leaving WORKING.

    Status-aware guidance instead of one-size-fits-all "re-link the
    phone": WAHA's own lifecycle says FAILED wants a restart (and a
    re-scan if that fails), while STARTING/SCAN_QR_CODE/PASSKEY_* are
    normal pairing states that want patience, not action. Every
    variant states the two facts the operator needs: the bot is muted
    (no replies go out) and recovery is being watched automatically.
    """
    if status == "FAILED":
        action = "restart the WAHA session (re-scan via logout + start if that fails)."
    elif status in (
        "STARTING",
        "SCAN_QR_CODE",
        "PASSKEY_REQUIRED",
        "PASSKEY_CONFIRMATION_REQUIRED",
    ):
        action = f"a pairing step ({status.lower()}) is in progress — give it a minute."
    elif status == "STOPPED":
        action = "start the WAHA session."
    else:
        action = f"check the WAHA session (status: {status})."
    return (
        f"WhatsApp session is {status} — bot muted, incoming messages are held"
        f" for retry. Watching for recovery and will report back. If it stays"
        f" down: {action}"
    )


async def session_recovered(waha: WahaClient, session: str) -> None:
    """Run the once-per-recovery duties; safe from any trigger.

    The event path, the poller and the message-triggered probe can all
    race to be the first to see recovery; each goes through here so the
    flag flip, the operator-target re-capture and the retried 🔵 notify
    happen exactly once no matter who wins.
    """
    async with _probe_lock:
        if session_healthy():
            return
        set_session_health(HEALTHY_STATUS)
        logger.info("WAHA session recovered")
        # Re-capture the operator target: the startup fetch may have
        # failed against a dead WAHA, and a re-linked account carries
        # a different JID.
        capture_operator_target(waha, session)
        await notify_recovery(waha, session)
    disarm_recovery_poller()


def arm_recovery_poller(waha: WahaClient, session: str) -> None:
    """Poll WAHA for the session status until it reports ``WORKING``.

    The active half of recovery: while muted, ask the source of truth
    every :data:`RECOVERY_POLL_S` — a hung engine that stopped
    emitting events (but reconnected underneath) is caught on the
    next probe instead of muting the bot until a human intervenes.
    Idempotent: arming while a poller already runs is a no-op.
    """
    global _recovery_task
    if _recovery_task is not None and not _recovery_task.done():
        return
    logger.info(
        "Arming session recovery poller (every {interval}s until WORKING)",
        interval=RECOVERY_POLL_S,
    )
    _recovery_task = asyncio.ensure_future(_poll_until_recovered(waha, session))


def disarm_recovery_poller() -> None:
    """Stop the background poller (recovery confirmed or shutdown)."""
    global _recovery_task
    if _recovery_task is not None:
        _recovery_task.cancel()
        _recovery_task = None


def recovery_poller_armed() -> bool:
    """Whether the recovery poller is currently running (test/view)."""
    return _recovery_task is not None and not _recovery_task.done()


#: Retry backoff between recovery-notify attempts; a module attribute so
#: tests (and operators reading the code) can adjust it without the
#: leading underscore.
NOTIFY_BACKOFF_S = _NOTIFY_BACKOFF_S


async def _poll_until_recovered(waha: WahaClient, session: str) -> None:
    """The poller loop: probe, sleep, repeat; exits on recovery."""
    try:
        while not session_healthy():
            await probe_session_recovery(waha, session)
            if session_healthy():
                break
            await asyncio.sleep(RECOVERY_POLL_S)
    except asyncio.CancelledError:
        raise
    except Exception:
        # Never die: a poller crash would silently disable active
        # recovery for the process lifetime.
        logger.exception("Session recovery poller crashed")


async def probe_session_recovery(waha: WahaClient, session: str) -> bool:
    """Ask WAHA directly whether the session is back; recover if so.

    The cheap probe behind both triggers: the poller's interval and a
    muted incoming message (webhooks being delivered at all is itself
    evidence the session is back — but only a ``WORKING`` answer from
    ``GET /api/sessions/{session}`` flips the flag). A failed probe is
    logged at debug and swallowed: WAHA restarting is the norm while
    unhealthy.
    """
    if session_healthy():
        return True
    try:
        info = await asyncio.to_thread(waha.get_session, session)
        status = str(info.get("status") or "")
    except Exception as exc:
        logger.debug("Session recovery probe failed: {exc}", exc=exc)
        return False
    if status != HEALTHY_STATUS:
        logger.debug("Session recovery probe: status is {status}", status=status)
        return False
    await session_recovered(waha, session)
    return True


async def notify_recovery(waha: WahaClient, session: str) -> None:
    """The 🔵 recovery notify, retried — a racing send must not lose it.

    The session just went ``WORKING``; WhatsApp's own reconnect can
    still fail the first send attempt. A failed notify backs off
    ``NOTIFY_BACKOFF_S`` and retries up to ``_NOTIFY_ATTEMPTS`` times
    instead of vanishing — the operator's "sometimes the back online
    message is not received" was exactly this single-shot send.
    """
    for attempt in range(1, _NOTIFY_ATTEMPTS + 1):
        if await _try_send_recovery_notify(waha, session):
            return
        if attempt < _NOTIFY_ATTEMPTS:
            logger.info(
                "Recovery notify attempt {attempt} failed; retrying in {backoff}s",
                attempt=attempt,
                backoff=NOTIFY_BACKOFF_S,
            )
            await asyncio.sleep(NOTIFY_BACKOFF_S)
    message = (
        f"Recovery notify failed after {_NOTIFY_ATTEMPTS} attempts; session IS"
        f" recovered, replies are flowing"
    )
    logger.warning("{message}", message=message)


async def _try_send_recovery_notify(waha: WahaClient, session: str) -> bool:
    """One recovery-notify send attempt; True when it landed."""
    message = (
        "WhatsApp session recovered — the bot is answering again; held"
        " messages were retried on WAHA redelivery."
    )
    try:
        await send_operator_notify(waha, session, message, kind="up")
        return True
    except Exception as exc:
        logger.warning("Recovery notify send failed: {exc}", exc=exc)
        return False


async def notify_operator(
    waha: WahaClient, session: str, message: str, kind: str = "down"
) -> None:
    """Best-effort WhatsApp message to the bot's own account.

    The operator reads it in the "Message yourself" chat. Fails soft:
    the session may be dead — the very thing being reported — so a
    failed send is logged and swallowed; the loud log line is the floor.
    """
    try:
        await send_operator_notify(waha, session, message, kind)
    except Exception as exc:
        logger.warning("Operator notification failed: {exc}", exc=exc)


async def send_operator_notify(
    waha: WahaClient, session: str, message: str, kind: str
) -> str:
    """Send one operator notification; raises on failure (the retry loop).

    Split from :func:`notify_operator` so the recovery path can
    *retry* a failed 🔵 — the single-shot swallow is right for the 🟠
    (the session is down, retrying is futile) but loses the recovery
    message when the send races WhatsApp's own reconnect.

    The prefix stays ``wahabot:`` alone — the old
    ``WAHA session '{session}' {message}`` prefix sat in front of
    LLM notices too, telling the operator their WhatsApp session had
    a problem when the LLM provider did. Each caller writes its own
    accurate subject line instead.
    """
    me = state.operator_jid
    if not me:
        return ""
    icon = "🔵" if kind == "up" else "🟠"
    text = f"{icon} wahabot: {message}"
    sent_id = await asyncio.to_thread(waha.send_text, session, me, text)
    # The notification lands in the self-chat; mark it so its echo
    # event is never parsed as an operator command.
    remember_self_echo(sent_id)
    return sent_id


def llm_endpoint_down(exc: Exception) -> bool:
    """Whether *exc* means the LLM endpoint (or its proxy) is unreachable.

    ``APIConnectionError`` covers DNS, refused and dropped connections.
    A reverse proxy in front of the provider answers instead when it is
    the one that is down — a 502/503/504 ``APIStatusError`` — which must
    classify the same: the runs it fails are just as lost, and the
    operator notification exists for exactly this class of outage.

    Timeouts do NOT belong here: ``APITimeoutError`` subclasses
    ``APIConnectionError`` but means the endpoint answered nothing
    within the per-request budget — the provider may be healthy and
    still generating (non-streaming calls send zero bytes until
    completion). A timeout is a slow generation or a hung endpoint,
    never a proved outage, so it gets its own classifier
    (:func:`llm_call_timed_out`) and its own operator message; treating
    it as "unreachable" would pause-replies-narrate an outage that
    never happened.
    """
    if isinstance(exc, openai.APITimeoutError):
        return False
    if isinstance(exc, openai.APIConnectionError):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500


def llm_call_timed_out(exc: Exception) -> bool:
    """Whether *exc* is the per-request LLM timeout firing.

    Distinct from an outage (:func:`llm_endpoint_down`): a timeout
    says the request lived out its budget without an answer — the
    model may still be generating on a healthy endpoint. The failure
    is just as fatal to the run, so the chat paths give it the same
    seen-marker drop (WAHA's redelivery retries), but the operator
    message must say what actually happened: a slow/hung generation,
    not an unreachable endpoint.
    """
    return isinstance(exc, openai.APITimeoutError)


async def mark_llm_unreachable(waha: WahaClient, session: str, exc: Exception) -> None:
    """Flip the LLM health flag down and notify the operator once.

    Called from the failure handlers of the chat and command paths on
    an outage-class exception: the first failure of an outage
    transitions the flag and sends the 🟠 notification; later failures
    (every message that arrives while the provider is down) only log —
    the operator already knows, and the chat path's dropped seen
    markers let WAHA's redelivery retry once the provider is back.
    """
    logger.warning(
        "LLM endpoint unreachable ({exc}); operator notified, messages await redelivery",
        exc=exc,
    )
    if state.llm_healthy:
        state.llm_healthy = False
        message = (
            "LLM provider is unreachable — replies are paused until it"
            " responds again. Incoming messages are held and retried"
            " automatically."
        )
        await notify_operator(waha, session, message, kind="down")


async def classify_llm_failure(
    waha: WahaClient, session: str, exc: Exception, timeout_s: float
) -> str:
    """Notify per an LLM failure's class; returns the class for the caller.

    One shared classifier for every run path (single message, burst,
    album, operator command): a timeout notifies as a timeout (the
    endpoint may be healthy and still generating), an outage-class
    failure flips the health flag and notifies "unreachable", and
    anything else is a bug — no notify at all, the caller's full
    traceback is the report. Callers still own the seen-marker drop;
    this only decides who gets told what.
    """
    if llm_call_timed_out(exc):
        await mark_llm_timed_out(waha, session, exc, timeout_s)
        return "timeout"
    if llm_endpoint_down(exc):
        await mark_llm_unreachable(waha, session, exc)
        return "outage"
    return "bug"


async def mark_llm_timed_out(
    waha: WahaClient, session: str, exc: Exception, timeout_s: float
) -> None:
    """Notify the operator that an LLM call timed out (once per slow stretch).

    A timeout is not an outage (:func:`llm_endpoint_down`): the outage
    health flag stays untouched — the very next run may succeed against
    the same healthy-but-slow endpoint, and "replies paused" would be
    the wrong message. The operator gets one 🟠 naming the actual
    budget and the knob (``WAHABOT_LLM_TIMEOUT``) on the first timeout
    of a stretch; repeats stay quiet until a successful run resets the
    latch (alongside :func:`mark_llm_recovered`). Callers drop seen
    markers exactly like an outage — a timed-out run is just as lost.
    """
    logger.warning(
        "LLM call timed out after {timeout}s ({exc}); messages await redelivery",
        timeout=timeout_s,
        exc=exc,
    )
    if not state.llm_timeout_notified:
        state.llm_timeout_notified = True
        message = (
            f"LLM call timed out after {timeout_s:.0f}s — the model was still"
            f" generating (or the provider hung). If your provider regularly"
            f" needs longer, raise WAHABOT_LLM_TIMEOUT; replies continue on"
            f" the next message."
        )
        await notify_operator(waha, session, message, kind="down")


async def mark_llm_recovered(waha: WahaClient, session: str) -> None:
    """Flip the LLM health flag back up and notify the operator once.

    Called after any run completes against the provider: the first
    success after an outage transitions the flag and sends the 🔵
    notification; the steady state stays silent. A success also
    re-arms the timeout notify latch (:func:`mark_llm_timed_out`) —
    the endpoint answering again means the next slow call is a fresh
    slow-stretch worth one new 🟠.
    """
    if not state.llm_healthy:
        state.llm_healthy = True
        await notify_operator(
            waha,
            session,
            "LLM provider is back — replies resumed.",
            kind="up",
        )
    state.llm_timeout_notified = False
