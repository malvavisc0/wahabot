import asyncio
from typing import Any

from loguru import logger

from wahabot.core.waha import WahaClient
from wahabot.status.notify import notify_operator, send_operator_notify
from wahabot.status.state import (
    HEALTHY_STATUS,
    capture_operator_target,
    session_healthy,
    set_session_health,
    state,
)
from wahabot.webhook import on_session_status

RECOVERY_POLL_S = 15.0

_NOTIFY_ATTEMPTS = 3

_NOTIFY_BACKOFF_S = 2.0

_recovery_task: asyncio.Task[None] | None = None

_probe_lock = asyncio.Lock()


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
            await notify_operator(waha, session, session_down_message(status))
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
    flag flip, the operator-target re-capture and the retried recovery notify
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
    """The recovery notify, retried — a racing send must not lose it.

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
        await send_operator_notify(waha, session, message)
        return True
    except Exception as exc:
        logger.warning("Recovery notify send failed: {exc}", exc=exc)
        return False
