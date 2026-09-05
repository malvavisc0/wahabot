"""Session health: gate agent runs on the WAHA session being WORKING.

``wahabot`` is only as alive as its WAHA session: when the session
leaves ``WORKING`` every send would fail, yet the agent would still
burn LLM tokens, tool calls and transcriptions producing replies that
can never be delivered. This module tracks session health (seeded from
``GET /api/sessions/{session}`` at startup, updated by every
``session.status`` event), mutes message/command handling while the
session is down, and notifies the operator once per transition.
"""

import asyncio
from typing import Any

from loguru import logger

from wahabot.core.waha import WahaClient
from wahabot.webhook import on_session_status

#: statuses reported by ``WASessionStatusBody`` that keep the bot muted;
#: ``WORKING`` (and nothing else) is healthy.
HEALTHY_STATUS = "WORKING"

_session_healthy: bool = True
_notification_target: dict[str, str] = {}


def session_healthy() -> bool:
    """True while the WAHA session is WORKING."""
    return _session_healthy


def set_session_health(status: str) -> None:
    """Update the health flag from a session status string."""
    global _session_healthy
    _session_healthy = status == HEALTHY_STATUS


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
    if not _session_healthy:
        logger.warning(
            "WAHA session {session} starts in {status} — bot muted",
            session=session,
            status=status,
        )
    _capture_operator_target(waha, session)
    return status


def _capture_operator_target(waha: WahaClient, session: str) -> None:
    """Remember the bot's own JID as the operator-notification target."""
    try:
        me = waha.get_me(session)
    except Exception as exc:
        logger.warning("Could not fetch own JID for operator alerts: {exc}", exc=exc)
        return
    own = str(me.get("id") or "")
    if own:
        _notification_target["me"] = own


def register_session_status_handler(waha: WahaClient, session: str) -> None:
    """Watch ``session.status`` events: mute/unmute, notify on transitions."""

    @on_session_status
    async def watch_session(event: Any) -> None:
        status = str(event.payload.get("status", ""))
        if not status:
            return
        was = session_healthy()
        set_session_health(status)
        if status == HEALTHY_STATUS:
            if not was:
                logger.info("WAHA session recovered")
                await notify_operator(waha, session, "recovered — back online")
            return
        if was:
            logger.warning("WAHA session status: {status}", status=status)
            await notify_operator(
                waha, session, f"session is {status} — bot muted, re-link the phone."
            )


async def notify_operator(waha: WahaClient, session: str, message: str) -> None:
    """Best-effort WhatsApp message to the bot's own account.

    The operator reads it in the "Message yourself" chat. Fails soft:
    the session may be dead — the very thing being reported — so a
    failed send is logged and swallowed; the loud log line is the floor.
    """
    me = _notification_target.get("me")
    if not me:
        return
    text = f"⚠️ wahabot: WAHA session '{session}' {message}"
    try:
        await asyncio.to_thread(waha.send_text, session, me, text)
    except Exception as exc:
        logger.warning("Operator notification failed: {exc}", exc=exc)
