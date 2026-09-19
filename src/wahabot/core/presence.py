"""Human-like chat presence: seen receipts and the typing indicator.

Two mechanical de-robotizers, both fail-soft by contract — presence is
cosmetic and must never break, delay, or duplicate a reply:

- :func:`mark_seen` — call ``sendSeen`` for a chat on each incoming
  message event, so the bot reads like a member even when its run ends
  silent.
- :func:`typing_pause` — show "typing…", wait a length-scaled random
  moment, then let the reply go out. A sub-second answer to a
  paragraph is the loudest machine tell there is (the group said it
  out loud: "pero responde muy rápido"); the pause makes the same
  answer read as composed.
- :func:`clear_typing` — the best-effort escape hatch for a send that
  fails after the indicator went on (WhatsApp clears it when the
  message lands; a message that never lands must clear it itself).

All sync by design: they run inside the worker threads that already
wrap every WAHA call (``asyncio.to_thread`` in the handler, the tool
fns' own thread), so the sleeps never touch the event loop. HTTP
errors are logged and swallowed, never raised.
"""

import random
import time

from loguru import logger

from wahabot.core.waha import WahaClient

__all__ = ["clear_typing", "mark_seen", "typing_pause"]


def mark_seen(waha: WahaClient, session: str, chat_id: str) -> None:
    """Mark a chat's pending messages as seen; swallow every failure.

    Runs inline before the agent run — the read receipt is the
    response to *their* message, not to the bot's reply, so it goes
    out even when the run later stays silent.
    """
    try:
        waha.send_seen(session, chat_id)
    except Exception as exc:
        logger.debug("sendSeen failed for {chat}: {exc}", chat=chat_id, exc=exc)


def typing_pause(
    waha: WahaClient,
    session: str,
    chat_id: str,
    reply: str,
    min_s: float,
    max_s: float,
) -> bool:
    """Show "typing…", wait a reply-scaled moment; the reply follows.

    The delay grows with the reply's length (a longer answer takes
    longer to "type"), jittered uniformly up to ``max_s`` so
    back-to-back replies don't share a signature timing. ``min_s <= 0``
    disables the routine — no typing call, no wait. The indicator is
    left ON: WhatsApp clears it the moment the message lands, and the
    caller that sends the reply owns that transition.

    Returns True when the indicator is on, so the caller can clear it
    via :func:`clear_typing` if its send fails; False in every other
    case (disabled, or ``startTyping`` itself failed).
    """
    if min_s <= 0:
        return False
    try:
        waha.set_typing(session, chat_id, True)
    except Exception as exc:
        logger.debug("startTyping failed for {chat}: {exc}", chat=chat_id, exc=exc)
        return False
    cap = max(max_s, min_s)
    delay = random.uniform(min_s, min(cap, min_s + len(reply) / 40.0))
    time.sleep(delay)
    return True


def clear_typing(waha: WahaClient, session: str, chat_id: str) -> None:
    """Clear the "typing…" indicator; swallow every failure.

    The send that follows a successful :func:`typing_pause` normally
    clears the indicator by landing; a send that raises must call this
    on the way out, and its own failure must never mask the original
    error (also why this never raises).
    """
    try:
        waha.set_typing(session, chat_id, False)
    except Exception as exc:
        logger.debug("stopTyping failed for {chat}: {exc}", chat=chat_id, exc=exc)
