"""Multi-message burst assembly: one sender's rapid messages, one turn.

Real requests arrive as bursts — "there is a leak", "flat 3B", a photo,
a voice note — and answering the first line prematurely (or once per
line) recreates the human's confusion. This module holds messages that
already passed every gate in a per-(chat, sender) buffer and flushes
them as ONE agent turn when the sender goes quiet.

Two deadlines, per docs/plans/commercial-roadmap.md "Multi-Message
Turns":

- **Inactivity window** (``burst_inactivity_s``): reset by every
  same-sender message, so gap-y bursts still hold together.
- **Hold cap** (``burst_hold_cap_s``): measured from the FIRST
  buffered message, so a sender cannot defer the response
  indefinitely.

A different participant never extends another sender's turn: the
buffer key is (session, chat, sender). The timer is a rescheduled
task, not a sleep — every arrival cancels and re-spawns the flush.

The completion handler (registered by ``handlers`` at boot) receives
the buffer and runs the agent once over all its messages. Like the
album buffer this is fire-and-forget: the handler owns failure
recovery (dropping every buffered message's seen marker so WAHA's
redelivery reprocesses them).
"""

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from wahabot.ai.messages import jid_string, message_kind
from wahabot.core.models import WahaEvent

#: Flush timing knobs are read from Settings by the caller
#: (``handlers``) and passed to :func:`configure` — the buffer module
#: itself stays import-light for unit tests.
_inactivity_s: float | None = None
_hold_cap_s: float | None = None

#: Fires when a buffer completes (inactivity elapsed or cap reached).
#: Set by handlers at registration; None in unit tests.
_on_complete: Callable[[BurstBuffer], Coroutine[Any, Any, None]] | None = None


@dataclass
class BurstBuffer:
    """One in-flight burst: its sender and the messages held so far."""

    key: tuple[str, str, str]
    sender: str
    messages: list[WahaEvent] = field(default_factory=list)
    #: Arrival wall-clock of the first buffered message (monotonic
    #: clock; the cap measures hold duration, not message timestamps).
    first_seen: float = field(default_factory=time.monotonic)

    def ids(self) -> list[str]:
        """The buffered messages' ids, in arrival order."""
        return [str(event.payload.get("id", "")) for event in self.messages]


#: Open burst buffers, keyed by (session, chat_id, sender). The sender
#: is the normalized participant JID in groups (LID groups report
#: participants as JID objects — naive stringifying would split one
#: sender into two buffers) and the chat id in 1:1 chats.
_bursts: dict[tuple[str, str, str], BurstBuffer] = {}

#: Flush timer task per buffer; cancelled and re-spawned on every
#: arrival (the resettable inactivity window).
_timers: dict[tuple[str, str, str], asyncio.Task[None]] = {}

#: Strong references to fire-and-forget completion runs so the GC
#: cannot collect them mid-flight; each task removes itself when done.
_tasks: set[asyncio.Task[None]] = set()


def configure(inactivity_s: float, hold_cap_s: float) -> None:
    """Set the flush timing knobs (called by handlers at boot)."""
    global _inactivity_s, _hold_cap_s
    _inactivity_s = inactivity_s
    _hold_cap_s = hold_cap_s


def set_completion_handler(
    handler: Callable[[BurstBuffer], Coroutine[Any, Any, None]] | None,
) -> None:
    """Register the coroutine called with each completed burst buffer."""
    global _on_complete
    _on_complete = handler


def burst_key(event: WahaEvent) -> tuple[str, str, str]:
    """The buffer key of an event: session, chat and sender.

    The sender is the normalized participant JID in groups; in 1:1
    chats the chat id doubles as the sender (the partner is the only
    possible sender).
    """
    chat_id = str(event.payload.get("from", ""))
    if chat_id.endswith("@g.us"):
        data = event.payload.get("_data", {})
        sender = jid_string(event.payload.get("participant") or data.get("author"))
    else:
        sender = chat_id
    return event.session, chat_id, sender


#: Message kinds that join a burst buffer. Standalone images, videos
#: and PTVs join the assembled turn — splitting "the ceiling is
#: leaking" from its attached evidence defeats the purpose. Albums are
#: excluded: the album buffer reassembles them on its own (arrival-
#: ordered, engine-dependent linkage) and a burst flush must never
#: inherit an album's attribution ambiguity. Unknown kinds fall
#: through and run the single-message path.
BUFFERABLE_KINDS = frozenset({"text", "audio", "image", "sticker", "video", "ptv"})


def is_bufferable(event: WahaEvent) -> bool:
    """Whether *event* joins the burst buffer instead of running now."""
    return message_kind(event) in BUFFERABLE_KINDS


def add_message(event: WahaEvent) -> bool:
    """Buffer *event* into its sender's burst; True when consumed.

    Opens the buffer on first sight and (re)arms the flush timer on
    every arrival. The deadline is ``min(inactivity, remaining cap)``
    so a long burst hits the hold cap even while the sender keeps
    typing; the cap fires the flush mid-burst by design — the roadmap
    prefers an early reply over a silent wait.
    """
    if _inactivity_s is None or _hold_cap_s is None:
        return False
    key = burst_key(event)
    buffer = _bursts.get(key)
    if buffer is None:
        buffer = BurstBuffer(key=key, sender=key[2])
        _bursts[key] = buffer
    buffer.messages.append(event)
    remaining_cap = _hold_cap_s - (time.monotonic() - buffer.first_seen)
    delay = max(min(_inactivity_s, remaining_cap), 0.0)
    timer = _timers.pop(key, None)
    if timer is not None and not timer.done():
        timer.cancel()
    _timers[key] = _spawn(_flush_later(key, delay))
    logger.debug(
        "Burst {sender} in {chat_id} holds {n} message(s), flush in {delay:.1f}s",
        sender=key[2],
        chat_id=key[1],
        n=len(buffer.messages),
        delay=delay,
    )
    return True


def pending(chat_id: str) -> bool:
    """True when *chat_id* has any open burst buffer (test/pause aid)."""
    return any(key[1] == chat_id for key in _bursts)


async def _flush_later(key: tuple[str, str, str], delay: float) -> None:
    """Complete the buffer after *delay*, whatever it holds.

    Re-spawned on every arrival, so this runs only when the sender
    went quiet (or the hold cap arrived — whichever deadline the
    ``add_message`` reschedule last set).
    """
    await asyncio.sleep(delay)
    registered = _timers.get(key)
    if registered is not None and asyncio.current_task() is not registered:
        # A newer timer owns this buffer now (arrival raced the wake);
        # let the newest deadline win.
        return
    _timers.pop(key, None)
    buffer = _bursts.pop(key, None)
    if buffer is not None:
        _complete(buffer)


def _complete(buffer: BurstBuffer) -> None:
    """Hand a finished buffer to the completion handler, if any."""
    if _on_complete is None or not buffer.messages:
        return
    logger.info(
        "Burst complete in {chat_id}: {n} message(s) from {sender}",
        chat_id=buffer.key[1],
        n=len(buffer.messages),
        sender=buffer.sender,
    )
    _spawn(_on_complete(buffer))


def _spawn(coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
    """Run *coro* in the background, keeping a reference until done."""
    task = asyncio.ensure_future(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task


def reset() -> None:
    """Drop buffers, timers and the configured knobs (test isolation)."""
    global _inactivity_s, _hold_cap_s
    for timer in _timers.values():
        timer.cancel()
    _timers.clear()
    _bursts.clear()
    _tasks.clear()
    _inactivity_s = None
    _hold_cap_s = None
