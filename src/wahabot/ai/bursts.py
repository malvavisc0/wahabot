"""Multi-message burst assembly: one sender's rapid messages, one turn."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from loguru import logger

from wahabot.ai.messages import jid_string, message_kind
from wahabot.core.models import WahaEvent


@dataclass
class BurstBuffer:
    """One in-flight burst: its sender and the intake ids held so far."""

    key: tuple[str, str, str]
    sender: str
    message_ids: list[str] = field(default_factory=list)
    first_seen: float = field(default_factory=time.monotonic)

    def ids(self) -> list[str]:
        """The buffered messages' ids, in arrival order."""
        return list(self.message_ids)

    def hold_seconds(self) -> float:
        """Time from the first buffered message to now (monotonic)."""
        return time.monotonic() - self.first_seen


class Bursts:
    """Per-registration burst buffers, timers and completion tasks."""

    def __init__(
        self,
        inactivity_s: float,
        hold_cap_s: float,
        on_complete: Callable[[BurstBuffer], Awaitable[None]],
    ) -> None:
        self.inactivity_s = inactivity_s
        self.hold_cap_s = hold_cap_s
        self.on_complete = on_complete
        self._buffers: dict[tuple[str, str, str], BurstBuffer] = {}
        self._timers: dict[tuple[str, str, str], asyncio.Task[None]] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def ids(self, key: tuple[str, str, str]) -> list[str]:
        """Every buffered id for one sender's current window."""
        buffer = self._buffers.get(key)
        return buffer.ids() if buffer is not None else []

    def add(self, message_id: str, event: WahaEvent) -> bool:
        """Buffer a message's intake id into its sender's burst."""
        key = burst_key(event)
        if not key[2]:
            return False
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = BurstBuffer(key=key, sender=key[2])
            self._buffers[key] = buffer
        buffer.message_ids.append(message_id)
        remaining_cap = self.hold_cap_s - (time.monotonic() - buffer.first_seen)
        delay = max(min(self.inactivity_s, remaining_cap), 0.0)
        timer = self._timers.pop(key, None)
        if timer is not None and not timer.done():
            timer.cancel()
        timer_task = self.spawn_buffer(self.flush_later(key, delay))
        if timer_task is not None:
            self._timers[key] = timer_task
        logger.debug(
            "Burst {sender} in {chat_id} holds {n} message(s), flush in {delay:.1f}s",
            sender=key[2],
            chat_id=key[1],
            n=len(buffer.message_ids),
            delay=delay,
        )
        return True

    def pending(self, chat_id: str) -> bool:
        """True when *chat_id* has any open burst buffer."""
        return any(key[1] == chat_id for key in self._buffers)

    async def flush_later(self, key: tuple[str, str, str], delay: float) -> None:
        await asyncio.sleep(delay)
        registered = self._timers.get(key)
        if registered is not None and asyncio.current_task() is not registered:
            return
        self._timers.pop(key, None)
        buffer = self._buffers.pop(key, None)
        if buffer is not None and buffer.message_ids:
            self.complete(buffer)

    def complete(self, buffer: BurstBuffer) -> None:
        logger.info(
            "Burst complete in {chat_id}: {n} message(s) from {sender}, held {hold:.1f}s",
            chat_id=buffer.key[1],
            n=len(buffer.message_ids),
            sender=buffer.sender,
            hold=buffer.hold_seconds(),
        )
        self.spawn_buffer(self.on_complete(buffer))

    def spawn_buffer(self, coro: Awaitable[None]) -> asyncio.Task[None] | None:
        try:
            task = asyncio.ensure_future(coro)
        except RuntimeError:
            close = getattr(coro, "close", None)
            if close is not None:
                close()
            return None
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def reset(self) -> None:
        """Drop buffers, timers and cancellation state (test isolation)."""
        for timer in self._timers.values():
            timer.cancel()
        for task in self._tasks:
            task.cancel()
        self._timers.clear()
        self._buffers.clear()
        self._tasks.clear()


def burst_key(event: WahaEvent) -> tuple[str, str, str]:
    """The buffer key of an event: session, chat and sender."""
    chat_id = str(event.payload.get("from", ""))
    if chat_id.endswith("@g.us"):
        data = event.payload.get("_data", {})
        sender = jid_string(event.payload.get("participant") or data.get("author"))
    else:
        sender = chat_id
    return event.session, chat_id, sender


BUFFERABLE_KINDS = frozenset({"text", "audio", "image", "sticker", "video", "ptv"})


def is_bufferable(event: WahaEvent) -> bool:
    """Whether *event* joins the burst buffer instead of running now."""
    return message_kind(event) in BUFFERABLE_KINDS
