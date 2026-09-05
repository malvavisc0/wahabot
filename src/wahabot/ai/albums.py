"""Reassemble WhatsApp media albums into a single agent turn.

WAHA NOWEB delivers an album as an ``album`` container event (carrying
``_data.expectedImageCount``) followed by its images as standalone
``image`` events with **no** linkage field (``parentMsgId`` is null on
this engine). Reassembly therefore keys on order, not ids: a container
opens a buffer for its chat, and the next ``expected`` image events in
that chat fill it. A timeout flushes whatever arrived so a dropped
image never hangs the album (or leaks the buffer) forever.
"""

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from wahabot.ai.messages import message_kind
from wahabot.core.models import WahaEvent

#: How long an open album buffer waits for its images before flushing.
_ALBUM_FLUSH_TIMEOUT_S = 3.0


@dataclass
class AlbumBuffer:
    """One in-flight album: the container event plus its images so far."""

    container: WahaEvent
    expected: int
    images: list[WahaEvent] = field(default_factory=list)
    opened_at: float = field(default_factory=time.monotonic)


#: Open album buffers, keyed by (session, chat_id) — one album per chat
#: at a time; a new container supersedes a stale buffer (which then
#: flushes empty, its images already attributed to the newer album).
_albums: dict[tuple[str, str], AlbumBuffer] = {}

#: Strong references to fire-and-forget tasks (flush timers, completion
#: runs) so the GC cannot collect them mid-flight; each task removes
#: itself when done.
_tasks: set[asyncio.Task[None]] = set()


def _spawn(coro: Coroutine[Any, Any, None]) -> None:
    """Run *coro* in the background, keeping a reference until done.

    Outside a running loop (unit tests poking the state machine
    synchronously) the coroutine is closed and skipped — timers and
    completion runs only make sense on the webhook's loop.
    """
    try:
        task = asyncio.ensure_future(coro)
    except RuntimeError:
        coro.close()
        return
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


#: Fires when a buffer completes (expected count reached or timeout).
#: Set by handlers at registration; None in unit tests.
_on_complete: Callable[[AlbumBuffer], Coroutine[Any, Any, None]] | None = None


def set_completion_handler(
    handler: Callable[[AlbumBuffer], Coroutine[Any, Any, None]] | None,
) -> None:
    """Register the coroutine called with each completed album buffer."""
    global _on_complete
    _on_complete = handler


def album_chat_key(event: WahaEvent) -> tuple[str, str]:
    """The buffer key of an event: its session and chat."""
    return event.session, str(event.payload.get("from", ""))


def expected_images(event: WahaEvent) -> int:
    """The container's declared image count (0 when unknown)."""
    data = event.payload.get("_data", {})
    try:
        return int(data.get("expectedImageCount") or 0)
    except TypeError, ValueError:
        return 0


def start_album(event: WahaEvent) -> None:
    """Open a buffer for an album container and schedule its flush."""
    key = album_chat_key(event)
    _albums[key] = AlbumBuffer(container=event, expected=expected_images(event))
    logger.debug(
        "Album opened in {chat_id} (expecting {n} images)",
        chat_id=key[1],
        n=expected_images(event),
    )
    _spawn(_flush_later(key))


def add_album_image(event: WahaEvent) -> bool:
    """Buffer an image into its chat's open album; True when consumed.

    Images are attributed to the chat's open buffer in arrival order —
    the engine gives no parent link, and WhatsApp delivers an album's
    images back-to-back, so order is the only reliable key. When the
    buffer reaches its expected count it completes immediately.
    """
    buffer = _albums.get(album_chat_key(event))
    if buffer is None:
        return False
    buffer.images.append(event)
    if buffer.expected and len(buffer.images) >= buffer.expected:
        _albums.pop(album_chat_key(event), None)
        _complete(buffer)
    return True


def pending_album(event: WahaEvent) -> bool:
    """True when this event's chat has an open album buffer."""
    return album_chat_key(event) in _albums


async def _flush_later(key: tuple[str, str]) -> None:
    """Complete the buffer after the timeout, whatever it holds."""
    await asyncio.sleep(_ALBUM_FLUSH_TIMEOUT_S)
    buffer = _albums.pop(key, None)
    if buffer is not None:
        _complete(buffer)


def _complete(buffer: AlbumBuffer) -> None:
    """Hand a finished buffer to the completion handler, if any."""
    if _on_complete is None or not buffer.images:
        return
    logger.debug(
        "Album complete in {chat_id} ({got}/{want} images)",
        chat_id=buffer.container.payload.get("from"),
        got=len(buffer.images),
        want=buffer.expected,
    )
    _spawn(_on_complete(buffer))


def is_album_container(event: WahaEvent) -> bool:
    """True for the album container event itself."""
    return message_kind(event) == "album"
