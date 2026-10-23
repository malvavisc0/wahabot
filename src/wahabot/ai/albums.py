"""Reassemble WhatsApp media albums into a single agent turn."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from loguru import logger

from wahabot.ai.messages import message_kind
from wahabot.core.models import WahaEvent

ALBUM_FLUSH_TIMEOUT_S = 3.0


@dataclass
class AlbumBuffer:
    """One in-flight album: its container id plus image intake ids."""

    container_id: str
    session: str
    chat_id: str
    expected: int
    image_ids: list[str] = field(default_factory=list)


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


def is_album_container(event: WahaEvent) -> bool:
    """True for the album container event itself."""
    return message_kind(event) == "album"


class Albums:
    """Per-registration album buffers and their flush timers."""

    def __init__(
        self,
        on_complete: Callable[[AlbumBuffer], Awaitable[None]],
        *,
        flush_timeout_s: float = ALBUM_FLUSH_TIMEOUT_S,
    ) -> None:
        self.on_complete = on_complete
        self.flush_timeout_s = flush_timeout_s
        self._albums: dict[tuple[str, str], AlbumBuffer] = {}
        self._consumed: set[str] = set()
        self._tasks: set[asyncio.Task[None]] = set()

    def buffer(self, event: WahaEvent) -> str | None:
        """Buffer an album container or image; its recovery key when consumed."""
        key = album_chat_key(event)
        if is_album_container(event):
            container_id = str(event.payload.get("id", ""))
            self._albums[key] = AlbumBuffer(
                container_id=container_id,
                session=event.session,
                chat_id=key[1],
                expected=expected_images(event),
            )
            logger.debug(
                "Album opened in {chat_id} (expecting {n} images)",
                chat_id=key[1],
                n=expected_images(event),
            )
            self._consumed.add(container_id)
            self._spawn(self._flush_later(key))
            return container_id
        if message_kind(event) not in ("image", "sticker"):
            return None
        buffer = self._albums.get(key)
        if buffer is None:
            return None
        message_id = str(event.payload.get("id", ""))
        self._consumed.add(message_id)
        buffer.image_ids.append(message_id)
        if buffer.expected and len(buffer.image_ids) >= buffer.expected:
            self._albums.pop(key, None)
            self._complete(buffer)
        return buffer.container_id

    async def _flush_later(self, key: tuple[str, str]) -> None:
        await asyncio.sleep(self.flush_timeout_s)
        buffer = self._albums.pop(key, None)
        if buffer is not None:
            self._complete(buffer)

    def _complete(self, buffer: AlbumBuffer) -> None:
        if not buffer.image_ids:
            return
        logger.debug(
            "Album complete in {chat_id} ({got}/{want} images)",
            chat_id=buffer.chat_id,
            got=len(buffer.image_ids),
            want=buffer.expected,
        )
        self._spawn(self.on_complete(buffer))

    def _spawn(self, coro: Awaitable[None]) -> asyncio.Task[None] | None:
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

    def consumed(self, event: WahaEvent) -> bool:
        """Whether this event already joined an open or completed album."""
        return str(event.payload.get("id", "")) in self._consumed

    def reset(self) -> None:
        """Drop all open album buffers and their flush timers."""
        for task in self._tasks:
            task.cancel()
        self._albums.clear()
        self._consumed.clear()
        self._tasks.clear()
