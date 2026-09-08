"""Bounded TTL cache: one eviction idiom for the bot's marker tables.

Insertion order is oldest-first (``dict``), so the cap evicts the
oldest entry and expired entries drop on access. Used for the seen-id
dedup cache, the self-echo tracker, and the participant-roster cache —
three tables that share "remember a value for a bounded time, capped"
and nothing else.
"""

import time
from collections.abc import Iterator

__all__ = ["TtlCache"]


class TtlCache[K, V]:
    """A capped mapping whose entries expire after *ttl* seconds."""

    def __init__(self, ttl: float, cap: int) -> None:
        self.ttl = ttl
        self.cap = cap
        self._entries: dict[K, tuple[float, V]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[K]:
        return iter(self._entries)

    def __contains__(self, key: K) -> bool:
        return self.get(key) is not None

    def get(self, key: K) -> V | None:
        """The live value for *key*, dropping it when expired."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.monotonic() - ts > self.ttl:
            del self._entries[key]
            return None
        return value

    def put(self, key: K, value: V) -> None:
        """Mark *key* with *value*, evicting the oldest entry at the cap."""
        while len(self._entries) >= self.cap:
            del self._entries[next(iter(self._entries))]
        self._entries[key] = (time.monotonic(), value)

    def drop(self, key: K) -> None:
        """Forget *key* (a no-op when absent)."""
        self._entries.pop(key, None)

    def clear(self) -> None:
        """Drop every entry."""
        self._entries.clear()
