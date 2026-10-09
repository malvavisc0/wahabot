"""Who is who on a shared account: message authors and display names.

The bot and its human operator write from the *same* WhatsApp account,
so every message they send carries the same JID. WAHA still tells them
apart at the moment a message is sent — the webhook's ``source`` field
is ``api`` for the bot (a send through WAHA's API) and ``app`` for the
human (phone or WhatsApp Web) — but that fact exists only in the live
event. Quotes, reactions and ``read_chat`` listings later point at the
message by id alone, and WAHA's history endpoints do not carry
``source``.

This module records the fact while it is available:

- :class:`AuthorBook` maps a message's short id to ``"bot"`` or
  ``"operator"``, so a quote of an account message can say *which*
  teammate wrote it instead of guessing.
- :class:`NameBook` maps a participant JID to the display name WhatsApp
  delivered with their last message (``notifyName``). WAHA's message
  history API returns no names at all in LID groups (verified: 0 of
  300 fetched messages carried ``notifyName``), so names learned from
  webhooks are the only reliable source for quoted senders, reaction
  notes, mention notes and ``read_chat`` results.

Both are small, bounded JSON files under ``<data_dir>/identity/``,
written atomically and fail-soft: a missing or corrupt file starts
empty, and a write failure is logged, never raised. Both are flushed
at interpreter exit so the rate-limited window's last write survives
a shutdown.
"""

import atexit
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

__all__ = [
    "AUTHOR_BOT",
    "AUTHOR_OPERATOR",
    "AuthorBook",
    "JsonBook",
    "NameBook",
    "authors",
    "configure",
    "display_name",
    "flush_on_exit",
    "message_key",
    "names",
]

AUTHOR_BOT = "bot"
AUTHOR_OPERATOR = "operator"

#: Bounds: enough for weeks of busy groups, small enough to rewrite whole.
_MAX_AUTHORS = 20_000
_MAX_NAMES = 5_000

#: Minimum seconds between disk writes; the in-memory view is always current.
_FLUSH_INTERVAL_S = 5.0


def message_key(message_id: str) -> str:
    """The stable short id inside any WAHA message id shape.

    Webhook payloads carry the serialized form
    ``true_<chat>_<SHORT>[_<participant>]``, while ``replyTo.id`` and
    reaction targets may carry only ``<SHORT>``. The short id is the one
    part every shape shares, so it is the key.
    """
    parts = message_id.split("_")
    if len(parts) >= 3 and parts[0] in ("true", "false"):
        return parts[2]
    return message_id


class JsonBook:
    """A bounded ``str → str`` mapping persisted as one JSON file.

    Writes are rate-limited (:data:`_FLUSH_INTERVAL_S`); the in-memory
    view is always current, so a crash loses at most a few seconds of
    learned identities, never correctness.
    """

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.path: Path | None = None
        self.data: dict[str, str] = {}
        self.lock = threading.Lock()
        self.dirty = False
        self.last_flush = 0.0

    def load(self, path: Path) -> None:
        """Bind to *path* and load it; a bad or missing file starts empty."""
        with self.lock:
            self.path = path
            self.data = {}
            # A freshly bound book has no recent write: its first put
            # must flush at once, not inherit some earlier book's timer.
            self.last_flush = 0.0
            try:
                raw: Any = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return
            except (OSError, ValueError) as exc:
                logger.warning("Ignoring unreadable {path}: {exc}", path=path, exc=exc)
                return
            if isinstance(raw, dict):
                self.data = {str(k): str(v) for k, v in raw.items() if isinstance(v, str)}

    def get(self, key: str) -> str:
        """The stored value for *key*, "" when absent."""
        with self.lock:
            return self.data.get(key, "")

    def put(self, key: str, value: str) -> None:
        """Store *value* under *key*, evicting the oldest entry at the cap."""
        if not key or not value:
            return
        with self.lock:
            if self.data.get(key) == value:
                return
            self.data.pop(key, None)
            while len(self.data) >= self.cap:
                del self.data[next(iter(self.data))]
            self.data[key] = value
            self.dirty = True
        self.flush()

    def flush(self, force: bool = False) -> None:
        """Write to disk when dirty, at most every few seconds unless forced."""
        with self.lock:
            if self.path is None or not self.dirty:
                return
            now = time.monotonic()
            if not force and now - self.last_flush < _FLUSH_INTERVAL_S:
                return
            data = dict(self.data)
            self.dirty = False
            self.last_flush = now
            path = self.path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("Could not save {path}: {exc}", path=path, exc=exc)
            with self.lock:
                # Keep the book dirty: the next flush (or the exit
                # hook) retries instead of silently dropping it.
                self.dirty = True

    def clear(self) -> None:
        """Forget every entry (tests, `wahabot forget`-style resets)."""
        with self.lock:
            self.data.clear()
            self.dirty = False
            self.last_flush = 0.0


class AuthorBook(JsonBook):
    """Short message id → ``bot`` or ``operator`` for the shared account."""

    def record(self, message_id: str, author: str) -> None:
        """Remember who wrote *message_id* (any WAHA id shape)."""
        self.put(message_key(message_id), author)

    def author(self, message_id: str) -> str:
        """``bot``, ``operator``, or "" when the message predates the record."""
        return self.get(message_key(message_id)) if message_id else ""


class NameBook(JsonBook):
    """Participant JID → last display name WhatsApp delivered for them."""

    def learn(self, jid: str, name: str) -> None:
        """Remember *name* as *jid*'s current display name."""
        name = name.strip()
        if jid and name:
            self.put(jid, name)

    def name(self, jid: str) -> str:
        """*jid*'s learned display name, "" when unknown."""
        return self.get(jid) if jid else ""


authors = AuthorBook(_MAX_AUTHORS)
names = NameBook(_MAX_NAMES)


def display_name(jid: str, roster_names: dict[str, str] | None = None) -> str:
    """A participant's display name: the chat's roster, else the name book.

    LID groups' rosters carry no names and WAHA's history API returns
    none either, so names learned from live webhooks are the only ones
    many members ever get. Fetched (roster) names win where both exist.
    """
    return (roster_names or {}).get(jid, "") or names.name(jid)


@atexit.register
def flush_on_exit() -> None:
    """Write both books at interpreter exit.

    ``put`` rate-limits disk writes; without this, the updates in the
    last window would only ever live in memory.
    """
    authors.flush(force=True)
    names.flush(force=True)


def configure(data_dir: Path, session: str) -> None:
    """Load both books for *session* from ``<data_dir>/identity/<session>/``."""
    base = data_dir / "identity" / session
    authors.load(base / "authors.json")
    names.load(base / "names.json")
