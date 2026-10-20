"""Per-chat serial registry: the ``#m<n>`` <-> WAHA message id map.

Every inbound message gets a small, chat-global serial the prompt can carry
without the ~100-character WAHA id (which embeds the chat's own JID). The
serial is assigned at ingest, stored next to the chat's memory, and used by
the send tools to resolve ``reply_to`` / ``react`` tokens back to real ids.

Storage is append-only JSONL, one line per message, written with a single
``O_APPEND`` ``write()`` — atomic for a sub-4 KB line, so the hot path takes
no lock. Compaction keeps the newest entries for a chat and runs through a
temp file plus ``os.replace`` under an exclusive ``flock``.

An unknown sender keeps a stable ``<member-n>`` ordinal in the same entry, so
the prompt never falls back to a JID. A learned name wins from then on; the
ordinal stays with the chat in case the name is lost.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from wahabot.core.identity import message_key

#: A prompt-side serial token, e.g. ``#m12``.
TOKEN_RE = re.compile(r"^#m(\d+)$")

#: How many serials a chat keeps across a compaction.
DEFAULT_KEEP = 2000


@dataclass(frozen=True)
class SerialEntry:
    """One message's serial, real id and identity at ingest."""

    serial: int
    message_id: str
    jid: str
    name: str
    ordinal: int = 0


def format_serial_token(serial: int) -> str:
    """The prompt-side token for a serial (``12`` -> ``#m12``)."""
    return f"#m{serial}"


def registry_file(data_dir: str | Path, session: str, chat_id: str) -> Path:
    """The chat's registry, next to its persisted memory."""
    return Path(data_dir) / "memory" / session / f"{chat_id}.ids.jsonl"


class SerialRegistry:
    """The ``#m<n>`` map for one chat, backed by one JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.entries: dict[int, SerialEntry] = {}
        self.by_message: dict[str, int] = {}
        self.by_hash: dict[str, int] = {}
        self.ordinals: dict[str, int] = {}
        self.next_ordinal = 1
        self.last_serial = 0
        self.loaded = False
        self.lock = threading.RLock()

    def load(self) -> None:
        """Read the file once, skipping a torn tail line."""
        with self.lock:
            if self.loaded:
                return
            if self.path.exists():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    self.add_line(line)
            self.loaded = True

    def add_line(self, line: str) -> None:
        """Fold one stored JSONL line into the maps; ignore a torn line."""
        line = line.strip()
        if not line:
            return
        try:
            raw = json.loads(line)
        except ValueError:
            return
        entry = SerialEntry(
            serial=int(raw["m"]),
            message_id=str(raw["id"]),
            jid=str(raw.get("jid", "")),
            name=str(raw.get("name", "")),
            ordinal=int(raw.get("ord", 0)),
        )
        self.entries[entry.serial] = entry
        self.by_message[entry.message_id] = entry.serial
        self.by_hash[message_key(entry.message_id)] = entry.serial
        self.last_serial = max(self.last_serial, entry.serial)
        if entry.ordinal and entry.jid and entry.jid not in self.ordinals:
            self.ordinals[entry.jid] = entry.ordinal
            self.next_ordinal = max(self.next_ordinal, entry.ordinal + 1)

    def ordinal_for(self, jid: str) -> int:
        """The stable ``<member-n>`` ordinal for *jid*, assigning one once."""
        if not jid:
            return 0
        with self.lock:
            self.load()
            ordinal = self.ordinals.get(jid)
            if ordinal is None:
                ordinal = self.next_ordinal
                self.next_ordinal += 1
                self.ordinals[jid] = ordinal
            return ordinal

    def known_ordinal(self, jid: str) -> int:
        """The stored ordinal for *jid*, or 0 when none was persisted.

        A read-only lookup: it never allocates, so a caller rendering a
        mere mention (not the sender of a stored message) cannot pin an
        ordinal that a restart would forget — it renders ``<member-?>``
        instead. Ordinals persist only as part of an ingested message's
        entry, so the sender path stays the one allocation site.
        """
        if not jid:
            return 0
        with self.lock:
            self.load()
            return self.ordinals.get(jid, 0)

    def next(
        self, *, message_id: str, jid: str = "", name: str = "", ordinal: int = 0
    ) -> int:
        """The serial for *message_id*, assigning one on first sight.

        A WAHA redelivery of the same message returns its existing serial
        and writes nothing.
        """
        with self.lock:
            self.load()
            existing = self.by_message.get(message_id)
            if existing is not None:
                return existing
            serial = self.last_serial + 1
            self.last_serial = serial
            entry = SerialEntry(serial, message_id, jid, name, ordinal)
            self.entries[serial] = entry
            self.by_message[message_id] = serial
            self.by_hash[message_key(message_id)] = serial
            if ordinal and jid and jid not in self.ordinals:
                self.ordinals[jid] = ordinal
                self.next_ordinal = max(self.next_ordinal, ordinal + 1)
            self.append(entry)
            return serial

    def serial_for(self, message_id: str) -> int:
        """The serial for a full id or a bare hash (``replyTo.id``), or 0."""
        with self.lock:
            self.load()
            serial = self.by_message.get(message_id)
            if serial is not None:
                return serial
            return self.by_hash.get(message_key(message_id), 0)

    def append(self, entry: SerialEntry) -> None:
        """Append one entry with a single atomic write."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "m": entry.serial,
                "id": entry.message_id,
                "jid": entry.jid,
                "name": entry.name,
                "ord": entry.ordinal,
            },
            ensure_ascii=False,
        )
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (line + "\n").encode("utf-8"))
        finally:
            os.close(fd)

    def resolve(self, token: str) -> SerialEntry | None:
        """The entry for a ``#m<n>`` token, or None if it is not one / unknown."""
        match = TOKEN_RE.match(token.strip())
        if match is None:
            return None
        with self.lock:
            self.load()
            return self.entries.get(int(match.group(1)))

    def max_serial(self) -> int:
        """The highest assigned serial, for continuing numbering across runs."""
        with self.lock:
            self.load()
            return self.last_serial

    def compact(self, keep: int = DEFAULT_KEEP) -> None:
        """Keep the newest *keep* entries, atomically and under a lock."""
        with self.lock:
            if not self.path.exists():
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock = self.path.with_name(f"{self.path.name}.lock")
            with open(lock, "w", encoding="utf-8") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                lines = [
                    line
                    for line in self.path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if len(lines) <= keep:
                    return
                self.rewrite(lines[-keep:])
            self.reset()

    def rewrite(self, lines: list[str]) -> None:
        """Replace the file with *lines* through a temp file and ``os.replace``."""
        fd, tmp = tempfile.mkstemp(
            dir=self.path.parent, prefix=f"{self.path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
            os.replace(tmp, self.path)
        except BaseException:
            os.unlink(tmp)
            raise

    def reset(self) -> None:
        """Forget the in-memory view so the next read reloads from disk."""
        with self.lock:
            self.entries = {}
            self.by_message = {}
            self.by_hash = {}
            self.ordinals = {}
            self.next_ordinal = 1
            self.last_serial = 0
            self.loaded = False


class SerialStore:
    """A process-wide cache of per-chat registries under one data directory."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.registries: dict[tuple[str, str], SerialRegistry] = {}
        self.lock = threading.Lock()

    def for_chat(self, session: str, chat_id: str) -> SerialRegistry:
        """The registry for one chat, loaded lazily and cached."""
        with self.lock:
            key = (session, chat_id)
            registry = self.registries.get(key)
            if registry is None:
                registry = SerialRegistry(registry_file(self.data_dir, session, chat_id))
                self.registries[key] = registry
            return registry

    def assign(
        self,
        session: str,
        chat_id: str,
        *,
        message_id: str,
        jid: str = "",
        name: str = "",
        ordinal: int = 0,
    ) -> SerialEntry:
        """Assign *message_id* a serial and return its stored entry."""
        registry = self.for_chat(session, chat_id)
        serial = registry.next(message_id=message_id, jid=jid, name=name, ordinal=ordinal)
        if registry.max_serial() % 500 == 0:
            registry.compact()
        registry.load()
        return registry.entries[serial]

    def resolve(self, session: str, chat_id: str, token: str) -> SerialEntry | None:
        """Resolve a ``#m<n>`` token in one chat, fail-soft."""
        return self.for_chat(session, chat_id).resolve(token)

    def find(self, session: str, token: str) -> SerialEntry | None:
        """Resolve a token across the session's chats; unique match only.

        Operator commands have no chat of their own. A token that names the
        same serial in more than one chat is ambiguous and resolves to None
        (the caller falls back to a soft error) rather than guessing.
        """
        match = TOKEN_RE.match(token.strip())
        if match is None:
            return None
        base = self.data_dir / "memory" / session
        if not base.is_dir():
            return None
        hits: list[SerialEntry] = []
        for path in sorted(base.glob("*.ids.jsonl")):
            entry = SerialRegistry(path).resolve(token)
            if entry is not None:
                hits.append(entry)
        return hits[0] if len(hits) == 1 else None


@lru_cache(maxsize=32)
def serial_store(data_dir: str) -> SerialStore:
    """The cached store for *data_dir* (one per process and data root)."""
    return SerialStore(data_dir)
