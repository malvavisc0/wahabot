"""Per-chat serial registry: the ``#m<n>`` <-> WAHA message id map.

Every inbound message gets a small, chat-global serial the prompt can carry
without the ~100-character WAHA id (which embeds the chat's own JID). The
serial is assigned at ingest, stored next to the chat's memory, and used by
the send tools to resolve ``reply_to`` / ``react`` tokens back to real ids.

Storage is append-only JSONL, one line per message, written with a single
``O_APPEND`` ``write()`` — atomic for a sub-4 KB line, so the hot path takes
no lock. Compaction keeps the newest entries for a chat and runs through a
temp file plus ``os.replace`` under an exclusive ``flock``.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

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


def format_serial_token(serial: int) -> str:
    """The prompt-side token for a serial (``12`` -> ``#m12``)."""
    return f"#m{serial}"


class SerialRegistry:
    """The ``#m<n>`` map for one chat, backed by one JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.entries: dict[int, SerialEntry] = {}
        self.by_message: dict[str, int] = {}
        self.last_serial = 0
        self.loaded = False

    def load(self) -> None:
        """Read the file once, skipping a torn tail line."""
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
        )
        self.entries[entry.serial] = entry
        self.by_message[entry.message_id] = entry.serial
        self.last_serial = max(self.last_serial, entry.serial)

    def next(self, *, message_id: str, jid: str = "", name: str = "") -> int:
        """The serial for *message_id*, assigning one on first sight.

        A WAHA redelivery of the same message returns its existing serial
        and writes nothing.
        """
        self.load()
        existing = self.by_message.get(message_id)
        if existing is not None:
            return existing
        serial = self.last_serial + 1
        self.last_serial = serial
        entry = SerialEntry(serial, message_id, jid, name)
        self.entries[serial] = entry
        self.by_message[message_id] = serial
        self.append(entry)
        return serial

    def append(self, entry: SerialEntry) -> None:
        """Append one entry with a single atomic write."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "m": entry.serial,
                "id": entry.message_id,
                "jid": entry.jid,
                "name": entry.name,
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
        self.load()
        return self.entries.get(int(match.group(1)))

    def max_serial(self) -> int:
        """The highest assigned serial, for continuing numbering across runs."""
        self.load()
        return self.last_serial

    def compact(self, keep: int = DEFAULT_KEEP) -> None:
        """Keep the newest *keep* entries, atomically and under a lock."""
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
        self.entries = {}
        self.by_message = {}
        self.last_serial = 0
        self.loaded = False
