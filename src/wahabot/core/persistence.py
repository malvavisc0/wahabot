"""On-disk persistence of per-chat conversation memory.

One JSON file per chat — ``<data_dir>/memory/<session>/<chat_id>.json`` —
holds that chat's ``ChatMemoryBuffer`` at its last save point, so a
chat's memory survives process restarts and LRU evictions. Plaintext on
purpose (matches the event journal); the wipe path is ``wahabot forget``.

Files are fail-soft throughout: a missing, corrupt, mis-versioned or
mismatched file starts the chat blank instead of crashing the webhook.
Only a file that provably cannot parse is renamed to ``.bad``; every
other condition leaves the file exactly as found.
"""

import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from llama_index.core.memory import ChatMemoryBuffer
from loguru import logger

#: Envelope format version. Unknown versions are ignored (blank start)
#: and left untouched — a future bot version may still read them.
VERSION = 1

#: Shape of a persistable chat key. Real chat JIDs (``…@g.us``,
#: ``…@c.us``, ``…@lid``) and the internal ``operator`` marker all
#: match; anything carrying path separators or traversal segments
#: does not and simply never persists (fail-soft, logged once).
_PERSISTABLE = re.compile(r"^[A-Za-z0-9.@_-]+$")

_ENVELOPE_VERSION = "version"
_ENVELOPE_SESSION = "session"
_ENVELOPE_CHAT_ID = "chat_id"
_ENVELOPE_SAVED_AT = "saved_at"
_ENVELOPE_MEMORY = "memory"


def persistable(chat_id: str) -> bool:
    """True when *chat_id* can safely become a filename under memory/."""
    return bool(_PERSISTABLE.fullmatch(chat_id)) and chat_id not in (".", "..")


def memory_file(data_dir: Path, session: str, chat_id: str) -> Path:
    """Path of a chat's memory file; JIDs are filesystem-safe everywhere."""
    return data_dir / "memory" / session / f"{chat_id}.json"


def load_memory(data_dir: Path, session: str, chat_id: str) -> ChatMemoryBuffer | None:
    """Load a chat's memory buffer, or None when absent, unreadable or invalid.

    Fail-soft: never raises. A file that cannot be JSON-parsed (or whose
    buffer cannot be reconstructed) is renamed to ``.bad`` for
    forensics; a wrong-version or mismatched-envelope file is left
    untouched as an inspectable artifact.
    """
    if not persistable(chat_id):
        return None
    path = memory_file(data_dir, session, chat_id)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("Unreadable memory file {path}: {exc}", path=path, exc=exc)
        return None
    try:
        envelope = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning(
            "Corrupt memory file {path}, renamed to .bad: {exc}", path=path, exc=exc
        )
        _rename_to_bad(path)
        return None
    if not isinstance(envelope, dict):
        logger.warning("Memory file {path} is not an object, renamed to .bad", path=path)
        _rename_to_bad(path)
        return None
    data = cast(dict[str, Any], envelope)
    if data.get(_ENVELOPE_VERSION) != VERSION:
        logger.info(
            "Ignoring memory file {path} with unknown version {version}",
            path=path,
            version=data.get(_ENVELOPE_VERSION),
        )
        return None
    if data.get(_ENVELOPE_SESSION) != session or data.get(_ENVELOPE_CHAT_ID) != chat_id:
        logger.info(
            "Ignoring memory file {path}: envelope does not match the path",
            path=path,
        )
        return None
    try:
        memory = cast(dict[str, Any], data[_ENVELOPE_MEMORY])
        return ChatMemoryBuffer.from_dict(memory)
    except Exception as exc:
        logger.warning(
            "Unparseable memory buffer in {path}, renamed to .bad: {exc}",
            path=path,
            exc=exc,
        )
        _rename_to_bad(path)
        return None


def save_memory(
    data_dir: Path, session: str, chat_id: str, memory: ChatMemoryBuffer
) -> None:
    """Write a chat's memory buffer to disk, atomically, swallowing failures.

    The payload goes to a ``.tmp`` sibling then ``os.replace``, so a
    crash mid-write leaves the previous good file. Any error is logged
    and swallowed — disk or serialization alike: a memory failure must
    never break a reply (a raised save would drop the message's seen
    marker and WAHA's redelivery would retry the run into the same
    failure forever). A non-persistable chat id (path separators,
    traversal segments) never touches the disk.

    A successful write logs one INFO line mirroring the restore log
    ("Saved memory for …"), so both halves of the persistence round
    trip are visible in the journal.
    """
    if not persistable(chat_id):
        logger.warning("Not persisting memory for bad chat id {id!r}", id=chat_id)
        return
    path = memory_file(data_dir, session, chat_id)
    tmp = path.with_name(path.name + ".tmp")
    to_dict = cast(Callable[..., dict[str, Any]], memory.to_dict)
    try:
        envelope = {
            _ENVELOPE_VERSION: VERSION,
            _ENVELOPE_SESSION: session,
            _ENVELOPE_CHAT_ID: chat_id,
            _ENVELOPE_SAVED_AT: int(time.time()),
            _ENVELOPE_MEMORY: to_dict(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(envelope), encoding="utf-8")
        os.replace(tmp, path)
        logger.info(
            "Saved memory for {chat_id}: {count} messages",
            chat_id=chat_id,
            count=len(memory.get_all()),
        )
    except Exception as exc:
        logger.warning("Failed to save memory to {path}: {exc}", path=path, exc=exc)


def forget_memory(data_dir: Path, session: str, chat_id: str) -> bool:
    """Delete a chat's memory file; True when a file was actually removed."""
    if not persistable(chat_id):
        logger.warning("Not forgetting non-persistable chat id {id!r}", id=chat_id)
        return False
    path = memory_file(data_dir, session, chat_id)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("Failed to delete memory file {path}: {exc}", path=path, exc=exc)
        return False
    return True


def _rename_to_bad(path: Path) -> None:
    """Move a known-garbage memory file aside for inspection."""
    target = path.with_name(path.name + ".bad")
    try:
        os.replace(path, target)
    except OSError as exc:
        logger.warning("Could not rename {path} to .bad: {exc}", path=path, exc=exc)
