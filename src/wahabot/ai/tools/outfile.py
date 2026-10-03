"""Filesystem sink for large tool outputs.

Tools that fan out a lot of data (a chat's whole history with
``limit=600``, a long web page, a YouTube transcript, shell output)
must not inline megabytes into the model's token budget and must not
silently truncate. Instead they write the full payload to a temporary
file and return just its metadata (``file.path``/``file.bytes``) plus a
bounded inline preview. The model keeps a working summary inline and can
read the rest in parts through its own tools (``run_shell_command``), or
the operator can open the path — without paying a base64/bloat tax on
every message.
"""

import contextlib
import json
import os
import tempfile
from typing import Any, BinaryIO

__all__ = [
    "file_metadata",
    "open_byte_output",
    "tool_output_dir",
    "write_json_output",
    "write_text_output",
]

#: Subdirectory under the system temp dir; all tool dumps land here so a
#: workspace scan finds them in one place, nothing touches data_dir, and
#: the OS's regular temp-dir cleaning sweeps them like any other
#: transient file.
_PREFIX = "wahabot-toolout"


def tool_output_dir() -> str:
    """The temp directory tool dumps are written to (created on demand)."""
    directory = os.path.join(tempfile.gettempdir(), _PREFIX)
    os.makedirs(directory, exist_ok=True)
    return directory


def write_json_output(label: str, payload: Any) -> dict[str, Any]:
    """Persist *payload* to a fresh ``.json`` file in the tool output dir.

    ``mkstemp`` guarantees an unused, atomically-created path so concurrent
    tool calls never collide; *label* only names the prefix for humans (the
    timestamp plus a random suffix) and is never trusted as a filename.
    Returns the file's metadata: absolute ``path``, bare ``filename`` and
    ``bytes`` written.
    """
    fd, path = tempfile.mkstemp(prefix=f"{label}.", suffix=".json", dir=tool_output_dir())
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except BaseException:
        _discard(path)
        raise
    return file_metadata(path)


def write_text_output(label: str, text: str) -> dict[str, Any]:
    """Persist *text* to a fresh ``.txt`` file in the tool output dir.

    The plain-text counterpart of :func:`write_json_output` — page
    bodies, transcripts, shell output. Same metadata shape.
    """
    fd, path = tempfile.mkstemp(prefix=f"{label}.", suffix=".txt", dir=tool_output_dir())
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
    except BaseException:
        _discard(path)
        raise
    return file_metadata(path)


def open_byte_output(label: str) -> tuple[BinaryIO, dict[str, Any]]:
    """Open a fresh spill file for streaming bytes into.

    The shell tool's reader threads use this to stream a command's full
    output straight to disk as it arrives — memory stays bounded no
    matter how much a command floods, and nothing captured is lost.
    Returns the open binary handle and its initial metadata dict (the
    same shape :func:`file_metadata` produces; the caller finalizes
    ``bytes`` after closing, since the file grows while streaming).
    """
    fd, path = tempfile.mkstemp(prefix=f"{label}.", suffix=".txt", dir=tool_output_dir())
    try:
        handle = os.fdopen(fd, "wb")
    except BaseException:
        _discard(path)
        raise
    return handle, {"path": path, "filename": os.path.basename(path), "bytes": 0}


def file_metadata(path: str) -> dict[str, Any]:
    """The envelope's file metadata for a spilled *path*."""
    return {
        "path": path,
        "filename": os.path.basename(path),
        "bytes": os.path.getsize(path),
    }


def _discard(path: str) -> None:
    """Remove a partially-written spill file so no garbage lingers."""
    with contextlib.suppress(OSError):
        os.remove(path)
