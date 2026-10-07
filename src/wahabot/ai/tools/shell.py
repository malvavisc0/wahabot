"""Host shell execution with bounded previews and best-effort capture.

Commands run through Bash without a sandbox. Each stdout/stderr preview
retains up to ``settings.shell_max_output`` bytes (at least 200), decoded
as UTF-8 with replacement and stripped of surrounding whitespace.
Overflow attempts a lazy spill file; successful files contain the raw
captured bytes. Capture failures keep the preview, mark truncation and
report ``capture_errors`` without changing the command's exit status.

A single wall-time deadline covers output reads and process completion.
Timeouts return available partial output after bounded process-group
termination and reader draining; cleanup can exceed the runtime deadline.
Descendants that leave the process group can escape termination. The tool
is off by default and only registered when ``shell_tool`` is enabled
(``WAHABOT_SHELL_TOOL=true``); the runtime budget is configured with
``WAHABOT_SHELL_TIMEOUT`` (at least one second).
"""

import contextlib
import os
import signal
import subprocess
import threading
import time
from typing import IO, Any

from loguru import logger

from wahabot.ai.tools.envelope import error, ok
from wahabot.ai.tools.outfile import open_byte_output
from wahabot.settings import Settings

__all__ = ["shell_command"]

_MIN_TIMEOUT_SECONDS = 1.0
_MIN_MAX_OUTPUT = 200
_GRACE_SECONDS = 1.0
_KILL_WAIT_SECONDS = 2.0
_READ_CHUNK = 65536
#: The shell every ``run_shell_command`` call executes through. Public:
#: ``core.host`` reports it in the ``{{host}}`` block so the prompt's
#: claim and the tool's behavior cannot drift apart.
SHELL = "/bin/bash"


def shell_command(settings: Settings, command: str) -> str:
    """Run an unsandboxed host command and return its JSON envelope.

    ``/bin/bash`` supports pipes and redirection; stdin is closed. Each
    stream keeps a preview of at most ``max(settings.shell_max_output, 200)``
    bytes, decoded with replacement and stripped. On overflow a lazy spill
    attempts to retain the raw capture with bounded memory. Capture faults
    mark truncation and add per-stream ``capture_errors``; failed spills
    are omitted. ``ok`` stays true for executed commands, including nonzero
    exits or capture faults; start failures/timeouts return an error envelope.

    One monotonic deadline (at least one second) covers stream reads and
    process completion. Timeout sends SIGTERM then SIGKILL to the original
    process group, with bounded wait/drain cleanup outside that budget.
    Available partial capture is returned; descendants that leave the
    process group are not guaranteed to be killed or reaped.
    """
    if not command.strip():
        return error("command cannot be empty")
    run_timeout = max(settings.shell_timeout, _MIN_TIMEOUT_SECONDS)
    max_output = max(settings.shell_max_output, _MIN_MAX_OUTPUT)
    logger.debug("Running shell command: {cmd}", cmd=command)
    deadline = time.monotonic() + run_timeout
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            executable=SHELL,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except Exception as exc:
        logger.warning("shell_command failed to start: {exc}", exc=exc)
        return error(f"command failed to run: {exc}")
    sinks = {
        "stdout": _StreamSink("shell.stdout", max_output),
        "stderr": _StreamSink("shell.stderr", max_output),
    }
    readers = [
        _start_reader(proc.stdout, sinks["stdout"]),
        _start_reader(proc.stderr, sinks["stderr"]),
    ]
    try:
        if not _join_readers(readers, deadline):
            raise subprocess.TimeoutExpired(command, run_timeout)
        proc.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        _join_readers(readers, time.monotonic() + _KILL_WAIT_SECONDS)
        logger.warning("shell_command timed out after {t}s", t=run_timeout)
        return _render(None, sinks, f"command timed out after {run_timeout}s")
    return _render(proc.returncode, sinks, None)


class _StreamSink:
    """One stream's byte-bounded preview and optional best-effort spill.

    The first *cap* bytes stay in memory. Overflow attempts a lazy spill,
    seeded with the preview to retain the raw capture when writes succeed.
    A capture failure disables spilling and hides incomplete file metadata.
    In-budget output creates no file; the preview never grows past *cap*.
    """

    def __init__(self, label: str, cap: int) -> None:
        self._cap = cap
        self._preview = bytearray()
        self._file: IO[bytes] | None = None
        self._meta: dict[str, Any] | None = None
        self._label = label
        self._finalized = False
        self.capture_error: str | None = None

    def append(self, chunk: bytes) -> None:
        """Keep a bounded prefix and attempt to spill overflow.

        After a capture failure chunks are discarded without retrying disk
        operations, allowing the reader to keep draining. A finalized sink
        ignores late chunks from a reader that outlived bounded cleanup.
        """
        if self._finalized or self.capture_error is not None:
            return
        if len(self._preview) < self._cap:
            take = chunk[: self._cap - len(self._preview)]
            self._preview.extend(take)
            chunk = chunk[len(take) :]
        if chunk:
            try:
                if self._file is None:
                    self._file, self._meta = open_byte_output(self._label)
                    self._file.write(bytes(self._preview))
                self._file.write(chunk)
            except Exception as exc:
                self.capture_failed("spill", exc)

    def capture_failed(self, operation: str, exc: Exception) -> None:
        """Record the first capture fault and discard any incomplete spill."""
        if self.capture_error is not None:
            return
        self.capture_error = f"{operation} failed: {type(exc).__name__}: {exc}"[:200]
        handle, meta = self._file, self._meta
        self._file = None
        self._meta = None
        if handle is not None:
            with contextlib.suppress(Exception):
                handle.close()
        if meta is not None:
            with contextlib.suppress(OSError):
                os.remove(meta["path"])

    @property
    def text(self) -> str:
        """The byte-bounded preview decoded as UTF-8 with replacement."""
        return bytes(self._preview).decode(errors="replace")

    @property
    def truncated(self) -> bool:
        """True when output overflowed the preview or capture may be incomplete."""
        return self._file is not None or self.capture_error is not None

    def file_meta(self) -> dict[str, Any] | None:
        """Finalize and return a successful spill's metadata, otherwise None.

        Closing flushes buffered writes; failure invalidates the spill and
        reports a capture fault. Idempotent; freezes even an inline-only sink.
        """
        if self._finalized:
            return self._meta
        self._finalized = True
        if self._file is not None:
            try:
                self._file.close()
                if self._meta is not None:
                    self._meta["bytes"] = os.path.getsize(self._meta["path"])
            except Exception as exc:
                self.capture_failed("spill finalization", exc)
        return self._meta


def _start_reader(stream: IO[bytes] | None, sink: _StreamSink) -> threading.Thread:
    """Start a daemon pipe-drain thread; report read failures on its sink."""

    def drain() -> None:
        if stream is None:
            return
        try:
            while chunk := stream.read(_READ_CHUNK):
                sink.append(chunk)
        except Exception as exc:
            sink.capture_failed("read", exc)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    return reader


def _join_readers(readers: list[threading.Thread], deadline: float) -> bool:
    """Join reader threads by a monotonic deadline; False if any remain alive."""
    for reader in readers:
        reader.join(timeout=max(0.0, deadline - time.monotonic()))
        if reader.is_alive():
            return False
    return True


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Signal the original process group with bounded waits for *proc*.

    After SIGTERM, wait up to the grace period for *proc*, then send SIGKILL
    to the group even if *proc* exited promptly. Only *proc* is waited on;
    descendants that leave its process group are not covered. Both waits
    are bounded so a stuck process cannot hang this cleanup indefinitely.
    """
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=_GRACE_SECONDS)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=_KILL_WAIT_SECONDS)


def _render(
    returncode: int | None,
    sinks: dict[str, _StreamSink],
    failure: str | None,
) -> str:
    """Render exit/timeout status separately from best-effort capture.

    stdout/stderr are decoded, whitespace-stripped byte-bounded previews.
    Successful spills use ``file``/``stderr_file``; capture faults mark
    ``truncated`` and attach ``capture_errors`` without publishing failed
    spills or changing command exit status. Timeouts return available
    partial capture on an error envelope, without ``exit_code``.
    """
    out_file = sinks["stdout"].file_meta()
    err_file = sinks["stderr"].file_meta()
    out = sinks["stdout"].text.strip()
    err = sinks["stderr"].text.strip()
    out_truncated = sinks["stdout"].truncated
    err_truncated = sinks["stderr"].truncated
    payload: dict[str, Any] = {
        "stdout": out,
        "stderr": err,
        "truncated": out_truncated or err_truncated,
    }
    if out_file is not None:
        payload["file"] = out_file
    if err_file is not None:
        payload["stderr_file"] = err_file
    capture_errors = {
        name: sink.capture_error
        for name, sink in sinks.items()
        if sink.capture_error is not None
    }
    if capture_errors:
        payload["capture_errors"] = capture_errors
    if failure is not None:
        return error(failure, **payload)
    payload["exit_code"] = returncode if returncode is not None else "unknown"
    return ok(**payload)
