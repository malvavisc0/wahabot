"""Rich shell execution tool.

Runs an arbitrary shell command on the host via ``subprocess`` and returns
a result in the shared JSON envelope — the same "never raises" contract
as the other tools. Inline stdout/stderr are bounded
(``settings.shell_max_output`` chars per stream), but nothing captured
is ever lost: past the inline cap a reader thread streams the rest
straight to a spill file (opened only on overflow, so small commands
touch no disk; memory stays bounded no matter how much a command
floods), and the envelope points at it via ``file.path`` /
``stderr_file.path``. Even a timeout reports the partial output
captured before the kill — inline previews plus spill files on the
``error`` envelope. Because this can do anything on the host, it is
**off by default**: the tool is only registered when ``shell_tool`` is
enabled in settings (``WAHABOT_SHELL_TOOL=true``). Operators must also
cap runtime (``WAHABOT_SHELL_TIMEOUT``) so a runaway command can never
hang the webhook.
"""

import contextlib
import os
import signal
import subprocess
import threading
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
    """Run a shell command and return its result as a JSON envelope.

    The command runs through ``/bin/bash`` so pipes, redirection and the
    usual shell features work; stdin is closed so a command that reads
    it cannot hang. Nothing captured is lost: each stream keeps an
    inline preview bounded by ``settings.shell_max_output`` chars, and
    whatever overflows it streams to a spill file — so a flooding
    command can exhaust neither memory nor the model's token budget.
    A non-zero exit code is reported in the envelope (``ok`` stays true
    — the command ran), while a start failure or timeout yields an
    ``error`` envelope; on timeout the partial output captured before
    the kill still rides that envelope. On timeout the shell and its
    whole process group are reaped (SIGTERM, then SIGKILL), so
    background children cannot survive orphaned on the host.
    """
    if not command.strip():
        return error("command cannot be empty")
    run_timeout = max(settings.shell_timeout, _MIN_TIMEOUT_SECONDS)
    max_output = max(settings.shell_max_output, _MIN_MAX_OUTPUT)
    logger.debug("Running shell command: {cmd}", cmd=command)
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
    if not _join_readers(readers, run_timeout):
        _kill_tree(proc)
        logger.warning("shell_command timed out after {t}s", t=run_timeout)
        return _render(None, sinks, f"command timed out after {run_timeout}s")
    proc.wait()
    return _render(proc.returncode, sinks, None)


class _StreamSink:
    """One command stream's bounded inline preview plus lazy spill file.

    The first *cap* bytes stay in memory as the envelope's inline
    preview. Only when a stream overflows that cap is a spill file
    opened — seeded with the buffered preview so the file always holds
    the complete stream — and every further chunk streams straight to
    disk. A command whose output fits inline therefore never touches
    disk, and a flooding command never grows memory past the cap.
    """

    def __init__(self, label: str, cap: int) -> None:
        self._cap = cap
        self._preview = bytearray()
        self._file: IO[bytes] | None = None
        self._meta: dict[str, Any] | None = None
        self._label = label
        self._finalized = False

    def append(self, chunk: bytes) -> None:
        """Add one chunk to the preview, spilling the remainder to disk.

        A no-op once the sink is finalized: on the timeout path the
        reader thread can still be draining pipe remnants when the
        envelope is rendered — writing to the closed handle would both
        crash the reader and lose the very tail the envelope promised.
        """
        if self._finalized:
            return
        if len(self._preview) < self._cap:
            take = chunk[: self._cap - len(self._preview)]
            self._preview.extend(take)
            chunk = chunk[len(take) :]
        if chunk:
            if self._file is None:
                self._file, self._meta = open_byte_output(self._label)
                self._file.write(bytes(self._preview))
            self._file.write(chunk)

    @property
    def text(self) -> str:
        """The bounded inline preview."""
        return bytes(self._preview).decode(errors="replace")

    @property
    def truncated(self) -> bool:
        """True when output past the inline cap exists (on the spill file)."""
        return self._file is not None

    def file_meta(self) -> dict[str, Any] | None:
        """The spill file's metadata, or None if the stream fit inline.

        Finalizes the file (closing the handle and computing its size);
        idempotent, so the timeout path can call it after the kill.
        """
        if self._file is None or self._finalized:
            return self._meta
        self._finalized = True
        with contextlib.suppress(OSError):
            self._file.close()
        meta = self._meta
        if meta is not None:
            with contextlib.suppress(OSError):
                meta["bytes"] = os.path.getsize(meta["path"])
        return self._meta


def _start_reader(stream: IO[bytes] | None, sink: _StreamSink) -> threading.Thread:
    """Start a daemon reader thread for one pipe and return its handle."""

    def drain() -> None:
        if stream is None:
            return
        while chunk := stream.read(_READ_CHUNK):
            sink.append(chunk)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    return reader


def _join_readers(readers: list[threading.Thread], timeout: float) -> bool:
    """Join reader threads within *timeout*; False when the deadline passed."""
    deadline = threading.Event()
    timer = threading.Timer(timeout, deadline.set)
    timer.start()
    try:
        for reader in readers:
            while reader.is_alive():
                if deadline.is_set():
                    return False
                reader.join(timeout=0.05)
    finally:
        timer.cancel()
    return True


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Terminate *proc* and its process group: SIGTERM, grace, then SIGKILL.

    SIGKILL goes to the whole group unconditionally after the grace
    period — a child that ignores SIGTERM must not survive orphaned just
    because the shell itself died promptly. The final wait is bounded so
    a child stuck in uninterruptible sleep cannot hang the tool thread.
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
    """Render the command's outcome into the JSON envelope.

    Inline stdout/stderr are the sinks' bounded previews; each stream
    that overflowed its cap carries the full capture as a spill file
    (``file`` for stdout, ``stderr_file`` for stderr) so the model can
    read the rest in parts (``head``/``tail``/``sed``/``rg`` on the
    path). A timed-out command renders an ``error`` envelope — but
    with the same previews and spill files, so only the missing tail is
    lost, never what was already captured.
    """
    out = sinks["stdout"].text.strip()
    err = sinks["stderr"].text.strip()
    out_truncated = sinks["stdout"].truncated
    err_truncated = sinks["stderr"].truncated
    payload: dict[str, Any] = {
        "stdout": out,
        "stderr": err,
        "truncated": out_truncated or err_truncated,
    }
    if out_truncated:
        payload["file"] = sinks["stdout"].file_meta()
    if err_truncated:
        payload["stderr_file"] = sinks["stderr"].file_meta()
    if failure is not None:
        return error(failure, **payload)
    payload["exit_code"] = returncode if returncode is not None else "unknown"
    return ok(**payload)
