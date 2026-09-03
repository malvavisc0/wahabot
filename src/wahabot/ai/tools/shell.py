"""Rich shell execution tool.

Runs an arbitrary shell command on the host via ``subprocess`` and returns
a bounded, status-style string — the same "never raises" contract as the
other tools. Because this can do anything on the host, it is **off by
default**: the tool is only registered when ``shell_tool`` is enabled in
settings (``WAHABOT_SHELL_TOOL=true``). Operators must also cap runtime
(``WAHABOT_SHELL_TIMEOUT``) and output size (``WAHABOT_SHELL_MAX_OUTPUT``)
so a runaway command can never hang the webhook or flood the model.
"""

import contextlib
import os
import signal
import subprocess
import threading
from typing import IO

from loguru import logger

from wahabot.settings import Settings

__all__ = ["shell_command"]

_MIN_TIMEOUT_SECONDS = 1.0
_MIN_MAX_OUTPUT = 200
_GRACE_SECONDS = 1.0
_KILL_WAIT_SECONDS = 2.0
_READ_CHUNK = 65536
_SHELL = "/bin/sh"


def shell_command(settings: Settings, command: str) -> str:
    """Run a shell command and return its trimmed output, prefixed with status.

    The command runs through ``/bin/sh`` so pipes, redirection and the
    usual shell features work; stdin is closed so a command that reads
    it cannot hang. Capture is bounded: reader threads stop collecting
    past ``settings.shell_max_output`` bytes per stream, so a flooding
    command cannot exhaust memory. A non-zero exit code is reported
    rather than raised. On timeout the shell and its whole process
    group are reaped (SIGTERM, then SIGKILL), so background children
    cannot survive orphaned on the host.
    """
    if not command.strip():
        return "Error: command cannot be empty."
    run_timeout = max(settings.shell_timeout, _MIN_TIMEOUT_SECONDS)
    max_output = max(settings.shell_max_output, _MIN_MAX_OUTPUT)
    logger.debug("Running shell command: {cmd}", cmd=command)
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            executable=_SHELL,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except Exception as exc:
        logger.warning("shell_command failed to start: {exc}", exc=exc)
        return f"Command failed to run: {exc}"
    readers = [
        _start_reader(proc.stdout, max_output),
        _start_reader(proc.stderr, max_output),
    ]
    if not _join_readers(readers, run_timeout):
        _kill_tree(proc)
        logger.warning("shell_command timed out after {t}s", t=run_timeout)
        return _render_result(
            proc.returncode,
            readers[0].text,
            readers[1].text,
            max_output,
            header=f"Command timed out after {run_timeout}s.",
        )
    proc.wait()
    return _render_result(proc.returncode, readers[0].text, readers[1].text, max_output)


class _StreamReader:
    """Drain one pipe on a daemon thread, keeping at most ``cap`` bytes."""

    def __init__(self, stream: IO[bytes] | None, cap: int) -> None:
        self._chunks: list[bytes] = []
        self._left = cap
        self.thread = threading.Thread(target=self._drain, args=(stream,), daemon=True)

    def _drain(self, stream: IO[bytes] | None) -> None:
        if stream is None:
            return
        while chunk := stream.read(_READ_CHUNK):
            keep = chunk[: self._left]
            if keep:
                self._chunks.append(keep)
                self._left -= len(keep)

    @property
    def text(self) -> str:
        return b"".join(self._chunks).decode(errors="replace")


def _start_reader(stream: IO[bytes] | None, cap: int) -> _StreamReader:
    """Start a daemon reader thread for one pipe and return its handle."""
    reader = _StreamReader(stream, cap)
    reader.thread.start()
    return reader


def _join_readers(readers: list[_StreamReader], timeout: float) -> bool:
    """Join reader threads within *timeout*; False when the deadline passed."""
    deadline = threading.Event()
    timer = threading.Timer(timeout, deadline.set)
    timer.start()
    try:
        for reader in readers:
            while reader.thread.is_alive():
                if deadline.is_set():
                    return False
                reader.thread.join(timeout=0.05)
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


def _render_result(
    returncode: int | None,
    stdout: str | None,
    stderr: str | None,
    max_output: int,
    header: str | None = None,
) -> str:
    """Render a completed command into the status string returned to the agent."""
    code = returncode if returncode is not None else "unknown"
    parts = [header] if header else []
    parts.append(f"exit code: {code}")
    parts.extend(_output_lines(stdout, stderr, max_output))
    return "\n".join(parts)


def _output_lines(stdout: str | None, stderr: str | None, max_output: int) -> list[str]:
    """Labelled, capped stdout/stderr blocks, or a no-output marker."""
    lines: list[str] = []
    out = (stdout or "").strip()
    err = (stderr or "").strip()
    if out:
        lines.append(f"stdout:\n{_cap(out, max_output)}")
    if err:
        lines.append(f"stderr:\n{_cap(err, max_output)}")
    return lines or ["(no output)"]


def _cap(text: str, max_output: int) -> str:
    """Trim *text* to *max_output* characters, flagging the truncation."""
    omitted = len(text) - max_output
    if omitted <= 0:
        return text
    return f"{text[:max_output]}\n... [truncated, {omitted} chars omitted]"
