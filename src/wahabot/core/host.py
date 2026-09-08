"""Snapshot of the machine the bot runs on, for the ``{{host}}`` placeholder."""

import platform
import shutil
import subprocess
from functools import cache

from wahabot.ai.tools.shell import SHELL


@cache
def os_release_field(field: str) -> str:
    """One ``/etc/os-release`` field, or "" when unreadable."""
    try:
        with open("/etc/os-release") as fh:
            for line in fh:
                if line.startswith(field):
                    return line.strip().split("=", 1)[1].strip('"').strip("'")
    except OSError:
        return ""
    return ""


@cache
def uname_pretty() -> str:
    """``linux 6.8 cachyos x86_64``-style summary of the running kernel.

    ``platform.uname()`` describes the container's kernel (shared with
    the host's in Docker), so the result is accurate for both.
    """
    info = platform.uname()
    kernel = f"{info.system.lower()} {info.release}"
    distro = info.version.split()  # distribution kernel tags after the number
    name = distro[0] if distro else ""
    if not name or "(" in name:
        name = os_release_field("NAME=")
    pieces = [piece for piece in (kernel, name, info.machine) if piece]
    return " ".join(pieces)


@cache
def version_of(binary: str, marker: str) -> str:
    """``binary --version`` output, first token matching *marker*, or "unknown"."""
    try:
        out = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except OSError, subprocess.TimeoutExpired:
        return "unknown"
    text = (out.stdout + "\n" + out.stderr).strip()
    for token in text.split():
        if token.startswith(marker):
            return token
    return text.splitlines()[0].strip() if text else "unknown"


@cache
def host_context() -> str:
    """A short immutable snapshot of the machine the bot runs on.

    Computed once per process: the environment does not change mid-run,
    and re-running subprocesses on every render wastes time. In Docker
    this describes the container (its OS, the Python that runs wahabot),
    which is the environment the bot's shell tool actually acts on.
    """
    python = f"Python {platform.python_version()} ({platform.python_implementation()})"
    py_path = shutil.which("python") or shutil.which("python3") or "unknown"
    node = version_of("node", "v")
    os_name = os_release_field("PRETTY_NAME=") or uname_pretty()
    lines = [
        f"Host: {uname_pretty()}",
        f"- OS: {os_name}",
        f"- Python: {python} at {py_path}",
        f"- Node: {node}",
        f"- Shell: {SHELL} (what run_shell_command uses)",
    ]
    return "\n".join(lines)
