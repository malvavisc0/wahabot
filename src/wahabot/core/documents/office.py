"""Office and plain-text document conversion."""

from pathlib import Path

from wahabot.core.documents.limits import DocumentLimits
from wahabot.core.documents.pdf import run_command

OFFICE_EXTENSIONS = frozenset({".docx", ".odt", ".rtf", ".doc"})


def pandoc_text(path: Path, limits: DocumentLimits) -> tuple[str, str]:
    """Convert an office document to plain text with pandoc."""
    code, stdout, stderr = run_command(
        ["pandoc", str(path), "-t", "plain"], limits.timeout_s
    )
    if code != 0:
        return "", stderr
    return stdout.decode(errors="replace"), ""


def plain_text(path: Path, limits: DocumentLimits) -> tuple[str, str]:
    """Read a plain text or CSV file up to the character cap."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        return "", str(exc)
    if len(data) > limits.max_bytes:
        data = data[: limits.max_bytes]
    return data.decode("utf-8", errors="replace"), ""
