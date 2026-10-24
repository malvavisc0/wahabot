"""PDF text and OCR extraction through poppler and tesseract."""

import subprocess
import tempfile
import time
from pathlib import Path

from wahabot.core.documents.limits import DocumentLimits


def run_command(command: list[str], timeout_s: float) -> tuple[int, bytes, str]:
    """Run one bounded external command; missing binaries are ``-1``."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError:
        return -1, b"", f"{command[0]} is not installed"
    except subprocess.TimeoutExpired:
        return -2, b"", f"{command[0]} timed out"
    except OSError as exc:
        return -3, b"", str(exc)
    return (
        completed.returncode,
        completed.stdout,
        completed.stderr.decode(errors="replace"),
    )


def pdf_page_count(path: Path, limits: DocumentLimits) -> int:
    """Total pages in a PDF, or 0 when ``pdfinfo`` cannot tell."""
    code, stdout, _ = run_command(["pdfinfo", str(path)], limits.timeout_s)
    if code != 0:
        return 0
    for line in stdout.decode(errors="replace").splitlines():
        key, _, value = line.partition(":")
        if key.strip().casefold() != "pages":
            continue
        try:
            return int(value.strip())
        except ValueError:
            return 0
    return 0


def pdf_text(path: Path, limits: DocumentLimits) -> tuple[str, int, str]:
    """``pdftotext -layout`` up to the page limit; ``(text, pages, error)``."""
    code, stdout, stderr = run_command(
        [
            "pdftotext",
            "-layout",
            "-enc",
            "UTF-8",
            "-f",
            "1",
            "-l",
            str(limits.max_pages),
            str(path),
            "-",
        ],
        limits.timeout_s,
    )
    if code != 0:
        return "", 0, stderr
    text = stdout.decode(errors="replace")
    pages = text.count("\f") or (1 if text.strip() else 0)
    return text, pages, ""


def pdf_ocr(path: Path, limits: DocumentLimits) -> tuple[str, int, str]:
    """Render the first N pages and OCR each with tesseract."""
    with tempfile.TemporaryDirectory(prefix="wahabot-ocr-") as raw:
        root = Path(raw)
        prefix = root / "page"
        code, _, stderr = run_command(
            [
                "pdftoppm",
                "-png",
                "-r",
                "150",
                "-f",
                "1",
                "-l",
                str(limits.max_pages),
                str(path),
                str(prefix),
            ],
            limits.timeout_s,
        )
        if code != 0:
            return "", 0, stderr
        pages = sorted(
            root.glob("page-*.png"),
            key=lambda item: int(item.stem.rsplit("-", 1)[-1]),
        )
        texts: list[str] = []
        deadline = time.monotonic() + limits.timeout_s
        for page in pages:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "", len(pages), "tesseract timed out"
            text_code, stdout, text_err = run_command(
                ["tesseract", str(page), "stdout"], remaining
            )
            if text_code != 0:
                return "", len(pages), text_err
            texts.append(stdout.decode(errors="replace"))
        return "\n".join(texts), len(pages), ""
