"""Document extraction limits and results."""

from dataclasses import dataclass

from wahabot.core.models import WahaEvent

#: The returned state of one extraction attempt.
STATUS_OK = "ok"
STATUS_OCR = "ocr"
STATUS_EMPTY = "empty"
STATUS_UNREADABLE = "unreadable"
STATUS_OVER_LIMIT = "over_limit"

#: The marker key the media pipeline stamps on an event before rendering.
DOCUMENT_MARKER_KEY = "_wahabot_document_marker"
DOCUMENT_TEXT_KEY = "_wahabot_document_text"


@dataclass(frozen=True, slots=True)
class DocumentLimits:
    """The byte, page, character and time caps for one document."""

    max_bytes: int = 16 * 1024 * 1024
    max_pages: int = 20
    max_chars: int = 12_000
    timeout_s: float = 20.0

    @classmethod
    def from_settings(cls, settings: object) -> DocumentLimits:
        """Build limits from the process settings."""
        return cls(
            max_bytes=int(getattr(settings, "max_file_bytes", cls.max_bytes)),
            max_pages=int(getattr(settings, "max_document_pages", cls.max_pages)),
            max_chars=int(getattr(settings, "max_document_chars", cls.max_chars)),
            timeout_s=float(getattr(settings, "document_timeout_s", cls.timeout_s)),
        )


@dataclass(frozen=True, slots=True)
class DocumentText:
    """One document's extracted text, bounded and labelled."""

    text: str
    status: str
    pages: int = 0
    notice: str = ""
    filename: str = ""
    mime: str = ""

    @property
    def readable(self) -> bool:
        """Whether the model can read any text from it."""
        return self.status in (STATUS_OK, STATUS_OCR, STATUS_OVER_LIMIT) and bool(
            self.text
        )


def trim_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Trim to *max_chars*; the flag says content was dropped."""
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean, False
    return clean[:max_chars], True


def over_limit_notice(filename: str) -> str:
    """A plain-language over-limit notice."""
    return (
        f"{filename or 'The document'} is too large to read here; "
        "only the beginning was reviewed. The original stays in the chat."
    )


def page_limit_notice(filename: str, max_pages: int, total_pages: int) -> str:
    """A plain-language notice for a PDF whose pages were truncated."""
    return (
        f"{filename or 'The document'} has {total_pages} pages; only the "
        f"first {max_pages} were read. The original stays in the chat."
    )


def download_limit_notice(filename: str) -> str:
    """A plain-language notice when a document exceeds the download cap."""
    return (
        f"{filename or 'The document'} is too large to download here; "
        "ask the sender to share a smaller copy."
    )


def unreadable_notice(filename: str) -> str:
    """A plain-language unreadable notice."""
    return (
        f"{filename or 'The document'} could not be read; the original stays in the chat."
    )


def partial_ocr_notice(error: str) -> str:
    """A notice for OCR that stopped after reading only some pages."""
    detail = " ".join(error.split())[:160] or "OCR stopped early"
    return (
        f"Only part of the document could be read ({detail}); "
        "the rest is not included. The original stays in the chat."
    )


def document_marker(result: DocumentText) -> str:
    """The turn payload line describing a document and its status."""
    label = result.filename or "document"
    detail = result.status
    if result.pages:
        detail = f"{detail}, {result.pages} page(s)"
    return f'[document: "{label}" ({result.mime or "unknown"}, {detail})]'


def document_text_payload(result: DocumentText) -> str:
    """The extracted text block carrying its untrusted-content label."""
    if not result.readable:
        return f"[document note] {result.notice}" if result.notice else ""
    payload = "[document text, untrusted; verify before relying] " + result.text.replace(
        "\x00", ""
    )
    if result.notice:
        payload = f"{payload}\n\n[document note] {result.notice}"
    return payload


def mark_document_failure(
    event: WahaEvent, filename: str, mime: str, *, too_large: bool
) -> None:
    """Stamp the turn when a document could not be downloaded."""
    result = DocumentText(
        text="",
        status=STATUS_OVER_LIMIT if too_large else STATUS_UNREADABLE,
        notice=(
            download_limit_notice(filename) if too_large else unreadable_notice(filename)
        ),
        filename=filename,
        mime=mime,
    )
    event.payload[DOCUMENT_MARKER_KEY] = document_marker(result)
    event.payload[DOCUMENT_TEXT_KEY] = document_text_payload(result)
