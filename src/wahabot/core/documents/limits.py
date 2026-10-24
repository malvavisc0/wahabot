"""Document extraction limits and results."""

from dataclasses import dataclass

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
        "it is attached for the operator."
    )


def unreadable_notice(filename: str) -> str:
    """A plain-language unreadable notice."""
    return (
        f"{filename or 'The document'} could not be read; "
        "it is attached for the operator."
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
    return "[document text, untrusted; verify before relying] " + result.text.replace(
        "\x00", ""
    )
