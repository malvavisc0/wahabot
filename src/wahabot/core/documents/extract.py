"""Document extraction: PDF text/OCR, office, plain text and CSV."""

import tempfile
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from wahabot.core.documents.limits import (
    STATUS_EMPTY,
    STATUS_OCR,
    STATUS_OK,
    STATUS_OVER_LIMIT,
    STATUS_UNREADABLE,
    DocumentLimits,
    DocumentText,
    over_limit_notice,
    page_limit_notice,
    trim_text,
    unreadable_notice,
)
from wahabot.core.documents.office import OFFICE_EXTENSIONS, pandoc_text, plain_text
from wahabot.core.documents.pdf import pdf_ocr, pdf_page_count, pdf_text

PDF_MIMES = frozenset({"application/pdf", "application/x-pdf"})
TEXT_MIMES = frozenset(
    {
        "text/plain",
        "text/csv",
        "text/tab-separated-values",
        "application/csv",
        "text/markdown",
    }
)
OFFICE_MIMES = frozenset(
    {
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.oasis.opendocument.text",
        "application/rtf",
        "text/rtf",
    }
)
TEXT_EXTENSIONS = frozenset({".txt", ".csv", ".tsv", ".md", ".log"})
PDF_EXTENSIONS = frozenset({".pdf"})
MIME_SUFFIXES = MappingProxyType(
    {
        "application/pdf": ".pdf",
        "text/plain": ".txt",
        "text/csv": ".csv",
        "text/markdown": ".md",
        (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ): ".docx",
        "application/vnd.oasis.opendocument.text": ".odt",
        "application/rtf": ".rtf",
        "application/msword": ".doc",
    }
)


def document_kind(mime: str, filename: str = "") -> str:
    """``pdf``, ``office``, ``text`` or ``other`` from mime and extension."""
    lowered = mime.split(";", 1)[0].strip().casefold()
    suffix = Path(filename).suffix.casefold()
    if lowered in PDF_MIMES or suffix in PDF_EXTENSIONS:
        return "pdf"
    if lowered in OFFICE_MIMES or suffix in OFFICE_EXTENSIONS:
        return "office"
    if lowered in TEXT_MIMES or suffix in TEXT_EXTENSIONS:
        return "text"
    return "other"


def is_document(mime: str, filename: str = "") -> bool:
    """Whether an attachment is a supported document."""
    return document_kind(mime, filename) != "other"


def bounded(
    text: str,
    status: str,
    pages: int,
    filename: str,
    mime: str,
    limits: DocumentLimits,
    notice: str = "",
) -> DocumentText:
    """Apply the character cap and build the result."""
    clean, dropped = trim_text(text, limits.max_chars)
    if dropped:
        return DocumentText(
            text=clean,
            status=STATUS_OVER_LIMIT,
            pages=pages,
            notice=over_limit_notice(filename),
            filename=filename,
            mime=mime,
        )
    if not clean:
        return DocumentText(
            text="",
            status=STATUS_EMPTY,
            pages=pages,
            notice=notice or unreadable_notice(filename),
            filename=filename,
            mime=mime,
        )
    return DocumentText(
        text=clean,
        status=status,
        pages=pages,
        notice=notice,
        filename=filename,
        mime=mime,
    )


def apply_page_limit(
    result: DocumentText, total_pages: int, filename: str, limits: DocumentLimits
) -> DocumentText:
    """Flag a readable result when only the first pages were extracted."""
    if total_pages <= limits.max_pages or not result.text:
        return result
    return replace(
        result,
        status=STATUS_OVER_LIMIT,
        pages=total_pages,
        notice=page_limit_notice(filename, limits.max_pages, total_pages),
    )


def extract_pdf(
    path: Path, filename: str, mime: str, limits: DocumentLimits
) -> DocumentText:
    """Extract PDF text, falling back to page OCR."""
    total_pages = pdf_page_count(path, limits)
    text, pages, error = pdf_text(path, limits)
    if text.strip():
        return apply_page_limit(
            bounded(text, STATUS_OK, pages, filename, mime, limits),
            total_pages,
            filename,
            limits,
        )
    ocr_text, ocr_pages, ocr_error = pdf_ocr(path, limits)
    if ocr_text.strip():
        return apply_page_limit(
            bounded(ocr_text, STATUS_OCR, ocr_pages, filename, mime, limits),
            total_pages,
            filename,
            limits,
        )
    notice = unreadable_notice(filename)
    if error or ocr_error:
        notice = f"{notice} ({error or ocr_error})"[:500]
    return DocumentText(
        text="",
        status=STATUS_UNREADABLE,
        pages=ocr_pages or pages or total_pages,
        notice=notice,
        filename=filename,
        mime=mime,
    )


def extract_document(
    path: Path,
    mime: str,
    filename: str = "",
    limits: DocumentLimits | None = None,
) -> DocumentText:
    """Extract one supported document within the hard limits."""
    caps = limits or DocumentLimits()
    label = filename or path.name
    problem = document_problem(path, label, mime, caps)
    if problem is not None:
        return problem
    kind = document_kind(mime, label)
    if kind == "pdf":
        return extract_pdf(path, label, mime, caps)
    if kind == "office":
        return extract_office_document(path, label, mime, caps)
    if kind == "text":
        return extract_text_document(path, label, mime, caps)
    return unsupported_document(label, mime)


def document_problem(
    path: Path, label: str, mime: str, caps: DocumentLimits
) -> DocumentText | None:
    """A pre-extraction failure (missing file or over the byte cap)."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        return DocumentText(
            text="",
            status=STATUS_UNREADABLE,
            notice=f"could not open the document ({exc})"[:300],
            filename=label,
            mime=mime,
        )
    if size > caps.max_bytes:
        return DocumentText(
            text="",
            status=STATUS_OVER_LIMIT,
            notice=over_limit_notice(label),
            filename=label,
            mime=mime,
        )
    return None


def extract_office_document(
    path: Path, label: str, mime: str, caps: DocumentLimits
) -> DocumentText:
    """Convert an office document with pandoc."""
    text, error = pandoc_text(path, caps)
    if error and not text.strip():
        return DocumentText(
            text="",
            status=STATUS_UNREADABLE,
            notice=f"{unreadable_notice(label)} ({error})"[:500],
            filename=label,
            mime=mime,
        )
    return bounded(text, STATUS_OK, 0, label, mime, caps)


def extract_text_document(
    path: Path, label: str, mime: str, caps: DocumentLimits
) -> DocumentText:
    """Read a plain text or CSV file."""
    text, error = plain_text(path, caps)
    if error:
        return DocumentText(
            text="",
            status=STATUS_UNREADABLE,
            notice=f"{unreadable_notice(label)} ({error})"[:500],
            filename=label,
            mime=mime,
        )
    return bounded(text, STATUS_OK, 0, label, mime, caps)


def unsupported_document(label: str, mime: str) -> DocumentText:
    """The result for a type no extractor handles."""
    return DocumentText(
        text="",
        status=STATUS_UNREADABLE,
        notice=unreadable_notice(label),
        filename=label,
        mime=mime,
    )


def suffix_for(mime: str, filename: str = "") -> str:
    """A safe temporary suffix for extraction by extension."""
    existing = Path(filename).suffix
    if existing and len(existing) <= 8:
        return existing.casefold()
    return MIME_SUFFIXES.get(mime.split(";", 1)[0].strip().casefold(), ".bin")


def extract_bytes(
    data: bytes,
    mime: str,
    filename: str = "",
    limits: DocumentLimits | None = None,
) -> DocumentText:
    """Extract a downloaded document from memory through a temp file."""
    caps = limits or DocumentLimits()
    label = filename or "document"
    if len(data) > caps.max_bytes:
        return DocumentText(
            text="",
            status=STATUS_OVER_LIMIT,
            notice=over_limit_notice(label),
            filename=label,
            mime=mime,
        )
    temp_path = Path()
    try:
        with tempfile.NamedTemporaryFile(
            prefix="wahabot-doc-", suffix=suffix_for(mime, filename), delete=False
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(data)
        return extract_document(temp_path, mime, label, caps)
    finally:
        temp_path.unlink(missing_ok=True)
