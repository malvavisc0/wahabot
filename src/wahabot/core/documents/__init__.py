"""Documents: bounded extraction and turn markers."""

from wahabot.core.documents.extract import (
    document_kind,
    extract_bytes,
    extract_document,
    is_document,
    suffix_for,
)
from wahabot.core.documents.limits import (
    DOCUMENT_MARKER_KEY,
    DOCUMENT_TEXT_KEY,
    STATUS_EMPTY,
    STATUS_OCR,
    STATUS_OK,
    STATUS_OVER_LIMIT,
    STATUS_UNREADABLE,
    DocumentLimits,
    DocumentText,
    document_marker,
    document_text_payload,
    over_limit_notice,
    trim_text,
    unreadable_notice,
)

__all__ = [
    "DOCUMENT_MARKER_KEY",
    "DOCUMENT_TEXT_KEY",
    "STATUS_EMPTY",
    "STATUS_OCR",
    "STATUS_OK",
    "STATUS_OVER_LIMIT",
    "STATUS_UNREADABLE",
    "DocumentLimits",
    "DocumentText",
    "document_kind",
    "document_marker",
    "document_text_payload",
    "extract_bytes",
    "extract_document",
    "is_document",
    "over_limit_notice",
    "suffix_for",
    "trim_text",
    "unreadable_notice",
]
