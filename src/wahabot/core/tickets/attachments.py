"""Persist customer media locally and link it to its ticket."""

import datetime
import mimetypes
from pathlib import Path

from loguru import logger

from wahabot.core.documents.extract import extract_bytes, is_document
from wahabot.core.documents.limits import (
    DOCUMENT_MARKER_KEY,
    DOCUMENT_TEXT_KEY,
    STATUS_UNREADABLE,
    DocumentLimits,
    DocumentText,
    document_marker,
    document_text_payload,
    mark_document_failure,
    unreadable_notice,
)
from wahabot.core.models import WahaEvent
from wahabot.core.tickets.models import (
    Actor,
    Attachment,
    EventKind,
    Ticket,
    text_field,
)
from wahabot.core.tickets.store import TicketStore
from wahabot.core.tickets.store_base import digest
from wahabot.core.transport import Transport
from wahabot.core.waha import MediaTooLargeError
from wahabot.settings import Settings

_MIME_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "audio/ogg": ".oga",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "video/mp4": ".mp4",
    "application/pdf": ".pdf",
}


def media_kind(event: WahaEvent) -> str:
    """The attachment kind carried by one event, or "" when it has none."""
    if not event.payload.get("hasMedia"):
        return ""
    return kind_from_parts(event_declared_type(event), media_mime(event))


def event_declared_type(event: WahaEvent) -> str:
    """WAHA's declared media type, or ""."""
    return text_field(event.payload.get("_data"), "type")


def kind_from_parts(declared: str, mimetype: str) -> str:
    """Map a declared type and mimetype to one attachment kind."""
    if declared in ("ptt", "audio", "voice") or mimetype.startswith("audio/"):
        return "voice"
    if declared in ("image", "sticker") or mimetype.startswith("image/"):
        return "image"
    if declared == "video" or mimetype.startswith("video/"):
        return "video"
    return "file"


def media_url(event: WahaEvent) -> str:
    """The event's media URL, or "" when it has none."""
    return text_field(event.payload.get("media"), "url")


def media_mime(event: WahaEvent) -> str:
    """The event's media mimetype, with a generic fallback."""
    return text_field(event.payload.get("media"), "mimetype")


def media_filename(event: WahaEvent) -> str:
    """The sender's document filename, or ""."""
    return text_field(event.payload.get("media"), "filename") or text_field(
        event.payload.get("_data"), "filename"
    )


def byte_cap(kind: str, settings: Settings) -> int:
    """The download cap for one media kind."""
    return {
        "image": settings.max_image_bytes,
        "voice": settings.max_audio_bytes,
        "video": settings.max_video_bytes,
    }.get(kind, settings.max_file_bytes)


def extension_for(mime: str, url: str) -> str:
    """A file extension for the stored copy."""
    base = mime.split(";", 1)[0].strip().lower()
    if base in _MIME_EXTENSIONS:
        return _MIME_EXTENSIONS[base]
    guessed = mimetypes.guess_extension(base) if base else None
    if guessed:
        return guessed
    suffix = Path(url.split("?", 1)[0]).suffix
    return suffix if suffix and len(suffix) <= 8 else ".bin"


def media_path(settings: Settings, sha256: str, extension: str, at: int) -> Path:
    """The dated local path for one stored attachment."""
    moment = datetime.datetime.fromtimestamp(at, tz=datetime.UTC)
    return (
        settings.data_dir
        / "media"
        / f"{moment:%Y}"
        / f"{moment:%m}"
        / f"{sha256}{extension}"
    )


def download_event_media(
    waha: Transport,
    event: WahaEvent,
    settings: Settings,
    kind: str,
    url: str,
    mime: str,
    filename: str,
) -> bytes | None:
    """Download one event's media, marking document failures on the turn."""
    message_id = str(event.payload.get("id", ""))
    try:
        return waha.download_media(url, max_bytes=byte_cap(kind, settings))
    except Exception as exc:
        logger.warning(
            "Ticket attachment download failed for {id}: {exc}", id=message_id, exc=exc
        )
        if kind == "file" and is_document(mime, filename):
            mark_document_failure(
                event, filename, mime, too_large=isinstance(exc, MediaTooLargeError)
            )
        return None


def extract_document_bytes(
    data: bytes, mime: str, filename: str, settings: Settings
) -> DocumentText:
    """Extract one document, falling back to an unreadable notice."""
    try:
        return extract_bytes(
            data,
            mime,
            filename,
            DocumentLimits.from_settings(settings),
        )
    except Exception as exc:
        logger.warning(
            "Document extraction failed for {name}: {exc}", name=filename, exc=exc
        )
        return DocumentText(
            text="",
            status=STATUS_UNREADABLE,
            notice=unreadable_notice(filename),
            filename=filename,
            mime=mime,
        )


def persist_event_attachments(
    store: TicketStore,
    waha: Transport,
    ticket: Ticket,
    event: WahaEvent,
    settings: Settings,
    *,
    at: int | None = None,
) -> list[Attachment]:
    """Download and store one event's media, deduped by content hash."""
    kind = media_kind(event)
    url = media_url(event)
    if not kind or not url:
        return []
    message_id = str(event.payload.get("id", ""))
    mime = media_mime(event)
    filename = media_filename(event)
    document = kind == "file" and is_document(mime, filename)
    data = download_event_media(waha, event, settings, kind, url, mime, filename)
    if data is None:
        return []
    derived_text = str(event.payload.get("body", "") or "")
    page_count = 0
    if document:
        result = extract_document_bytes(data, mime, filename, settings)
        derived_text = result.text or result.notice
        if result.notice and result.notice not in derived_text:
            derived_text = f"{derived_text}\n\n{result.notice}".strip()
        page_count = result.pages
        event.payload[DOCUMENT_MARKER_KEY] = document_marker(result)
        event.payload[DOCUMENT_TEXT_KEY] = document_text_payload(result)
    return persist_bytes(
        store,
        ticket,
        message_id,
        kind,
        mime,
        data,
        settings,
        derived_text=derived_text,
        page_count=page_count,
        at=at,
    )


def persist_bytes(
    store: TicketStore,
    ticket: Ticket,
    message_id: str,
    kind: str,
    mime: str,
    data: bytes,
    settings: Settings,
    *,
    derived_text: str = "",
    page_count: int = 0,
    at: int | None = None,
) -> list[Attachment]:
    """Store already-downloaded bytes, linking them to ticket and message."""
    if not data:
        return []
    sha256 = digest(data)
    existing = [item for item in store.attachments(ticket.id) if item.sha256 == sha256]
    if any(item.message_id == message_id for item in existing):
        return []
    moment = store.now() if at is None else at
    if existing:
        path = Path(existing[0].path)
    else:
        path = media_path(settings, sha256, extension_for(mime, message_id), moment)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
    attachment = store.add_attachment(
        ticket.id,
        message_id,
        kind,
        mime=mime,
        size=len(data),
        sha256=sha256,
        path=str(path),
        derived_text=derived_text,
        page_count=page_count,
    )
    store.add_event(
        ticket.id,
        Actor.CUSTOMER.value,
        EventKind.ATTACHMENT.value,
        derived_text,
        [message_id],
        at=moment,
    )
    return [attachment]


def attachment_metadata(attachments: list[Attachment]) -> list[dict[str, object]]:
    """Model/operator-facing attachment facts without local paths."""
    return [
        {
            "kind": item.kind,
            "mime": item.mime,
            "bytes": item.bytes,
            "message_id": item.message_id,
            "text": item.derived_text,
            "page_count": item.page_count,
        }
        for item in attachments
    ]
