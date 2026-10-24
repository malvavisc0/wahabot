"""Document extraction: generated fixtures, OCR, limits and turn rendering."""

import asyncio
import shutil
import zipfile
from pathlib import Path

import pytest

from wahabot.ai.serials import SerialRegistry
from wahabot.ai.turn_records import RecordEntry, Slots, render_entry
from wahabot.core.documents.extract import (
    DocumentLimits,
    apply_page_limit,
    extract_document,
)
from wahabot.core.documents.limits import (
    DOCUMENT_MARKER_KEY,
    DOCUMENT_TEXT_KEY,
    DocumentText,
    document_marker,
    document_text_payload,
    page_limit_notice,
)
from wahabot.core.documents.pdf import run_command
from wahabot.core.models import WahaEvent
from wahabot.core.tickets.attachments import (
    attachment_metadata,
    persist_bytes,
    persist_event_attachments,
)
from wahabot.core.tickets.models import Ticket
from wahabot.core.tickets.store import TicketStore
from wahabot.core.waha import MediaTooLargeError
from wahabot.handlers.single_mixin import SingleMixin
from wahabot.settings import Settings


def write_text_pdf(path: Path, text: str) -> None:
    """A one-page PDF with real selectable text."""
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        None,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode()
    objects[3] = (
        f"<< /Length {len(stream)} >>\nstream\n" + stream.decode() + "\nendstream"
    )
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{index} 0 obj\n{body}\nendobj\n".encode())
    xref = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode())
    out.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode()
    )
    path.write_bytes(bytes(out))


def write_docx(path: Path, text: str) -> None:
    """A minimal DOCX with one paragraph."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types"><Default Extension="rels" ContentType="'
            'application/vnd.openxmlformats-package.relationships+xml"/><Default '
            'Extension="xml" ContentType="application/xml"/><Override PartName="'
            '/word/document.xml" ContentType="application/vnd.openxmlformats-'
            'officedocument.wordprocessingml.document.main+xml"/></Types>',
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.'
            'openxmlformats.org/package/2006/relationships"><Relationship Id="rId1"'
            ' Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            'relationships/officeDocument" Target="word/document.xml"/></Relationships>',
        )
        archive.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.'
            'openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>'
            + text
            + "</w:t></w:r></w:p></w:body></w:document>",
        )


def test_pdf_text_plain_text_and_unreadable(tmp_path: Path) -> None:
    if shutil.which("pdftotext") is None:
        pytest.skip("pdftotext is not installed")
    pdf = tmp_path / "policy.pdf"
    write_text_pdf(pdf, "Refund policy thirty days")
    result = extract_document(pdf, "application/pdf", "policy.pdf")
    assert result.status == "ok"
    assert "Refund" in result.text
    assert result.pages == 1

    text = tmp_path / "note.csv"
    text.write_text("order,status\n42,open\n", encoding="utf-8")
    result = extract_document(text, "text/csv", "note.csv")
    assert "order,status" in result.text

    unknown = tmp_path / "archive.bin"
    unknown.write_bytes(b"\x00\x01")
    result = extract_document(unknown, "application/octet-stream", "archive.bin")
    assert result.status == "unreadable" and result.notice


def test_office_conversion_uses_pandoc(tmp_path: Path) -> None:
    if shutil.which("pandoc") is None:
        pytest.skip("pandoc is not installed")
    docx = tmp_path / "notes.docx"
    write_docx(docx, "Delivery arrives Friday")
    result = extract_document(
        docx,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "notes.docx",
    )
    assert "Delivery" in result.text


def test_scanned_pdf_falls_back_to_ocr(tmp_path: Path) -> None:
    if shutil.which("pdftoppm") is None or shutil.which("tesseract") is None:
        pytest.skip("poppler/tesseract are not installed")
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (800, 240), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=64)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((40, 80), "INVOICE 42", fill="black", font=font)
    pdf = tmp_path / "scan.pdf"
    image.save(pdf, "PDF")
    result = extract_document(pdf, "application/pdf", "scan.pdf")
    assert result.status == "ocr"
    assert "INVOICE" in result.text.upper()


def test_limits_truncate_and_binary_timeout(tmp_path: Path) -> None:
    text = tmp_path / "big.txt"
    text.write_text("x" * 9000, encoding="utf-8")
    result = extract_document(
        text,
        "text/plain",
        "big.txt",
        DocumentLimits(max_chars=100, max_pages=5, timeout_s=20),
    )
    assert len(result.text) <= 100
    assert result.status == "over_limit" and result.notice

    code, _, error = run_command(["sleep", "2"], 0.01)
    assert code == -2 and "timed out" in error


def test_over_limit_payload_keeps_the_notice() -> None:
    result = DocumentText(
        text="only the beginning",
        status="over_limit",
        pages=100,
        notice="The document has 100 pages; only the first 20 were read.",
        filename="policy.pdf",
        mime="application/pdf",
    )
    payload = document_text_payload(result)
    assert "only the beginning" in payload
    assert "only the first 20 were read" in payload


def test_page_limit_marks_the_result() -> None:
    result = DocumentText(
        text="page text",
        status="ok",
        pages=20,
        filename="policy.pdf",
        mime="application/pdf",
    )
    limited = apply_page_limit(result, 100, "policy.pdf", DocumentLimits(max_pages=20))
    assert limited.status == "over_limit"
    assert limited.pages == 100
    assert limited.notice == page_limit_notice("policy.pdf", 20, 100)


def test_uncaptioned_document_reaches_the_agent_gate() -> None:
    event = WahaEvent.model_validate(
        {
            "event": "message",
            "session": "default",
            "payload": {
                "id": "false_123@g.us_DOC",
                "from": "123@g.us",
                "fromMe": False,
                "body": "",
                "timestamp": 1,
                "_data": {"type": "document"},
            },
        }
    )

    class Gate:
        settled = False

        async def ticket_gate_events(self, events: list[WahaEvent]) -> bool:
            events[0].payload[DOCUMENT_MARKER_KEY] = "[document: policy.pdf]"
            return False

        def settle_intake(self, *_args: object, **_kwargs: object) -> None:
            self.settled = True

    gate = Gate()
    handled = asyncio.run(
        SingleMixin.ticket_or_content_gate(gate, event, None, None, None, None)  # type: ignore[arg-type]
    )
    assert handled is False
    assert gate.settled is False


def test_empty_turn_without_a_document_still_settles_silent() -> None:
    event = WahaEvent.model_validate(
        {
            "event": "message",
            "session": "default",
            "payload": {
                "id": "false_123@g.us_EMPTY",
                "from": "123@g.us",
                "fromMe": False,
                "body": "",
                "timestamp": 1,
                "_data": {"type": "text"},
            },
        }
    )

    class Gate:
        settled = False

        async def ticket_gate_events(self, _events: list[WahaEvent]) -> bool:
            return False

        def settle_intake(self, *_args: object, **_kwargs: object) -> None:
            self.settled = True

    gate = Gate()
    handled = asyncio.run(
        SingleMixin.ticket_or_content_gate(gate, event, None, None, None, None)  # type: ignore[arg-type]
    )
    assert handled is True
    assert gate.settled is True


def test_too_large_download_marks_the_turn(
    store: TicketStore, settings: Settings
) -> None:
    contact = store.upsert_contact("491555000001@c.us")
    ticket = store.create_ticket(contact.id, "491555000001@c.us")
    event = WahaEvent.model_validate(
        {
            "event": "message",
            "session": "default",
            "payload": {
                "id": "msg-big",
                "from": contact.chat_id,
                "fromMe": False,
                "body": "",
                "hasMedia": True,
                "media": {
                    "url": "http://example.test/big.pdf",
                    "mimetype": "application/pdf",
                    "filename": "big.pdf",
                },
                "_data": {"type": "document"},
            },
        }
    )

    class TooLarge:
        def download_media(self, *_args: object, **_kwargs: object) -> bytes:
            raise MediaTooLargeError("http://example.test/big.pdf", 1)

    assert persist_event_attachments(store, TooLarge(), ticket, event, settings) == []  # type: ignore[arg-type]
    assert "over_limit" in str(event.payload[DOCUMENT_MARKER_KEY])
    assert "too large" in str(event.payload[DOCUMENT_TEXT_KEY]).casefold()


def test_document_payload_and_attachment_page_count(
    store: TicketStore, settings: Settings
) -> None:
    contact = store.upsert_contact("491555000001@c.us", display_name="Smoke Sender")
    ticket: Ticket = store.create_ticket(contact.id, "491555000001@c.us")
    persist_bytes(
        store,
        ticket,
        "msg-doc",
        "file",
        "application/pdf",
        b"%PDF-fake",
        settings,
        derived_text="Extracted body",
        page_count=3,
    )
    attachments = store.attachments(ticket.id)
    assert attachments[0].page_count == 3
    assert attachment_metadata(attachments)[0]["page_count"] == 3

    result = DocumentText(
        text="Body text",
        status="ok",
        pages=3,
        filename="policy.pdf",
        mime="application/pdf",
    )
    event = WahaEvent.model_validate(
        {
            "event": "message",
            "session": "default",
            "payload": {
                "id": "false_123@g.us_DOC",
                "from": "123@g.us",
                "fromMe": False,
                "body": "read this",
                "timestamp": 1,
                DOCUMENT_MARKER_KEY: document_marker(result),
                DOCUMENT_TEXT_KEY: document_text_payload(result),
                "_data": {"type": "document", "notifyName": "Smoke Sender"},
            },
        }
    )
    registry = SerialRegistry(store.path.parent / "serials.jsonl")
    line, _, _ = render_entry(
        RecordEntry(event=event),
        registry=registry,
        names={},
        tz_name="UTC",
        slots=Slots(),
    )
    assert "document#1" in line
    assert "policy.pdf" in line
    assert "Body text" in line
