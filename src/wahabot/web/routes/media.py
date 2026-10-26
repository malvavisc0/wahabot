"""Authenticated attachment streaming; no ticket page redirect."""

import re
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, Response

from wahabot.web.deps import current_settings, media_auth, ticket_store_for

router = APIRouter()
SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
INLINE_KINDS = frozenset({"image", "video", "voice", "audio"})
INLINE_MIME = "application/pdf"
UNSAFE_INLINE_MIME = "image/svg+xml"


def inside_data_dir(path: Path, root: Path) -> bool:
    """Whether an attachment path stays under the configured data directory."""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError, ValueError:
        return False


def base_mime(mime: str) -> str:
    """The MIME type without parameters or whitespace."""
    return mime.split(";", 1)[0].strip().lower()


def safe_mime(base: str) -> bool:
    """Whether a normalized MIME type may be rendered inline by a browser.

    The MIME comes from the inbound payload, so anything the browser
    would execute (HTML, SVG) must never be served same-origin inline.
    """
    if not base or base == UNSAFE_INLINE_MIME:
        return False
    return base.startswith(("image/", "video/", "audio/")) or base == INLINE_MIME


@router.api_route("/app/media/{attachment_id}", methods=["GET", "HEAD"])
def media(request: Request, attachment_id: int) -> Response:
    """Stream one stored attachment only to a signed-in browser."""
    settings = current_settings()
    refused = media_auth(request, settings)
    if refused is not None:
        return refused
    stored = ticket_store_for(settings).attachment(attachment_id)
    if stored is None or not stored.path:
        return Response(status_code=404)
    path = Path(stored.path)
    if not inside_data_dir(path, settings.data_dir) or not path.is_file():
        return Response(status_code=404)
    filename = SAFE_FILENAME.sub("_", path.name) or "attachment"
    base = base_mime(stored.mime)
    inline = stored.kind in INLINE_KINDS and safe_mime(base)
    disposition = "inline" if inline else "attachment"
    media_type = base if inline else "application/octet-stream"
    headers = {
        "Content-Disposition": (
            f'{disposition}; filename="{filename}"; '
            + f"filename*=UTF-8''{quote(filename)}"
        ),
        "Content-Security-Policy": "sandbox",
    }
    return FileResponse(
        path,
        media_type=media_type,
        headers=headers,
    )
