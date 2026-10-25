"""Authenticated attachment streaming; no ticket page redirect."""

import re
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, Response

from wahabot.web.deps import current_settings, media_auth, ticket_store_for

router = APIRouter()
SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def inside_data_dir(path: Path, root: Path) -> bool:
    """Whether an attachment path stays under the configured data directory."""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError, ValueError:
        return False


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
    inline = stored.kind in {"image", "video", "voice", "audio"}
    disposition = "inline" if inline else "attachment"
    headers = {
        "Content-Disposition": (
            f'{disposition}; filename="{filename}"; '
            + f"filename*=UTF-8''{quote(filename)}"
        )
    }
    return FileResponse(
        path,
        media_type=stored.mime or "application/octet-stream",
        headers=headers,
    )
