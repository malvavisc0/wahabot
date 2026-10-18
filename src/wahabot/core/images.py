"""Image generation and editing via the OpenAI-compatible image service.

Expression counterpart to :mod:`wahabot.core.tts`: sync functions
called from the tool fns' worker threads like every other network call,
fail-soft — a render failure must degrade to the model's own message,
never crash the run.
"""

import base64
import struct
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from wahabot.settings import Settings

__all__ = ["edit", "generate", "load_source", "metadata", "prune_images", "save"]

#: Generated images are scratch; ones older than this are pruned on save.
IMAGE_TTL_S = 24 * 3600
#: Cap on a fetched source image for an edit.
MAX_SOURCE_BYTES = 10 * 1024 * 1024


def generate(settings: Settings, prompt: str, size: str) -> bytes | None:
    """Render *prompt* as PNG bytes; None on any failure.

    ``POST {image_api_base}/v1/images/generations`` with the OpenAI
    image shape. The service does not expand prompts, so *prompt* must
    already be a full scene caption.
    """
    if not settings.image_api_base:
        return None
    url = f"{settings.image_api_base.rstrip('/')}/v1/images/generations"
    return post_image(
        settings,
        lambda client: client.post(url, json={"prompt": prompt, "size": size}),
        "Image generation",
    )


def edit(
    settings: Settings,
    prompt: str,
    name: str,
    data: bytes,
    mimetype: str,
    strength: float | None,
) -> bytes | None:
    """Repaint *data* per *prompt*; None on any failure.

    ``POST {image_api_base}/v1/images/edits`` (multipart): *prompt*
    describes the finished scene, not only the change. *strength* rides
    the inline override tag the service parses.
    """
    if not settings.image_api_base:
        return None
    url = f"{settings.image_api_base.rstrip('/')}/v1/images/edits"
    text = prompt
    if strength is not None:
        text = (
            f'{prompt} <sd_cpp_extra_args>{{"strength":{strength}}}</sd_cpp_extra_args>'
        )
    files = {"image": (name or "source.png", data, mimetype or "image/png")}
    return post_image(
        settings,
        lambda client: client.post(url, data={"prompt": text}, files=files),
        "Image edit",
    )


def post_image(
    settings: Settings,
    send: Callable[[httpx.Client], httpx.Response],
    label: str,
) -> bytes | None:
    """POST through *send* and decode the OpenAI image envelope; None on failure."""
    try:
        with httpx.Client(timeout=settings.image_timeout) as client:
            response = send(client)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("{label} failed: {exc}", label=label, exc=exc)
        return None
    return decode(payload)


def decode(payload: Any) -> bytes | None:
    """The base64 PNG in an OpenAI image envelope, or None."""
    try:
        encoded = payload["data"][0]["b64_json"]
        data = base64.b64decode(encoded)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning("Image response carried no image: {exc}", exc=exc)
        return None
    if not data:
        logger.warning("Image service returned an empty image")
        return None
    return data


def load_source(
    source: str, max_bytes: int = MAX_SOURCE_BYTES
) -> tuple[str, bytes, str] | str:
    """Read a source image from a local path or http(s) URL.

    Returns ``(name, data, mimetype)`` or an error string. The mimetype
    is sniffed from the content, not the filename: the temp store names
    everything ``.png``, so an inbound JPEG (saved as ``gen-*.png``) must
    still be declared as one. The size cap is a refusal, never a
    truncation — a cut file would not decode.
    """
    if "://" in source:
        name = source.split("?", 1)[0].rsplit("/", 1)[-1] or "source.png"
        try:
            with httpx.Client(timeout=60) as client:
                fetched = fetch_capped(client, source, max_bytes)
        except httpx.HTTPError as exc:
            return f"could not fetch the source image: {exc}"
        if isinstance(fetched, str):
            return fetched
        data = fetched
    else:
        path = Path(source)
        name = path.name or "source.png"
        try:
            data = path.read_bytes()
        except OSError as exc:
            return f"could not read the source image: {exc}"
        if len(data) > max_bytes:
            return f"source image exceeds {max_bytes} bytes"
    return name, data, sniff_mime(data)


def fetch_capped(client: httpx.Client, url: str, max_bytes: int) -> bytes | str:
    """Download *url*, refusing (not truncating) past *max_bytes*."""
    chunks: list[bytes] = []
    total = 0
    with client.stream("GET", url) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > max_bytes:
                return f"source image exceeds {max_bytes} bytes"
            chunks.append(chunk)
    return b"".join(chunks)


def sniff_mime(data: bytes) -> str:
    """Image format from magic bytes (default PNG)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/png"


def save(data: bytes) -> Path:
    """Write rendered PNG *data* to the temp image directory; return its path.

    Generated images are scratch: they exist to be sent, not kept, so
    they land in a temp directory rather than the data dir. Stale ones
    are pruned on each save so the directory cannot grow unbounded.
    """
    directory = Path(tempfile.gettempdir()) / "wahabot-images"
    directory.mkdir(parents=True, exist_ok=True)
    prune_images(directory, time.time())
    path = directory / f"gen-{time.time_ns()}.png"
    path.write_bytes(data)
    return path


def prune_images(directory: Path, now: float) -> None:
    """Delete generated images older than ``IMAGE_TTL_S``; best-effort."""
    for path in directory.glob("gen-*.png"):
        try:
            if now - path.stat().st_mtime > IMAGE_TTL_S:
                path.unlink()
        except OSError:
            continue


def metadata(data: bytes) -> dict[str, Any]:
    """Format, pixel dimensions, and byte size of a rendered image."""
    info: dict[str, Any] = {
        "bytes": len(data),
        "format": "unknown",
        "width": 0,
        "height": 0,
    }
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        info.update(format="png", width=width, height=height)
    return info
