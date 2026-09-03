"""Sniff image URLs out of message text and fetch them for vision.

A bare link in a chat ("look at this
https://host/path/pic.png/revision/latest") is just text to the
handler — this module finds those URLs, streams the bytes with the
same Chrome TLS impersonation ``visit_url`` uses (many CDNs block bare
clients), and hands them to the workflow as image blocks so the model
actually sees the picture. The Content-Type header is the source of
truth for "is this an image" (the wikia example above serves
``image/webp`` with a ``.png`` path); extensions only steer the
initial URL pick, never the verdict.
"""

import re
from typing import Any
from urllib.parse import urlsplit

from curl_cffi import requests as cffi_requests
from loguru import logger

from wahabot.settings import Settings

__all__ = ["fetch_url_images", "image_urls"]

_IMPERSONATE = "chrome"

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_IMAGE_EXT_RE = re.compile(r"\.(png|jpe?g|gif|webp|bmp|avif)(?=$|[/?#])", re.IGNORECASE)

#: Content-Types worth downloading when the URL carries no image extension.
_SNIFFABLE_CONTENT_TYPES = (
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/avif",
)


def image_urls(text: str, limit: int) -> list[str]:
    """Extract up to *limit* plausible image URLs from *text*.

    An URL qualifies when any path segment ends in an image extension —
    ``pic.png``, but also ``pic.png/revision/latest`` (wikia-style
    derivative paths). URLs with a known non-image extension are
    excluded so an HTML page linked as ``.html`` is never fetched.
    """
    urls: list[str] = []
    for match in _URL_RE.finditer(text):
        url = match.group().rstrip(").,;:!?\"'>]}")
        path = urlsplit(url).path
        if _IMAGE_EXT_RE.search(path) and url not in urls:
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def fetch_url_images(settings: Settings, urls: list[str]) -> list[dict[str, Any]]:
    """Download image bytes for *urls*, skipping failures.

    Returns ``{"data": bytes, "mimetype": str}`` dicts for the URLs that
    turned out to serve images under ``max_image_bytes``; a miss costs
    nothing beyond a log line. Content-Type decides image-ness.
    """
    images: list[dict[str, Any]] = []
    for url in urls[: max(settings.max_url_images, 0)]:
        image = _fetch_one(settings, url)
        if image is not None:
            images.append(image)
    return images


def _fetch_one(settings: Settings, url: str) -> dict[str, Any] | None:
    """Stream one URL; None unless it is an image within the size cap."""
    kwargs: dict[str, Any] = {"impersonate": _IMPERSONATE}
    if settings.web_search_proxy:
        kwargs["proxies"] = {
            "http": settings.web_search_proxy,
            "https": settings.web_search_proxy,
        }
    timeout = max(settings.web_search_timeout, 2.0)
    response = None
    try:
        response = cffi_requests.get(url, stream=True, timeout=timeout, **kwargs)
        response.raise_for_status()
        mimetype = _image_mimetype(response.headers.get("content-type", ""))
        if mimetype is None:
            logger.debug(
                "URL {url} is not an image ({ctype})",
                url=url,
                ctype=response.headers.get("content-type", "?"),
            )
            return None
        data = _read_capped(response, settings.max_image_bytes)
    except Exception as exc:
        logger.warning("Image URL fetch failed for {url}: {exc}", url=url, exc=exc)
        return None
    finally:
        if response is not None:
            response.close()
    logger.info(
        "Fetched image URL {url} ({mime}, {size} B)",
        url=url,
        mime=mimetype,
        size=len(data),
    )
    return {"data": data, "mimetype": mimetype}


def _image_mimetype(content_type: str) -> str | None:
    """The image/* mimetype a Content-Type header describes, or None."""
    mimetype = content_type.split(";", 1)[0].strip().lower()
    return mimetype if mimetype in _SNIFFABLE_CONTENT_TYPES else None


def _read_capped(response: Any, max_bytes: int) -> bytes:
    """Read a streamed body, aborting with an error once past *max_bytes*."""
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_content():
        size += len(chunk)
        if size > max_bytes:
            raise OSError(f"image exceeds {max_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)
