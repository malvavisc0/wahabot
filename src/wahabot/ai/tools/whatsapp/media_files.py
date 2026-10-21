import base64
import io
import mimetypes
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from PIL import Image

_UNCAPTIONABLE_KINDS = frozenset({"voice", "sticker"})

_IMAGE_MIME_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".svg": "image/svg+xml",
    ".avif": "image/avif",
}

_DOC_MIME_BY_EXT: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".zip": "application/zip",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_VIDEO_MIME_BY_EXT: dict[str, str] = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".3gp": "video/3gpp",
    ".3g2": "video/3gpp2",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".ogv": "video/ogg",
}

_AUDIO_MIME_BY_EXT: dict[str, str] = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}


def normalize_media_source(source: str | None) -> str | None:
    """Strip a media source, treating missing or blank values as absent."""
    if source is None:
        return None
    return str(source).strip() or None


def media_source_error(
    url: str | None, path: str | None, text: str | None, kind: str
) -> str | None:
    """The argument-error message for send_media's source, or None.

    Exactly one source (url/path/text) per call, none blank (the fn
    normalizes blank to None first); `text` is only valid for
    ``kind=voice`` — a uniform rule with no odd-one-out (bug 1's
    lesson).
    """
    if sum(source is not None for source in (url, path, text)) != 1:
        return "pass exactly one of url, path or text"
    if text is not None and kind != "voice":
        return "text is only valid for kind=voice"
    return None


def media_argument_error(
    url: str | None,
    path: str | None,
    text: str | None,
    kind: str,
    caption: str,
    filename: str | None,
    language: str | None,
) -> str | None:
    """Validate normalized media sources and kind-specific options in order."""
    source_error = media_source_error(url, path, text, kind)
    if source_error:
        return source_error
    if caption and kind in _UNCAPTIONABLE_KINDS:
        return f"caption is not supported for kind={kind}"
    if filename and kind != "file":
        return "filename is only valid for kind=file"
    if language and (kind != "voice" or text is None):
        return "language is only valid for kind=voice with text"
    return None


def image_file(name_or_url: str, max_file_bytes: int) -> dict[str, Any] | str:
    """A WAHA image payload for a URL or local path, or an error string.

    Same shape as :func:`video_file` but typed with the image MIME
    map; a shell-tool render (``.png``, ``.webp``) must not ride the
    wire stamped ``application/octet-stream``.
    """
    if "://" in name_or_url:
        return remote_file(name_or_url, _IMAGE_MIME_BY_EXT, "image/jpeg")
    loaded = local_file(name_or_url, max_file_bytes)
    if isinstance(loaded, str):
        return loaded
    return loaded | {
        "mimetype": infer_mimetype(name_or_url, _IMAGE_MIME_BY_EXT, "image/jpeg"),
    }


def infer_mimetype(name_or_url: str, curated: dict[str, str], default: str) -> str:
    """Best-effort mimetype from a filename's or URL's extension.

    Prefers the curated extension→MIME map (avoids unregistered types
    such as ``image/jpg`` and ``image/tif``, which WAHA/WhatsApp
    reject), then ``mimetypes``, then *default* when nothing is known.
    """
    path = PurePosixPath(urlsplit(name_or_url).path)
    ext = path.suffix.lower()
    mapped = curated.get(ext)
    if mapped:
        return mapped
    guessed = mimetypes.guess_type(path.name or "")[0]
    if guessed and default.split("/")[0] == "image" and not guessed.startswith("image/"):
        return default
    return guessed or default


def remote_file(
    url: str,
    curated: dict[str, str] | None = None,
    default: str = "application/octet-stream",
) -> dict[str, Any]:
    """A WAHA RemoteFile for a URL, typed by the extension of its path.

    *curated*/*default* select the MIME map — documents by default,
    videos pass the video map. This declares a best-effort MIME type;
    it neither downloads nor validates the remote content or size.
    """
    file: dict[str, Any] = {
        "mimetype": infer_mimetype(url, curated or _DOC_MIME_BY_EXT, default),
        "url": url,
    }
    name = PurePosixPath(urlsplit(url).path).name
    if name:
        file["filename"] = name
    return file


def local_file(path: str, max_file_bytes: int) -> dict[str, Any] | str:
    """A WAHA BinaryFile for a local path, or an error message string."""
    local = Path(path)
    try:
        data = local.read_bytes()
    except OSError as exc:
        return f"cannot read {path}: {exc}"
    if len(data) > max_file_bytes:
        return f"file is {len(data)} B, over the {max_file_bytes} B cap"
    return {
        "mimetype": infer_mimetype(
            local.name, _DOC_MIME_BY_EXT, "application/octet-stream"
        ),
        "filename": local.name,
        "data": base64.b64encode(data).decode(),
    }


def voice_payload(audio: bytes) -> dict[str, Any]:
    """A WAHA ``VoiceBinaryFile`` for synthesized mp3 bytes.

    No disk touch: the bytes the TTS service returned ride straight to
    the send as base64. The synthesized form is always mp3
    (``response_format``), so the mimetype is fixed; the filename is
    required by the wire schema (WAHA's convert path names its temp
    file from it).
    """
    return {
        "mimetype": "audio/mpeg",
        "filename": "voice-note.mp3",
        "data": base64.b64encode(audio).decode(),
    }


def voice_file(name_or_url: str, max_file_bytes: int) -> dict[str, Any] | str:
    """A WAHA voice payload for a URL or local path, or an error string.

    Same shape as :func:`video_file` but typed with the audio MIME map;
    a shell-tool render (``.mp3``, ``.wav``) must not ride the wire
    stamped ``application/octet-stream``.
    """
    if "://" in name_or_url:
        return remote_file(name_or_url, _AUDIO_MIME_BY_EXT, "audio/mpeg")
    loaded = local_file(name_or_url, max_file_bytes)
    if isinstance(loaded, str):
        return loaded
    return loaded | {
        "mimetype": infer_mimetype(name_or_url, _AUDIO_MIME_BY_EXT, "audio/mpeg")
    }


def squared_sticker_payload(path: str, max_sticker_bytes: int) -> dict[str, Any] | str:
    """A square WAHA sticker payload for a local path, or an error string.

    WhatsApp renders stickers on a square canvas: a 1080x1360 meme
    sent as-is arrives squashed, and the model had no way to know
    A non-square image is
    letterboxed onto a square canvas (white bars where the image does
    not reach) and re-encoded as WebP; a square one rides the wire
    unchanged. Unreadable/corrupt images surface PIL's error as the
    error string — the model gets a diagnosis, not a crash.
    """
    loaded = local_file(path, max_sticker_bytes)
    if isinstance(loaded, str):
        return loaded
    try:
        with Image.open(io.BytesIO(base64.b64decode(loaded["data"]))) as im:
            width, height = im.size
            if width == height:
                return loaded | {
                    "mimetype": infer_mimetype(path, _IMAGE_MIME_BY_EXT, "image/webp")
                }
            side = max(width, height)
            canvas = Image.new(im.mode, (side, side), "white")
            canvas.paste(im, ((side - width) // 2, (side - height) // 2))
            buf = io.BytesIO()
            canvas.save(buf, "WEBP", lossless=True)
    except Exception as exc:
        return f"cannot decode {path} as an image: {exc}"
    data = buf.getvalue()
    if len(data) > max_sticker_bytes:
        return (
            f"padded sticker is {len(data)} B, over the "
            f"{max_sticker_bytes} B cap; downscale before sending"
        )
    return {
        "mimetype": "image/webp",
        "filename": PurePosixPath(path).name,
        "data": base64.b64encode(data).decode(),
    }


def sticker_file(name_or_url: str, max_file_bytes: int) -> dict[str, Any] | str:
    """A WAHA sticker payload for a URL or local path, or an error string.

    Typed with the image MIME map. This helper does not decode, square,
    or convert the source to WebP; local tool sends use
    :func:`squared_sticker_payload` for square-padding.
    """
    if "://" in name_or_url:
        return remote_file(name_or_url, _IMAGE_MIME_BY_EXT, "image/webp")
    loaded = local_file(name_or_url, max_file_bytes)
    if isinstance(loaded, str):
        return loaded
    return loaded | {
        "mimetype": infer_mimetype(name_or_url, _IMAGE_MIME_BY_EXT, "image/webp")
    }


def video_file(name_or_url: str, max_file_bytes: int) -> dict[str, Any] | str:
    """A WAHA video payload for a URL or local path, or an error string.

    A URL becomes a ``RemoteFile`` (WAHA downloads and transcodes it);
    a local path a ``BinaryFile`` via :func:`local_file`, re-typed to
    the video MIME map — a ``.mp4`` produced by the shell tool must
    not ride the wire stamped ``application/octet-stream``.
    """
    if "://" in name_or_url:
        return remote_file(name_or_url, _VIDEO_MIME_BY_EXT, "video/mp4")
    loaded = local_file(name_or_url, max_file_bytes)
    if isinstance(loaded, str):
        return loaded
    return loaded | {
        "mimetype": infer_mimetype(name_or_url, _VIDEO_MIME_BY_EXT, "video/mp4")
    }
