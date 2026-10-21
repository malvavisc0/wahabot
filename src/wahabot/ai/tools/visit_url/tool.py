from urllib.parse import urlsplit

from wahabot.ai.tools.envelope import error
from wahabot.ai.tools.visit_url.page import page_envelope
from wahabot.ai.tools.visit_url.video import media_envelope
from wahabot.settings import Settings


def visit_url(settings: Settings, url: str) -> str:
    """Fetch HTTP(S) text or available video/post metadata, without rendering.

    Media-host URLs attempt yt-dlp metadata without downloading video;
    YouTube links attempt captions as ``transcript``. Extraction failures,
    unresolved placeholders, and ordinary exceptions during metadata or
    caption processing fall back to an HTTP page fetch. Unavailable or
    malformed caption tracks normally just omit ``transcript``.

    Returns:
        A JSON envelope with the page ``text`` (a bounded preview of up
        to ~4000 chars), its final ``url``, HTTP ``status`` and a
        ``truncated`` flag — and, when the body was cut, a ``file``
        (metadata containing the extracted text's file.path) if writing succeeds.
        A metadata result has ``kind="video"`` or ``kind="post"`` and
        ``title``/``description``/``uploader`` fields. Posts carry
        ``item_count``; ``item_count_truncated`` marks counts and summed
        durations limited to the first 100 inspected entries (unless
        the parent supplies its own duration). Captions add ``transcript``/
        ``transcript_truncated`` and (on long captions) a ``transcript_file`` with
        the fetched caption text, which can be incomplete or generated.
        Descriptions cap at 800 characters with description_truncated.
        HTML text is best-effort tag stripping, not verified visible text;
        other decoded bodies are preserved. Preview limits are not download
        caps. For string inputs, ordinary HTTP fetch/response-processing/
        envelope failures return an ``error`` envelope; unreadable bodies
        yield a placeholder and spill failures keep the preview.
        Process-level interrupts propagate.
    """
    url = url.strip()
    if not url:
        return error("url cannot be empty")
    try:
        parts = urlsplit(url)
    except ValueError:
        return error("url must be a valid HTTP(S) URL")
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return error("url must be a valid HTTP(S) URL")
    media = media_envelope(url, settings)
    if media is not None:
        return media
    return page_envelope(url, settings)
