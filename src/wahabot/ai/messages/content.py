from typing import Any

from wahabot.ai.messages.constants import NON_REPLYABLE_SUFFIXES, REACTION_ONLY_RE
from wahabot.core.models import WahaEvent
from wahabot.core.waha import media_with_url


def is_replyable(event: WahaEvent) -> bool:
    """Check whether the message comes from a chat the bot can answer.

    Status updates (`status@broadcast`) and newsletters arrive as
    `message` events but are not replyable conversations.
    """
    sender = str(event.payload.get("from", ""))
    return not sender.endswith(NON_REPLYABLE_SUFFIXES)


def message_kind(event: WahaEvent) -> str:
    """Classify an incoming message: text, image, video, audio, sticker, ..."""
    kind = event.payload.get("_data", {}).get("type")
    if kind == "ptt":
        return "audio"
    if kind in ("image", "video", "ptv", "audio", "sticker", "document"):
        return kind
    if kind == "album":
        return "album"
    return "text"


def is_reaction_only(event: WahaEvent) -> bool:
    """True for a text message whose body is only emoji/punctuation.

    A bare ``🤣`` carries no request: it is a reaction, not an opening.
    In ``judicious`` groups it must not *start* a turn — the model reads
    the whole merged backlog as its current message and answers it (the
    emoji-to-essay failure). A media message is never reaction-only, so
    questions and statements always wake the agent.
    """
    if message_kind(event) != "text":
        return False
    return bool(REACTION_ONLY_RE.match(str(event.payload.get("body", "")).strip()))


def extract_text(event: WahaEvent) -> str | None:
    """Return replyable text, or None when there is nothing to answer.

    Album containers carry no text of their own. Every other message —
    text, voice transcription, image/video caption — stores its text in
    the top-level `body`, so return it directly instead of skipping
    messages simply because they carry media.
    """
    if message_kind(event) == "album":
        return None
    body = str(event.payload.get("body", "")).strip()
    return body or None


def image_media(event: WahaEvent) -> dict[str, Any] | None:
    """The media dict of an image message, or None.

    ``image`` messages and ``sticker`` messages qualify — a sticker is
    a (possibly animated) webp image, and the vision model can comment
    on its first frame. Videos and documents are excluded. The dict
    holds ``url``, ``mimetype`` and optionally ``filename``.
    """
    if message_kind(event) not in ("image", "sticker"):
        return None
    return media_with_url(event.payload)


def video_media(event: WahaEvent) -> dict[str, Any] | None:
    """The media dict of a video message (kinds ``video``/``ptv``), or None.

    Like :func:`image_media` but for moving pictures; only entries with
    a URL qualify — the download path checks the mimetype/size caps.
    """
    if message_kind(event) not in ("video", "ptv"):
        return None
    return media_with_url(event.payload)
