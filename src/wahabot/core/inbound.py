"""Shared routing and media gates for inbound message workflows."""

from wahabot.ai.messages import bot_jids, conversation_jid, message_kind
from wahabot.core.echoes import is_self_echo
from wahabot.core.models import WahaEvent
from wahabot.settings import Settings


def self_chat_needs_transcript(
    event: WahaEvent, body: str | None, settings: Settings
) -> bool:
    """Allow early voice transcription only for a non-echo self-chat turn."""
    return (
        body is None
        and message_kind(event) == "audio"
        and bool(settings.transcribe_url)
        and bool(event.payload.get("fromMe"))
        and conversation_jid(event) in bot_jids(event)
        and not is_self_echo(str(event.payload.get("id", "")))
    )


def can_prepare_video(event: WahaEvent, settings: Settings) -> bool:
    """Require both video and vision support for an attached video."""
    return message_kind(event) in ("video", "ptv") and settings.video and settings.vision


def url_video_enabled(settings: Settings) -> bool:
    """Whether inbound text may resolve video links into frames and transcripts."""
    return settings.video and settings.vision and settings.max_url_videos > 0
