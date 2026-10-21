"""Reply context rendering and the agent entrypoint."""

from wahabot.ai.context.addressing import (
    mentions_note,
    quoted_participant,
    reply_context,
    sender_tag,
)
from wahabot.ai.context.participants import (
    chat_rules,
    participant_names,
    roster_cache,
    roster_names,
)
from wahabot.ai.context.prompt import render_system_prompt
from wahabot.ai.context.turn import handle_message
from wahabot.ai.context.user_message import final_reply
from wahabot.ai.history import (
    chat_visible_text,
    is_emoji_narration,
    is_error_narration,
    is_silence_narration,
    is_single_emoji,
    narration_kind,
)

__all__ = [
    "chat_rules",
    "chat_visible_text",
    "final_reply",
    "handle_message",
    "is_emoji_narration",
    "is_error_narration",
    "is_silence_narration",
    "is_single_emoji",
    "mentions_note",
    "narration_kind",
    "participant_names",
    "quoted_participant",
    "render_system_prompt",
    "reply_context",
    "roster_cache",
    "roster_names",
    "sender_tag",
]
