"""Message handler connecting the webhook to the LlamaIndex agent."""

from wahabot.core.runs import (
    chat_lock,
    context_for,
    contexts,
    persist_memory,
)
from wahabot.handlers.dedup import (
    forget_seen,
    seen_recently,
)
from wahabot.handlers.memory import append_to_memory
from wahabot.handlers.registration import (
    register_agent_handler,
    register_forget_handler,
)
from wahabot.handlers.replies import finish_agent_reply

__all__ = [
    "append_to_memory",
    "chat_lock",
    "context_for",
    "contexts",
    "finish_agent_reply",
    "forget_seen",
    "persist_memory",
    "register_agent_handler",
    "register_forget_handler",
    "seen_recently",
]
