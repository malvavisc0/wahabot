"""LlamaIndex agent wiring for WhatsApp messages.

The agent is a base: it loads an OpenAI-compatible LLM from settings
and exposes a single entrypoint, :func:`handle_message`. Bot behavior
lives in the handlers registered on the workflow.
"""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings as LlamaSettings
from llama_index.llms.openai_like import OpenAILike
from loguru import logger

from whabot.core.models import WahaEvent
from whabot.settings import Settings

NON_REPLYABLE_SUFFIXES = ("@broadcast", "@newsletter")


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
    if kind in ("image", "video", "ptv", "audio", "sticker", "document"):
        return kind
    if kind == "album":
        return "album"
    return "text"


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


def is_group_addressed(event: WahaEvent) -> bool:
    """Check whether a group message mentions or is aimed at us.

    In 1:1 chats every message is for us. In groups, only messages
    that mention our JID should wake the agent.
    """
    payload = event.payload
    if not str(payload.get("from", "")).endswith("@g.us"):
        return True
    if payload.get("fromMe"):
        return False
    me = (event.me or {}).get("id")
    if me is None:
        return False
    mentions = payload.get("_data", {}).get("mentionedJidList", [])
    return me in mentions


def load_llm(settings: Settings) -> OpenAILike:
    """Configure the OpenAI-compatible LLM from settings."""
    return OpenAILike(
        model=settings.llm_model,
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key,
        is_function_calling_model=True,
    )


def build_agent(settings: Settings) -> FunctionAgent:
    """Build the agent with the configured LLM and system prompt."""
    llm = load_llm(settings)
    LlamaSettings.llm = llm
    return FunctionAgent(tools=[], llm=llm, system_prompt=settings.agent_system_prompt)


async def handle_message(event: WahaEvent, agent: FunctionAgent) -> str:
    """Run the agent over an incoming message event and return its reply."""
    chat_id = str(event.payload.get("from", ""))
    body = str(event.payload.get("body", ""))
    logger.info("Agent handling message from {chat_id}", chat_id=chat_id)
    result = await agent.run(user_msg=body)
    return str(result)
