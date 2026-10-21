import re
from collections.abc import Callable, Sequence
from typing import Any, cast, override

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import BaseTool
from llama_index.llms.openai_like import OpenAILike
from loguru import logger

from wahabot.ai.history import (
    INBOUND_ID_RE,
    chat_visible_text,
)
from wahabot.ai.tools.notebook import MAX_NOTE_TOKENS
from wahabot.ai.tools.slim import slim_tool_specs
from wahabot.core.embeddings import LanEmbedding
from wahabot.core.notebook import (
    MAX_NOTE_LENGTH,
)
from wahabot.core.thread_index import (
    ThreadIndexer,
)
from wahabot.core.tokens import text_tokens, truncate_to_tokens
from wahabot.settings import Settings

EARLY_STOPPING_PROMPT = (
    "You have reached the maximum number of tool rounds ({limit}). "
    "Based on the information gathered so far, provide a helpful final "
    "response to the user's original message. Do not attempt to use any "
    "more tools. If you already sent your reply, say nothing more — "
    "answer with an empty message."
)


POST_DELIVERY_WRAP_UP_PROMPT = (
    "You already delivered your reply. Do not send more replies or use any tools. "
    "Review this run for a verified, reusable lesson, not an activity summary. "
    "Record only a new correction, confirmed preference, or demonstrated "
    "workaround missing from the notebook. Return two lines: first "
    "`topic: <short-slug>` naming the lesson's subject in a few hyphenated "
    "words, then the lesson — the situation, what was wrong or clarified, the "
    "evidence, and what to do differently next time, in at most two concise "
    "sentences. Reuse the topic of an existing lesson when the subject matches. "
    "Do not infer success from an attempted action, invent facts, or turn your "
    "inference into operator policy. This is a private chat-scoped lesson for "
    "later runs, never a message to the chat. If nothing new and lasting was "
    "learned, return an empty message."
)


TRUNCATED_NOTE_SUFFIX = "\n[Truncated wrap-up note]"


CORRECTION_REWRITE_PROMPT = (
    "You are rewriting one WhatsApp reply. The draft below tells the "
    "recipient they are wrong. If the conversation corrects the assistant, "
    "rewrite the draft to admit the assistant's own mistake plainly. If the "
    "recipient is genuinely wrong, keep the correction but remove the "
    "accusation. Return only the rewritten reply, in the same language, with "
    "no preamble."
)


TOPIC_PREFIX = "topic:"


def topic_slug(line: str) -> str:
    """Normalise a wrap-up `topic:` line to a short, stable key slug."""
    value = line.split(":", 1)[1] if ":" in line else line
    value = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return value[:40].strip("-")


def parse_wrap_up(text: str) -> tuple[str, str]:
    """Split a wrap-up reply into ``(topic slug, lesson)``.

    A leading ``topic: <slug>`` line names the lesson; the line names
    the key, never the lesson, so it is dropped from the stored body in
    every outcome. Without a usable topic line the slug is empty and the
    caller keys by content instead; a topic line without a lesson body
    leaves nothing to store (``("", "")``).
    """
    lines = text.strip().splitlines()
    if lines and lines[0].strip().casefold().startswith(TOPIC_PREFIX):
        slug = topic_slug(lines[0])
        lesson = "\n".join(lines[1:]).strip()
        if slug and lesson:
            return slug, lesson
        return "", lesson
    return "", text.strip()


def truncate_note(note: str) -> str:
    """Cut a note to the notebook's caps, marking that storage shortened it."""
    if text_tokens(note) <= MAX_NOTE_TOKENS and len(note) <= MAX_NOTE_LENGTH:
        return note
    room = MAX_NOTE_LENGTH - len(TRUNCATED_NOTE_SUFFIX)
    return (
        truncate_to_tokens(
            note[:room], MAX_NOTE_TOKENS - text_tokens(TRUNCATED_NOTE_SUFFIX)
        )
        + TRUNCATED_NOTE_SUFFIX
    )


def raise_missing_llm() -> FunctionCallingLLM:
    """Fail fast when no llm was passed to the workflow."""
    raise ValueError("FunctionCallingAgentWorkflow requires an llm")


def recap_line(message: ChatMessage) -> str:
    """One recap line, or "" when the message has no visible text.

    The recap must never carry tool-call arguments or narration: a
    model's tool payload is not conversation, and a private self-report
    was never sent to the chat. ``message_text`` is the token counter's
    view (content plus tool args), not this one. The inbound
    ``[message id: …]`` note is metadata this code stamped on the
    turn, not conversation — it is stripped so the recap carries words
    only.
    """
    if message.role not in (MessageRole.USER, MessageRole.ASSISTANT):
        return ""
    text = INBOUND_ID_RE.sub("", str(message.content or "")).strip()
    if message.role == MessageRole.ASSISTANT:
        text = chat_visible_text(text)
    return f"[{message.role.value}] {text}" if text else ""


def build_thread_index(
    settings: Settings | None, thread_index: ThreadIndexer | None
) -> ThreadIndexer | None:
    """The per-chat recall index, or None when its kill switch is off.

    An injected index wins (tests, alternate wiring). Otherwise the index
    requires ``memory_persist`` (it is a file next to the buffer) and
    ``thread_recall``; a missing ``embed_api_base`` leaves it FTS5-only,
    so the exact-token floor works without the LAN embedding service.
    """
    if thread_index is not None:
        return thread_index
    if settings is None or not settings.memory_persist or not settings.thread_recall:
        return None
    embed_model = (
        LanEmbedding(
            api_base=settings.embed_api_base,
            model_name=settings.embed_model,
            timeout=settings.embed_timeout,
        )
        if settings.embed_api_base
        else None
    )
    return ThreadIndexer(
        settings.data_dir,
        embed_model,
        query_timeout=max(0.1, min(settings.embed_timeout, 2.0)),
    )


class ObservableOpenAILike(OpenAILike):
    """OpenAILike whose instrumentation payload names model/temperature.

    Exists because the OTel llama-index instrumentor reads
    ``model_dict["model"]``/``["temperature"]`` for its span attributes
    but the base ``to_payload`` exposes neither (external constraint).
    Also slims the tool specs of every request (``wahabot.ai.tools.slim``):
    the base ``_prepare_chat_with_tools`` builds them from Pydantic's
    JSON schema. Titles and redundant ``strict: false`` are removed, and
    nullable unions are
    compacted without dropping explicit null support.
    """

    @override
    def to_payload(self) -> dict[str, Any]:
        return {
            **super().to_payload(),
            "model": self.model,
            "temperature": self.temperature,
        }

    @override
    def _prepare_chat_with_tools(
        self,
        tools: Sequence[BaseTool],
        user_msg: str | ChatMessage | None = None,
        chat_history: list[ChatMessage] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        tool_choice: str | dict[str, Any] | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        base = cast(
            Callable[..., dict[str, Any]],
            super()._prepare_chat_with_tools,
        )
        prepared = base(
            tools,
            user_msg=user_msg,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            tool_required=tool_required,
            tool_choice=tool_choice,
            strict=strict,
            **kwargs,
        )
        specs = prepared.get("tools")
        if isinstance(specs, list):
            prepared["tools"] = slim_tool_specs(specs)
        return prepared


def load_llm(settings: Settings) -> FunctionCallingLLM:
    """Configure the OpenAI-compatible chat LLM from settings.

    Sampling options follow the model card (see ``Settings``).
    ``top_p``/``presence_penalty`` ride ``additional_kwargs`` (merged
    into the request body); ``top_k``/``min_p``/``repetition_penalty``
    ride ``extra_body`` because the OpenAI SDK's typed ``create()``
    signature rejects them (external constraint).
    """
    kwargs: dict[str, Any] = {
        "top_p": settings.llm_top_p,
        "presence_penalty": settings.llm_presence_penalty,
        "extra_body": {
            "top_k": settings.llm_top_k,
            "min_p": settings.llm_min_p,
            "repetition_penalty": settings.llm_repetition_penalty,
        },
    }
    if settings.llm_reasoning_effort:
        kwargs["reasoning_effort"] = settings.llm_reasoning_effort
    if settings.llm_auto_cache:
        # Requesty gateway flag: adds explicit cache breakpoints on the
        # largest content blocks, so repeated prefixes (the system
        # prompt, early history) are billed at cached rates.
        extra_body = dict(kwargs.get("extra_body", {}))
        extra_body["requesty"] = {"auto_cache": True}
        kwargs["extra_body"] = extra_body
    return ObservableOpenAILike(
        model=settings.llm_model,
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
        temperature=settings.llm_temperature,
        additional_kwargs=kwargs,
        # No client retries: a hung endpoint must fail fast within the
        # workflow timeout, not stack 60s attempts until it blows up.
        max_retries=0,
        is_chat_model=True,
        is_function_calling_model=True,
    )


def warn_context_mismatch(context_window: int, memory_token_limit: int) -> bool:
    """Warn loudly when the memory budget exceeds the model's context window.

    The aria-ai lesson: a budget larger than what the server can actually
    hold is silently clamped (or truncated) by the provider, and the
    memory system quietly believes it has room
    it does not. The window is the operator-declared
    ``WAHABOT_LLM_CONTEXT_WINDOW`` — llama-index's metadata default
    (3900) describes no real deployment, so it is never consulted.
    Returns True when the mismatch was found; 0 (unknown) is not one.
    """
    if context_window > 0 and memory_token_limit > context_window:
        logger.warning(
            "Memory token limit {limit} exceeds the model context window {window}",
            limit=memory_token_limit,
            window=context_window,
        )
        return True
    return False
