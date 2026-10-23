"""LLM observability: export agent traces to Langfuse when configured.

Two pieces connect wahabot's llama-index workflow to Langfuse, sharing the
global OpenTelemetry tracer provider without either side calling the
other directly:

1. ``LlamaIndexInstrumentor`` (from ``opentelemetry-instrumentation-llamaindex``)
   makes every LLM/tool/workflow call emit OTel spans with prompts,
   completions and token counts.
2. The ``Langfuse`` client installs its span processor on the same
   provider, batches finished spans and ships them to the Langfuse API
   from background threads.

The feature is strictly opt-in: without ``LANGFUSE_PUBLIC_KEY`` /
``LANGFUSE_SECRET_KEY`` nothing is initialized and no spans leave the
process. ``LANGFUSE_BASE_URL`` selects the region (default EU cloud;
``https://us.cloud.langfuse.com`` etc.). Everything is fail-soft —
observability can never take the bot down.

WhatsApp chat ids (``<phone>@c.us`` / ``<phone>-<ts>@g.us``) are PII, so
the export hook masks them in span attributes before they leave.

Credentials come from ``Settings`` (``LANGFUSE_PUBLIC_KEY`` /
``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_BASE_URL`` via the ``.env`` file),
keeping every knob in one place.
"""

import atexit
import hashlib
import re
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any, Literal, cast

from loguru import logger

from wahabot.settings import Settings

if TYPE_CHECKING:
    from langfuse import Langfuse

__all__ = [
    "chat_trace_attributes",
    "enable_langfuse",
    "flush",
    "masked_chat_id",
    "observation",
]

#: The observation types Langfuse accepts for a started observation.
ObservationType = Literal[
    "generation",
    "embedding",
    "span",
    "agent",
    "tool",
    "chain",
    "retriever",
    "evaluator",
    "guardrail",
]

#: The configured client, or None while tracing is disabled.
_client: Langfuse | None = None

#: True once ``enable_langfuse`` succeeded; guards duplicate atexit hooks.
_enabled = False

#: WhatsApp JIDs — ``1234567890@c.us``, ``1234-567890123@g.us``,
#: ``123456789012345@lid`` (the linked-device identity), ``@broadcast`` —
#: plus the bare mention-token shape (``@1234567890``): group chats write
#: mentions as bare LID/phone user parts, and a mention-resolving send
#: carries them in tool arguments and message text.
_JID_PATTERN = (
    r"\b\d{6,}(?:-\d+)?@(?:c|g)\.us\b"
    r"|\b\d{6,}@lid\b"
    r"|\b[\w.-]+@broadcast\b"
    r"|(?<![\w@])@\d{6,}\b"
)

_JID_RE = re.compile(_JID_PATTERN)


def enable_langfuse(settings: Settings) -> bool:
    """Initialise Langfuse tracing for llama-index agent runs.

    No-op returning False when credentials are missing. Idempotent.
    Credential problems surface as warnings, never exceptions.
    """
    global _client, _enabled
    if _enabled:
        return True
    public_key = settings.langfuse_public_key.strip()
    secret_key = settings.langfuse_secret_key.strip()
    if not public_key or not secret_key:
        return False
    base_url = settings.langfuse_base_url.strip() or None
    environment = settings.langfuse_tracing_environment.strip() or None
    from langfuse import Langfuse as Client
    from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import set_tracer_provider

    try:
        provider = TracerProvider(resource=Resource.create({"service.name": "wahabot"}))
        set_tracer_provider(provider)

        _client = Client(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            environment=environment,
            mask_otel_spans=mask_otel_spans,
        )
        LlamaIndexInstrumentor().instrument()
        _dedupe_tool_spans()
    except Exception:
        _client = None
        logger.exception("Failed to initialise Langfuse tracing; continuing without it")
        return False
    _warn_if_auth_fails(_client)
    atexit.register(flush)
    _enabled = True
    logger.info("Langfuse tracing enabled for llama-index agent runs")
    return True


def _warn_if_auth_fails(client: Langfuse) -> None:
    """Best-effort credential check; a failure warns but never disables."""
    try:
        authenticated = client.auth_check()
    except Exception:
        logger.warning("Langfuse auth check raised; exports may fail")
        return
    if not authenticated:
        logger.warning(
            "Langfuse auth failed; check the LANGFUSE_* credentials — exports may fail"
        )


def _dedupe_tool_spans() -> None:
    """Strip the dispatcher span off ``FunctionTool.__call__``.

    LlamaIndex's ``__init_subclass__`` auto-decorates every public
    method with ``dispatcher.span``, so a tool call nests two identical
    spans — one around ``__call__``, one around the ``call`` it
    delegates to — doubling the trace payload for every tool execution.
    Removing the outer
    decoration leaves the single ``call`` span, which carries the same
    input/output and is the one the failure marking (bug 4) targets.
    Fail-soft: an unexpected layout logs and leaves the duplicate in
    place rather than breaking tracing.
    """
    try:
        from llama_index.core.tools.function_tool import FunctionTool

        wrapper = FunctionTool.__dict__.get("__call__")
        if wrapper is None or not hasattr(wrapper, "__wrapped__"):
            logger.debug("FunctionTool.__call__ not dispatcher-wrapped; no dedup")
            return
        FunctionTool.__call__ = wrapper.__wrapped__
    except Exception:
        logger.debug("Could not dedupe FunctionTool tracing spans")


#: Span attributes whose value is the session id we set ourselves —
#: masking them would collapse every chat into one anonymous session.
_SESSION_ID_KEYS = frozenset({"session.id", "langfuse.session.id"})


def mask_otel_spans(*, params: Any) -> Any:
    """Export-stage hook: mask WhatsApp JIDs in span attributes."""
    try:
        from langfuse.types import MaskOtelSpansResult, OtelSpanPatch
    except ImportError:  # pragma: no cover - SDK versions without the hook
        return None
    patches = {}
    for identifier, span in params.spans.items():
        masked = {
            key: _mask_value(value)
            for key, value in span.attributes.items()
            if key not in _SESSION_ID_KEYS and _JID_RE.search(_stringify(value))
        }
        if masked:
            patches[identifier] = OtelSpanPatch(set_attributes=masked)
    return MaskOtelSpansResult(span_patches=patches)


def _stringify(value: Any) -> str:
    """Span attribute value as a string for the cheap JID pre-check."""
    return value if isinstance(value, str) else str(value)


def _mask_value(value: Any) -> Any:
    """Mask JIDs inside a string, recursing into lists/dicts otherwise."""
    if isinstance(value, str):
        return _JID_RE.sub("[jid redacted]", value)
    if isinstance(value, list):
        return [_mask_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _mask_value(item) for key, item in value.items()}
    return value


def _opaque(value: str) -> str:
    """A stable, non-reversible id for trace metadata."""
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def masked_chat_id(chat_id: str) -> str:
    """A stable, non-reversible chat id for span metadata."""
    return _opaque(chat_id)


def trace_metadata(
    *,
    message_ids: Sequence[str] = (),
    serial_tokens: Sequence[str] = (),
    reply_to: str = "",
) -> dict[str, str]:
    """Trace metadata for one run: ids are opaque, serial tokens are not PII.

    The outbound id never reaches the traces: it is known only after the
    run's span closed, so the local audit journal is its home.
    """
    metadata: dict[str, str] = {}
    opaque_ids = [_opaque(value) for value in message_ids if value]
    if opaque_ids:
        metadata["wahabot.message_ids"] = ",".join(opaque_ids)
    tokens = [token for token in serial_tokens if token]
    if tokens:
        metadata["wahabot.serial_tokens"] = ",".join(tokens)
    if reply_to:
        metadata["wahabot.reply_to"] = _opaque(reply_to)
    return metadata


@asynccontextmanager
async def observation(
    name: str,
    *,
    as_type: ObservationType = "generation",
    model: str | None = None,
    input: Any = None,
    metadata: Any = None,
) -> AsyncIterator[Any]:
    """One Langfuse observation; a no-op when tracing is off.

    Fail-soft: a tracing error never fails the call it wraps. Yields
    ``None`` when no client is configured or the observation could not
    start, so callers must guard their ``update`` calls. The span runs
    as the current OTel observation, so it groups under the active
    trace (the agent's propagation context when one is recording).
    """
    if _client is None:
        yield None
        return
    try:
        span = _client.start_as_current_observation(
            name=name,
            # Langfuse overloads split the literal per observation kind.
            as_type=cast("Any", as_type),
            model=model,
            input=input,
            metadata=metadata,
        )
    except Exception:
        logger.debug("Langfuse observation failed to start: {name}", name=name)
        yield None
        return
    with span as current:
        yield current


@contextmanager
def chat_trace_attributes(
    chat_id: str,
    *,
    message_ids: Sequence[str] = (),
    serial_tokens: Sequence[str] = (),
    reply_to: str = "",
) -> Iterator[None]:
    """Group one run's LLM traces under a Langfuse session per chat.

    A no-op when tracing is disabled, so the handler never needs to
    check. Each WhatsApp chat becomes a stable, re-findable Langfuse
    session (``wa:<chat_id>``); turns within it show up as traces.
    Message ids are hashed before leaving the process: a WhatsApp id
    embeds the chat JID, so the raw id belongs only in the local audit.
    """
    if _client is None:
        yield
        return
    try:
        from langfuse import propagate_attributes

        metadata = trace_metadata(
            message_ids=message_ids,
            serial_tokens=serial_tokens,
            reply_to=reply_to,
        )
        propagation = propagate_attributes(
            session_id=f"wa:{chat_id}",
            tags=["wahabot"],
            metadata=metadata or None,
        )
    except Exception:
        logger.exception("Failed to start Langfuse attribute propagation")
        yield
        return
    with propagation:
        yield


def flush() -> None:
    """Flush buffered Langfuse events; safe at process exit."""
    if _client is not None:
        try:
            _client.flush()
        except Exception:
            logger.exception("Failed to flush Langfuse events")
