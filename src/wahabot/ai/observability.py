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
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from loguru import logger

from wahabot.settings import Settings

if TYPE_CHECKING:
    from langfuse import Langfuse

__all__ = ["chat_trace_attributes", "enable_langfuse", "flush"]

#: The configured client, or None while tracing is disabled.
_client: Langfuse | None = None

#: True once ``enable_langfuse`` succeeded; guards duplicate atexit hooks.
_enabled = False

#: WhatsApp JIDs — ``1234567890@c.us``, ``1234-567890123@g.us``, ``@broadcast``.
_JID_RE = re.compile(r"\b\d{6,}(?:-\d+)?@(?:c|g)\.us\b|\b[\w.-]+@broadcast\b")


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
    try:
        from langfuse import Langfuse as Client
        from opentelemetry.instrumentation.llamaindex import (
            LlamaIndexInstrumentor,
        )

        _client = Client(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            environment=environment,
            mask_otel_spans=mask_otel_spans,
        )
        LlamaIndexInstrumentor().instrument()
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


@contextmanager
def chat_trace_attributes(chat_id: str) -> Iterator[None]:
    """Group one run's LLM traces under a Langfuse session per chat.

    A no-op when tracing is disabled, so the handler never needs to
    check. Each WhatsApp chat becomes a stable, re-findable Langfuse
    session (``wa:<chat_id>``); turns within it show up as traces.
    """
    if _client is None:
        yield
        return
    try:
        from langfuse import propagate_attributes

        propagation = propagate_attributes(
            session_id=f"wa:{chat_id}",
            tags=["wahabot"],
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
