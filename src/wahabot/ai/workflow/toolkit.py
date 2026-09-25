"""Tool-call execution helpers for the agent workflow.

``run_tool_call`` plus the model-facing failure envelopes, log extras
and outcome classifiers used by ``handle_tool_calls`` and the
``FunctionCallingAgentWorkflow`` audit paths. Distinguished from
:mod:`delivery` (history folding) by substance: this module is about
*invoking* tools — signature validation, span marking, worker-thread
execution — and describing their outcomes in logs and envelopes.
"""

import asyncio
import inspect
import json
from collections.abc import Callable
from functools import partial
from typing import Any, cast

import pydantic
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.tools import BaseTool, ToolOutput, ToolSelection
from loguru import logger

#: A TypeError raised by the tool call carries this marker when the
#: arguments never matched the tool's signature — the wrapper below
#: re-raises signature failures with it so the envelope can name the
#: offending argument instead of quoting a raw Python message.
_SIGNATURE_MARKER = "invalid tool arguments:"


#: How many chars of a tool call's rendered arguments reach the log: an
#: audit line, not a transcript. The reason kwarg stays uncapped up to
#: this budget — it is the justification, and action tools cap theirs at
#: 200 chars anyway (``_REASON_LOG_CAP``).
_ARGS_LOG_CAP = 300


def tool_call_key(tool_call: ToolSelection) -> str:
    """A comparable identity for one tool call: name + sorted arguments.

    Two calls are "the same" when they target the same tool with the
    same arguments — the loop detector compares consecutive rounds by
    these keys. Arguments are sorted so dict key order in the model's
    JSON does not defeat the comparison.
    """
    kwargs = dict(tool_call.tool_kwargs)
    return f"{tool_call.tool_name}:{sorted(kwargs.items())}"


def tool_call_log_extra(tool_call: ToolSelection) -> str:
    """The per-call context for the ``run_tool_call`` log lines.

    The reason kwarg first when present (the model's *why*), then the
    remaining arguments (the *what* — a shell command, a search
    query). A missing reason is flagged in place: the operator's audit
    goal is "read the log and know why", so a call the model never
    justified must be visible as such, not just shorter. Empty for a
    bare call with no arguments at all.
    """
    kwargs = dict(tool_call.tool_kwargs)
    parts: list[str] = []
    if reason := str(kwargs.pop("reason", "")).strip():
        parts.append(f"reason: {reason[:_ARGS_LOG_CAP]}")
    else:
        parts.append("reason: (model gave none)")
    if kwargs:
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(kwargs.items()))
        parts.append(f"args: {args[:_ARGS_LOG_CAP]}")
    return f" ({'; '.join(parts)})" if parts else ""


def tool_fn(tool: BaseTool) -> Callable[..., Any]:
    """The wrapped tool function (FunctionTool exposes it as ``fn``)."""
    fn = getattr(tool, "fn", None)
    if fn is None:
        raise TypeError("tool carries no callable function to validate")
    return cast("Callable[..., Any]", fn)


def validate_tool_kwargs(tool: BaseTool, tool_kwargs: dict[str, Any]) -> str | None:
    """A model-facing error message for bad tool arguments, or None.

    Two checks, cheapest first: the function signature (catches unknown
    and missing arguments) and the pydantic schema (catches wrong
    argument *types* the signature can't see). Both produce the
    ``{"ok": false, "error": ...}`` envelope shape every tool already
    uses, with the valid argument names listed so the model can retry
    with corrected arguments — the raw ``TypeError`` it replaced named
    no alternatives and cost a full recovery round in the send_image
    incident (docs/bug-report-2c665d8.md, bug 2).
    """
    valid = tool_argument_names(tool)
    try:
        inspect.signature(tool_fn(tool)).bind(**tool_kwargs)
    except TypeError as exc:
        return (
            f"invalid arguments for {tool.metadata.get_name()}: {exc}; "
            f"valid arguments: {', '.join(valid)}"
        )
    schema = tool.metadata.fn_schema
    if schema is None:
        return None
    try:
        schema.model_validate(tool_kwargs)
    except pydantic.ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        return (
            f"invalid arguments for {tool.metadata.get_name()}: {details}; "
            f"valid arguments: {', '.join(valid)}"
        )
    return None


def tool_argument_names(tool: BaseTool) -> list[str]:
    """The tool's argument names: schema fields, or the fn signature."""
    schema = tool.metadata.fn_schema
    if schema is not None:
        return list(schema.model_fields)
    return [
        name
        for name in inspect.signature(tool_fn(tool)).parameters
        if name not in ("self", "ctx", "context")
    ]


def tool_call_failure_envelope(tool_call: ToolSelection, exc: Exception) -> str:
    """A model-facing failure envelope for an exception in a tool call.

    Signature mismatches (the message starts with ``_SIGNATURE_MARKER``)
    get the valid-arguments hint; every other exception keeps its
    ``str(exc)`` — the model still needs the actual error text to
    diagnose WAHA/network failures — but wrapped in the ``ok: false``
    envelope so the chat template's error detector sees it (it scans
    for ``"error":`` and misses free-form exception text entirely).
    """
    message = str(exc)
    if message.startswith(_SIGNATURE_MARKER):
        message = message[len(_SIGNATURE_MARKER) :].strip()
    return json.dumps(
        {"ok": False, "error": message, "tool": tool_call.tool_name},
        ensure_ascii=False,
    )


def mark_active_span_error(tool_call: ToolSelection, exc: Exception) -> None:
    """Flag the active tracing span ERROR and end it for a failed tool call.

    The exception never propagates out of the instrumented tool layer
    (``run_tool_call`` catches it), so the OTel span ends ``UNSET`` —
    and worse, the LlamaIndex dispatcher's drop path never *ends* the
    span at all, so it leaks unexported and the Langfuse trace showed a
    0.0s gap where the send_image TypeError happened
    (docs/bug-report-2c665d8.md, bug 4). This must run in the worker
    thread while the tool span's context is still attached: there it
    sets ERROR plus ``tool.name``/``tool.error`` attributes and ends
    the span, which both records and exports the failure. Fail-soft:
    tracing may be disabled, or no span may be active.
    """
    try:
        from opentelemetry.trace import Status, StatusCode, get_current_span

        span = get_current_span()
        if not span.is_recording():
            return
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        span.set_attribute("tool.name", tool_call.tool_name)
        span.set_attribute("tool.error", str(exc))
        span.end()
    except Exception:
        logger.debug("Could not mark tracing span for failed tool call")


def run_tool_guarded(tool: BaseTool, tool_call: ToolSelection) -> ToolOutput:
    """Call the tool, marking its tracing span ERROR on exception.

    Runs inside the ``asyncio.to_thread`` worker so the span the
    dispatcher opened around ``FunctionTool.call`` is still the active
    one when the exception is caught — closing it there is the only
    chance, because the dispatcher's drop path leaks the span open and
    the event-loop thread's context carries no tool span.
    """
    try:
        return tool(**tool_call.tool_kwargs)
    except Exception as exc:
        mark_active_span_error(tool_call, exc)
        raise


async def run_tool_call(
    tools_by_name: dict[str, BaseTool], tool_call: ToolSelection
) -> ChatMessage:
    """Run one tool call, returning its output (or the failure) as a tool message.

    Tool functions are sync and do network I/O (WAHA, web), so they run
    in a worker thread via ``asyncio.to_thread`` to keep the event loop
    responsive. Tool outputs are not truncated here: every tool bounds
    its own payload at the source (``visit_url`` capping its text,
    ``web_search`` its per-result snippets, the list tools slimming
    ``_data`` and fitting to a whole-message budget), so a blunt
    workflow-level char cutoff would only mangle already-curated JSON
    envelopes.

    Arguments are validated against the tool's signature and schema
    *before* the call: a mismatch returns the error envelope with the
    valid argument names instead of letting a raw ``TypeError`` through
    (bug 2 of docs/bug-report-2c665d8.md — the model got "unexpected
    keyword argument 'path'" with no hint of what was valid, and burned
    a whole round improvising a workaround).

    Each call is logged: one INFO line with its outcome (completed,
    unknown) and its context (the reason when the model gave one, the
    remaining arguments), or WARNING with the exception when the tool
    raised — tool failures are what gets grepped for, so they carry
    the error detail.
    """
    tool = tools_by_name.get(tool_call.tool_name)
    kwargs = {"tool_call_id": tool_call.tool_id, "name": tool_call.tool_name}
    extra = tool_call_log_extra(tool_call)
    failure: Exception | None = None
    if tool is None:
        content = f"Tool {tool_call.tool_name} does not exist"
        outcome = "unknown"
    else:
        invalid = validate_tool_kwargs(tool, dict(tool_call.tool_kwargs))
        if invalid is not None:
            failure = TypeError(f"{_SIGNATURE_MARKER}{invalid}")
            content = tool_call_failure_envelope(tool_call, failure)
            outcome = "failed"
        else:
            try:
                guard = partial(run_tool_guarded, tool, tool_call)
                called = await asyncio.to_thread(guard)
                content = called.content
                outcome = "completed"
            except TypeError as exc:
                # A TypeError here is also a signature mismatch (the
                # model's JSON decoded but the fn rejected the shape):
                # route it through the envelope with the valid-args hint.
                failure = TypeError(f"{_SIGNATURE_MARKER}{exc}")
                content = tool_call_failure_envelope(tool_call, failure)
                outcome = "failed"
            except Exception as exc:
                content = tool_call_failure_envelope(tool_call, exc)
                outcome = "failed"
                failure = exc
    if failure is not None:
        logger.warning(
            "Tool call {tool}{extra} failed: {exc}",
            tool=tool_call.tool_name,
            extra=extra,
            exc=failure,
        )
    else:
        logger.info(
            "Tool call {tool}{extra}: {outcome}",
            tool=tool_call.tool_name,
            extra=extra,
            outcome=outcome,
        )
    return ChatMessage(role="tool", content=content, additional_kwargs=kwargs)


def tool_outcome(content: str) -> str:
    """The audit outcome of a finished tool call: completed/failed/unknown.

    Every tool reports failures as the ``{"ok": false, ...}``
    envelope (``wahabot.ai.tools.envelope``), and since the exception
    wrapper also produces that envelope, it is the single
    authoritative signal for every failure path.
    """
    if content.startswith("Tool ") and content.endswith(" does not exist"):
        return "unknown"
    try:
        envelope: Any = json.loads(content)
    except json.JSONDecodeError:
        return "completed"
    if isinstance(envelope, dict):
        record: dict[str, Any] = envelope
        if record.get("ok") is False:
            return "failed"
    return "completed"


def tool_outcome_ok(content: str) -> bool:
    """Whether a finished tool call succeeded, per :func:`tool_outcome`."""
    return tool_outcome(content) == "completed"
