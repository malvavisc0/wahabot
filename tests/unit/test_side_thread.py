import asyncio
import json
import unittest.mock
from typing import Any

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool, ToolSelection

from tests.harness import (
    CHAT_ID,
    SESSION,
)


def test_side_thread_fence_works_without_a_gate() -> None:
    from llama_index.core.workflow import Context

    from wahabot.ai.tools.whatsapp import RunTarget
    from wahabot.ai.workflow import FunctionCallingAgentWorkflow

    llm = unittest.mock.Mock()
    llm.metadata.is_function_calling_model = True
    agent = FunctionCallingAgentWorkflow(llm=llm)
    context = Context(agent)
    target = RunTarget(session=SESSION, chat_id=CHAT_ID)

    assert (
        asyncio.run(
            agent.start_triage(
                context,
                target,
                "Beto, you never answered me",
                False,
                True,
                quoted_other=True,
            )
        )
        is False
    )
    assert (
        asyncio.run(
            agent.start_triage(
                context,
                target,
                "I am quoting you",
                False,
                True,
                quoted_bot=True,
            )
        )
        is True
    )


def test_run_tool_call_returns_envelope_for_bad_arguments() -> None:
    """Bad tool arguments get a model-facing envelope, not a raw TypeError.

    The send_image incident: the
    model passed ``path`` to a URL-only tool and got
    "unexpected keyword argument 'path'" — no valid-arguments hint, no
    ``ok: false`` envelope for the chat template's error detector, and a
    full wasted recovery round. The wrapper must now validate first
    and answer with the envelope naming the valid arguments.
    """
    from pydantic import BaseModel

    from wahabot.ai.workflow import run_tool_call, tool_outcome

    class UrlOnlySchema(BaseModel):
        url: str | None = None
        reason: str = ""

    def url_only_fn(url: str | None = None, reason: str = "") -> str:
        return '{"ok": true}'

    tool = FunctionTool.from_defaults(
        fn=url_only_fn,
        fn_schema=UrlOnlySchema,
        name="send_image",
        description="Send an image from a URL.",
    )

    async def call(kwargs: dict[str, str]) -> str:
        selection = ToolSelection(
            tool_id="tc", tool_name="send_image", tool_kwargs=kwargs
        )
        message = await run_tool_call({"send_image": tool}, selection)
        return str(message.content)

    # Unknown argument: envelope with the valid-arguments hint.
    content = asyncio.run(call({"path": "/tmp/x.png", "reason": "meme"}))
    envelope = json.loads(content)
    assert envelope["ok"] is False
    assert "unexpected keyword argument 'path'" in envelope["error"]
    assert "valid arguments: url, reason" in envelope["error"]
    assert tool_outcome(content) == "failed"
    # The chat template's error detector scans for '"error":' in the
    # first 120 chars — the envelope must be visible to it.
    assert '"error":' in content[:120].lower()

    # Missing required argument (fn has no defaults for it).
    def needs_id_fn(message_id: str, reason: str = "") -> str:
        return '{"ok": true}'

    class NeedsIdSchema(BaseModel):
        message_id: str
        reason: str = ""

    strict_tool = FunctionTool.from_defaults(
        fn=needs_id_fn,
        fn_schema=NeedsIdSchema,
        name="react_to_message",
        description="React.",
    )
    selection = ToolSelection(
        tool_id="tc", tool_name="react_to_message", tool_kwargs={"reason": "r"}
    )
    message = asyncio.run(run_tool_call({"react_to_message": strict_tool}, selection))
    assert "missing a required argument: 'message_id'" in str(message.content)

    # Wrong type: the pydantic schema catches what the signature can't.
    selection = ToolSelection(
        tool_id="tc",
        tool_name="react_to_message",
        tool_kwargs={"message_id": 123, "reason": "r"},
    )
    message = asyncio.run(run_tool_call({"react_to_message": strict_tool}, selection))
    assert "message_id: Input should be a valid string" in str(message.content)


def test_tool_validation_rejects_coercion_before_execution() -> None:
    from pydantic import BaseModel

    from wahabot.ai.workflow.toolkit import run_tool_call

    class CountSchema(BaseModel):
        count: int

    called: list[int] = []

    def count_fn(count: int) -> str:
        called.append(count)
        return '{"ok": true}'

    tool = FunctionTool.from_defaults(
        fn=count_fn, fn_schema=CountSchema, name="count", description="Count."
    )
    selection = ToolSelection(
        tool_id="count", tool_name="count", tool_kwargs={"count": "5"}
    )
    outcome = asyncio.run(run_tool_call({"count": tool}, selection))
    assert json.loads(str(outcome.content))["ok"] is False
    assert called == []
    selection.tool_kwargs = {"count": 5}
    outcome = asyncio.run(run_tool_call({"count": tool}, selection))
    assert json.loads(str(outcome.content))["ok"] is True
    assert called == [5]


def test_run_tool_call_wraps_runtime_exceptions_in_envelope() -> None:
    """A tool that raises gets the ``ok: false`` envelope, with the error text.

    The model still needs the actual exception message to diagnose
    WAHA/network failures — but wrapped in the envelope so the chat
    template's error detector (bug 2b) sees it.
    """
    from pydantic import BaseModel

    from wahabot.ai.workflow import run_tool_call, tool_outcome

    def exploding_fn(url: str | None = None, reason: str = "") -> str:
        raise RuntimeError("WAHA connection refused")

    class UrlSchema(BaseModel):
        url: str | None = None
        reason: str = ""

    tool = FunctionTool.from_defaults(
        fn=exploding_fn, fn_schema=UrlSchema, name="fetch_thing", description="Fetch."
    )
    selection = ToolSelection(
        tool_id="tc", tool_name="fetch_thing", tool_kwargs={"url": "http://x"}
    )
    message = asyncio.run(run_tool_call({"fetch_thing": tool}, selection))
    content = str(message.content)
    envelope = json.loads(content)
    assert envelope["ok"] is False
    assert "WAHA connection refused" in envelope["error"]
    assert envelope["tool"] == "fetch_thing"
    assert tool_outcome(content) == "failed"


def test_tool_tracing_marks_errors_and_dedupes() -> None:
    """Tool tracing: failures export an ERROR span, successes one span.

    Two trace bugs from the meme incident:

    - Bug 4: the LlamaIndex dispatcher's drop path never ends the span
      on exception, so failed calls leaked unexported — the Langfuse
      trace showed a 0.0s gap where the send_image TypeError happened.
      ``run_tool_guarded`` must set ERROR plus the tool attributes and
      end the span while it is still active, inside the worker thread.
    - Bug 5: LlamaIndex auto-decorates both ``FunctionTool.__call__``
      and ``FunctionTool.call`` with the dispatcher span, doubling
      every tool span. ``_dedupe_tool_spans`` must leave exactly one.

    One provider + one instrument() for both scenarios: OTel allows a
    single global tracer provider per process, so a second test-local
    provider would be silently ignored anyway.
    """
    from opentelemetry import trace as trace_api
    from opentelemetry.instrumentation.llamaindex import (
        LlamaIndexInstrumentor,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from pydantic import BaseModel

    from wahabot.ai.observability import (
        _dedupe_tool_spans,  # pyright: ignore[reportPrivateUsage]
    )
    from wahabot.ai.workflow import run_tool_call

    exported: list[Any] = []

    class _Exporter:
        def export(self, batch: Any) -> None:
            exported.extend(batch)

        def shutdown(self) -> None:
            return None

    class _Processor(SimpleSpanProcessor):
        def __init__(self) -> None:
            super().__init__(_Exporter())  # pyright: ignore[reportArgumentType]

    class UrlSchema(BaseModel):
        url: str | None = None
        reason: str = ""

    def exploding_fn(url: str | None = None, reason: str = "") -> str:
        raise RuntimeError("WAHA connection refused")

    def quiet_fn(url: str | None = None, reason: str = "") -> str:
        return '{"ok": true}'

    provider = TracerProvider()
    provider.add_span_processor(_Processor())
    trace_api.set_tracer_provider(provider)
    LlamaIndexInstrumentor().instrument()
    _dedupe_tool_spans()

    fail_tool = FunctionTool.from_defaults(
        fn=exploding_fn, fn_schema=UrlSchema, name="fetch_thing", description="Fetch."
    )
    ok_tool = FunctionTool.from_defaults(
        fn=quiet_fn, fn_schema=UrlSchema, name="quiet", description="Quiet."
    )

    fail_sel = ToolSelection(
        tool_id="tc", tool_name="fetch_thing", tool_kwargs={"url": "http://x"}
    )
    message = asyncio.run(run_tool_call({"fetch_thing": fail_tool}, fail_sel))
    ok_tool(url="http://x")
    provider.force_flush()

    tool_spans = [s for s in exported if s.name.startswith("FunctionTool")]
    # Bug 5: one span per tool execution, not two.
    assert len(tool_spans) == 2, [s.name for s in tool_spans]
    by_status: dict[str, list[Any]] = {}
    for span in tool_spans:
        by_status.setdefault(span.status.status_code.name, []).append(span)
    # Bug 4: the failed call's span carries ERROR and the tool attributes.
    assert "ERROR" in by_status, by_status.keys()
    attrs: dict[str, Any] = dict(by_status["ERROR"][0].attributes or {})
    assert attrs["tool.name"] == "fetch_thing"
    assert "WAHA connection refused" in str(attrs["tool.error"])
    assert json.loads(str(message.content))["ok"] is False


def test_token_count_is_honest_not_char_equivalent() -> None:
    """The trim counts real tokens, so the budget buys real history.

    The old 1-char≈1-token estimate overcounted ~4x and starved the
    model to a handful of visible turns per run; the tokenizer-backed
    counter must bring a typical Spanish line down to its true size,
    and never exceed the char length (the old upper bound).
    """
    from wahabot.ai.workflow import message_text, token_count

    msg = ChatMessage(
        role=MessageRole.USER,
        content=(
            "[Member <111222333444555@lid>] montaje en techo, 3.5 a 5 m: "
            "a esa altura el cuerpo humano no te come la señal"
        ),
    )
    counted = token_count(msg)
    assert 0 < counted <= len(message_text(msg))
    # A tokenizer that fits ~78 chars of Spanish in ~27 tokens: the
    # honest count must be well under the char count (the 4x that
    # starved the window), with slack for tokenizer variance.
    assert counted < len(message_text(msg)) // 2
    # Tool-call kwargs ride the counted text too — a tool-heavy
    # history cannot masquerade as free.
    call = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="",
        additional_kwargs={"tool_calls": [{"name": "send_message", "arguments": "{}"}]},
    )
    assert token_count(call) > 0


def test_degrade_old_history_squeezes_old_keeps_fresh() -> None:
    """Old turns lose thinking and tool payloads; fresh turns stay verbatim.

    The meme incident: replayed
    history was dominated by old reasoning blocks and raw ``stdout``
    — ~12.4k tokens per run. Degradation keeps the social content
    (who said what, what was delivered) and the tool verdicts, drops
    the operational scaffolding from slices older than the last
    ``FRESH_TURNS`` user turns, and never touches the fresh window.
    """
    import json as _json

    from llama_index.core.base.llms.types import (
        TextBlock,
        ThinkingBlock,
        ToolCallBlock,
    )

    from wahabot.ai.history import degrade_old_history
    from wahabot.ai.workflow import token_count

    def user(text: str) -> ChatMessage:
        return ChatMessage(role=MessageRole.USER, content=text)

    def assistant(text: str) -> ChatMessage:
        return ChatMessage(role=MessageRole.ASSISTANT, blocks=[TextBlock(text=text)])

    def tool_result(payload: dict[str, Any]) -> ChatMessage:
        return ChatMessage(role=MessageRole.TOOL, content=_json.dumps(payload))

    messages = [
        user("revisa mi wifi"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ThinkingBlock(content="deliberation " * 200),
                ToolCallBlock(
                    tool_name="run_shell_command",
                    tool_call_id="c1",
                    tool_kwargs={
                        "command": "iw dev wlan0 link && " + "x" * 3000,
                        "reason": "Revisar el estado del wifi",
                    },
                ),
            ],
        ),
        tool_result({"ok": True, "exit_code": 0, "stdout": "wlan0 ok\n" + "x" * 3000}),
        assistant("todo bien con tu wifi"),
        user("crea un meme"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ThinkingBlock(content="meme plan " * 200),
                ToolCallBlock(
                    tool_name="run_shell_command",
                    tool_call_id="c2",
                    tool_kwargs={"command": "fc-list | head", "reason": "Fuentes"},
                ),
            ],
        ),
        tool_result({"ok": True, "exit_code": 0, "stdout": "fonts ok"}),
        assistant("meme enviado"),
        user("jajaja"),
        assistant("jaj"),
    ]
    before = sum(token_count(m) for m in messages)
    degraded = degrade_old_history(messages)
    after = sum(token_count(m) for m in degraded)

    # The old wifi slice: thinking gone, call body trimmed, stdout gone,
    # verdicts kept.
    old_slice = degraded[:4]
    assert not any(isinstance(b, ThinkingBlock) for m in old_slice for b in m.blocks)
    (old_call,) = [
        b
        for m in old_slice
        for b in m.blocks
        if isinstance(b, ToolCallBlock) and b.tool_call_id == "c1"
    ]
    # Every key survives (the model copies call shapes from history);
    # only the long value is cut to a preview.
    assert set(old_call.tool_kwargs) == {"command", "reason"}
    assert old_call.tool_kwargs["reason"] == "Revisar el estado del wifi"
    assert old_call.tool_kwargs["command"].startswith("iw dev wlan0 link")
    assert len(old_call.tool_kwargs["command"]) <= 81
    wifi_tool = next(m for m in old_slice if m.role == MessageRole.TOOL)
    assert '"ok": true' in str(wifi_tool.content)
    assert "wlan0" not in str(wifi_tool.content)
    # Delivered words survive.
    assert any(
        m.role == MessageRole.ASSISTANT and "todo bien" in str(m.content)
        for m in old_slice
    )
    # The fresh window (last FRESH_TURNS user turns and after) untouched:
    # the fresh meme call keeps its full kwargs.
    fresh_from = next(i for i, m in enumerate(messages) if m.content == "crea un meme")
    assert degraded[fresh_from:] == messages[fresh_from:]
    # Old payload bulk actually shrank.
    assert after < before // 2
    # Idempotent: degrading again changes nothing.
    assert degrade_old_history(degraded) == degraded


def test_degrade_old_history_small_conversation_untouched() -> None:
    """Fewer user turns than the fresh window: nothing is degraded."""

    from wahabot.ai.history import FRESH_TURNS, degrade_old_history

    messages = [
        ChatMessage(role=MessageRole.USER, content="hola"),
        ChatMessage(role=MessageRole.ASSISTANT, content="hola!"),
    ]
    assert len(messages) <= FRESH_TURNS
    assert degrade_old_history(messages) == messages


def test_squeeze_tool_result_keeps_verdict_only() -> None:
    """The tool-result squeeze: verdict keys stay, payload goes.

    A failure keeps its diagnosis (``error``), a success keeps its
    ``ok``; non-JSON tool text passes through — a plain answer is its
    own verdict — and a verdict no smaller than the original is left
    alone (no point swapping identical bulk).
    """
    import json as _json

    from wahabot.ai.history import squeeze_tool_result

    big = ChatMessage(
        role=MessageRole.TOOL,
        content=_json.dumps({"ok": False, "error": "cooldown", "stdout": "x" * 2000}),
    )
    squeezed = squeeze_tool_result(big)
    assert '"error": "cooldown"' in str(squeezed.content)
    assert "xxxx" not in str(squeezed.content)

    plain = ChatMessage(role=MessageRole.TOOL, content="the answer is 42")
    assert squeeze_tool_result(plain).content == plain.content

    already_small = ChatMessage(
        role=MessageRole.TOOL, content='{"ok": true, "chat": "c@g.us"}'
    )
    assert squeeze_tool_result(already_small).content == already_small.content
    failed_command = ChatMessage(
        role=MessageRole.TOOL,
        content=json.dumps({"ok": True, "exit_code": 7, "stdout": "x" * 2000}),
    )
    verdict = json.loads(str(squeeze_tool_result(failed_command).content))
    assert verdict == {"ok": True, "exit_code": 7}
