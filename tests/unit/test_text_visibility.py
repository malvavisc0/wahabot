import asyncio
from typing import Any

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer

from tests.harness import (
    CHAT_ID,
)
from tests.unit.support import (
    _PURGE_SCRIPT,
)
from wahabot.ai.history import (
    chat_visible_text,
    deduplicate_messages,
)
from wahabot.core.models import WahaEvent
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings


def test_chat_visible_text_self_report_grammar() -> None:
    """Only third-person run reports strip; customer answers survive.

    The filter is defence in depth behind the structural fix (notes are
    never stored as turns), so it must be precision-first: a false
    positive deletes a real reply line, a false negative leaks a
    cosmetic self-report. First-person answers ("I sent you the PDF")
    and the German/Spanish reply forms are legitimate customer-facing
    lines; the wrap-up prompt elicits third-person reports ("Sent
    <name> a final confirmation that …"), which carry a run cue.
    """
    # Legitimate reply lines survive even in second position.
    for answer in (
        "I've sent the confirmation.",
        "I have already sent the confirmation.",
        "I sent you the PDF.",
        "Ich habe geantwortet.",
        "Auf deine Frage habe ich geantwortet.",
        "Respondí a tu mensaje de ayer.",
        "Sent the invoice and the receipt.",
    ):
        assert chat_visible_text(f"Hola.\n{answer}") == f"Hola.\n{answer}"
    # The observed leak shapes strip.
    for note in (
        "Sent Tobias a final confirmation that I will no longer reply, as requested.",
        "Sent a text-based Spanish mini-lesson after the voice synthesis failed.",
        "Sent him the files instead of messaging again.",
        "Auf Tobias' Scherz über KI-Stimmen mit trockenem Selbstkommentar geantwortet.",
    ):
        assert chat_visible_text(f"Hola.\n{note}") == "Hola."


def test_final_reply_strips_an_appended_self_report() -> None:
    from llama_index.core.base.llms.types import ChatResponse

    from wahabot.ai.context import final_reply

    result = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content=(
                "Understood, I will stay out of it.\n"
                "Sent a confirmation that the request to stop was honored."
            ),
        )
    )
    assert final_reply(result) == "Understood, I will stay out of it."


def test_deduplicate_refuses_to_merge_a_wrap_up_note_into_a_reply() -> None:
    """A private note is never folded into the assistant's sent turn.

    The merge is what taught the model to append notes to its replies
    The reply and the note stay separate
    turns, in both orders, while ordinary same-role turns still merge.
    """
    from wahabot.ai.history import merge_pair
    from wahabot.ai.messages import WRAP_UP_NOTE_KWARG

    reply = ChatMessage(role=MessageRole.ASSISTANT, content="Delivered the form.")
    note = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="Sent the form and confirmed receipt.",
        additional_kwargs={WRAP_UP_NOTE_KWARG: True},
    )

    forward = deduplicate_messages([reply, note])
    assert [m.content for m in forward] == ["Delivered the form.", note.content]
    assert WRAP_UP_NOTE_KWARG not in forward[0].additional_kwargs
    assert forward[1].additional_kwargs.get(WRAP_UP_NOTE_KWARG) is True

    backward = deduplicate_messages([note, reply])
    assert [m.content for m in backward] == [note.content, "Delivered the form."]
    assert backward[0].additional_kwargs.get(WRAP_UP_NOTE_KWARG) is True

    # Two notes may still fold: the boundary is private-vs-public, not
    # note-vs-note.
    both = deduplicate_messages([note, note])
    assert len(both) == 1 and both[0].content == f"{note.content}\n{note.content}"

    # Ordinary same-role turns still merge.
    other = ChatMessage(role=MessageRole.ASSISTANT, content="Second line.")
    merged = deduplicate_messages([reply, other])
    assert [m.content for m in merged] == ["Delivered the form.\nSecond line."]

    # The merge helper refuses the crossing itself (invariant).
    with pytest.raises(ValueError, match="wrap-up note"):
        merge_pair(reply, note)


def test_remember_drops_an_appended_self_report(unit_settings: Settings) -> None:
    """``remember`` stores the reply, never the appended private note.

    Memory mirrors the chat: the note was never sent, so storing it would
    put it back in front of the model and re-teach the pattern.
    """
    from llama_index.core.base.llms.types import ChatResponse
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.core.workflow import Context

    from wahabot.ai.workflow import FunctionCallingAgentWorkflow, load_llm

    async def stored() -> list[str]:
        wf = FunctionCallingAgentWorkflow(llm=load_llm(unit_settings))
        ctx = Context(wf)
        await ctx.store.set(
            "memory",
            ChatMemoryBuffer.from_defaults(),  # pyright: ignore[reportUnknownMemberType]
        )
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=(
                "Sounds good, I will hold the reply until tomorrow.\n"
                "Sent a confirmation that the reply is held until tomorrow."
            ),
        )
        await FunctionCallingAgentWorkflow.remember(
            wf, ctx, ChatResponse(message=message), []
        )
        memory = await ctx.store.get("memory")
        return [str(m.content) for m in await memory.aget_all()]

    assert asyncio.run(stored()) == ["Sounds good, I will hold the reply until tomorrow."]


def test_remember_filters_undelivered_leak(unit_settings: Settings) -> None:
    """``remember`` stores nothing when the final text is a leaked token."""

    from llama_index.core.base.llms.types import ChatResponse
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.core.workflow import Context

    from wahabot.ai.workflow import FunctionCallingAgentWorkflow, load_llm

    async def stored() -> list[Any]:
        wf = FunctionCallingAgentWorkflow(llm=load_llm(unit_settings))
        ctx = Context(wf)
        await ctx.store.set(
            "memory",
            ChatMemoryBuffer.from_defaults(),  # pyright: ignore[reportUnknownMemberType]
        )
        for text in ("stay_silent", "I'll stay silent here"):
            message = ChatMessage(role=MessageRole.ASSISTANT, content=text)
            await FunctionCallingAgentWorkflow.remember(
                wf, ctx, ChatResponse(message=message), []
            )
        real = ChatMessage(role=MessageRole.ASSISTANT, content="real words")
        await FunctionCallingAgentWorkflow.remember(
            wf, ctx, ChatResponse(message=real), []
        )
        memory = await ctx.store.get("memory")
        return [str(m.content) for m in await memory.aget_all()]

    assert asyncio.run(stored()) == ["real words"]


@pytest.mark.skipif(not _PURGE_SCRIPT.exists(), reason="scripts/ is local-only")
def test_purge_script_drops_leaked_tokens() -> None:
    """The purge script produces sanitized, alternating history."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("purge_leaked_silence", _PURGE_SCRIPT)
    assert spec is not None and spec.loader is not None
    purge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(purge)

    memory = ChatMemoryBuffer.from_defaults(token_limit=8000)  # pyright: ignore[reportUnknownMemberType]

    def U(c: str) -> ChatMessage:
        return ChatMessage(role=MessageRole.USER, content=c)

    def A(c: str) -> ChatMessage:
        return ChatMessage(role=MessageRole.ASSISTANT, content=c)

    memory.put(U("[Ana] q1"))
    memory.put(A("stay_silent"))
    memory.put(U("[Ana] q2"))
    memory.put(A("real answer"))
    memory.put(A("stay_silent"))
    kept, dropped = purge.purge_buffer(memory)
    assert dropped == 2
    # The purge removed the assistant turns between/before user turns;
    # the re-sanitize merges the now-consecutive user messages (every
    # word kept, kwargs merged) so alternation holds.
    assert [str(m.content) for m in kept] == ["[Ana] q1\n[Ana] q2", "real answer"]
    roles = [m.role for m in kept]
    assert roles[0] == MessageRole.USER and roles[-1] == MessageRole.ASSISTANT
    # Idempotent: a second pass over already-clean history drops nothing
    # and keeps the merged shape.
    again = ChatMemoryBuffer.from_defaults(token_limit=8000)  # pyright: ignore[reportUnknownMemberType]
    for message in kept:
        again.put(message)
    kept2, dropped2 = purge.purge_buffer(again)
    assert dropped2 == 0
    assert [str(m.content) for m in kept2] == ["[Ana] q1\n[Ana] q2", "real answer"]


def test_dispatch_contains_handler_failure() -> None:
    """A failing handler is contained: logged, other handlers still run.

    Regression for the unhandled traceback that reached uvicorn's
    ServerErrorMiddleware: a handler crash must not 500 the webhook
    (WAHA redelivers non-200s → endless loop) or drop the event's
    other handlers.
    """
    import wahabot.webhook as webhook_module
    from tests.harness import reset_handlers

    reset_handlers()
    try:
        calls: list[str] = []

        async def boom(_event: WahaEvent) -> None:
            calls.append("boom")
            raise RuntimeError("handler exploded")

        async def after(_event: WahaEvent) -> None:
            calls.append("after")

        webhook_module.on_message(boom)
        webhook_module.on_message(after)
        event = WahaEvent(
            id="evt_test_containment",
            timestamp=1,
            event="message",
            session="default",
            payload={"id": "msg_test_containment", "from": CHAT_ID},
        )
        asyncio.run(webhook_module.dispatch(event))
        assert calls == ["boom", "after"]
    finally:
        reset_handlers()


def test_slim_tool_spec_strips_padding_keeps_semantics() -> None:
    """Slimming removes titles/redundant flags and preserves nullable semantics.

    The bundled tools ride every request as serialized JSON schemas;
    Pydantic pads them with auto-generated titles and redundant
    ``strict: false``. Nullable unions become equivalent type arrays
    only when unconstrained. Names, descriptions, defaults, and enums
    survive, and the result is a deep copy: the
    original spec object is never mutated.
    """
    import json as _json

    from wahabot.ai.tools.slim import slim_tool_spec

    spec: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "send_text",
            "description": "Send a text reply.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "chat": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "description": "Optional chat JID.",
                        "title": "Chat",
                    },
                    "text": {
                        "description": "Text to send.",
                        "title": "Text",
                        "type": "string",
                    },
                    "kind": {
                        "anyOf": [
                            {"enum": ["a", "b"], "type": "string"},
                            {"type": "null"},
                        ],
                        "default": "a",
                        "title": "Kind",
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    }
    slimmed = slim_tool_spec(spec)

    fn = slimmed["function"]
    assert "strict" not in fn
    params = fn["parameters"]
    assert "title" not in _json.dumps(slimmed)
    assert params["properties"]["chat"] == {
        "type": ["string", "null"],
        "default": None,
        "description": "Optional chat JID.",
    }
    assert params["properties"]["text"] == {
        "description": "Text to send.",
        "type": "string",
    }
    assert params["properties"]["kind"] == {
        "anyOf": [{"enum": ["a", "b"], "type": "string"}, {"type": "null"}],
        "default": "a",
    }
    assert params["required"] == ["text"]
    assert params["additionalProperties"] is False
    # A non-nullable anyOf (a real union) must survive.
    real_union = slim_tool_spec(
        {
            "type": "function",
            "function": {
                "name": "u",
                "parameters": {
                    "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "int"}]}}
                },
            },
        }
    )
    assert real_union["function"]["parameters"]["properties"]["x"]["anyOf"] == [
        {"type": "string"},
        {"type": "int"},
    ]
    nullable_array = slim_tool_spec(
        {
            "type": "function",
            "function": {
                "name": "array",
                "strict": True,
                "parameters": {
                    "properties": {
                        "x": {
                            "anyOf": [
                                {"type": "array", "items": {"type": "string"}},
                                {"type": "null"},
                            ]
                        }
                    }
                },
            },
        }
    )
    assert nullable_array["function"]["strict"] is True
    assert nullable_array["function"]["parameters"]["properties"]["x"] == {
        "type": ["array", "null"],
        "items": {"type": "string"},
    }
    # The input spec is untouched.
    assert spec["function"]["parameters"]["properties"]["chat"]["title"] == "Chat"
    assert spec["function"]["strict"] is False


def test_prepared_tool_specs_are_slimmed_and_valid() -> None:
    """The LLM request's tool specs are slimmed, all tools present.

    End to end through ``ObservableOpenAILike``: the override the chat
    path uses must slim the specs ``_prepare_chat_with_tools`` builds
    from the real bundled tools — every tool survives, no ``title``
    or ``strict: false`` padding rides the payload, and the schemas
    still validate a sample call (the wire format the model answers
    in is unchanged, only the documentation padding shrank).
    """

    from wahabot.ai.tools import build_default_tools
    from wahabot.ai.workflow import ObservableOpenAILike
    from wahabot.settings import Settings

    settings = Settings(
        waha_url="http://x",
        waha_api_key="k",
        webhook_hmac_key="h",
        llm_api_base="http://llm.invalid",
        llm_api_key="k",
        shell_tool=True,
        _env_file=None,
    )
    llm = ObservableOpenAILike(
        model="test-model",
        api_base="http://llm.invalid",
        api_key="k",
        is_chat_model=True,
        is_function_calling_model=True,
    )
    tools = build_default_tools(WahaClient("http://x", "k"), settings)
    prepared = llm._prepare_chat_with_tools(  # pyright: ignore[reportPrivateUsage]
        tools, chat_history=[]
    )
    specs = prepared["tools"]

    assert len(specs) == len(tools)
    names = {spec["function"]["name"] for spec in specs}
    assert names == {tool.metadata.name for tool in tools}

    def json_keys(spec: dict[str, Any]) -> set[str]:
        keys: set[str] = set()

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                keys.update(node)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(spec)
        return keys

    all_keys = set().union(*(json_keys(spec) for spec in specs))
    assert "title" not in all_keys, "title padding survived"
    assert "strict" not in all_keys, "strict flag survived"
    assert "anyOf" not in all_keys, "nullable unions survived"
    for spec in specs:
        for field in spec["function"]["parameters"]["properties"].values():
            if field.get("default", object()) is None:
                assert "null" in field["type"]

    # A slimmed schema still accepts a real call's arguments: a
    # minimal valid call (every optional field at its default, every
    # required field filled) must validate against the unchanged
    # Pydantic model — the wire format the model answers in.
    sample: dict[str, Any] = {
        "text": "hola",
        "command": "ls",
        "url": "https://x.example",
        "query": "q",
        "report": "r",
        "message_id": "m",
        "kind": "image",
        "mode": "list",
    }
    for tool in tools:
        schema = tool.metadata.fn_schema
        assert schema is not None, f"{tool.metadata.name}: no fn_schema"
        call = {
            name: sample.get(name, "x") if field.is_required() else field.default
            for name, field in schema.model_fields.items()
        }
        call["reason"] = "prueba"
        if tool.metadata.name == "notebook":
            call["mode"] = "read"
        schema.model_validate(call)
