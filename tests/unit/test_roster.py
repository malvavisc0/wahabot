import asyncio
import shutil
import unittest.mock
from typing import Any, cast

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer

from tests.harness import (
    CHAT_ID,
    SESSION,
)
from wahabot.ai.context import (
    render_system_prompt,
)
from wahabot.ai.tools.whatsapp import (
    mention_tokens,
    sender_names,
)
from wahabot.core.host import host_context
from wahabot.settings import Settings


def test_sender_names_reads_notify_name() -> None:
    """``sender_names`` extracts, strips, dedups and drops nameless jids.

    A stub feeds messages with the real production shapes; the loop must
    strip whitespace, collapse duplicate senders, and ignore entries with
    no ``notifyName``.
    """

    class StubWaha:
        def fetch_chat_messages(
            self,
            _session: str,
            _chat_id: str,
            limit: int = 100,
        ) -> list[dict[str, Any]]:
            return [
                {
                    "participant": "491555000001@c.us",
                    "_data": {"notifyName": " Smoke "},
                },
                {
                    "participant": "491555000001@c.us",
                    "_data": {"notifyName": " Smoke "},
                },
                {"participant": "491555000002@c.us", "_data": {"notifyName": ""}},
                {
                    "participant": {"_serialized": "491555000003@lid"},
                    "_data": {"notifyName": "Lidler"},
                },
            ]

    names = sender_names(cast(Any, StubWaha()), SESSION, CHAT_ID)
    assert names == {  # stripped, no dup, nameless dropped, jid object resolved
        "491555000001@c.us": "Smoke",
        "491555000003@lid": "Lidler",
    }


def test_host_placeholder() -> None:
    """``{{host}}`` resolves to the machine snapshot in the prompt."""
    rendered = render_system_prompt("Host info:\n{{host}}")
    assert "{{host}}" not in rendered
    assert rendered == f"Host info:\n{host_context()}"
    assert "Host: " in rendered
    assert "- Python: " in rendered


def test_host_lists_only_present_binaries() -> None:
    """The Extras line advertises what shutil.which finds — nothing else.

    The model plans shell commands around this line, so a claimed-but-
    missing binary is a prompt lie: the Dockerfile set must render,
    and a stripped PATH must omit the line wholesale rather than list
    names the environment lacks.
    """
    from wahabot.core import host as host_module

    real_which = shutil.which

    def which(name: str, path: Any = None) -> str | None:
        if name in ("ffmpeg", "jq", "magick"):
            return f"/usr/bin/{name}"
        return real_which(name) if name == "bash" else None

    with (
        unittest.mock.patch("shutil.which", which),
        unittest.mock.patch.dict("os.environ", {"PATH": "/usr/bin"}, clear=False),
    ):
        host_module.available_binaries.cache_clear()
        host_module.host_context.cache_clear()
        try:
            snapshot = host_module.host_context()
        finally:
            host_module.available_binaries.cache_clear()
            host_module.host_context.cache_clear()
    assert "- Extras: ffmpeg, magick (ImageMagick), jq" in snapshot
    # absent binaries are not claimed
    assert "pandoc" not in snapshot
    assert "tesseract" not in snapshot


def test_host_omits_binaries_line_when_none_present() -> None:
    """A bare host renders no Extras line at all, not an empty one."""
    from wahabot.core import host as host_module

    with unittest.mock.patch("shutil.which", return_value=None):
        host_module.available_binaries.cache_clear()
        host_module.host_context.cache_clear()
        try:
            snapshot = host_module.host_context()
        finally:
            host_module.available_binaries.cache_clear()
            host_module.host_context.cache_clear()
    assert "Extras" not in snapshot


def test_remember_strips_thinking_separator(unit_settings: Settings) -> None:
    """``remember`` stores the reply text without the thinking separator.

    Reasoning models split thinking from text with a leading blank line
    inside the text block; memory must keep the words, not the
    separator — the stored prefix costs tokens every turn and teaches
    the model to keep emitting it.
    """

    from llama_index.core.base.llms.types import (
        ChatMessage,
        ChatResponse,
        MessageRole,
        TextBlock,
        ThinkingBlock,
    )
    from llama_index.core.workflow import Context

    from wahabot.ai.workflow import FunctionCallingAgentWorkflow, load_llm

    async def stored_texts() -> list[str]:
        wf = FunctionCallingAgentWorkflow(llm=load_llm(unit_settings))
        ctx = Context(wf)
        await ctx.store.set(
            "memory",
            ChatMemoryBuffer.from_defaults(),  # pyright: ignore[reportUnknownMemberType]
        )
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ThinkingBlock(content="reasoning"),
                TextBlock(text="\n\njaj real reply"),
            ],
        )
        await FunctionCallingAgentWorkflow.remember(
            wf, ctx, ChatResponse(message=message), []
        )
        memory = await ctx.store.get("memory")
        return [
            block.text or ""
            for m in await memory.aget_all()
            for block in m.blocks
            if isinstance(block, TextBlock)
        ]

    assert asyncio.run(stored_texts()) == ["jaj real reply"]


def test_post_delivery_wrap_up_note_becomes_a_notebook_fact(
    unit_settings: Settings,
) -> None:
    """Post-delivery final text is a notebook fact, never an assistant turn.

    Storing the note in history taught the model to append notes to its
    replies; the durable record is the
    notebook copy, so memory keeps only the delivered turn.
    """
    from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
    from llama_index.core.workflow import Context

    from wahabot.ai.messages import WRAP_UP_NOTE_KWARG
    from wahabot.ai.workflow import FunctionCallingAgentWorkflow, load_llm
    from wahabot.core.notebook import Notebook

    async def scenario() -> tuple[list[ChatMessage], str, str]:
        wf = FunctionCallingAgentWorkflow(
            llm=load_llm(unit_settings), notebook=Notebook(unit_settings.data_dir)
        )
        ctx = Context(wf)
        await ctx.store.set(
            "memory",
            ChatMemoryBuffer.from_defaults(),  # pyright: ignore[reportUnknownMemberType]
        )
        # The delivery already fired this run (sticker sent).
        from wahabot.ai.tools.whatsapp import RunTarget, bind_target, reset_target

        target = RunTarget(session=SESSION, chat_id=CHAT_ID)
        token = bind_target(target)
        try:
            target.sent = CHAT_ID
            message = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=(
                    "Made the millennial starter pack meme with PIL and sent "
                    "it as a sticker; send_image rejected the local path."
                ),
            )
            response = ChatResponse(message=message)
            delivered = wf.any_delivery()
            await wf.remember(ctx, response, [], skip_text=delivered)
            if delivered:
                await wf.store_wrap_up_note(str(message.content))
            memory = await ctx.store.get("memory")
            stored = await memory.aget_all()
            final = wf.drop_post_delivery_text(response)
        finally:
            reset_target(token)
        return stored, str(final.message.content or ""), str(message.content or "")

    stored, final_content, note = asyncio.run(scenario())
    # The note never becomes an assistant turn in memory.
    assert all(WRAP_UP_NOTE_KWARG not in m.additional_kwargs for m in stored)
    assert all(note not in str(m.content) for m in stored)
    # The durable record is the notebook copy.
    notes = Notebook(unit_settings.data_dir).read(SESSION, CHAT_ID)
    assert [n.text for n in notes] == [note]
    # The reply to the chat stays empty: the latch holds.
    assert final_content == ""


def test_wrap_up_prompt_used_after_delivery(unit_settings: Settings) -> None:
    """After a delivery, wrap-up calls ask for the self-record lesson.

    ``wrap_up_response`` must pick ``POST_DELIVERY_WRAP_UP_PROMPT``
    over the round-limit prompt when the delivery latch fired, and write
    the produced lesson to the notebook — never to memory.
    """
    from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
    from llama_index.core.workflow import Context

    from wahabot.ai.messages import WRAP_UP_NOTE_KWARG
    from wahabot.ai.workflow import (
        POST_DELIVERY_WRAP_UP_PROMPT,
        FunctionCallingAgentWorkflow,
        load_llm,
    )
    from wahabot.core.notebook import Notebook

    async def scenario() -> tuple[str, list[ChatMessage]]:
        seen: dict[str, str] = {}

        async def fake_achat(self_llm: Any, messages: Any, **kwargs: Any) -> Any:
            seen["prompt"] = str(messages[-1].content or "")
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="Sent the meme as a sticker after the image path failed",
                )
            )

        from wahabot.ai.workflow import ObservableOpenAILike

        original = ObservableOpenAILike.achat
        ObservableOpenAILike.achat = fake_achat  # pyright: ignore[reportAttributeAccessIssue]
        try:
            wf = FunctionCallingAgentWorkflow(
                llm=load_llm(unit_settings),
                notebook=Notebook(unit_settings.data_dir),
            )
            ctx = Context(wf)
            await ctx.store.set(
                "memory",
                ChatMemoryBuffer.from_defaults(),  # pyright: ignore[reportUnknownMemberType]
            )
            from wahabot.ai.tools.whatsapp import RunTarget, bind_target, reset_target

            target = RunTarget(session=SESSION, chat_id=CHAT_ID)
            token = bind_target(target)
            try:
                target.sent = CHAT_ID  # delivery fired
                await wf.wrap_up_response(
                    ctx, "non-delivery round after completed delivery"
                )
                memory = await ctx.store.get("memory")
                return seen["prompt"], await memory.aget_all()
            finally:
                reset_target(token)
        finally:
            ObservableOpenAILike.achat = original

    prompt, stored = asyncio.run(scenario())
    assert POST_DELIVERY_WRAP_UP_PROMPT[:30] in prompt
    assert all(WRAP_UP_NOTE_KWARG not in m.additional_kwargs for m in stored)
    notes = Notebook(unit_settings.data_dir).read(SESSION, CHAT_ID)
    assert len(notes) == 1
    assert "sticker" in notes[0].text


def test_remember_drops_lone_emoji_reply(unit_settings: Settings) -> None:
    """``remember`` never stores a lone-emoji reply as the bot's text.

    The handler converts a lone emoji into a reaction (~50 %) or drops
    it as silence — the chat never sees it as a text reply, so memory
    must not record it (memory mirrors the chat). Delivery still
    receives it: the conversion happens handler-side. Multi-emoji
    strings are real chat text and must stay.
    """

    from llama_index.core.base.llms.types import (
        ChatMessage,
        ChatResponse,
        MessageRole,
        TextBlock,
    )
    from llama_index.core.workflow import Context

    from wahabot.ai.workflow import FunctionCallingAgentWorkflow, load_llm

    async def stored_count(reply: str) -> int:
        wf = FunctionCallingAgentWorkflow(llm=load_llm(unit_settings))
        ctx = Context(wf)
        await ctx.store.set(
            "memory",
            ChatMemoryBuffer.from_defaults(),  # pyright: ignore[reportUnknownMemberType]
        )
        message = ChatMessage(role=MessageRole.ASSISTANT, blocks=[TextBlock(text=reply)])
        await FunctionCallingAgentWorkflow.remember(
            wf, ctx, ChatResponse(message=message), []
        )
        memory = await ctx.store.get("memory")
        return len(await memory.aget_all())

    assert asyncio.run(stored_count("👋")) == 0
    assert asyncio.run(stored_count("🤣🤣🤣")) == 1


def test_warn_accidental_silence() -> None:
    """``warn_accidental_silence`` fires only on the accidental-empty shape.

    The trace-audit case: a reasoning model spends its tokens thinking,
    returns an empty final answer after tool rounds, nothing is
    delivered — indistinguishable at run time from chosen silence, so
    this warning is the only signal in the logs. Chosen silence (first
    round, or a delivery already made) must stay unlogged.
    """
    from collections.abc import Iterator
    from contextlib import contextmanager

    from llama_index.core.base.llms.types import ChatResponse
    from loguru import logger

    from wahabot.ai.tools.whatsapp import RunTarget, bind_target, reset_target
    from wahabot.ai.workflow import FunctionCallingAgentWorkflow

    wf = FunctionCallingAgentWorkflow.__new__(FunctionCallingAgentWorkflow)
    empty = ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""))
    text = ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content="hi"))
    quiet = RunTarget(session=SESSION, chat_id=CHAT_ID)
    delivered = RunTarget(session=SESSION, chat_id=CHAT_ID, sent=CHAT_ID)

    @contextmanager
    def bound(target: RunTarget) -> Iterator[None]:
        token = bind_target(target)
        try:
            yield
        finally:
            reset_target(token)

    logs: list[str] = []

    def record(message: str, **_: Any) -> None:
        logs.append(message)

    with (
        unittest.mock.patch.object(logger, "warning", record),
        bound(quiet),
    ):
        wf.warn_accidental_silence(empty, rounds=8)  # the audit shape: warns
        wf.warn_accidental_silence(empty, rounds=1)  # first-round silence: no
        wf.warn_accidental_silence(text, rounds=8)  # visible answer: no
    with unittest.mock.patch.object(logger, "warning", record), bound(delivered):
        wf.warn_accidental_silence(empty, rounds=8)  # post-delivery quiet: no
    assert logs == [
        "".join(
            (
                "Stopping run: empty final answer after {rounds} rounds ",
                "(nothing delivered, no stay_silent)",
            )
        )
    ]


def test_mention_tokens() -> None:
    """``mention_tokens`` extracts only user-part and JID-shaped tokens."""
    text = "Para @111222333444555 y @491555000001@c.us, no @ana ni mail@x.com"
    assert mention_tokens(text) == ["111222333444555", "491555000001@c.us"]
    assert mention_tokens("no ats here") == []


def test_operator_tools_placeholder_renders() -> None:
    """``{{operator_tools}}`` in the system prompt renders the fence rules.

    The operator-only reach (other chats, contact-book resolution,
    recent conversations) is stated once in the prompt instead of being
    repeated in every tool description — the render must substitute it
    and never leave the placeholder verbatim, on any turn.
    """
    from wahabot.ai.context import render_system_prompt

    out = render_system_prompt("Rules:\n{{operator_tools}}")
    assert "reserved to `[operator command]` turns" in out
    assert "omit `chat` or use its exact JID" in out
    assert "`escalate` is the fixed-destination operator exception" in out
    assert "{{operator_tools}}" not in out
    # A prompt not carrying the placeholder is untouched.
    plain = render_system_prompt("You are {{bot_name}}.")
    assert "operator command" not in plain


def test_resolve_chat_chat_run_matches_roster(unit_settings: Settings) -> None:
    """A chat run resolves names against the chat's own roster only.

    The contact book and chat list never enter a chat-run search: the
    search space is the current chat's participants (the people the
    asker already shares a conversation with), so a participant cannot
    enumerate the operator's contacts. A match returns JID+name for
    mentions; a miss says no match was found in available participant names.
    """
    import json as _json

    from wahabot.ai.tools.whatsapp import (
        RunTarget,
        bind_target,
        read_chat,
        reset_target,
    )

    class RosterWaha:
        def get_chat_overview(self, session: str, chat_id: str) -> Any:
            return {
                "id": chat_id,
                "participants": [
                    {"id": "111222333444555@lid"},
                    {"id": "491555000001@c.us", "name": "Smoke Sender"},
                ],
            }

        def fetch_chat_messages(
            self, session: str, chat_id: str, limit: int = 50
        ) -> list[dict[str, Any]]:
            return [
                {
                    "participant": "111222333444555@lid",
                    "_data": {"notifyName": "Alex Rivers"},
                }
            ]

    tool = cast(Any, read_chat(cast(Any, RosterWaha()), unit_settings)).fn
    token = bind_target(RunTarget(session=SESSION, chat_id=CHAT_ID))
    try:
        hit = _json.loads(tool(mode="resolve", name="  alex rivers  "))
        miss = _json.loads(tool(mode="resolve", name="Family"))
        outsider = _json.loads(tool(mode="resolve", name="kai's operator friend"))
    finally:
        reset_target(token)
    assert hit["ok"] and hit["matches"] == [
        {"id": "111222333444555@lid", "name": "Alex Rivers", "tag": "@111222333444555"}
    ]
    assert not miss["ok"] and "named members" in miss["error"]
    assert not outsider["ok"]
