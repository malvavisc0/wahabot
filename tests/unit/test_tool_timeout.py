import asyncio
import json
import unittest.mock
from pathlib import Path
from typing import Any

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from tests.harness import (
    CHAT_ID,
    OWN_LID,
    SESSION,
)
from tests.unit.support import (
    member_event,
)
from wahabot.ai.messages import (
    addressed_note,
    jid_string,
)
from wahabot.ai.tools.whatsapp import (
    fenced_chat,
)
from wahabot.core.filters import chat_allowed
from wahabot.core.models import WahaEvent
from wahabot.settings import Settings
from wahabot.status import (
    run_timed_out,
)


def test_run_timeout_cap_and_context_reuse(unit_settings: Settings) -> None:
    """The run cap is a per-run wall clock, and a cancelled run leaves
    the shared per-chat Context usable.

    The incident: the library
    enforces ``timeout=`` as a cumulative alive-time budget over the
    whole Context, so on the reused per-chat Context a busy chat's
    budget hit zero and every later message died within seconds. The
    cap now lives around ``agent.run`` in ``handle_message`` — one
    wall clock per run — and the library timeout stays ``None``. Two
    properties must hold:

    1. A run whose LLM call outlives ``WAHABOT_RUN_TIMEOUT`` is
       cancelled and surfaces as ``TimeoutError`` (the class
       ``run_timed_out`` classifies, so the operator notify and
       seen-marker drop behave like the old ``WorkflowTimeoutError``).
    2. Cancellation mid-run does not poison the Context: the next
       message in the same chat runs normally against the same ctx.
    """

    from llama_index.core.workflow import Context

    from wahabot.ai.context import handle_message
    from wahabot.ai.workflow import ObservableOpenAILike, build_agent

    async def scenario() -> tuple[Any, Any]:
        capped = unit_settings.model_copy(update={"run_timeout": 1})
        agent = build_agent(capped, system_prompt="You are a bot.")

        slow: dict[str, Any] = {}

        async def hanging_achat(self_llm: Any, messages: Any, **kwargs: Any) -> Any:
            slow["started"] = True
            await asyncio.sleep(30)  # outlives the 1s run cap
            raise AssertionError("unreachable: the cap cancels first")

        async def quick_achat(self_llm: Any, messages: Any, **kwargs: Any) -> Any:
            from llama_index.core.base.llms.types import ChatResponse

            return ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content="fine")
            )

        ctx = Context(agent)
        original = ObservableOpenAILike.achat
        ObservableOpenAILike.achat = hanging_achat  # pyright: ignore[reportAttributeAccessIssue]
        timed_out: Any = None
        try:
            slow_event = WahaEvent(
                id="slow",
                timestamp=1,
                event="message",
                session=SESSION,
                me={},
                payload={"from": "491555000001@c.us", "body": "hang please"},
            )
            # waha=None: the DM path skips the roster lookup entirely.
            timed_out = await handle_message(slow_event, agent, ctx=ctx, settings=capped)
        except Exception as exc:
            timed_out = exc
        finally:
            ObservableOpenAILike.achat = quick_achat  # pyright: ignore[reportAttributeAccessIssue]
            try:
                next_event = WahaEvent(
                    id="next",
                    timestamp=2,
                    event="message",
                    session=SESSION,
                    me={},
                    payload={"from": "491555000001@c.us", "body": "still there?"},
                )
                reply, target = await handle_message(
                    next_event, agent, ctx=ctx, settings=capped
                )
                recovered = (reply, target)
            except Exception as exc:
                recovered = exc
            ObservableOpenAILike.achat = original
        assert slow.get("started"), "the slow run never reached the LLM call"
        return timed_out, recovered

    timed_out, recovered = asyncio.run(scenario())
    # Property 1: the cap fires as TimeoutError, not WorkflowTimeoutError.
    assert isinstance(timed_out, TimeoutError), timed_out
    assert run_timed_out(timed_out)
    # Property 2: the same Context serves the next run cleanly.
    assert not isinstance(recovered, Exception), recovered
    reply, _target = recovered
    assert reply == "fine"


def test_jid_string_strips_device_suffix() -> None:
    """A linked-device suffix names a device, not a person."""

    assert jid_string("491555000000:41@lid") == "491555000000@lid"
    assert jid_string({"_serialized": "491555000000:3@c.us"}) == "491555000000@c.us"
    assert jid_string("1234567890-1234567890@g.us") == "1234567890-1234567890@g.us"


def test_conversation_jid_uses_to_for_fromme() -> None:
    """WAHA puts the account in ``from`` of its own messages; the chat is ``to``."""
    from wahabot.ai.messages import conversation_jid

    own = WahaEvent(
        id="e",
        timestamp=1,
        event="message.any",
        session=SESSION,
        me={"id": "491555000000@c.us", "lid": OWN_LID},
        payload={"fromMe": True, "from": "491555000000@lid", "to": CHAT_ID},
    )
    assert conversation_jid(own) == CHAT_ID
    incoming = own.model_copy(update={"payload": {"from": CHAT_ID, "fromMe": False}})
    assert conversation_jid(incoming) == CHAT_ID


def test_chat_allowed_whitelists_fromme_by_chat() -> None:
    """An operator message in a whitelisted group passes the whitelist."""

    own = WahaEvent(
        id="e",
        timestamp=1,
        event="message.any",
        session=SESSION,
        me={"id": "491555000000@c.us", "lid": OWN_LID},
        payload={
            "fromMe": True,
            "from": "491555000000@lid",
            "to": CHAT_ID,
            "participant": "491555000000@lid",
        },
    )
    assert chat_allowed(own, {CHAT_ID}, set())


def test_addressed_note_distinguishes_shared_account_cases() -> None:
    """Name, account tag and quote-reply each say what they mean."""
    from wahabot.core.identity import AUTHOR_BOT, AUTHOR_OPERATOR, authors

    named = member_event(body="kai, una pregunta")
    assert "names you" in addressed_note(named, bot_name="kai")

    tagged = member_event(
        body="@491555000000 mira", _data={"mentionedJidList": ["491555000000@lid"]}
    )
    assert "may mean you or your operator" in addressed_note(tagged, bot_name="kai")

    authors.record("true_x_BOTMSG", AUTHOR_BOT)
    authors.record("true_x_OPMSG", AUTHOR_OPERATOR)
    to_bot = member_event(
        body="no estoy de acuerdo",
        replyTo={"id": "BOTMSG", "participant": "491555000000@lid"},
    )
    assert "replies to your message" in addressed_note(to_bot, bot_name="kai")
    to_operator = member_event(
        body="jaja", replyTo={"id": "OPMSG", "participant": "491555000000@lid"}
    )
    assert "your operator typed" in addressed_note(to_operator, bot_name="kai")
    unknown = member_event(
        body="ok", replyTo={"id": "OLD", "participant": "491555000000@lid"}
    )
    assert "you or your operator" in addressed_note(unknown, bot_name="kai")
    other = member_event(
        body="ok", replyTo={"id": "X", "participant": "491555000002@lid"}
    )
    assert addressed_note(other, bot_name="kai") == ""


def test_mentions_note_names_tagged_members() -> None:
    """Bare ``@digits`` get a name, and the shared account is called out."""
    from wahabot.ai.context import mentions_note
    from wahabot.core.identity import names as name_book

    name_book.learn("491555000002@lid", "Ada Lovelace")
    event = member_event(
        body="@491555000002 y @491555000000 y @491555000009",
        _data={
            "mentionedJidList": [
                "491555000002@lid",
                {"_serialized": "491555000000@lid"},
                "491555000009@lid",
            ]
        },
    )
    note = mentions_note(event)
    assert note.startswith("\n[mentions: ")
    assert "@491555000002 is Ada Lovelace <491555000002@lid>" in note
    assert "@491555000000 is the shared account" in note
    assert "@491555000009 is <491555000009@lid>" in note
    assert mentions_note(member_event(body="sin menciones")) == ""


def test_quoted_account_message_names_the_author() -> None:
    """A quote of the shared account says which teammate wrote it."""
    from wahabot.ai.context import reply_context
    from wahabot.core.identity import AUTHOR_BOT, AUTHOR_OPERATOR, authors

    own = {"491555000000@lid"}
    authors.record("BOT1", AUTHOR_BOT)
    authors.record("OP1", AUTHOR_OPERATOR)
    bot_quote = {"id": "BOT1", "participant": "491555000000@lid", "body": "hola"}
    op_quote = {"id": "OP1", "participant": "491555000000@lid", "body": "hey"}
    old_quote = {"id": "OLD", "participant": "491555000000@lid", "body": "hm"}
    assert reply_context(bot_quote, {}, own).startswith("you <491555000000@lid>")
    assert reply_context(op_quote, {}, own).startswith("your operator <")
    assert reply_context(old_quote, {}, own).startswith("this account, you or your")


def test_prepare_mentions_rewrites_names_and_jid_tails() -> None:
    """``@Name`` and ``@digits@lid`` become the ``@digits`` WhatsApp tags."""
    from wahabot.ai.tools.whatsapp import prepare_mentions
    from wahabot.core.identity import names as name_book

    name_book.learn("491555000002@lid", "Ada Lovelace")
    name_book.learn("491555000003@lid", "Ada Byron")
    name_book.learn("491555000004@lid", "Grace Hopper")
    waha = unittest.mock.Mock()
    waha.lid_phone.return_value = ""
    roster = ["491555000002@lid", "491555000003@lid", "491555000004@lid"]

    text, _ = prepare_mentions(waha, SESSION, "hola @491555000004@lid", roster)
    assert text == "hola @491555000004"
    text, _ = prepare_mentions(waha, SESSION, "gracias @Grace, bien", roster)
    assert text == "gracias @491555000004, bien"
    text, _ = prepare_mentions(waha, SESSION, "@Ada Lovelace tiene razón", roster)
    assert text == "@491555000002 tiene razón"
    # Ambiguous first name: left as typed rather than tagging a guess.
    text, _ = prepare_mentions(waha, SESSION, "@Ada?", roster)
    assert text == "@Ada?"
    # Not a mention: an @ glued to a word.
    text, _ = prepare_mentions(waha, SESSION, "L@s Grace", roster)
    assert text == "L@s Grace"
    # Not a mention: an @-handle inside a URL, even when the name matches.
    text, _ = prepare_mentions(
        waha, SESSION, "mira https://example.com/@grace ahora", roster
    )
    assert text == "mira https://example.com/@grace ahora"


def test_prepare_mentions_bridges_lid_to_phone_roster() -> None:
    """A LID token missing from a phone-JID roster is confirmed via WAHA."""
    from wahabot.ai.tools.whatsapp import prepare_mentions

    waha = unittest.mock.Mock()
    waha.lid_phone.return_value = "491555000005@c.us"
    _, roster = prepare_mentions(
        waha, SESSION, "@777888999000111 mira", ["491555000005@c.us"]
    )
    assert "777888999000111@lid" in roster


def test_operator_run_refuses_chatless_tools() -> None:
    """An operator command must name its chat; "operator" never reaches WAHA."""
    from wahabot.ai.tools.whatsapp import RunTarget

    target = RunTarget(session=SESSION, chat_id="operator", armed=True)
    chat, err = fenced_chat(None, target)
    assert chat is None and err and "no default chat" in err
    chat, err = fenced_chat(CHAT_ID, target)
    assert chat == CHAT_ID and err is None


def test_degrade_repairs_reason_only_calls_from_original() -> None:
    """Calls an older squeeze cut to ``reason`` regain their argument keys."""
    from llama_index.core.base.llms.types import ToolCallBlock

    from wahabot.ai.history import degrade_message

    msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[
            ToolCallBlock(
                tool_call_id="c-old",
                tool_name="read_chat",
                tool_kwargs={"reason": "leer el hilo"},
            )
        ],
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "c-old",
                    "type": "function",
                    "function": {
                        "name": "read_chat",
                        "arguments": json.dumps(
                            {"mode": "list", "limit": 30, "reason": "leer el hilo"}
                        ),
                    },
                }
            ]
        },
    )
    (block,) = [b for b in degrade_message(msg).blocks if isinstance(b, ToolCallBlock)]
    assert block.tool_kwargs == {"mode": "list", "limit": 30, "reason": "leer el hilo"}


def test_identity_books_persist(tmp_path: Path) -> None:
    """Authors and names survive a restart; a corrupt file starts empty."""
    from wahabot.core import identity

    identity.configure(tmp_path, SESSION)
    identity.authors.record("true_g@g.us_ABC_491555000000@lid", identity.AUTHOR_BOT)
    identity.names.learn("491555000002@lid", "Ada Lovelace")
    identity.authors.flush(force=True)
    identity.names.flush(force=True)
    identity.authors.clear()
    identity.names.clear()
    identity.configure(tmp_path, SESSION)
    assert identity.authors.author("ABC") == identity.AUTHOR_BOT
    assert identity.names.name("491555000002@lid") == "Ada Lovelace"
    (tmp_path / "identity" / SESSION / "names.json").write_text("{nope")
    identity.configure(tmp_path, SESSION)
    assert identity.names.name("491555000002@lid") == ""


def test_identity_flush_window_saved_by_exit_hook(tmp_path: Path) -> None:
    """A put inside the rate-limit window defers to disk, not to memory.

    The first write flushes at once; a second put within
    ``_FLUSH_INTERVAL_S`` leaves the file stale while the in-memory
    view is current — the exit hook's forced flush is what saves the
    window's last write.
    """
    from wahabot.core import identity

    identity.configure(tmp_path, SESSION)
    names_json = tmp_path / "identity" / SESSION / "names.json"
    identity.names.learn("491555000002@lid", "Ada Lovelace")
    assert names_json.exists()
    identity.names.learn("491555000002@lid", "Ada Byron")
    assert identity.names.name("491555000002@lid") == "Ada Byron"
    assert "Ada Byron" not in names_json.read_text()
    identity.flush_on_exit()
    assert "Ada Byron" in names_json.read_text()


def test_identity_failed_write_stays_dirty(tmp_path: Path) -> None:
    """A failed flush keeps the book dirty, so a later flush retries.

    Without the retry flag, a transient write error (a full disk, a
    blocked path) would silently drop every entry learned since the
    last good write.
    """
    from wahabot.core import identity

    identity.configure(tmp_path, SESSION)
    names_json = tmp_path / "identity" / SESSION / "names.json"
    blocker = names_json.with_suffix(".tmp")
    blocker.mkdir(parents=True)  # write_text onto a directory fails
    identity.names.learn("491555000002@lid", "Ada Lovelace")
    assert not names_json.exists()
    blocker.rmdir()
    identity.names.flush(force=True)
    assert "Ada Lovelace" in names_json.read_text()


def test_identity_book_evicts_oldest_at_cap() -> None:
    """The cap evicts least-recently-put entries; a re-put refreshes."""
    from wahabot.core.identity import JsonBook

    book = JsonBook(3)
    for i in range(3):
        book.put(f"k{i}", "v")
    book.put("k3", "v")
    assert book.get("k0") == ""
    assert book.get("k3") == "v"
    book.put("k1", "v2")
    book.put("k4", "v")
    assert book.get("k2") == ""
    assert book.get("k1") == "v2"


def test_sender_fields_name_and_author() -> None:
    """Fetched messages carry participant, learned name, and account author."""
    from wahabot.ai.tools.whatsapp import sender_fields
    from wahabot.core.identity import AUTHOR_BOT, authors
    from wahabot.core.identity import names as name_book

    name_book.learn("491555000002@lid", "Ada Lovelace")
    member = {"participant": "491555000002@lid", "_data": {}}
    assert sender_fields(member) == {
        "participant": "491555000002@lid",
        "name": "Ada Lovelace",
    }
    account = {"id": "true_g@g.us_ACC", "fromMe": True, "_data": {}}
    assert sender_fields(account)["author"] == "unknown"
    authors.record("ACC", AUTHOR_BOT)
    assert sender_fields(account)["author"] == AUTHOR_BOT


@pytest.mark.parametrize("api", ["", "http://img.invalid:9196"])
def test_image_tool_registered_only_with_an_image_api(
    unit_settings: Settings, api: str
) -> None:
    from wahabot.ai.tools import build_default_tools

    settings = unit_settings.model_copy(update={"image_api_base": api})
    tools = build_default_tools(unittest.mock.Mock(), settings)
    names = {tool.metadata.name for tool in tools}
    assert ("generate_image" in names) is bool(api)
    assert ("edit_image" in names) is bool(api)


def test_image_generate_decodes_the_openai_envelope(
    unit_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import base64 as b64

    from wahabot.core import images

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"data": [{"b64_json": b64.b64encode(b"PNG").decode()}]}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def post(self, url: str, json: dict[str, object]) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(images.httpx, "Client", FakeClient)
    settings = unit_settings.model_copy(
        update={"image_api_base": "http://img.invalid:9196"}
    )
    assert images.generate(settings, "a caption", "1024x1024") == b"PNG"
    off = unit_settings.model_copy(update={"image_api_base": ""})
    assert images.generate(off, "a caption", "") is None


def test_image_metadata_and_temp_save() -> None:
    """metadata reads PNG dimensions; save lands in the temp image dir."""
    import struct

    from wahabot.core import images

    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", 1024, 768)
    png += b"\x00" * 8
    assert images.metadata(png) == {
        "bytes": len(png),
        "format": "png",
        "width": 1024,
        "height": 768,
    }
    assert images.metadata(b"not an image")["format"] == "unknown"
    path = images.save(b"PNG")
    assert path.exists() and path.parent.name == "wahabot-images"
    assert path.read_bytes() == b"PNG"
