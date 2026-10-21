import asyncio
import datetime
import unittest.mock
from typing import Any, cast

import pytest
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.tools import ToolSelection

from tests.harness import (
    CHAT_ID,
    SESSION,
)
from tests.unit.support import (
    _conflicting_waha,
    _group_event,
    _group_waha,
)
from wahabot.ai.context import (
    is_emoji_narration,
    is_silence_narration,
    is_single_emoji,
    participant_names,
    quoted_participant,
    render_system_prompt,
    roster_cache,
    sender_tag,
)
from wahabot.ai.history import (
    chat_visible_text,
)
from wahabot.core.models import WahaEvent
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings


def test_log_action_reason() -> None:
    """``log_action_reason`` logs the justification, warns when absent.

    The reason is capped at 200 chars and rides the log kwargs; a
    missing/blank reason is the model acting without articulating why
    — worth a WARNING so the operator sees it in the audit trail.
    """
    from loguru import logger

    from wahabot.ai.tools.whatsapp import log_action_reason

    infos: list[tuple[str, dict[str, Any]]] = []
    warnings: list[tuple[str, dict[str, Any]]] = []

    def record(message: str, **kwargs: Any) -> None:
        (warnings if "carried no reason" in message else infos).append((message, kwargs))

    with (
        unittest.mock.patch.object(logger, "info", record),
        unittest.mock.patch.object(logger, "warning", record),
    ):
        log_action_reason("send_message", "directly asked by name", chat=CHAT_ID)
        log_action_reason("stay_silent", "  ")
        log_action_reason("react_to_message", "x" * 500)
    assert [tool for _, kw in infos for tool in [kw["tool"]]] == [
        "send_message",
        "react_to_message",
    ]
    assert infos[0][1]["reason"] == "directly asked by name"
    assert infos[0][1]["suffix"] == f" {{'chat': '{CHAT_ID}'}}"
    assert len(infos[1][1]["reason"]) == 200  # the 500-char reason was capped
    assert warnings[0][0] == "Tool call {tool} carried no reason{suffix}"
    assert warnings[0][1] == {"tool": "stay_silent", "suffix": ""}


def test_every_tool_takes_a_reason() -> None:
    """Every bundled tool exposes a ``reason`` parameter end to end.

    The operator's audit goal — read the log and know what the bot did
    and why — needs the model to justify every call. ``reason`` must
    ride the schema (so the LLM sees it) and the function (so the call
    executes); a tool missing either breaks the audit trail silently.
    """
    import inspect

    from wahabot.ai.tools import build_default_tools

    settings = Settings(
        waha_url="http://x",
        waha_api_key="k",
        webhook_hmac_key="h",
        llm_api_base="http://llm.invalid",
        llm_api_key="k",
        shell_tool=True,
        _env_file=None,
    )
    tools = build_default_tools(WahaClient("http://x", "k"), settings)
    assert tools, "no tools built"
    for tool in tools:
        name = tool.metadata.name
        schema = tool.metadata.fn_schema
        assert schema is not None and "reason" in schema.model_fields, (
            f"{name}: schema missing reason"
        )
        fn = cast(Any, tool).fn
        assert "reason" in inspect.signature(fn).parameters, (
            f"{name}: function missing reason"
        )
        assert set(schema.model_fields) == set(inspect.signature(fn).parameters), (
            f"{name}: schema and callable parameters differ"
        )
        assert schema.model_fields["reason"].default == ""
        assert "internal logs" in (schema.model_fields["reason"].description or "")
        spec = tool.metadata.to_openai_tool()["function"]
        assert spec["description"] == tool.metadata.description
        for field, parameter in inspect.signature(fn).parameters.items():
            if parameter.default is inspect.Parameter.empty:
                assert schema.model_fields[field].is_required(), (name, field)
            elif not schema.model_fields[field].is_required():
                assert schema.model_fields[field].default == parameter.default, (
                    name,
                    field,
                )


def test_silence_cancels_other_tools_in_the_same_batch() -> None:
    from llama_index.core.base.llms.types import ChatResponse
    from llama_index.core.workflow import StopEvent

    from wahabot.ai.workflow import FunctionCallingAgentWorkflow

    llm = unittest.mock.Mock()
    llm.metadata.is_function_calling_model = True
    agent = FunctionCallingAgentWorkflow(llm=llm)
    response = ChatResponse(message=ChatMessage(role="assistant", content=""))
    calls = [
        ToolSelection(
            tool_id="read", tool_name="read_chat", tool_kwargs={"mode": "list"}
        ),
        ToolSelection(
            tool_id="silent",
            tool_name="stay_silent",
            tool_kwargs={"reason": "not invited"},
        ),
    ]
    with (
        unittest.mock.patch.object(
            agent, "stop_with", new_callable=unittest.mock.AsyncMock
        ) as stop,
        unittest.mock.patch.object(
            agent, "remember", new_callable=unittest.mock.AsyncMock
        ) as remember,
        unittest.mock.patch.object(agent, "log_silence_reason"),
    ):
        asyncio.run(agent.route_tool_calls(unittest.mock.Mock(), response, calls, 1))
        remember.assert_not_called()
        event = stop.call_args.args[1]
        assert isinstance(event, StopEvent)
        # Nothing is staged for storage: the wrap-up note is notebook-only.
        assert stop.call_args.kwargs == {"silent": True}


def test_tool_call_log_extra() -> None:
    """The per-call log context: reason first, then args; gaps flagged.

    A missing reason must be visible as such — the audit line's whole
    job is answering "why", so a call the model never justified prints
    ``reason: (model gave none)`` instead of quietly looking bare.
    """
    from llama_index.core.tools import ToolSelection

    from wahabot.ai.workflow import tool_call_log_extra

    def call(name: str, **kwargs: Any) -> ToolSelection:
        return ToolSelection(tool_id="id", tool_name=name, tool_kwargs=kwargs)

    assert (
        tool_call_log_extra(
            call("run_shell_command", command="docker logs wahabot", reason="check crash")
        )
        == " (reason: check crash; args: command='docker logs wahabot')"
    )
    assert tool_call_log_extra(call("web_search", query="python 3.14")) == (
        " (reason: (model gave none); args: query='python 3.14')"
    )
    assert tool_call_log_extra(call("recent_chats")) == (" (reason: (model gave none))")
    # A lone reason with no other arguments: no trailing args part.
    assert (
        tool_call_log_extra(call("stay_silent", reason="banter between others"))
        == " (reason: banter between others)"
    )


def test_is_silence_narration_catches_leaked_tool_token() -> None:
    """A leaked ``stay_silent`` tool token is silence chatter, not an answer.

    The trace shows the model emitting the literal tool name ``stay_silent``
    as a plain-text reply instead of calling the tool; those go straight to
    the chat, so ``is_silence_narration`` must treat the token as silence.
    It stays anchored to the *whole* reply — a real sentence containing the
    word must still go through.
    """
    assert is_silence_narration("stay_silent")
    assert is_silence_narration(" stay_silent ")
    assert not is_silence_narration(
        "I could say stay_silent here but I'll answer anyway."
    )


def test_is_silence_narration_catches_spanish() -> None:
    """Spanish silence narration is the same bug as the English one.

    The chat is bilingual (English and Spanish), and a model that
    narrates "I'll stay silent" in one language will do it in the
    other: ``Sin respuesta.`` reaching the chat is the identical
    failure to ``No response.``. The Spanish patterns mirror the
    English ones shape for shape — a real Spanish sentence must never
    match, same as English.
    """
    assert is_silence_narration("Sin respuesta.")
    assert is_silence_narration("Me quedo callado.")
    assert is_silence_narration("Prefiero quedarme en silencio.")
    assert is_silence_narration("Nada que añadir.")
    assert is_silence_narration("No tengo nada más que decir.")
    assert is_silence_narration("No voy a responder.")
    assert not is_silence_narration("Me quedo callado entonces, pero mañana te cuento")
    assert not is_silence_narration("el silencio de la noche me gusta")
    assert not is_silence_narration("sin respuesta tuya no puedo decidir, dime tú")
    assert not is_silence_narration("nada que decirte aún, espera la noticia")


def test_is_silence_narration_catches_german() -> None:
    """German silence narration is the same bug as the English one.

    The chat's languages are covered pattern for pattern: ``Keine
    Antwort.`` reaching the chat is the identical failure to ``No
    response.``. The German patterns mirror the English/Spanish
    shapes — a real German sentence must never match.
    """
    assert is_silence_narration("Keine Antwort.")
    assert is_silence_narration("Nichts zu sagen.")
    assert is_silence_narration("Ich bleibe still.")
    assert is_silence_narration("Nicht an mich gerichtet.")
    assert is_silence_narration("Ich werde nicht antworten.")
    assert is_silence_narration("Ich habe reagiert, damit bin ich fertig.")
    assert not is_silence_narration("Keine Antwort von dir gestern, alles gut?")
    assert not is_silence_narration("ich bleibe still wenn du das willst, aber sag mir")
    assert not is_silence_narration("die Stille hier drüben ist seltsam")


def test_is_single_emoji_catches_lone_emoji() -> None:
    """A lone emoji reply is intercepted — it's a reaction, not a message.

    The model sometimes outputs a single emoji as text instead of calling
    ``react_to_message``.  ``is_single_emoji`` catches these so the handler
    can convert them to reactions or silence.
    """
    assert is_single_emoji("👋")
    assert is_single_emoji("🤣")
    assert is_single_emoji("😂")
    assert is_single_emoji("👍")
    assert is_single_emoji("❤")
    assert is_single_emoji("🔥")
    # Surrounding whitespace is tolerated.
    assert is_single_emoji(" 👋 ")
    assert is_single_emoji("  🤣\n")


def test_is_single_emoji_passes_multi_emoji() -> None:
    """Multi-emoji strings are real messages — they must NOT be intercepted."""
    assert not is_single_emoji("🤣🤣🤣")
    assert not is_single_emoji("😂👍")
    assert not is_single_emoji("🤣😂")
    assert not is_single_emoji("👋🔥❤")


def test_is_single_emoji_passes_text() -> None:
    """Plain text and text-with-emoji must pass through as normal messages."""
    assert not is_single_emoji("hola 👋")
    assert not is_single_emoji("me sirve pa' reírme 🤣")
    assert not is_single_emoji("hello")
    assert not is_single_emoji("stay_silent")
    assert not is_single_emoji("")
    assert not is_single_emoji("   ")


def test_is_emoji_narration_catches_reaction_report() -> None:
    """Emoji + a report about that reaction is never a chat message.

    A 2026-09-25 production trace shows the model *writing*
    ``👍\\nReaccioné con 👍 a…`` instead of calling
    ``react_to_message`` — the exact leak ``is_single_emoji`` cannot
    catch (the narration rides the emoji) and ``is_silence_narration``
    cannot catch (Spanish, and not about staying silent). The shape —
    one emoji, then a line about the bot's own action — is language-
    agnostic: the verb list covers the chat's English and Spanish.
    The strings below are reconstructed examples of the pattern, not
    verbatim chat content.
    """
    assert is_emoji_narration(
        "👍\nReaccioné con 👍 al comentario anterior, sin escribir texto."
    )
    assert is_emoji_narration(
        "🙄\nReaccioné con 🙄 a la broma de antes para no romper el hilo."
    )
    assert is_emoji_narration("👀\nEnvié una reacción a la pregunta.")
    assert is_emoji_narration("😅 I reacted with 😅 to that one, no text.")
    assert is_emoji_narration("👍\nI already reacted, so I'm done here.")
    assert is_emoji_narration("👍\nHabe mit 👍 reagiert, ohne zu schreiben.")
    assert is_emoji_narration("🙄 Ich habe mit 🙄 reagiert und bleibe still.")


def test_is_emoji_narration_passes_real_messages() -> None:
    """An emoji opening a real message is a message, not a self-report.

    The narration filter is anchored to the *second line's verb*: it
    must not swallow genuine chat text that happens to start with an
    emoji — the line after the emoji has to be about the bot's own
    reaction/send for the reply to count as narration.
    """
    assert not is_emoji_narration("😂 esa fue buena, me acordé de ayer")
    assert not is_emoji_narration("😅 pues yo lo vi ayer, fue hace un año")
    assert not is_emoji_narration("👍")
    assert not is_emoji_narration("no tengo ni idea, pregunta mañana")
    assert not is_emoji_narration("🤣🤣🤣")
    assert not is_emoji_narration("")
    assert not is_emoji_narration("   ")


def test_participant_names_backfills_from_messages() -> None:
    """A bare LID roster gains names from recent messages' notifyName."""
    roster_cache.clear()
    names = participant_names(_group_waha(), SESSION, CHAT_ID)
    assert names == {
        "491555000007@lid": "Milo Petrov",
        "491555000009@lid": "Tavo Ríos",
    }


def test_participant_names_roster_wins_over_backfill() -> None:
    """Roster names are authoritative; the message walk only backfills."""
    roster_cache.clear()
    names = participant_names(_conflicting_waha(), SESSION, CHAT_ID)
    assert names == {"491555000007@lid": "Roster Name"}


def test_sender_tag_group_renders_name_and_jid() -> None:
    event = _group_event(name="Milo Petrov")
    assert sender_tag(event) == "[Milo Petrov <491555000007@lid>]"


def test_sender_tag_group_resolves_name_from_roster() -> None:
    event = _group_event()  # no notifyName on the event itself
    names = {"491555000007@lid": "Milo Petrov"}
    assert sender_tag(event, names) == "[Milo Petrov <491555000007@lid>]"


def test_sender_tag_group_unknown_name_falls_back_to_jid() -> None:
    assert sender_tag(_group_event()) == "[491555000007@lid]"


def test_sender_tag_dm_keeps_bare_name() -> None:
    dm = WahaEvent(
        id="dm-tag",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={
            "from": "491555000001@c.us",
            "body": "hi",
            "_data": {"notifyName": "Smoke Sender"},
        },
    )
    assert sender_tag(dm) == "[Smoke Sender]"


def test_sender_tag_dm_without_name_yields_empty() -> None:
    """A DM with neither notifyName nor participant has nothing to show.

    Today's documented fallback: an empty tag (should not happen —
    real DM events always carry one of the two).
    """
    dm = WahaEvent(
        id="dm-tag2",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={"from": "491555000001@c.us", "body": "hi"},
    )
    assert sender_tag(dm) == ""


def test_sender_tag_normalizes_jid_object() -> None:
    event = _group_event(participant={"_serialized": "491555000007@lid"})
    assert sender_tag(event) == "[491555000007@lid]"


def test_quoted_participant_renders_name_and_jid() -> None:
    reply = {
        "participant": {"_serialized": "491555000007@lid"},
    }
    names = {"491555000007@lid": "Milo Petrov"}
    assert quoted_participant(reply, names) == "Milo Petrov <491555000007@lid>"


def test_quoted_participant_notify_name_wins() -> None:
    reply = {
        "participant": "491555000009@lid",
        "_data": {"notifyName": "Tavo Ríos"},
    }
    assert quoted_participant(reply, {"491555000009@lid": "wrong"}) == (
        "Tavo Ríos <491555000009@lid>"
    )


def test_quoted_participant_unknown_falls_back_to_bare_id() -> None:
    assert quoted_participant({"participant": "491555000009@lid"}) == "491555000009"


def test_render_system_prompt_substitutes_own_identities() -> None:
    prompt = "You are {{own_identities}} — a message that names you.\n{{date}}"
    out = render_system_prompt(prompt, own_jid="4915@c.us", own_lid="4915@lid")
    assert "You are `4915@c.us` or `4915@lid`" in out
    assert "{{own" not in out


def test_render_system_prompt_drops_identity_lines_when_unknown() -> None:
    prompt = "# Groups\nYou are {{own_identities}} — addresses you.\nRule stays.\n"
    out = render_system_prompt(prompt)
    assert "{{own" not in out
    assert "addresses you" not in out
    assert "Rule stays." in out


def test_render_system_prompt_drops_own_jid_line_when_unknown() -> None:
    out = render_system_prompt("id: {{own_jid}}\nkeep")
    assert "{{own_jid}}" not in out
    assert "id:" not in out
    assert "keep" in out


def test_render_system_prompt_substitutes_operator_name() -> None:
    out = render_system_prompt("Defer to {{operator_name}}.", operator_name="Ada")
    assert out == "Defer to Ada."


def test_render_system_prompt_operator_name_empty_falls_back() -> None:
    out = render_system_prompt("Defer to {{operator_name}}.")
    assert out == "Defer to the operator."
    out = render_system_prompt("Defer to {{operator_name}}.", operator_name="  ")
    assert out == "Defer to the operator."


def test_render_system_prompt_operator_name_in_goal() -> None:
    out = render_system_prompt(
        "Rules.", goal="Serve {{operator_name}}", operator_name="Ada"
    )
    assert out.startswith("# Goal\n\nServe Ada\n\n")


def test_render_system_prompt_memory_section() -> None:
    out = render_system_prompt("Rules.", goal="Goal.", memory="Never mention X.")
    assert out == "# Goal\n\nGoal.\n\n# Memory\n\nNever mention X.\n\nRules."


def test_render_system_prompt_memory_empty_omitted() -> None:
    assert render_system_prompt("Rules.", goal="Goal.") == "# Goal\n\nGoal.\n\nRules."
    assert render_system_prompt("Rules.", memory="   ") == "Rules."


def test_render_system_prompt_memory_placeholders() -> None:
    now = datetime.datetime(2026, 10, 14, tzinfo=datetime.UTC)
    with unittest.mock.patch("wahabot.ai.context.prompt.datetime.datetime") as clock:
        clock.now.return_value = now
        out = render_system_prompt("Rules.", memory="Today is {{date}}.", tz_name="UTC")
    assert out == "# Memory\n\nToday is Wednesday 14th October, 2026.\n\nRules."
    assert "{{date}}" not in out


def test_render_system_prompt_memory_identity_placeholders() -> None:
    jid = "491555000000@c.us"
    lid = "491555000000@lid"
    out = render_system_prompt(
        "Rules.",
        goal="Account {{own_jid}}.",
        memory="Account {{own_jid}} and {{own_lid}}: {{own_identities}}.",
        own_jid=jid,
        own_lid=lid,
    )
    assert out == (
        f"# Goal\n\nAccount {jid}.\n\n# Memory\n\n"
        f"Account {jid} and {lid}: `{jid}` or `{lid}`.\n\nRules."
    )
    assert "{{own_" not in out


def test_render_system_prompt_memory_unknown_identity() -> None:
    out = render_system_prompt(
        "Rules.",
        memory=(
            "Account {{own_jid}}.\nLID {{own_lid}}.\nIDs {{own_identities}}.\nKeep this."
        ),
    )
    assert out == "# Memory\n\nKeep this.\n\nRules."


def test_render_system_prompt_rejects_oversized_memory() -> None:
    with pytest.raises(ValueError, match="Session memory exceeds"):
        render_system_prompt("Rules.", memory="fact " * 8100)


def test_chat_visible_text_filters_leaked_tokens() -> None:
    assert chat_visible_text("stay_silent") == ""
    assert chat_visible_text('{"error": {"message": "boom"}}') == ""
    assert chat_visible_text("real answer") == "real answer"
    assert chat_visible_text("") == ""


def test_chat_visible_text_strips_an_appended_self_report() -> None:
    """A private note appended to a real reply is dropped, the reply kept.

    The observed leak shape: the reply, a
    newline, then the model's own third-person action report, often in a
    different language from the chat.
    """
    reply = (
        "Klar, ich schweige ab jetzt zu dir.\n"
        "Sent a final confirmation that I will no longer reply, as requested."
    )
    assert chat_visible_text(reply) == "Klar, ich schweige ab jetzt zu dir."
    # A normal multi-sentence reply is untouched.
    normal = "Yes, that works. I will send it tomorrow."
    assert chat_visible_text(normal) == normal
    # A single line is never stripped: it may be the reply itself.
    assert chat_visible_text("Sent it yesterday.") == "Sent it yesterday."
