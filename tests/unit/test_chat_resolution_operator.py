import json
import unittest.mock
from pathlib import Path
from typing import Any, cast, override

import httpx
import pytest

from tests.harness import (
    CHAT_ID,
    FOREIGN_JID,
    OWN_LID,
    SESSION,
)
from tests.unit.support import (
    _journal_fixture_lines,
    _write_journal,
)
from wahabot.ai.tools.whatsapp import (
    OPERATOR_ARMED,
    OPERATOR_KEY,
    dangling_mentions,
    deliver_chat_text,
    ordered_merge,
    resolve_mentions,
)
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings


def test_resolve_chat_operator_run_searches_contacts(unit_settings: Settings) -> None:
    """An operator run can use the bounded contact-book fallback.

    ``wahabot tell`` commands resolve against the operator's chat list
    first (contacts as fallback) — the fence opens for the trusted
    channel alone, so the cross-chat JIDs the send tools need stay
    reachable exactly there.
    """
    import json as _json

    from wahabot.ai.tools.whatsapp import (
        RunTarget,
        bind_target,
        read_chat,
        reset_target,
    )

    class ContactBookWaha:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def list_chats(self, session: str) -> list[dict[str, Any]]:
            self.calls.append("chats")
            return []

        def list_contacts(self, session: str) -> list[dict[str, Any]]:
            self.calls.append("contacts")
            return [{"id": "491999999999@c.us", "name": "Family"}]

    waha = ContactBookWaha()
    tool = cast(Any, read_chat(cast(Any, waha), unit_settings)).fn
    token = bind_target(RunTarget(session=SESSION, chat_id=CHAT_ID, armed=True))
    try:
        hit = _json.loads(tool(mode="resolve", name="Family"))
    finally:
        reset_target(token)
    assert hit["ok"] and hit["matches"] == [{"id": "491999999999@c.us", "name": "Family"}]
    assert waha.calls == ["chats", "contacts"]
    assert OPERATOR_KEY and OPERATOR_ARMED == "armed"


@pytest.mark.parametrize("shell_enabled", [True, False])
def test_chat_reader_documents_available_spill_access(
    unit_settings: Settings, shell_enabled: bool
) -> None:
    from wahabot.ai.tools import build_default_tools

    settings = unit_settings.model_copy(update={"shell_tool": shell_enabled})
    tools = build_default_tools(unittest.mock.Mock(), settings)
    names = {tool.metadata.name for tool in tools}
    description = next(
        tool.metadata.description for tool in tools if tool.metadata.name == "read_chat"
    )
    assert ("run_shell_command" in names) is shell_enabled
    assert "not all history" in description
    assert "#m<n>" in description
    assert "Never quote unseen content" in description
    assert ("Read the raw spill via run_shell_command" in description) is shell_enabled


def test_chat_reader_mode_limits_and_name_caps(unit_settings: Settings) -> None:
    from wahabot.ai.tools.whatsapp import RunTarget, bind_target, read_chat, reset_target

    waha = unittest.mock.Mock()
    waha.list_chats.return_value = [
        {"id": f"{i}@g.us", "name": f"Family {i}"} for i in range(10)
    ]
    token = bind_target(RunTarget(session=SESSION, chat_id=CHAT_ID, armed=True))
    try:
        tool = cast(Any, read_chat(waha, unit_settings))
        recent = json.loads(str(tool(mode="recent", limit=100).content))
        assert recent["ok"] is True
        waha.list_chats.assert_called_with(SESSION, limit=30)
        resolved = json.loads(str(tool(mode="resolve", name="  Family  ").content))
        assert resolved["ok"] is True and len(resolved["matches"]) == 5
        waha.list_contacts.assert_not_called()
        for mode in ("list", "search"):
            refused = json.loads(str(tool(mode=mode, query="q", limit=0).content))
            assert refused["ok"] is False and "positive" in refused["error"]
    finally:
        reset_target(token)
    waha.fetch_chat_messages.assert_not_called()
    waha.search_messages.assert_not_called()


def test_journal_chat_messages_keeps_both_directions_deduped(tmp_path: Path) -> None:
    """The reader collects both directions: one filter is the trap.

    A member's row has ``from`` = the chat, an account row has ``to`` =
    the chat with ``fromMe`` — collecting only ``from``-side rows is
    exactly how a shared account silently disappears from a register.
    Reactions, other chats and redelivered duplicates stay out; a torn
    line from a crashed append is skipped, not raised.
    """
    from wahabot.core.journal import chat_messages

    events = tmp_path / "events" / SESSION
    events.mkdir(parents=True)
    _write_journal(events, _journal_fixture_lines(), torn=True)

    payloads = chat_messages(tmp_path / "events", SESSION, CHAT_ID)

    ids = [payload["id"] for payload in payloads]
    assert ids == [
        f"true_{CHAT_ID}_OP_{OWN_LID}",
        f"true_{CHAT_ID}_BOT_{OWN_LID}",
        f"false_{CHAT_ID}_MEMBER_491555000001@c.us",
    ]


def test_read_chat_export_labels_teammates_and_spills(
    tmp_path: Path, unit_settings: Settings
) -> None:
    """The export keeps both teammates visible, per name and count.

    The envelope names the account's presence itself (one count per
    author) and the rows carry identity: operator rows keep the
    account's display name, bot rows drop it — the shared name belongs
    to the operator, and on a bot row it would read as his words.
    """
    from wahabot.ai.tools.whatsapp import RunTarget, bind_target, read_chat, reset_target

    settings = unit_settings.model_copy(update={"data_dir": tmp_path})
    events = tmp_path / "events" / SESSION
    events.mkdir(parents=True)
    _write_journal(events, _journal_fixture_lines())

    waha = unittest.mock.Mock()
    tool = cast(Any, read_chat(waha, settings)).fn
    token = bind_target(RunTarget(session=SESSION, chat_id=CHAT_ID))
    try:
        out = json.loads(tool(mode="export"))
    finally:
        reset_target(token)

    assert out["ok"] is True
    assert out["recorded"] == 3 and out["returned"] == 3
    assert out["account_messages"] == {"operator": 1, "bot": 1}
    assert out["members"] == {"Alex Rivers": 1}
    assert out["span"] == [100, 300]
    rows = json.loads(Path(out["file"]["path"]).read_text(encoding="utf-8"))
    operator_row, bot_row = rows[0], rows[1]
    assert operator_row["author"] == "operator"
    assert operator_row["name"] == "Operator Human"
    assert bot_row["author"] == "bot"
    assert "name" not in bot_row
    assert len(bot_row["body"]) > 200
    waha.fetch_chat_messages.assert_not_called()


def test_read_chat_export_limit_caps_rejects_and_fences(
    tmp_path: Path, unit_settings: Settings
) -> None:
    """Limit keeps the newest recorded rows; the fence still applies.

    A chat run may export only its own conversation; an operator run
    keeps the cross-chat reach every other read mode has. The limit
    caps file rows while ``recorded`` keeps the whole truth.
    """
    from wahabot.ai.tools.whatsapp import RunTarget, bind_target, read_chat, reset_target

    settings = unit_settings.model_copy(update={"data_dir": tmp_path})
    events = tmp_path / "events" / SESSION
    events.mkdir(parents=True)
    _write_journal(events, _journal_fixture_lines())

    tool = cast(Any, read_chat(unittest.mock.Mock(), settings)).fn
    token = bind_target(RunTarget(session=SESSION, chat_id=CHAT_ID))
    try:
        refused = json.loads(tool(mode="export", limit=0))
        fenced = json.loads(tool(mode="export", chat=FOREIGN_JID))
        capped = json.loads(tool(mode="export", limit=1))
    finally:
        reset_target(token)
    assert refused["ok"] is False and "positive" in refused["error"]
    assert fenced["ok"] is False

    assert capped["ok"] is True
    assert capped["returned"] == 1 and capped["recorded"] == 3
    rows = json.loads(Path(capped["file"]["path"]).read_text(encoding="utf-8"))
    assert rows[0]["id"] == f"false_{CHAT_ID}_MEMBER_491555000001@c.us"

    armed_token = bind_target(RunTarget(session=SESSION, armed=True))
    try:
        across = json.loads(tool(mode="export", chat=CHAT_ID))
    finally:
        reset_target(armed_token)
    assert across["ok"] is True and across["recorded"] == 3


def test_chat_search_scans_one_window_and_filters_available_fields() -> None:
    messages = [
        {"id": "body", "body": "Needle in body"},
        {"id": "filename", "media": {"filename": "NEEDLE.pdf"}},
        {"id": "mimetype", "media": {"mimetype": "text/needle"}},
        {"id": "nested", "_data": {"media": {"filename": "needle.png"}}},
        {"id": "attachment", "media": {"content": "needle"}},
        {"id": "other", "body": "something else"},
    ]
    requests: list[httpx.Request] = []

    def window(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=messages)

    waha = WahaClient("http://waha.invalid", "k", transport=httpx.MockTransport(window))
    matches = waha.search_messages(SESSION, "needle", CHAT_ID, limit=6)
    assert [message["id"] for message in matches] == [
        "body",
        "filename",
        "mimetype",
        "nested",
    ]
    assert len(requests) == 1
    assert requests[0].url.path == "/api/messages"
    assert requests[0].url.params["limit"] == "6"
    assert "query" not in requests[0].url.params


@pytest.mark.parametrize(
    ("source_chat", "destination", "operator", "allowed"),
    [
        (CHAT_ID, CHAT_ID, False, True),
        (FOREIGN_JID, CHAT_ID, False, False),
        (CHAT_ID, FOREIGN_JID, False, False),
        (FOREIGN_JID, FOREIGN_JID, True, True),
    ],
)
def test_forwarding_documents_and_fences_source_and_destination(
    source_chat: str, destination: str, operator: bool, allowed: bool
) -> None:
    from wahabot.ai.tools.whatsapp import (
        RunTarget,
        bind_target,
        forward_message,
        reset_target,
    )

    source = f"false_{source_chat}_SOURCE"
    waha = unittest.mock.Mock()
    waha.forward_message.return_value = f"true_{destination}_FORWARDED"
    target = RunTarget(
        session=SESSION,
        chat_id=CHAT_ID,
        armed=operator,
        allowed_chats={destination} if operator else set(),
    )
    token = bind_target(target)
    try:
        outcome = cast(Any, forward_message(waha))(message_id=source, chat=destination)
    finally:
        reset_target(token)
    envelope = json.loads(str(outcome.content))
    assert envelope["ok"] is allowed
    if allowed:
        assert envelope["message_id"] == source
        assert envelope["chat"] == destination
        assert target.sent == destination
        waha.forward_message.assert_called_once_with(SESSION, destination, source)
    else:
        assert target.sent == ""
        waha.forward_message.assert_not_called()


@pytest.mark.parametrize(
    "message_id",
    ["sent-as-string", {"_serialized": "sent-nested"}, None, {"_serialized": None}],
)
def test_waha_send_ids_accept_documented_shapes(message_id: Any) -> None:
    expected = (
        message_id
        if isinstance(message_id, str)
        else cast(dict[str, Any], message_id or {}).get("_serialized") or ""
    )

    def forwarded(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["chatId"] == CHAT_ID
        assert payload["messageId"] == "source"
        return httpx.Response(200, json={"id": message_id})

    waha = WahaClient(
        "http://waha.invalid", "k", transport=httpx.MockTransport(forwarded)
    )
    assert waha.forward_message(SESSION, CHAT_ID, "source") == expected


def test_resolve_mentions() -> None:
    """Tokens resolve against roster user parts; non-members stay out."""
    roster = [
        "111222333444555@lid",
        "491555000001@c.us",
        "666777888999000@lid",
    ]
    assert resolve_mentions("Para @111222333444555", roster) == ["111222333444555@lid"]
    # Full-JID token resolves to the roster's canonical form.
    assert resolve_mentions("hi @491555000001@c.us", roster) == ["491555000001@c.us"]
    # Multiple tokens, roster order, no duplicates.
    assert resolve_mentions(
        "@666777888999000 mira @111222333444555 y @111222333444555", roster
    ) == ["666777888999000@lid", "111222333444555@lid"]
    # Token order wins over roster order; duplicates collapse.
    assert resolve_mentions("@111222333444555 luego @666777888999000", roster) == [
        "111222333444555@lid",
        "666777888999000@lid",
    ]
    # A token naming nobody on the roster invents no mention.
    assert resolve_mentions("cc @999999999999", roster) == []


def test_dangling_mentions() -> None:
    """``dangling_mentions`` returns exactly the tokens the resolver drops."""
    roster = ["111222333444555@lid", "491555000001@c.us"]
    assert dangling_mentions("cc @999999999999 y @111222333444555", roster) == [
        "999999999999"
    ]
    # Stylized tokens dangle — the model's failed tags. A bare @Name is
    # not token-shaped at all (the regex never extracts it), so it
    # neither resolves nor dangles.
    assert dangling_mentions("hey @L@s y @Lorenzo", roster) == ["L@s"]
    # Duplicates collapse; no tokens means nothing dangles.
    assert dangling_mentions("@999999999999 y @999999999999", roster) == ["999999999999"]
    assert dangling_mentions("sin tokens", roster) == []
    # An empty roster (DMs, outages) leaves every token dangling.
    assert dangling_mentions("cc @111222333444555", []) == ["111222333444555"]


def test_ordered_merge() -> None:
    """Explicit mentions keep their order; resolved additions follow."""
    assert ordered_merge(
        ["491555000001@c.us"], ["111222333444555@lid", "491555000001@c.us"]
    ) == ["491555000001@c.us", "111222333444555@lid"]
    assert ordered_merge([], []) == []


def test_deliver_chat_text_resolves_mentions() -> None:
    """Text delivery resolves ``@``-tokens into real mention JIDs.

    The handler's final-reply send and the ``send_message`` tool share
    one delivery core: a ``@<number>`` token naming a roster member
    rides the send as a mention JID, without verifying notification, and text
    naming nobody mentions nobody. A roster outage fails soft — the
    text still goes out, unmentioned.
    """

    class RosterWaha(WahaClient):
        """A WAHA recording sends, whose overview serves a LID roster."""

        def __init__(self) -> None:
            super().__init__(base_url="http://waha.invalid", api_key="k")
            self.sent: list[tuple[str, str, str, list[str] | None]] = []

        @override
        def get_chat_overview(self, session: str, chat_id: str) -> Any:
            return {
                "id": chat_id,
                "participants": [
                    {"id": {"_serialized": "111222333444555@lid"}},
                    {"id": {"_serialized": "491555000000@c.us"}},
                ],
            }

        @override
        def send_text(
            self,
            session: str,
            chat_id: str,
            text: str,
            reply_to: str | None = None,
            mentions: list[str] | None = None,
        ) -> str:
            self.sent.append((session, chat_id, text, mentions))
            return f"true_{chat_id}_SENT{len(self.sent)}"

    class BrokenRosterWaha(RosterWaha):
        @override
        def get_chat_overview(self, session: str, chat_id: str) -> Any:
            raise RuntimeError("roster unavailable")

    waha = RosterWaha()
    text = "listado vacío, @111222333444555, pero documentado"
    sent_id, mentions, dangling = deliver_chat_text(waha, "default", "123@g.us", text)
    assert mentions == ["111222333444555@lid"]
    assert dangling == []
    assert waha.sent == [("default", "123@g.us", text, ["111222333444555@lid"])]
    assert sent_id == "true_123@g.us_SENT1"
    # No tokens naming members: nothing to mention.
    _, mentions, dangling = deliver_chat_text(
        waha, "default", "123@g.us", "sin menciones aquí"
    )
    assert mentions == []
    assert dangling == []
    # A token naming nobody on the roster dangles — the send still goes
    # out, and the caller learns which tokens tagged nobody.
    _, mentions, dangling = deliver_chat_text(
        waha, "default", "123@g.us", "cc @999999999999 y @L@s"
    )
    assert mentions == []
    assert dangling == ["999999999999", "L@s"]
    # A roster outage fails soft: the text still goes out.
    _, mentions, _ = deliver_chat_text(BrokenRosterWaha(), "default", "123@g.us", text)
    assert mentions == []


@pytest.mark.parametrize(
    ("text", "explicit", "expected", "warning"),
    [
        (
            "Hello @491555000001 and @999999999999",
            None,
            ["491555000001@c.us"],
            "auto-resolved",
        ),
        (
            "Hello @Alice",
            ["491555000001@c.us"],
            ["491555000001@c.us"],
            "not confirmed",
        ),
        (
            "Hello @999999999999",
            ["999999999999@c.us"],
            ["999999999999@c.us"],
            "auto-resolved",
        ),
    ],
)
def test_send_message_warning_does_not_claim_notification_failure(
    text: str, explicit: list[str] | None, expected: list[str], warning: str
) -> None:
    from wahabot.ai.tools.whatsapp import (
        RunTarget,
        bind_target,
        reset_target,
        send_message,
    )

    waha = unittest.mock.Mock()
    waha.get_chat_overview.return_value = {"participants": [{"id": "491555000001@c.us"}]}
    waha.fetch_chat_messages.return_value = []
    waha.lid_phone.return_value = ""
    waha.send_text.return_value = f"true_{CHAT_ID}_SENT"
    token = bind_target(RunTarget(session=SESSION, chat_id=CHAT_ID))
    try:
        out = cast(Any, send_message(waha))(text=text, mentions=explicit)
    finally:
        reset_target(token)
    envelope = json.loads(str(out.content))
    assert envelope["ok"] is True
    assert envelope["mentions"] == expected
    assert warning in envelope["warning"]
    assert "nobody was notified" not in envelope["warning"]
    assert waha.send_text.call_args.kwargs["mentions"] == expected


def test_send_message_refuses_blame_text() -> None:
    from wahabot.ai.tools.whatsapp import send_message

    waha = unittest.mock.Mock()

    out = cast(Any, send_message(waha))(text="Te has equivocado, no era eso.")

    envelope = json.loads(str(out.content))
    assert envelope["ok"] is False
    assert "correction-guard" in envelope["error"]
    waha.send_text.assert_not_called()
