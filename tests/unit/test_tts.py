import base64
import json
import tempfile
import unittest.mock
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast, override

import httpx
import pytest

from tests.harness import (
    CHAT_ID,
    FOREIGN_JID,
    SESSION,
)
from tests.unit.support import (
    _PIN_MESSAGES,
    _mock_httpx_client,
    _PinWaha,
)
from wahabot.ai.resolve import chat_context_note, resolve_last_message
from wahabot.ai.tools.whatsapp import (
    fit_messages,
    image_file,
    infer_mimetype,
    local_file,
    remote_file,
    search_matches,
    video_file,
)
from wahabot.ai.tools.whatsapp.media_files import (
    _DOC_MIME_BY_EXT as DOC_MIME_BY_EXT,  # pyright: ignore[reportPrivateUsage]
)
from wahabot.ai.tools.whatsapp.media_files import (
    _VIDEO_MIME_BY_EXT as VIDEO_MIME_BY_EXT,  # pyright: ignore[reportPrivateUsage]
)
from wahabot.commands import build_command_event, chat_display_name
from wahabot.core.tts import synthesize
from wahabot.reactions import is_own_message_id
from wahabot.settings import Settings


def test_synthesize_wire_shape(unit_settings: Settings) -> None:
    """synthesize POSTs the OpenAI speech shape and returns the mp3 bytes.

    The voice and instruct come from config per language — never from
    the model — and an empty instruct entry means the key is omitted
    entirely (the `_casual` voices carry their own delivery).
    """
    unit_settings.tts_url = "http://tts.invalid"
    unit_settings.tts_voices = {"es": "vd_spanish_male", "en": "vd_british_male_casual"}
    unit_settings.tts_instruct = {
        "es": "spoken casually, like teasing a friend in a group chat",
        "en": "",
    }
    unit_settings.tts_timeout = 7.5
    requests: list[httpx.Request] = []

    def speech_wire(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"ID3mp3")

    with unittest.mock.patch.object(httpx, "Client", _mock_httpx_client(speech_wire)):
        audio = synthesize(unit_settings, "ya voy", "es")
    assert audio == b"ID3mp3"
    (request,) = requests
    assert request.url.path == "/v1/audio/speech"
    assert json.loads(request.content) == {
        "model": "tts-1",
        "input": "ya voy",
        "voice": "vd_spanish_male",
        "response_format": "mp3",
        "instruct": "spoken casually, like teasing a friend in a group chat",
    }
    # English: the _casual voice, instruct omitted (empty map entry).
    with unittest.mock.patch.object(httpx, "Client", _mock_httpx_client(speech_wire)):
        synthesize(unit_settings, "on it", "en")
    assert json.loads(requests[1].content) == {
        "model": "tts-1",
        "input": "on it",
        "voice": "vd_british_male_casual",
        "response_format": "mp3",
    }


def test_synthesize_default_language_fallback(unit_settings: Settings) -> None:
    """An unmapped language falls back to the configured default's voice."""
    unit_settings.tts_url = "http://tts.invalid"
    unit_settings.tts_voices = {"es": "vd_spanish_male", "en": "vd_british_male_casual"}
    unit_settings.tts_default_language = "es"
    requests: list[httpx.Request] = []

    def speech_wire(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"ID3mp3")

    with unittest.mock.patch.object(httpx, "Client", _mock_httpx_client(speech_wire)):
        audio = synthesize(unit_settings, "bonjour tout le monde", "fr")
    assert audio == b"ID3mp3"
    assert json.loads(requests[0].content)["voice"] == "vd_spanish_male"


def test_synthesize_fail_soft(unit_settings: Settings) -> None:
    """Every failure returns None: off, HTTP error, timeout, empty body."""

    def responder(
        status: int = 200, content: bytes = b"ID3mp3"
    ) -> Callable[[httpx.Request], httpx.Response]:
        def wire(_request: httpx.Request) -> httpx.Response:
            if status == -1:
                raise httpx.ConnectTimeout("timed out")
            return httpx.Response(status, content=content)

        return wire

    unit_settings.tts_url = ""  # feature off
    assert synthesize(unit_settings, "hola", "es") is None
    unit_settings.tts_url = "http://tts.invalid"
    unit_settings.tts_voices = {"es": "vd_spanish_male"}
    for status, content in ((500, b"boom"), (200, b"")):
        with unittest.mock.patch.object(
            httpx,
            "Client",
            _mock_httpx_client(responder(status, content)),
        ):
            assert synthesize(unit_settings, "hola", "es") is None, (status, content)
    with unittest.mock.patch.object(
        httpx,
        "Client",
        _mock_httpx_client(responder(-1)),
    ):
        assert synthesize(unit_settings, "hola", "es") is None


def test_synthesize_unmapped_no_default(unit_settings: Settings) -> None:
    """No voice for the language and no usable default: None, no request."""
    unit_settings.tts_url = "http://tts.invalid"
    unit_settings.tts_voices = {"es": "vd_spanish_male"}
    unit_settings.tts_default_language = "zz"
    requests: list[httpx.Request] = []

    def speech_wire(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"ID3mp3")

    with unittest.mock.patch.object(httpx, "Client", _mock_httpx_client(speech_wire)):
        assert synthesize(unit_settings, "hello", "en") is None
    assert requests == []


def test_search_matches_ranking() -> None:
    roster = [
        {"id": "1@g.us", "name": "Family"},
        {"id": "2@c.us", "name": "Family Schmit"},
        {"id": "3@c.us", "name": "Ana"},
        {"id": "4@g.us", "name": "Work"},
        {
            "id": {
                "server": "g.us",
                "user": "1809-1373",
                "_serialized": "1809-1373@g.us",
            },
            "name": "Family",
        },
        {"id": "", "name": "no id, dropped"},
    ]
    matches = search_matches(roster, "Family")
    assert [m["id"] for m in matches] == ["1@g.us", "1809-1373@g.us", "2@c.us"]
    assert all(set(m) == {"id", "name"} for m in matches)


def test_chat_display_name_resolves_and_fails_soft() -> None:
    """The delivered-notice renders names; WAHA down falls back to the JID."""
    waha = unittest.mock.Mock()
    waha.get_chat_overview.return_value = {"id": CHAT_ID, "name": "Book Circle"}
    assert chat_display_name(waha, SESSION, CHAT_ID) == "Book Circle"
    waha.get_chat_overview.return_value = {"id": CHAT_ID}
    assert chat_display_name(waha, SESSION, CHAT_ID) == CHAT_ID
    waha.get_chat_overview.side_effect = httpx.ConnectError("waha gone")
    assert chat_display_name(waha, SESSION, CHAT_ID) == CHAT_ID


def test_command_event_shape() -> None:
    command = build_command_event(SESSION, "send the plan to Family")
    command_payload = cast(dict[str, Any], command["payload"])
    assert command["event"] == "command"
    assert command_payload["body"] == "[operator command] send the plan to Family"
    assert cast(dict[str, Any], command_payload["_data"])["notifyName"] == "operator"


def test_fit_messages_spills_full_history_to_file() -> None:
    """Every fetched message rides inline slimmed; the full raw window in a file.

    Nothing requested is silently dropped (outfile.py's contract): all
    20 messages appear in ``messages`` (bodies capped at 200 chars),
    and the *entire* raw window — full bodies, no slimming — is
    persisted to ``file.path``, so a long read is never truncated at
    the source.
    """
    messages = [
        {"id": f"false_{CHAT_ID}_M{i}", "body": f"message {i}" * 40} for i in range(20)
    ]
    fitted = fit_messages(messages)
    assert fitted["message_count"] == 20
    assert fitted["returned"] == len(fitted["messages"]) == 20
    assert "file" in fitted
    assert fitted["file"]["bytes"] > 0
    newest = messages[0]["id"]
    assert fitted["messages"][0]["id"] == newest
    assert fitted["messages"][-1]["id"] == messages[-1]["id"]
    spill = json.loads(Path(fitted["file"]["path"]).read_text(encoding="utf-8"))
    assert len(spill) == 20
    assert spill[0]["id"] == newest
    assert spill[-1]["id"] == messages[-1]["id"]
    # the spill holds the full bodies, not the slimmed inline previews
    assert spill[0]["body"] == messages[0]["body"]
    # the inline slice caps bodies; the cut is flagged, not hidden
    assert fitted["messages"][0]["body_truncated"] is True


def test_fit_messages_short_list_rides_inline() -> None:
    """A short history fits inline: everything returned, still spilled.

    Even when everything fits inline the full bodies ride the file (the
    inline slice is slimmed — bodies cut at 200 chars), and
    `returned` matches `message_count`: nothing is dropped.
    """
    messages = [{"id": f"false_{CHAT_ID}_M{i}", "body": f"note {i}"} for i in range(3)]
    fitted = fit_messages(messages)
    assert fitted["message_count"] == 3
    assert fitted["returned"] == 3
    assert "file" in fitted


def test_chat_preview_body_and_message_truncation_are_independent() -> None:
    with unittest.mock.patch("wahabot.ai.tools.whatsapp.read.write_json_output") as spill:
        spill.return_value = {"path": "/tmp/mock-chat.json"}
        long_body = [{"id": "one", "body": "x" * 300}]
        fitted = fit_messages(long_body)
        assert fitted["returned"] == fitted["message_count"] == 1
        assert fitted["messages"][0]["body_truncated"] is True
        assert len(fitted["messages"][0]["body"]) == 201
        spill.assert_called_once()
        assert spill.call_args.args[1] == long_body
        assert fit_messages([]) == {"messages": [], "returned": 0}
        spill.assert_called_once()
    with unittest.mock.patch(
        "wahabot.ai.tools.whatsapp.read.write_json_output",
        side_effect=OSError("unavailable"),
    ):
        fitted = fit_messages(long_body)
        assert "file" not in fitted
        assert fitted["messages"][0]["body_truncated"] is True


def test_resolve_last_message_pins_named_chat() -> None:
    """The chat a command names resolves to its own real last message.

    The incident's failure shape: even with a fresher, unrelated
    conversation in the list, a command naming a chat must resolve to
    *that* chat and fetch *its* last message — the ground truth the
    shared operator history cannot provide.
    """
    waha = _PinWaha(_PIN_MESSAGES)
    outcome = resolve_last_message(
        cast(Any, waha), SESSION, "responde el ultimo mensaje en Bridge Club"
    )
    assert outcome is not None
    assert outcome.chat_id == CHAT_ID
    assert outcome.chat_name == "Bridge Club"
    assert outcome.candidates == ()
    assert waha.fetched == [CHAT_ID]
    assert outcome.last_message is not None
    assert outcome.last_message["id"] == f"false_{CHAT_ID}_LAST"


def test_resolve_last_message_no_named_chat() -> None:
    """A command naming no known chat resolves to nothing.

    No pin, no fetch: the command runs exactly as before, so 'what is
    the CPU temperature' (the command the incident's answer belonged
    to) never drags an unrelated chat into the turn.
    """
    waha = _PinWaha(_PIN_MESSAGES)
    assert (
        resolve_last_message(cast(Any, waha), SESSION, "how are the temperatures doing?")
        is None
    )
    assert waha.fetched == []


def test_resolve_last_message_ambiguous_keeps_candidates() -> None:
    """Several chats sharing the name surface as candidates.

    The resolver cannot know which 'Nadia' the operator means — but
    instead of guessing it presents every match with its JID, so the
    model picks from evidence instead of inferring from history.
    """

    class TwoNadias(_PinWaha):
        @override
        def list_chats(self, session: str, limit: int = 200) -> list[dict[str, Any]]:
            return [
                {"id": FOREIGN_JID, "name": "Nadia"},
                {"id": "491555000002@c.us", "name": "Nadia Ferrer"},
            ]

    outcome = resolve_last_message(
        cast(Any, TwoNadias()), SESSION, "responde a Nadia Ferrer"
    )
    assert outcome is not None
    ids = [m["id"] for m in outcome.candidates]
    assert len(ids) == 2
    # Longest name first: the most specific mention outranks a bare
    # substring of another chat's name.
    assert ids[0] == "491555000002@c.us"
    assert outcome.chat_id == "491555000002@c.us"


def test_resolve_last_message_fetch_fails_soft() -> None:
    """An unreadable chat still resolves — without a pinned message."""

    class Unreadable(_PinWaha):
        @override
        def fetch_chat_messages(
            self, session: str, chat_id: str, limit: int = 50
        ) -> list[dict[str, Any]]:
            raise RuntimeError("engine hiccup")

    outcome = resolve_last_message(
        cast(Any, Unreadable()), SESSION, "responde en Bridge Club"
    )
    assert outcome is not None
    assert outcome.chat_id == CHAT_ID
    assert outcome.last_message is None
    note = chat_context_note(outcome)
    assert "could not be fetched" in note
    assert "read_chat" in note
    assert "(mode=list)" in note


def test_resolve_last_message_dict_shaped_jid() -> None:
    """A dict-shaped WAHA chat id resolves to its serialized JID.

    The incident's 500s: WAHA's chat list carries group ids as objects
    (``{'server': 'g.us', 'user': …, '_serialized': …}``), and
    stringifying the dict raw built a URL the messages endpoint could
    never answer — every pinned-note fetch failed and the model had to
    guess the chat's last message from stale tool output.
    """

    class DictJids(_PinWaha):
        dict_jid = "491555000009-123456789@g.us"

        def __init__(self) -> None:
            super().__init__(
                [
                    {
                        "id": f"false_{self.dict_jid}_LAST",
                        "from": FOREIGN_JID,
                        "body": "cual es la disponibilidad del salon para el viernes",
                    },
                    {
                        "id": f"false_{self.dict_jid}_OLD1",
                        "from": FOREIGN_JID,
                        "body": "older message",
                    },
                ]
            )

        @override
        def list_chats(self, session: str, limit: int = 200) -> list[dict[str, Any]]:
            return [
                {
                    "id": {
                        "server": "g.us",
                        "user": "491555000009-123456789",
                        "_serialized": "491555000009-123456789@g.us",
                    },
                    "name": "Bridge Club",
                },
                {"id": FOREIGN_JID, "name": "Nadia"},
            ]

    waha = DictJids()
    outcome = resolve_last_message(
        cast(Any, waha), SESSION, "responde el ultimo mensaje en Bridge Club"
    )
    assert outcome is not None
    assert outcome.chat_id == "491555000009-123456789@g.us"
    assert waha.fetched == ["491555000009-123456789@g.us"]
    assert outcome.last_message is not None
    assert outcome.last_message["id"] == "false_491555000009-123456789@g.us_LAST"


def test_resolve_last_message_chats_down_contacts_fallback() -> None:
    """A dead chat list falls back to the contact book."""

    class ChatsDown(_PinWaha):
        @override
        def list_chats(self, session: str, limit: int = 200) -> list[dict[str, Any]]:
            raise RuntimeError("chats endpoint down")

        @override
        def list_contacts(self, session: str, limit: int = 500) -> list[dict[str, Any]]:
            return [{"id": FOREIGN_JID, "name": "Nadia"}]

    outcome = resolve_last_message(cast(Any, ChatsDown()), SESSION, "responde a Nadia")
    assert outcome is not None
    assert outcome.chat_id == FOREIGN_JID


def test_chat_context_note_shape() -> None:
    """The pinned note names the chat, carries the message id, and
    instructs the model to treat it as ground truth — the antidote to
    history-inferred targets from the incident."""
    waha = _PinWaha(_PIN_MESSAGES)
    outcome = resolve_last_message(
        cast(Any, waha), SESSION, "responde el ultimo mensaje en Bridge Club"
    )
    assert outcome is not None
    note = chat_context_note(outcome)
    assert note.startswith("\n[chat context]")
    assert "Bridge Club" in note and CHAT_ID in note
    assert f"false_{CHAT_ID}_LAST" in note
    assert "reply_to" in note
    assert "Never infer the target from conversation history" in note
    assert "only asks ABOUT the chat" in note


def test_chat_context_note_with_serial_names_the_token() -> None:
    """A known serial rides the note so the prompt's token rule and the
    pin agree; an unknown serial keeps the raw id alone."""
    waha = _PinWaha(_PIN_MESSAGES)
    outcome = resolve_last_message(
        cast(Any, waha), SESSION, "responde el ultimo mensaje en Bridge Club"
    )
    assert outcome is not None
    known = chat_context_note(outcome, serial=7)
    assert "Its last message is `#m7` id" in known
    unknown = chat_context_note(outcome, serial=0)
    assert "Its last message is id" in unknown
    assert "#m" not in unknown.split("Its last message is")[1].splitlines()[0]


def test_own_message_id_detection() -> None:
    assert is_own_message_id(f"true_{CHAT_ID}_X")
    assert not is_own_message_id(f"false_{CHAT_ID}_X")


@pytest.mark.parametrize(
    ("url", "mimetype", "filename"),
    [
        ("https://files.invalid/image.PNG?download=1", "image/png", "image.PNG"),
        (
            "https://files.invalid/image.unknown_image",
            "image/jpeg",
            "image.unknown_image",
        ),
        ("https://files.invalid/", "image/jpeg", None),
        ("https://files.invalid/report.pdf", "image/jpeg", "report.pdf"),
    ],
)
def test_image_url_payloads(url: str, mimetype: str, filename: str | None) -> None:
    expected = {"mimetype": mimetype, "url": url}
    if filename:
        expected["filename"] = filename
    assert image_file(url, max_file_bytes=1024) == expected


def test_send_file_payloads() -> None:
    remote = remote_file("http://files.invalid/q3/report.pdf")
    assert (
        remote["mimetype"] == "application/pdf"
        and remote["url"] == "http://files.invalid/q3/report.pdf"
        and remote["filename"] == "report.pdf"
    )
    assert (
        infer_mimetype(
            "http://x.invalid/noext", DOC_MIME_BY_EXT, "application/octet-stream"
        )
        == "application/octet-stream"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf = Path(tmpdir) / "out.pdf"
        pdf.write_bytes(b"%PDF-1.4 smoke bytes")
        local = local_file(str(pdf), max_file_bytes=1024)
        assert not isinstance(local, str)
        assert local["mimetype"] == "application/pdf"
        assert local["filename"] == "out.pdf"
        assert base64.b64decode(local["data"]) == b"%PDF-1.4 smoke bytes"
        assert isinstance(local_file(str(pdf), max_file_bytes=4), str)
        assert isinstance(local_file(str(Path(tmpdir) / "missing.pdf"), 1024), str)


def test_send_video_payloads() -> None:
    """The video payloads for both send_video sources.

    A URL must carry the video MIME (not a guessed type) and the path's
    basename; a local file must be re-typed from the document default
    to ``video/mp4`` — a shell-tool ``.mp4`` must not ride the wire
    stamped ``application/octet-stream``. Oversize/missing paths come
    back as the shared error-string shape, same as ``send_file``.
    """
    remote = video_file("http://files.invalid/q4/clip.webm", max_file_bytes=1024)
    assert not isinstance(remote, str)
    assert remote == {
        "mimetype": "video/webm",
        "url": "http://files.invalid/q4/clip.webm",
        "filename": "clip.webm",
    }
    assert (
        infer_mimetype("http://x.invalid/noext", VIDEO_MIME_BY_EXT, "video/mp4")
        == "video/mp4"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        mp4 = Path(tmpdir) / "out.mp4"
        mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42 smoke bytes")
        local = video_file(str(mp4), max_file_bytes=1024)
        assert not isinstance(local, str)
        assert local["mimetype"] == "video/mp4"
        assert local["filename"] == "out.mp4"
        assert base64.b64decode(local["data"]).startswith(b"\x00\x00\x00\x18ftyp")
        assert isinstance(video_file(str(mp4), max_file_bytes=4), str)
        assert isinstance(video_file(str(Path(tmpdir) / "missing.mp4"), 1024), str)
