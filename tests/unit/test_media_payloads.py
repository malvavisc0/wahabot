import base64
import json
import tempfile
import unittest.mock
from pathlib import Path
from typing import Any, cast, override

import httpx
import pytest

from tests.harness import (
    CHAT_ID,
    SESSION,
)
from wahabot.ai.tools.whatsapp import (
    deliver_chat_text,
    local_file,
    probe_media_url,
    sticker_file,
    voice_file,
)
from wahabot.core.presence import clear_typing, mark_seen, typing_pause
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings


def test_send_voice_payloads() -> None:
    """The voice payloads for both send_voice sources.

    A URL must carry the audio MIME (not a guessed type) and the path's
    basename; a local file must be re-typed from the document default to
    ``audio/mpeg`` — a shell-tool ``.mp3`` must not ride the wire stamped
    ``application/octet-stream``. Oversize/missing paths come back as the
    shared error-string shape, same as ``send_file``.
    """
    remote = voice_file("http://files.invalid/q5/note.mp3", max_file_bytes=1024)
    assert not isinstance(remote, str)
    assert remote == {
        "mimetype": "audio/mpeg",
        "url": "http://files.invalid/q5/note.mp3",
        "filename": "note.mp3",
    }
    remote_ogg = voice_file("http://files.invalid/q5/note.opus", max_file_bytes=1024)
    assert not isinstance(remote_ogg, str)
    assert remote_ogg["mimetype"] == "audio/ogg"
    with tempfile.TemporaryDirectory() as tmpdir:
        mp3 = Path(tmpdir) / "out.mp3"
        mp3.write_bytes(b"\xff\xf3 smoke mpeg frames")
        local = voice_file(str(mp3), max_file_bytes=1024)
        assert not isinstance(local, str)
        assert local["mimetype"] == "audio/mpeg"
        assert local["filename"] == "out.mp3"
        assert base64.b64decode(local["data"]).startswith(b"\xff\xf3")
        assert isinstance(voice_file(str(mp3), max_file_bytes=4), str)
        assert isinstance(voice_file(str(Path(tmpdir) / "missing.mp3"), 1024), str)


def test_send_sticker_payloads() -> None:
    """The sticker payloads for both send_sticker sources.

    Stickers are WebP stills: a URL must carry the image MIME map, a
    local ``.webp``/``.png`` must not ride the wire stamped
    ``application/octet-stream``. Oversize/missing paths come back as
    the shared error-string shape.
    """
    remote = sticker_file("http://files.invalid/q6/laugh.webp", max_file_bytes=1024)
    assert not isinstance(remote, str)
    assert remote == {
        "mimetype": "image/webp",
        "url": "http://files.invalid/q6/laugh.webp",
        "filename": "laugh.webp",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        webp = Path(tmpdir) / "smile.webp"
        webp.write_bytes(b"RIFF\x00\x00\x00WEBPVP8 smoke")
        local = sticker_file(str(webp), max_file_bytes=1024)
        assert not isinstance(local, str)
        assert local["mimetype"] == "image/webp"
        assert local["filename"] == "smile.webp"
        assert base64.b64decode(local["data"]).startswith(b"RIFF")
        assert isinstance(sticker_file(str(webp), max_file_bytes=4), str)
        assert isinstance(sticker_file(str(Path(tmpdir) / "missing.webp"), 1024), str)


def test_squared_sticker_payload_pads_nonsquare() -> None:
    """Non-square local stickers are letterboxed to square, not squashed.

    The meme incident: a 1080x1360
    meme sent via the sticker workaround arrived distorted because
    WhatsApp renders stickers on a square canvas and nothing padded or
    refused the image. The payload helper must center the image on a
    square canvas; a square input rides the wire unchanged; a corrupt
    file surfaces a diagnosis instead of raising.
    """
    import io

    from PIL import Image

    from wahabot.ai.tools.whatsapp import squared_sticker_payload

    with tempfile.TemporaryDirectory() as tmpdir:
        meme = Path(tmpdir) / "meme.webp"
        Image.new("RGB", (1080, 1360), "white").save(meme, "WEBP")
        padded = squared_sticker_payload(str(meme), max_sticker_bytes=1024 * 1024)
        assert not isinstance(padded, str)
        assert padded["mimetype"] == "image/webp"
        with Image.open(io.BytesIO(base64.b64decode(padded["data"]))) as im:
            assert im.size == (1360, 1360)

        square = Path(tmpdir) / "square.webp"
        Image.new("RGB", (512, 512), "red").save(square, "WEBP")
        kept = squared_sticker_payload(str(square), max_sticker_bytes=1024 * 1024)
        assert not isinstance(kept, str)
        with Image.open(io.BytesIO(base64.b64decode(kept["data"]))) as im:
            assert im.size == (512, 512)

        wide = Path(tmpdir) / "wide.webp"
        Image.new("RGB", (800, 400), "blue").save(wide, "WEBP")
        wide_padded = squared_sticker_payload(str(wide), max_sticker_bytes=1024 * 1024)
        assert not isinstance(wide_padded, str)
        with Image.open(io.BytesIO(base64.b64decode(wide_padded["data"]))) as im:
            assert im.size == (800, 800)

        corrupt = Path(tmpdir) / "bad.webp"
        corrupt.write_bytes(b"not an image")
        failed = squared_sticker_payload(str(corrupt), max_sticker_bytes=1024 * 1024)
        assert isinstance(failed, str)
        assert "cannot decode" in failed


def test_send_sticker_pads_local_path_before_sending() -> None:
    """The tool letterboxes a non-square local path before the wire send."""
    import io

    from PIL import Image

    from wahabot.ai.tools.whatsapp import (
        RunTarget,
        bind_target,
        reset_target,
        send_media,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        meme = Path(tmpdir) / "meme.webp"
        Image.new("RGB", (1080, 1360), "white").save(meme, "WEBP")
        waha = unittest.mock.Mock()
        waha.send_sticker.return_value = "id"
        settings = unittest.mock.Mock(max_sticker_bytes=1024 * 1024)
        target = RunTarget(session=SESSION, chat_id=CHAT_ID)
        token = bind_target(target)
        try:
            tool = send_media(waha, settings)
            with unittest.mock.patch(
                "wahabot.ai.tools.whatsapp.media_files.local_file", wraps=local_file
            ) as load:
                out = tool(kind="sticker", path=str(meme), reason="meme as sticker")
            load.assert_called_once_with(str(meme), settings.max_sticker_bytes)
        finally:
            reset_target(token)
        envelope: dict[str, Any] = json.loads(cast("str", out.content))
        assert envelope["ok"] is True
        assert envelope["square"] is True
        file = waha.send_sticker.call_args.kwargs["file"]
        with Image.open(io.BytesIO(base64.b64decode(file["data"]))) as im:
            assert im.size == (1360, 1360)


def test_remote_sticker_does_not_claim_verified_shape(unit_settings: Settings) -> None:
    from wahabot.ai.tools.whatsapp import (
        RunTarget,
        bind_target,
        reset_target,
        send_media,
    )

    waha = unittest.mock.Mock()
    waha.send_sticker.return_value = "id"
    token = bind_target(RunTarget(session=SESSION, chat_id=CHAT_ID))
    try:
        with unittest.mock.patch(
            "wahabot.ai.tools.whatsapp.media_send.probe_media_url"
        ) as probe:
            probe.return_value = None
            out = cast(Any, send_media(waha, unit_settings))(
                kind="sticker", url="https://media.invalid/wide.webp"
            )
    finally:
        reset_target(token)
    envelope = json.loads(str(out.content))
    assert envelope["ok"] is True
    assert "square" not in envelope
    assert waha.send_sticker.call_args.kwargs["file"]["url"] == (
        "https://media.invalid/wide.webp"
    )


@pytest.mark.parametrize(
    ("body", "expected_error"),
    [(None, "cannot read"), (b"large", "over the 4 B cap")],
)
def test_send_sticker_refuses_bad_paths(
    tmp_path: Path, unit_settings: Settings, body: bytes | None, expected_error: str
) -> None:
    from wahabot.ai.tools.whatsapp import (
        RunTarget,
        bind_target,
        reset_target,
        send_media,
    )

    path = tmp_path / "sticker.webp"
    if body is not None:
        path.write_bytes(body)
    settings = unit_settings.model_copy(update={"max_sticker_bytes": 4})
    waha = unittest.mock.Mock()
    token = bind_target(RunTarget(session=SESSION, chat_id=CHAT_ID))
    try:
        out = cast(Any, send_media(waha, settings))(kind="sticker", path=str(path))
    finally:
        reset_target(token)
    envelope = json.loads(str(out.content))
    assert envelope["ok"] is False
    assert expected_error in envelope["error"]
    waha.send_sticker.assert_not_called()


def test_send_media_rejects_invalid_sources() -> None:
    """The merged tool's source rules (bug 1's lesson) refuse bad args.

    Zero, multiple, or misapplied sources must return an error envelope
    naming the rule — never reach a WAHA call, never raise.
    """
    from wahabot.ai.tools.whatsapp import (
        RunTarget,
        bind_target,
        reset_target,
        send_media,
    )

    waha = unittest.mock.Mock()
    waha.send_image.return_value = "id"
    settings = unittest.mock.Mock(max_image_bytes=1024 * 1024)
    target = RunTarget(session=SESSION, chat_id=CHAT_ID)
    token = bind_target(target)
    cases: list[tuple[dict[str, Any], str]] = [
        # zero sources, known kind
        ({"kind": "image"}, "exactly one of url, path or text"),
        # multiple sources
        (
            {"kind": "image", "url": "https://x.invalid/a.png", "path": "/tmp/a.png"},
            "exactly one of url, path or text",
        ),
        # text is voice-only
        (
            {"kind": "image", "text": "not an image"},
            "text is only valid for kind=voice",
        ),
        # unknown kind
        ({"kind": "meme", "url": "https://x.invalid/a.png"}, "unknown kind"),
        # caption on a kind whose WAHA call has none
        (
            {
                "kind": "sticker",
                "url": "https://x.invalid/a.webp",
                "caption": "nope",
            },
            "caption is not supported for kind=sticker",
        ),
        (
            {"kind": "image", "path": "/tmp/image.png", "filename": "renamed.png"},
            "filename is only valid for kind=file",
        ),
        (
            {"kind": "video", "path": "/tmp/clip.mp4", "language": "es"},
            "language is only valid for kind=voice with text",
        ),
        (
            {"kind": "voice", "path": "/tmp/note.mp3", "language": "es"},
            "language is only valid for kind=voice with text",
        ),
    ]
    try:
        tool = send_media(waha, settings)
        for kwargs, needle in cases:
            out = tool(**kwargs)  # type: ignore[arg-type]
            envelope = json.loads(str(out.content))
            assert envelope["ok"] is False, kwargs
            assert needle in envelope["error"], (kwargs, envelope["error"])
    finally:
        reset_target(token)
    waha.send_image.assert_not_called()
    waha.send_sticker.assert_not_called()


def test_send_media_rejects_blank_and_dubious_sources() -> None:
    """Blank sources are refused like missing ones, before any WAHA call.

    A whitespace-only ``url``/``path``/``text`` passes a naive truthiness
    check, and the fn must not let it through to a wire send of "" —
    bug 1's lesson again: bad arguments get named, early.
    """
    from wahabot.ai.tools.whatsapp import (
        RunTarget,
        bind_target,
        reset_target,
        send_media,
    )

    waha = unittest.mock.Mock()
    settings = unittest.mock.Mock(max_file_bytes=1024 * 1024)
    target = RunTarget(session=SESSION, chat_id=CHAT_ID)
    token = bind_target(target)
    try:
        tool = send_media(waha, settings)
        for kwargs in (
            {"kind": "file", "path": " "},
            {"kind": "file", "url": " "},
            {"kind": "voice", "text": "   "},
        ):
            out = tool(**kwargs)  # type: ignore[arg-type]
            envelope = json.loads(str(out.content))
            assert envelope["ok"] is False, kwargs
            assert "exactly one of url, path or text" in envelope["error"], kwargs
    finally:
        reset_target(token)
    waha.send_file.assert_not_called()
    waha.send_voice.assert_not_called()


def test_send_media_blank_source_yields_to_real_source() -> None:
    """A whitespace-only source must not outrank a real one.

    The XOR check counts non-blank sources, but the dispatch keys on
    which argument was passed — a blank ``url`` next to a real
    ``path`` used to win the dispatch and probe " ". Normalizing
    blanks to None first keeps validation and dispatch agreed.
    """
    from wahabot.ai.tools.whatsapp import (
        RunTarget,
        bind_target,
        reset_target,
        send_media,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        doc = Path(tmpdir) / "report.pdf"
        doc.write_bytes(b"%PDF-1.4 fake")
        waha = unittest.mock.Mock()
        waha.send_file.return_value = "id"
        settings = unittest.mock.Mock(max_file_bytes=1024 * 1024)
        target = RunTarget(session=SESSION, chat_id=CHAT_ID)
        token = bind_target(target)
        try:
            tool: Any = send_media(waha, settings)
            out = tool(kind="file", url="   ", path=str(doc))
            envelope: dict[str, Any] = json.loads(str(out.content))
            assert envelope["ok"] is True, envelope
            assert envelope["mimetype"] == "application/pdf"
        finally:
            reset_target(token)
        waha.send_file.assert_called_once()
        assert waha.send_file.call_args.kwargs["file"]["mimetype"] == "application/pdf"


def test_read_chat_rejects_chat_in_operator_modes(unit_settings: Settings) -> None:
    """mode=resolve/recent ignore chats by design; a passed `chat` is refused.

    Both modes work on the whole roster/contact book, so a `chat`
    argument is meaningless there — refuse it instead of silently
    dropping it (bug 2's lesson: bad arguments get named).
    """
    import json as _json

    from wahabot.ai.tools.whatsapp import (
        RunTarget,
        bind_target,
        read_chat,
        reset_target,
    )

    waha = unittest.mock.Mock()
    target = RunTarget(session=SESSION, chat_id=CHAT_ID, armed=True)
    token = bind_target(target)
    try:
        tool = cast("Any", read_chat(cast("Any", waha), unit_settings)).fn
        for mode in ("resolve", "recent"):
            out = _json.loads(tool(mode=mode, chat=CHAT_ID, name="Family"))
            assert out["ok"] is False, mode
            assert f"`chat` is not valid for mode={mode}" in out["error"]
        # without `chat` the modes run their normal paths
        ok_out = _json.loads(tool(mode="resolve", name="Family"))
        assert ok_out["ok"] is False  # empty roster → no match, but not refused
        assert "not valid for mode" not in ok_out["error"]
    finally:
        reset_target(token)


def test_mark_seen_swallows_failures() -> None:
    """mark_seen never raises: a presence hiccup must not kill a run."""
    waha = unittest.mock.Mock()
    waha.send_seen.side_effect = httpx.ConnectError("waha gone")
    mark_seen(waha, SESSION, CHAT_ID)  # must not raise
    waha.send_seen.assert_called_once_with(SESSION, CHAT_ID)


def test_typing_pause_disabled() -> None:
    """min_s <= 0 skips the whole routine — no typing call, no wait."""
    waha = unittest.mock.Mock()
    typing_pause(waha, SESSION, CHAT_ID, "hello", min_s=0.0, max_s=3.0)
    waha.set_typing.assert_not_called()


def test_typing_pause_start_wait_order() -> None:
    """Normal pass: typing on, then a scaled wait; the reply follows."""
    waha = unittest.mock.Mock()
    order: list[str] = []

    def record(session: str, chat_id: str, typing: bool) -> None:
        order.append("on" if typing else "off")

    waha.set_typing.side_effect = record
    with unittest.mock.patch("wahabot.core.presence.time.sleep") as sleep:
        typing_pause(waha, SESSION, CHAT_ID, "a" * 400, min_s=0.5, max_s=3.0)
        (delay,), _ = sleep.call_args
    assert 0.5 <= delay <= 3.0  # length-scaled, clamped to the window
    assert order == ["on"]  # left ON: the send itself clears the indicator
    sleep.assert_called_once()


def test_typing_pause_failure_never_raises() -> None:
    """startTyping raising skips the wait and stays quiet."""
    waha = unittest.mock.Mock()
    waha.set_typing.side_effect = httpx.ConnectError("waha gone")
    with unittest.mock.patch("wahabot.core.presence.time.sleep") as sleep:
        typing_pause(waha, SESSION, CHAT_ID, "hello", min_s=0.1, max_s=3.0)
    sleep.assert_not_called()  # the early return path: no wait without typing


def test_clear_typing_swallows_failures() -> None:
    """clear_typing never raises — it runs on an error path already."""
    waha = unittest.mock.Mock()
    waha.set_typing.side_effect = httpx.ConnectError("waha gone")
    clear_typing(waha, SESSION, CHAT_ID)  # must not raise
    waha.set_typing.assert_called_once_with(SESSION, CHAT_ID, False)


def test_deliver_chat_text_clears_typing_on_send_failure() -> None:
    """A send that raises after typing-on clears the indicator on the way out.

    The landed message normally clears "typing…"; a failed send never
    lands, so ``deliver_chat_text`` must clear it itself — and never
    mask the original error doing so.
    """

    class FailingSendWaha(WahaClient):
        """A WAHA whose roster works but text sends always fail."""

        def __init__(self) -> None:
            super().__init__(base_url="http://waha.invalid", api_key="k")
            self.typing: list[bool] = []

        @override
        def get_chat_overview(self, session: str, chat_id: str) -> Any:
            return {"id": chat_id, "participants": []}

        @override
        def set_typing(self, session: str, chat_id: str, typing: bool) -> None:
            self.typing.append(typing)

        @override
        def send_text(
            self,
            session: str,
            chat_id: str,
            text: str,
            reply_to: str | None = None,
            mentions: list[str] | None = None,
        ) -> str:
            raise httpx.ConnectError("waha gone")

    waha = FailingSendWaha()
    with pytest.raises(httpx.ConnectError):
        deliver_chat_text(waha, SESSION, CHAT_ID, "hello", typing=(0.1, 0.2))
    assert waha.typing == [True, False]  # lit, then cleared best-effort


def test_probe_media_url_refuses_malformed() -> None:
    assert "not an http(s) URL" in (probe_media_url("not a url") or "")
    assert "not an http(s) URL" in (probe_media_url("ftp://x/y.png") or "")


def test_probe_media_url_refuses_missing() -> None:
    """A definitive 404 refuses the send — the URL was likely invented."""
    gone = unittest.mock.Mock(status_code=404)
    with (
        unittest.mock.patch.object(httpx, "head", return_value=gone),
        unittest.mock.patch.object(httpx, "get", return_value=gone),
    ):
        error = probe_media_url("https://x.invalid/pic.png")
    assert error is not None and "does not exist" in error
