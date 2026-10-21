import json
import tempfile
import time
import unittest.mock
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from tests.harness import (
    CHAT_ID,
    SESSION,
)
from wahabot.ai.resolve import resolve_last_message
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings


def test_generate_image_tool_envelope_has_path_and_metadata(
    unit_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import base64 as b64
    import struct

    from wahabot.ai.tools.image import image_builder
    from wahabot.core import images

    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", 512, 512) + b"\x00" * 8

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"data": [{"b64_json": b64.b64encode(png).decode()}]}

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
    tool = {t.metadata.name: t for t in image_builder(settings)}["generate_image"]
    payload = json.loads(tool.fn(prompt="a red fox", size="512x512"))
    assert payload["ok"] is True
    assert payload["format"] == "png"
    assert payload["width"] == 512 and payload["height"] == 512
    assert Path(payload["path"]).read_bytes() == png


def test_save_prunes_stale_images() -> None:
    import os

    from wahabot.core import images

    directory = Path(tempfile.gettempdir()) / "wahabot-images"
    directory.mkdir(parents=True, exist_ok=True)
    stale = directory / "gen-stale.png"
    stale.write_bytes(b"old")
    old = time.time() - images.IMAGE_TTL_S - 10
    os.utime(stale, (old, old))
    images.save(b"fresh")
    assert not stale.exists()


def test_edit_image_tool_uses_a_local_source(
    unit_settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import base64 as b64
    import struct

    from wahabot.ai.tools.image import image_builder
    from wahabot.core import images

    out = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", 256, 256) + b"\x00" * 8
    seen: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"data": [{"b64_json": b64.b64encode(out).decode()}]}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def post(
            self,
            url: str,
            data: dict[str, object] | None = None,
            files: dict[str, object] | None = None,
            json: dict[str, object] | None = None,
        ) -> FakeResponse:
            seen.update(url=url, data=data, files=files)
            return FakeResponse()

    monkeypatch.setattr(images.httpx, "Client", FakeClient)
    source = tmp_path / "source.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    settings = unit_settings.model_copy(
        update={"image_api_base": "http://img.invalid:9196"}
    )
    tool = {t.metadata.name: t for t in image_builder(settings)}["edit_image"]
    payload = json.loads(
        tool.fn(prompt="make the mug bright blue", source=str(source), strength=0.6)
    )
    assert payload["ok"] is True
    assert payload["width"] == 256 and payload["height"] == 256
    assert str(seen["url"]).endswith("/v1/images/edits")
    assert "strength" in str(seen["data"])
    assert seen["files"]["image"][0] == "source.png"
    assert Path(payload["path"]).read_bytes() == out


def test_load_source_sniffs_content_not_the_png_name(tmp_path: Path) -> None:
    """A mislabeled source is declared by its bytes, not its .png name."""
    from wahabot.core import images

    mislabeled = tmp_path / "gen-1.png"
    mislabeled.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    loaded = images.load_source(str(mislabeled))
    assert isinstance(loaded, tuple)
    _, _, mimetype = loaded
    assert mimetype == "image/jpeg"
    assert images.load_source(str(tmp_path / "missing.png")) != images.load_source(
        str(mislabeled)
    )


def test_one_letter_chat_name_does_not_resolve_from_prose() -> None:
    """A one-letter chat name must not match every sentence's letters."""
    waha = unittest.mock.Mock()
    waha.list_chats.return_value = [{"id": CHAT_ID, "name": "M"}]
    waha.list_contacts.return_value = []
    assert (
        resolve_last_message(
            cast(Any, waha), SESSION, "reply to me: the smoke test works"
        )
        is None
    )


def test_profile_picture_prefers_contact_then_chat() -> None:
    """Contact endpoint first, chat endpoint as fallback, "" when hidden."""
    import httpx

    def contact(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/contacts/profile-picture":
            return httpx.Response(
                200, json={"profilePictureURL": "https://pic.invalid/a.png"}
            )
        return httpx.Response(404)

    client = WahaClient(
        "http://waha.invalid", "k", transport=httpx.MockTransport(contact)
    )
    assert client.get_profile_picture(SESSION, CHAT_ID) == "https://pic.invalid/a.png"

    def chat(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/picture"):
            return httpx.Response(200, json={"url": "https://pic.invalid/b.png"})
        return httpx.Response(404)

    client = WahaClient("http://waha.invalid", "k", transport=httpx.MockTransport(chat))
    assert client.get_profile_picture(SESSION, CHAT_ID) == "https://pic.invalid/b.png"

    client = WahaClient(
        "http://waha.invalid",
        "k",
        transport=httpx.MockTransport(lambda _: httpx.Response(404)),
    )
    assert client.get_profile_picture(SESSION, CHAT_ID) == ""


def test_profile_picture_tool_envelope() -> None:
    """The tool returns the URL, or available=false when there is none."""
    from wahabot.ai.tools.whatsapp import (
        RunTarget,
        bind_target,
        profile_picture_builder,
        reset_target,
    )

    waha = unittest.mock.Mock()
    waha.get_profile_picture.return_value = "https://pic.invalid/a.png"
    target = RunTarget(session=SESSION, chat_id=CHAT_ID)
    token = bind_target(target)
    try:
        payload = json.loads(profile_picture_builder(cast(Any, waha)).fn(reason="check"))
    finally:
        reset_target(token)
    assert payload["ok"] is True and payload["url"] == "https://pic.invalid/a.png"
    waha.get_profile_picture.assert_called_once_with(SESSION, CHAT_ID)

    waha.get_profile_picture.return_value = ""
    token = bind_target(target)
    try:
        hidden = json.loads(profile_picture_builder(cast(Any, waha)).fn())
    finally:
        reset_target(token)
    assert hidden["ok"] is True and hidden["available"] is False


def test_reaction_targets_the_bot_by_bare_hash() -> None:
    from wahabot.core.identity import AUTHOR_BOT, authors
    from wahabot.reactions import reaction_targets_bot

    raw = "true_491555000010@g.us_BAREHASH_491555000010@g.us"
    authors.record(raw, AUTHOR_BOT)
    assert reaction_targets_bot("BAREHASH")
