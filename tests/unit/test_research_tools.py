import json
import unittest.mock
from typing import Any, ClassVar

import httpx
import pytest

from tests.harness import (
    CHAT_ID,
    SESSION,
)
from tests.unit.support import (
    STAGE_MODULES,
    _fake_ydl,
)
from wahabot.ai.messages import (
    video_media,
)
from wahabot.core.models import WahaEvent
from wahabot.core.transcribe import fetch_transcript
from wahabot.settings import Settings


def test_web_search_short_snippets_stay_inline(unit_settings: Settings) -> None:
    """Nothing cut means no spill: the envelope rides alone."""
    from wahabot.ai.tools import web_search as web_search_mod

    webserp_output = json.dumps(
        {
            "results": [
                {"url": "https://a.example", "title": "Hit A", "content": "tiny"},
            ]
        }
    )
    with unittest.mock.patch.object(
        web_search_mod, "_run_webserp", return_value=webserp_output
    ):
        result = json.loads(web_search_mod.web_search(unit_settings, "query"))
    assert result["ok"] is True
    assert "file" not in result
    assert "content_truncated" not in result["results"][0]


def test_web_search_caps_total_valid_results_and_spill(unit_settings: Settings) -> None:
    from wahabot.ai.tools import web_search as search

    raw = [{"title": "invalid", "url": None}] + [
        {"title": f"Hit {i}", "url": f"https://page.invalid/{i}", "content": "x" * 700}
        for i in range(5)
    ]
    with (
        unittest.mock.patch.object(
            search, "_run_webserp", return_value=json.dumps({"results": raw})
        ),
        unittest.mock.patch.object(search, "write_json_output") as spill,
    ):
        spill.return_value = {"path": "/tmp/mock-search.json"}
        result = json.loads(search.web_search(unit_settings, "q", max_results=2))
    assert result["count"] == 2
    assert [hit["title"] for hit in result["results"]] == ["Hit 0", "Hit 1"]
    stored = spill.call_args.args[1]["results"]
    assert len(stored) == 2 and stored[0]["content"] == "x" * 700


@pytest.mark.parametrize("has_results", [True, False])
def test_web_search_exposes_incomplete_engine_results(
    unit_settings: Settings, has_results: bool
) -> None:
    from wahabot.ai.tools import web_search as search

    output = {
        "results": [{"url": "https://page.invalid/", "title": "Hit"}]
        if has_results
        else [],
        "unresponsive_engines": [["google", "timeout"], ["brave", "blocked"]],
    }
    with unittest.mock.patch.object(
        search, "_run_webserp", return_value=json.dumps(output)
    ):
        result = json.loads(search.web_search(unit_settings, "query"))
    assert result["ok"] is has_results
    assert result["partial"] is True
    assert result["failed_engines"] == [
        {"engine": "google", "error": "timeout"},
        {"engine": "brave", "error": "blocked"},
    ]


def test_web_search_cli_treats_option_shaped_query_as_data() -> None:
    from wahabot.ai.tools.web_search import (
        _run_webserp,  # pyright: ignore[reportPrivateUsage]
    )

    with (
        unittest.mock.patch(
            "wahabot.ai.tools.web_search.shutil.which", return_value="/bin/webserp"
        ),
        unittest.mock.patch("wahabot.ai.tools.web_search.subprocess.run") as run,
    ):
        run.return_value = unittest.mock.Mock(returncode=0, stdout="{}")
        assert (
            _run_webserp(query="--version", max_results=2, timeout=2.5, proxy=None)
            == "{}"
        )
    command = run.call_args.args[0]
    assert command[-2:] == ["--", "--version"]
    assert "--timeout" not in command
    assert run.call_args.kwargs["timeout"] == 2.5


@pytest.mark.parametrize("shell_enabled", [True, False])
def test_research_tools_document_available_spill_access(
    unit_settings: Settings, shell_enabled: bool
) -> None:
    from wahabot.ai.tools.external import visit_url_builder, web_search_builder

    settings = unit_settings.model_copy(update={"shell_tool": shell_enabled})
    for tool in (visit_url_builder(settings), web_search_builder(settings)):
        description = tool.metadata.description
        assert ("Only previews are accessible" in description) is not shell_enabled
        assert ("run_shell_command" in description) is shell_enabled


@pytest.mark.parametrize(
    ("output", "expected_error"),
    [
        ("{", "webserp returned invalid JSON:"),
        ("{}", "webserp output missing 'results' list"),
        ('{"results": null}', "webserp output missing 'results' list"),
        ("[]", "webserp output must be a JSON object"),
        (
            json.dumps(
                {
                    "results": [
                        None,
                        "invalid",
                        {"url": "https://page.invalid/", "title": "Result"},
                        {"title": "missing URL"},
                    ]
                }
            ),
            None,
        ),
    ],
)
def test_web_search_parses_output_once(
    unit_settings: Settings, output: str, expected_error: str | None
) -> None:
    from wahabot.ai.tools import web_search as web_search_mod

    with (
        unittest.mock.patch.object(web_search_mod, "_run_webserp", return_value=output),
        unittest.mock.patch(
            "wahabot.ai.tools.web_search.json.loads", wraps=json.loads
        ) as load,
    ):
        rendered = web_search_mod.web_search(unit_settings, "sample")
        load.assert_called_once_with(output)
    result = json.loads(rendered)
    if expected_error is not None:
        assert result["ok"] is False
        assert result["error"].startswith(f"web_search failed: {expected_error}")
        return
    assert result == {
        "ok": True,
        "query": "sample",
        "count": 1,
        "results": [{"url": "https://page.invalid/", "title": "Result"}],
    }


def test_visit_url_media_bare_info_retries_processed(unit_settings: Settings) -> None:
    """A bare raw dict (Facebook share redirect) retries format processing.

    The raw tier alone returns url/id with no title — an all-null
    envelope masquerading as success that blocks the HTML fallback.
    The processed retry must reach the real metadata.
    """

    class RetryYDL:
        calls: ClassVar[list[str]] = []

        def __init__(self, opts: dict[str, Any]) -> None:
            return None

        def __enter__(self) -> RetryYDL:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def extract_info(
            self, url: str, download: bool = True, process: bool = True
        ) -> dict[str, Any] | None:
            assert download is False
            RetryYDL.calls.append("raw" if not process else "processed")
            if not process:
                return {"id": "1Do4fWrozJ", "url": url, "webpage_url": url}
            return {
                "title": "Bad robot fo' lifes",
                "uploader": "RizzBot",
                "duration": 27.333,
                "id": "1Do4fWrozJ",
            }

    from wahabot.ai.tools.visit_url import visit_url

    with unittest.mock.patch(
        "wahabot.ai.tools.visit_url.page.yt_dlp.YoutubeDL", RetryYDL
    ):
        result = json.loads(
            visit_url(unit_settings, "https://www.facebook.com/share/r/x/")
        )
    assert RetryYDL.calls == ["raw", "processed"]
    assert result["title"] == "Bad robot fo' lifes"
    assert result["uploader"] == "RizzBot"
    assert result["duration_s"] == 27.333


def test_unresolved_video_retry_falls_back_to_page(unit_settings: Settings) -> None:
    from wahabot.ai.tools.visit_url import visit_url

    class BareRetryYDL:
        def __init__(self, opts: Any) -> None:
            return None

        def __enter__(self) -> BareRetryYDL:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def extract_info(self, url: str, download: bool, process: bool = True) -> Any:
            if process:
                raise ValueError("metadata retry failed")
            return {"id": "placeholder", "url": url}

    response = unittest.mock.Mock(
        url="https://facebook.com/share/r/example/",
        status_code=200,
        text="<p>Readable fallback.</p>",
        headers={"content-type": "text/html"},
    )
    with (
        unittest.mock.patch(
            "wahabot.ai.tools.visit_url.page.yt_dlp.YoutubeDL", BareRetryYDL
        ),
        unittest.mock.patch(
            "wahabot.ai.tools.visit_url.page.fetch_page", return_value=response
        ) as fetch,
    ):
        result = json.loads(visit_url(unit_settings, response.url))
    fetch.assert_called_once()
    assert result["ok"] is True and result["text"] == "Readable fallback."
    assert "kind" not in result


@pytest.mark.parametrize("url", ["file:///etc/hosts", "not a url", "https://[invalid"])
def test_visit_url_refuses_non_web_sources(unit_settings: Settings, url: str) -> None:
    from wahabot.ai.tools.visit_url import visit_url

    with unittest.mock.patch("wahabot.ai.tools.visit_url.page.fetch_page") as fetch:
        result = json.loads(visit_url(unit_settings, url))
    assert result["ok"] is False and "HTTP(S)" in result["error"]
    fetch.assert_not_called()


def test_youtube_automatic_captions_and_language_fallback(
    unit_settings: Settings,
) -> None:
    from wahabot.ai.tools.visit_url import visit_url

    info = {
        "title": "Multiple languages",
        "id": "captions",
        "description": "x" * 900,
        "automatic_captions": {
            "es": [{"ext": "json3", "url": "https://captions.invalid/es"}],
            "fr": [{"ext": "json3", "url": "https://captions.invalid/fr"}],
        },
    }
    caption = json.dumps({"events": [{"segs": [{"utf8": "Una explicación clara."}]}]})
    with (
        unittest.mock.patch(
            "wahabot.ai.tools.visit_url.page.yt_dlp.YoutubeDL", _fake_ydl(info)
        ),
        unittest.mock.patch(
            "wahabot.ai.tools.visit_url.captions.fetch_captions", return_value=caption
        ) as fetch,
    ):
        result = json.loads(
            visit_url(unit_settings, "https://youtube.com/watch?v=captions")
        )
    fetch.assert_called_once_with("https://captions.invalid/es", unit_settings)
    assert result["transcript"] == "Una explicación clara."
    assert result["description_truncated"] is True
    assert len(result["description"]) == 800


def test_caption_hls_is_not_mistaken_for_transcript() -> None:
    from wahabot.ai.tools.visit_url import youtube_transcript

    tracks = {"en": [{"ext": "vtt", "url": "https://captions.invalid/playlist"}]}
    with unittest.mock.patch(
        "wahabot.ai.tools.visit_url.captions.fetch_captions",
        return_value="#EXTM3U\nsegment.vtt",
    ):
        assert youtube_transcript("https://youtube.com/watch?v=x", tracks) == ""


def test_caption_empty_parse_tries_another_available_format() -> None:
    from wahabot.ai.tools.visit_url import youtube_transcript

    tracks = {
        "en": [
            {"ext": "json3", "url": "https://captions.invalid/empty"},
            {"ext": "vtt", "url": "https://captions.invalid/text"},
        ]
    }
    with unittest.mock.patch(
        "wahabot.ai.tools.visit_url.captions.fetch_captions",
        side_effect=["{}", "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nUseful words."],
    ):
        assert (
            youtube_transcript("https://youtube.com/watch?v=x", tracks) == "Useful words."
        )


@pytest.mark.parametrize("stage", list(STAGE_MODULES))
def test_video_postprocessing_failure_falls_back_to_http(
    unit_settings: Settings, stage: str
) -> None:
    from wahabot.ai.tools.visit_url import visit_url

    info = {
        "title": "Clip",
        "entries": [{"duration": 1}],
        "subtitles": {"en": [{"ext": "vtt", "url": "https://captions.invalid/en"}]},
    }
    response = unittest.mock.Mock(
        url="https://youtube.com/watch?v=fallback",
        status_code=200,
        text="<p>Fallback content.</p>",
        headers={"content-type": "text/html"},
    )
    with (
        unittest.mock.patch(
            "wahabot.ai.tools.visit_url.page.yt_dlp.YoutubeDL", _fake_ydl(info)
        ),
        unittest.mock.patch(
            "wahabot.ai.tools.visit_url.captions.fetch_captions", return_value="Words."
        ),
        unittest.mock.patch(
            f"wahabot.ai.tools.visit_url.{STAGE_MODULES[stage]}.{stage}",
            side_effect=ValueError("bad metadata"),
        ),
        unittest.mock.patch(
            "wahabot.ai.tools.visit_url.page.fetch_page", return_value=response
        ),
    ):
        result = json.loads(visit_url(unit_settings, response.url))
    assert result["ok"] is True and result["text"] == "Fallback content."


def test_visit_url_media_downloader_error_falls_back(unit_settings: Settings) -> None:
    """A DownloadError in extract_info falls through to the HTML path."""

    class ExplodingYDL:
        def __init__(self, opts: dict[str, Any]) -> None:
            return None

        def __enter__(self) -> ExplodingYDL:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def extract_info(
            self, url: str, download: bool = True, process: bool = True
        ) -> dict[str, Any]:
            raise RuntimeError("Requested format is not available")

    from wahabot.ai.tools.visit_url import visit_url

    class FakeResponse:
        url = "https://www.instagram.com/p/abc/"
        status_code = 200
        text = "<html>wall</html>"
        headers: ClassVar[dict[str, str]] = {"content-type": "text/html"}

    def fake_fetch(url: str, settings: Settings) -> Any:
        return FakeResponse()

    with (
        unittest.mock.patch(
            "wahabot.ai.tools.visit_url.page.yt_dlp.YoutubeDL", ExplodingYDL
        ),
        unittest.mock.patch("wahabot.ai.tools.visit_url.page.fetch_page", fake_fetch),
    ):
        result = json.loads(visit_url(unit_settings, "https://www.instagram.com/p/abc/"))
    assert result["ok"] is True
    assert result["status"] == 200


def test_video_media_kind_guard() -> None:
    video_evt = WahaEvent(
        id="e7",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={
            "id": f"false_{CHAT_ID}_VID",
            "from": CHAT_ID,
            "body": "",
            "hasMedia": True,
            "media": {"url": "http://waha.invalid/v.mp4", "mimetype": "video/mp4"},
            "_data": {"type": "video"},
        },
    )
    assert video_media(video_evt) is not None
    ptv = WahaEvent(
        id="e8",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={
            "id": f"false_{CHAT_ID}_PTV",
            "from": CHAT_ID,
            "body": "",
            "media": {"url": "http://waha.invalid/p.mp4", "mimetype": "video/mp4"},
            "_data": {"type": "ptv"},
        },
    )
    assert video_media(ptv) is not None
    text_evt = WahaEvent(
        id="e1",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={"from": CHAT_ID, "body": "hi"},
    )
    assert video_media(text_evt) is None
    no_url = WahaEvent(
        id="e9",
        timestamp=1,
        event="message",
        session=SESSION,
        me={},
        payload={"id": "x", "from": CHAT_ID, "body": "", "_data": {"type": "video"}},
    )
    assert video_media(no_url) is None


def test_fetch_transcript_joins_segments(unit_settings: Settings) -> None:
    def _fake_post(_self: httpx.Client, *_args: object, **_kwargs: object) -> Any:
        resp = unittest.mock.Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "segments": [
                {"text": " hello "},
                {"text": " world"},
                {"text": ""},
            ]
        }
        return resp

    with unittest.mock.patch("httpx.Client.post", _fake_post):
        transcript = fetch_transcript(unit_settings, b"audio", "n.oga")
    assert transcript == "hello world"
