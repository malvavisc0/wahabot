import json
import time
import unittest.mock
from pathlib import Path
from typing import Any, ClassVar

import pytest

from tests.unit.support import (
    _fake_ydl,
)
from wahabot.ai.tools.shell import shell_command
from wahabot.settings import Settings


def test_visit_url_long_page_spills_to_file(unit_settings: Settings) -> None:
    """A page body over the cap keeps a bounded preview + full-body file.

    Reading a whole long article inline would cost the token budget;
    the envelope shows a ``text`` preview, flags ``truncated``, and
    spills the full body to ``file`` so the model can read the rest
    in parts via the shell tool.
    """
    from wahabot.ai.tools.visit_url import (
        MAX_CHARS as PAGE_CAP,
    )
    from wahabot.ai.tools.visit_url import (
        visit_url,
    )

    long_body = "all the words " * 900  # ~13.5k chars, over the 4k inline cap

    class FakeResponse:
        url = "https://example.com/very-long"
        status_code = 200
        text = long_body
        headers: ClassVar[dict[str, str]] = {"content-type": "text/html"}

    with unittest.mock.patch(
        "wahabot.ai.tools.visit_url.page.fetch_page",
        lambda url, settings: FakeResponse(),  # pyright: ignore[reportUnknownLambdaType]
    ):
        result = json.loads(visit_url(unit_settings, "https://example.com/very-long"))
    assert result["ok"] is True
    assert result["truncated"] is True
    assert result["text"] == long_body.strip()[:PAGE_CAP]
    assert "file" in result
    spill = Path(result["file"]["path"]).read_text(encoding="utf-8")
    assert spill == long_body.strip()
    assert result["file"]["bytes"] > PAGE_CAP


def test_visit_url_short_page_stays_inline(unit_settings: Settings) -> None:
    """A short page rides the envelope whole: no file, no flag."""
    from wahabot.ai.tools.visit_url import visit_url

    class FakeResponse:
        url = "https://example.com/short"
        status_code = 200
        text = "just a couple of words"
        headers: ClassVar[dict[str, str]] = {"content-type": "text/html"}

    with unittest.mock.patch(
        "wahabot.ai.tools.visit_url.page.fetch_page",
        lambda url, settings: FakeResponse(),  # pyright: ignore[reportUnknownLambdaType]
    ):
        result = json.loads(visit_url(unit_settings, "https://example.com/short"))
    assert result["ok"] is True
    assert result["truncated"] is False
    assert result["text"] == "just a couple of words"
    assert "file" not in result


def test_visit_url_js_shell_page_uses_meta_description(unit_settings: Settings) -> None:
    """A JS-shell page's caption rides the envelope, not its site name.

    Instagram and X serve non-logged-in fetches an empty render: the
    tag strip leaves the site name while the ``og:`` meta tags hold
    the post's real title and caption (the live 14:41 failure). The
    page's own description of itself beats the empty render.
    """
    from wahabot.ai.tools.visit_url import visit_url

    body = (
        "<head><title>Instagram</title>"
        '<meta property="og:title" content="Revista Pacto on Instagram:'
        ' &quot;El DANE report&#xf3; la tasa de desocupaci&#xf3;n&quot;" />'
        '<meta property="og:image" content="https://x.jpg" />'
        '<meta property="og:description" content="39K likes, 3,028 comments -'
        ' revista_pacto on October 1, 2026: &quot;la tasa subi&#xf3; 0,8 puntos&quot;" />'
        "</head><body><script>/* 700KB of JS shell */</script></body>"
    )

    class FakeResponse:
        url = "https://www.instagram.com/p/fake/"
        status_code = 200
        text = body
        headers: ClassVar[dict[str, str]] = {"content-type": "text/html"}

    with unittest.mock.patch(
        "wahabot.ai.tools.visit_url.page.fetch_page",
        lambda url, settings: FakeResponse(),  # pyright: ignore[reportUnknownLambdaType]
    ):
        result = json.loads(visit_url(unit_settings, "https://www.instagram.com/p/fake/"))
    assert result["ok"] is True
    assert "Revista Pacto" in result["text"]
    assert "39K likes" in result["text"]
    assert "0,8 puntos" in result["text"]
    assert result["text"] != "Instagram"


def test_visit_url_rendered_text_beats_meta_when_richer(unit_settings: Settings) -> None:
    """A page with real content never trades it for its meta tags."""
    from wahabot.ai.tools.visit_url import visit_url

    body = (
        '<html><head><meta name="description" content="a short blurb">'
        "</head><body><p>A full article with paragraphs of researched text that"
        " is far longer than the meta description could ever be.</p></body></html>"
    )

    class FakeResponse:
        url = "https://example.com/article"
        status_code = 200
        text = body
        headers: ClassVar[dict[str, str]] = {"content-type": "text/html"}

    with unittest.mock.patch(
        "wahabot.ai.tools.visit_url.page.fetch_page",
        lambda url, settings: FakeResponse(),  # pyright: ignore[reportUnknownLambdaType]
    ):
        result = json.loads(visit_url(unit_settings, "https://example.com/article"))
    assert result["ok"] is True
    assert "full article" in result["text"]
    assert "a short blurb" not in result["text"]


@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        ("application/json", '{"text": "two  spaces and <tag>", "lines": "a\\nb"}'),
        ("text/plain", "  Text with <tag> and\n\nspacing.  "),
    ],
)
def test_visit_url_non_html_preserves_response_text(
    unit_settings: Settings, content_type: str, body: str
) -> None:
    from wahabot.ai.tools.visit_url import visit_url

    response = unittest.mock.Mock(
        url="https://page.invalid/data",
        status_code=200,
        text=body,
        headers={"content-type": content_type},
    )
    with unittest.mock.patch(
        "wahabot.ai.tools.visit_url.page.fetch_page", return_value=response
    ):
        result = json.loads(visit_url(unit_settings, response.url))
    assert result["text"] == body


def test_caption_fetch_uses_configured_proxy_and_timeout(unit_settings: Settings) -> None:
    from wahabot.ai.tools.visit_url.captions import fetch_captions

    settings = unit_settings.model_copy(
        update={
            "web_search_proxy": "http://proxy.invalid:8080",
            "web_search_timeout": 3.5,
        }
    )
    response = unittest.mock.Mock(text="captions")
    with unittest.mock.patch(
        "wahabot.ai.tools.visit_url.captions.httpx.get", return_value=response
    ) as get:
        assert fetch_captions("https://captions.invalid/data", settings) == "captions"
    assert get.call_args.kwargs["proxy"] == settings.web_search_proxy
    assert get.call_args.kwargs["timeout"] == 3.5


def test_caption_fetch_empty_proxy_sends_none(unit_settings: Settings) -> None:
    from wahabot.ai.tools.visit_url.captions import fetch_captions

    settings = unit_settings.model_copy(update={"web_search_proxy": ""})
    response = unittest.mock.Mock(text="captions")
    with unittest.mock.patch(
        "wahabot.ai.tools.visit_url.captions.httpx.get", return_value=response
    ) as get:
        assert fetch_captions("https://captions.invalid/data", settings) == "captions"
    assert get.call_args.kwargs["proxy"] is None
    assert get.call_args.kwargs["timeout"] == max(settings.web_search_timeout, 2.0)


@pytest.mark.parametrize(
    "body",
    [
        "[]",
        '{"events": [null]}',
        '{"events": "bad"}',
        '{"events": [{"segs": {}}]}',
        '{"events": [{"segs": [null]}]}',
        '{"events": [{"segs": [{"utf8": 5}]}]}',
    ],
)
def test_malformed_caption_json_fails_soft(body: str) -> None:
    from wahabot.ai.tools.visit_url import (
        join_json3,  # pyright: ignore[reportPrivateUsage]
    )

    assert join_json3(body) == ""


def test_caption_vtt_blocks_and_markup_are_not_speech() -> None:
    from wahabot.ai.tools.visit_url import (
        strip_vtt,  # pyright: ignore[reportPrivateUsage]
    )

    text = (
        "WEBVTT\n\nNOTE ignored\nnot speech\n\nSTYLE\n::cue {color:red}\n\n"
        "00:00:00.000 --> 00:00:01.000\n<v Speaker><b>Hello</b> &amp; goodbye.\n"
    )
    assert strip_vtt(text) == "Hello & goodbye."


def test_post_metadata_keeps_fractional_duration_and_bounds_entries(
    unit_settings: Settings,
) -> None:
    from wahabot.ai.tools.visit_url import visit_url

    fetched: list[int] = []

    def entries() -> Any:
        for i in range(200):
            fetched.append(i)
            yield {"id": str(i), "duration": 0.25}

    info = {"title": "Many items", "id": "post", "entries": entries()}
    with unittest.mock.patch(
        "wahabot.ai.tools.visit_url.page.yt_dlp.YoutubeDL", _fake_ydl(info)
    ):
        result = json.loads(visit_url(unit_settings, "https://instagram.com/p/many/"))
    assert result["kind"] == "post"
    assert result["item_count"] == 100 and result["item_count_truncated"] is True
    assert result["duration_s"] == 25
    assert len(fetched) == 101


def test_shell_result_spills_over_cap() -> None:
    """A flooding command spills its full output to files (end to end).

    ``run_shell_command`` caps the inline stdout/stderr but must not
    lose output: each overflowing stream carries a spill file —
    `file` for stdout, `stderr_file` for stderr — holding the
    complete capture (seeded with the inline preview) so the agent
    can read the rest in parts. The whole path is exercised here
    (reader threads, spill, envelope) — a `_render_result`-only test
    once passed while the readers dropped the tail in production.
    """
    settings = Settings(
        webhook_hmac_key="k",
        waha_url="http://waha.invalid",
        waha_api_key="k",
        llm_api_base="http://llm.invalid",
        llm_api_key="k",
        shell_tool=True,
        shell_max_output=200,
        shell_timeout=30,
        _env_file=None,
    )
    loud = "loud" * 2500  # 10k chars, far past the 200-char inline cap
    command = f"printf %s '{loud}'; echo boom >&2; echo boom >&2"
    rendered = json.loads(shell_command(settings, command))
    assert rendered["ok"] is True
    assert rendered["exit_code"] == 0
    assert rendered["truncated"] is True
    assert rendered["stdout"] == loud[:200]
    assert rendered["stderr"] == "boom\nboom"  # small stderr rides inline
    assert "file" in rendered
    spill = Path(rendered["file"]["path"]).read_text(encoding="utf-8")
    assert loud in spill  # the complete stream, head to tail
    assert rendered["file"]["bytes"] == len(loud.encode())
    assert "stderr_file" not in rendered  # stderr fit inline: no disk touch


def test_shell_small_command_never_touches_disk() -> None:
    """Output within the inline cap rides the envelope alone: no files."""
    settings = Settings(
        webhook_hmac_key="k",
        waha_url="http://waha.invalid",
        waha_api_key="k",
        llm_api_base="http://llm.invalid",
        llm_api_key="k",
        shell_tool=True,
        shell_max_output=200,
        shell_timeout=30,
        _env_file=None,
    )
    rendered = json.loads(shell_command(settings, "echo small; echo oops >&2"))
    assert rendered["ok"] is True
    assert rendered["truncated"] is False
    assert rendered["stdout"] == "small"
    assert rendered["stderr"] == "oops"
    assert "file" not in rendered
    assert "stderr_file" not in rendered


def test_shell_sink_finalize_drops_late_reader_chunks() -> None:
    """A late reader write after finalize must not crash or corrupt.

    The timeout path renders the envelope while a reader daemon thread
    can still be draining pipe remnants; before the finalize flag that
    write hit a closed handle — crashing the reader and losing the
    partial tail the envelope promised. Finalized sinks drop late
    chunks instead.
    """
    from wahabot.ai.tools.shell import (
        _StreamSink,  # pyright: ignore[reportPrivateUsage]
    )

    sink = _StreamSink("race", 10)
    sink.append(b"x" * 10)  # fill the preview
    sink.append(b"spilled")  # open the spill file
    meta = sink.file_meta()  # finalize (the timeout path's render)
    assert meta is not None
    sink.append(b"late-tail")  # a reader racing past the kill
    assert sink.file_meta() is meta  # idempotent
    content = Path(meta["path"]).read_bytes()
    assert b"late-tail" not in content  # dropped, not crashed
    assert content == b"x" * 10 + b"spilled"


def test_shell_timeout_keeps_partial_output() -> None:
    """A timed-out command reports the partial output it managed to print.

    The old behavior threw everything away on timeout; the kill costs
    the model the missing tail, never what was already captured: the
    error envelope carries the inline previews (and spill files for
    anything that overflowed them).
    """
    settings = Settings(
        webhook_hmac_key="k",
        waha_url="http://waha.invalid",
        waha_api_key="k",
        llm_api_base="http://llm.invalid",
        llm_api_key="k",
        shell_tool=True,
        shell_max_output=200,
        shell_timeout=1,
        _env_file=None,
    )
    rendered = json.loads(shell_command(settings, 'bash -c "echo started; sleep 30"'))
    assert rendered["ok"] is False
    assert "timed out" in rendered["error"]
    assert rendered["stdout"] == "started"
    assert rendered["truncated"] is False


def test_shell_timeout_covers_process_after_output_eof(unit_settings: Settings) -> None:
    settings = unit_settings.model_copy(update={"shell_timeout": 1})
    started = time.monotonic()
    rendered = json.loads(
        shell_command(settings, "printf started; sleep 0.6; exec 1>&- 2>&-; exec sleep 3")
    )
    assert time.monotonic() - started < 1.5
    assert rendered["ok"] is False
    assert "timed out" in rendered["error"]
    assert rendered["stdout"] == "started"


def test_shell_spill_failure_keeps_exit_and_drains_output(
    unit_settings: Settings,
) -> None:
    settings = unit_settings.model_copy(update={"shell_max_output": 200})
    with unittest.mock.patch(
        "wahabot.ai.tools.shell.open_byte_output", side_effect=OSError("disk unavailable")
    ):
        result = json.loads(shell_command(settings, "printf '%010000d' 1; exit 7"))
    assert result["ok"] is True and result["exit_code"] == 7
    assert result["stdout"] == "0" * 200
    assert result["truncated"] is True
    assert "disk unavailable" in result["capture_errors"]["stdout"]
    assert "file" not in result


@pytest.mark.parametrize("failure", ["seed", "overflow", "later", "close"])
def test_shell_failed_spills_are_not_published(failure: str, tmp_path: Path) -> None:
    from wahabot.ai.tools.shell import (
        _StreamSink,  # pyright: ignore[reportPrivateUsage]
    )

    handle = unittest.mock.Mock()
    if failure == "close":
        handle.close.side_effect = OSError("flush failed")
    else:
        before = {"seed": 0, "overflow": 1, "later": 2}[failure]
        handle.write.side_effect = [None] * before + [OSError("write failed")]
    meta = {"path": str(tmp_path / "incomplete.txt"), "bytes": 0}
    sink = _StreamSink("test", 10)
    with unittest.mock.patch(
        "wahabot.ai.tools.shell.open_byte_output", return_value=(handle, meta)
    ):
        sink.append(b"x" * 20)
        sink.append(b"later")
        assert sink.file_meta() is None
        writes = handle.write.call_count
        sink.append(b"discarded")
        assert handle.write.call_count == writes
    assert sink.text == "x" * 10
    assert sink.truncated is True and sink.capture_error is not None


def test_shell_preview_byte_limit_handles_multibyte_text() -> None:
    from wahabot.ai.tools.shell import (
        _StreamSink,  # pyright: ignore[reportPrivateUsage]
    )

    raw = (" " + "\u00e9" * 101).encode()
    sink = _StreamSink("unicode", 200)
    sink.append(raw)
    assert sink.text.strip() == raw[:200].decode(errors="replace").strip()
    assert sink.text.endswith("\ufffd")
    meta = sink.file_meta()
    assert meta is not None and Path(meta["path"]).read_bytes() == raw
    assert sink.capture_error is None


def test_shell_pipe_read_failure_preserves_available_preview() -> None:
    from wahabot.ai.tools.shell import (
        _start_reader,  # pyright: ignore[reportPrivateUsage]
        _StreamSink,  # pyright: ignore[reportPrivateUsage]
    )

    stream = unittest.mock.Mock()
    stream.read.side_effect = [b"started", OSError("pipe failed")]
    sink = _StreamSink("read", 200)
    reader = _start_reader(stream, sink)
    reader.join(timeout=2)
    assert not reader.is_alive()
    assert sink.text == "started"
    assert sink.truncated is True
    assert sink.capture_error is not None and "pipe failed" in sink.capture_error
    assert sink.file_meta() is None


def test_web_search_spills_cut_snippets(unit_settings: Settings) -> None:
    """A snippet cut past the inline cap is not lost: full text on file.

    The inline content cap keeps the envelope small; a cut snippet is flagged
    `content_truncated` with the *full* findings riding `file` so the
    model can read the rest via the shell tool.
    """
    from wahabot.ai.tools import web_search as web_search_mod

    long_snippet = "detail " * 300  # 2.1k chars, past the 600-char cap
    webserp_output = json.dumps(
        {
            "results": [
                {"url": "https://a.example", "title": "Hit A", "content": long_snippet},
                {"url": "https://b.example", "title": "Hit B", "content": "short"},
            ]
        }
    )
    with unittest.mock.patch.object(
        web_search_mod, "_run_webserp", return_value=webserp_output
    ):
        result = json.loads(web_search_mod.web_search(unit_settings, "query"))
    assert result["ok"] is True
    assert result["count"] == 2
    finding = result["results"][0]
    assert finding["content_truncated"] is True
    assert (
        finding["content"]
        == long_snippet[
            : web_search_mod._MAX_CONTENT_CHARS  # pyright: ignore[reportPrivateUsage]
        ]
    )
    assert result["results"][1].get("content_truncated") is None
    assert "file" in result
    spill = json.loads(Path(result["file"]["path"]).read_text(encoding="utf-8"))
    assert spill["query"] == "query"
    assert spill["results"][0]["content"] == long_snippet  # uncut
