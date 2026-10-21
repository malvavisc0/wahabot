import asyncio
import inspect
import json
from typing import Any, cast

import httpx
from fastapi import FastAPI, HTTPException

from tests.harness.constants import FOREIGN_JID
from tests.harness.events import is_caption_request


def tool_call_response(
    name: str,
    arguments: dict[str, Any],
    *,
    call_id: str,
    response_id: str = "chatcmpl-smoke-tool",
    created: int = 1788525831,
    usage: tuple[int, int] = (10, 5),
) -> dict[str, Any]:
    """A ``chat.completion`` answer whose assistant turn emits one tool call."""
    prompt_tokens, completion_tokens = usage
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": "smoke-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


FirstResponse = tool_call_response(
    "send_message",
    {"text": "smoke reply one", "reason": "directly addressed by name"},
    call_id="call_send_1",
    response_id="chatcmpl-smoke-1",
    created=1788525828,
)

SecondResponse = {
    "id": "chatcmpl-smoke-2",
    "object": "chat.completion",
    "created": 1788525829,
    "model": "smoke-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "smoke final answer"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
}

FileResponse = tool_call_response(
    "send_media",
    {
        "kind": "file",
        "url": "http://files.invalid/q3/report.pdf",
        "caption": "the report",
    },
    call_id="call_file_1",
    response_id="chatcmpl-smoke-3",
    created=1788525830,
)

VideoResponse = tool_call_response(
    "send_media",
    {
        "kind": "video",
        "url": "http://files.invalid/q4/clip.mp4",
        "caption": "the clip",
    },
    call_id="call_video_1",
    response_id="chatcmpl-smoke-4",
    created=1788525833,
)

VoiceResponse = tool_call_response(
    "send_media",
    {"kind": "voice", "url": "http://files.invalid/q5/note.mp3"},
    call_id="call_voice_1",
    response_id="chatcmpl-smoke-5",
    created=1788525834,
)

VoiceTextResponse = tool_call_response(
    "send_media",
    {
        "kind": "voice",
        "text": "ya voy, un momento",
        "language": "es",
        "reason": "answering a voice note in kind",
    },
    call_id="call_voice_text_1",
    response_id="chatcmpl-smoke-voice-text",
    created=1788525835,
)

StickerResponse = tool_call_response(
    "send_media",
    {"kind": "sticker", "url": "http://files.invalid/q6/laugh.webp"},
    call_id="call_sticker_1",
    response_id="chatcmpl-smoke-6",
    created=1788525835,
)

FenceRefusalResponse = tool_call_response(
    "send_message",
    {"text": "leak", "chat": FOREIGN_JID},
    call_id="call_fence_1",
    response_id="chatcmpl-smoke-fence",
    created=1788525832,
)

CaptionResponse = {
    "id": "chatcmpl-smoke-caption",
    "object": "chat.completion",
    "created": 1788525831,
    "model": "smoke-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "two smoke-test squares on a white background",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14},
}

EscalateResponse = tool_call_response(
    "escalate",
    {"report": "Smoke Sender wants a human"},
    call_id="call_escalate_1",
    response_id="chatcmpl-smoke-escalate",
    created=1788525836,
)


def mock_transport() -> httpx.MockTransport:
    """A transport that answers every request with an empty 200 list."""
    return httpx.MockTransport(lambda _request: httpx.Response(200, json=[]))


class FakeLlm:
    """The scripted OpenAI-compatible chat completions server.

    Stateful, instance-scoped: ``requests`` records every body,
    ``override`` scripts the first agent-turn answer of a scenario, and
    ``selector`` routes every agent request through a callable when the
    round counter cannot tell interleaved runs apart (the concurrency
    checks). ``fail_status`` (0 = off) answers every request with that
    HTTP status instead, simulating a dead proxy in front of the
    provider, and ``hang_seconds`` (0 = off) sleeps before answering,
    simulating a healthy-but-slow provider for timeout tests (paired
    with a tiny ``llm_timeout`` so the client's budget fires). The app
    closure reads these attributes, never module globals, so a test
    sets ``fake_llm.override = …`` directly and the ``global``
    declarations of the old script disappear.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.override: dict[str, Any] | None = None
        self.selector: Any = None
        self.fail_status = 0
        self.hang_seconds = 0.0

    def app(self) -> FastAPI:
        """An OpenAI-compatible chat completions server with scripted answers."""
        return _make_llm_app(self)

    def clear(self) -> None:
        """Drop recorded requests and any per-scenario scripting."""
        self.requests.clear()
        self.override = None
        self.selector = None
        self.fail_status = 0
        self.hang_seconds = 0.0


def _make_llm_app(llm: FakeLlm) -> FastAPI:
    """An OpenAI-compatible chat completions server bound to *llm*."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def create(body: dict[str, Any]) -> Any:
        llm.requests.append(body)
        if llm.hang_seconds:
            # A slow generation: the connection is healthy, nothing is
            # sent until the sleep ends — the client's per-request
            # timeout (a tiny llm_timeout in tests) fires mid-sleep.
            await asyncio.sleep(llm.hang_seconds)
        if llm.fail_status:
            # A dead proxy answers before the provider ever sees the
            # request; the body argument is irrelevant to the failure.
            raise HTTPException(status_code=llm.fail_status)
        if is_caption_request(body):
            return CaptionResponse
        if llm.selector is not None:
            selected = llm.selector(body)
            if inspect.isawaitable(selected):
                selected = await selected
            return cast(dict[str, Any], selected)
        agent_round = sum(
            1 for request in llm.requests if not is_caption_request(request)
        )
        first = llm.override if llm.override else FirstResponse
        return first if agent_round == 1 else SecondResponse

    return app
