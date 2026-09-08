"""Shared harness for the wahabot test suite.

Importable helpers only — no process-global test state lives here
(that belongs to the fixtures). Everything is a plain function, class
or factory so a test can build any piece it needs without touching a
module global.

This is the harness of the old ``scripts/smoke_test.py`` extracted and
relandscaped: the script's mutable globals (``llm_requests``,
``first_response_override``, ``first_response_selector``) become the
``FakeLlm`` instance, ``serving``/the event factories/``smoke_settings``
are unchanged, and the cached artifacts become instance attributes on
``RecordingWaha``.
"""

import contextlib
import hashlib
import hmac
import inspect
import json
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast, override

import httpx
import uvicorn
from fastapi import FastAPI

import wahabot.webhook as webhook_module
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings

CHAT_ID = "1234567890-1234567890@g.us"
SESSION = "default"
ME_JID = "491555000000@c.us"

#: A foreign JID nobody in a test run inhabits — the cross-chat target
#: the fence must refuse on chat runs.
FOREIGN_JID = "19999999999@c.us"

#: Model answers: the first round requests the send_message tool, the
#: second round (after the tool result) is the final text. The JSON is
#: what an OpenAI-compatible provider sends on the wire. No ``chat``
#: argument: a chat-run send to the *current* conversation (an explicit
#: foreign JID would hit the cross-chat fence).
FirstResponse = {
    "id": "chatcmpl-smoke-1",
    "object": "chat.completion",
    "created": 1788525828,
    "model": "smoke-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_send_1",
                        "type": "function",
                        "function": {
                            "name": "send_message",
                            "arguments": json.dumps({"text": "smoke reply one"}),
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

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

#: Third scenario: the model delivers a document via send_file (URL form).
FileResponse = {
    "id": "chatcmpl-smoke-3",
    "object": "chat.completion",
    "created": 1788525830,
    "model": "smoke-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_file_1",
                        "type": "function",
                        "function": {
                            "name": "send_file",
                            "arguments": json.dumps(
                                {
                                    "url": "http://files.invalid/q3/report.pdf",
                                    "caption": "the report",
                                }
                            ),
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

#: Fence regression: the model (misleadingly instructed by a chat
#: participant) tries to send to a chat outside the conversation.
FenceRefusalResponse = {
    "id": "chatcmpl-smoke-fence",
    "object": "chat.completion",
    "created": 1788525832,
    "model": "smoke-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_fence_1",
                        "type": "function",
                        "function": {
                            "name": "send_message",
                            "arguments": json.dumps(
                                {"text": "leak", "chat": FOREIGN_JID}
                            ),
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

#: Answer for image-caption requests (a request carrying an image part
#: but no tool schemas is the vision captioner, not an agent run). One
#: short sentence, exactly what ``caption_image`` expects back.
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

#: Escalate wire scenario: the model calls the escalate tool.
EscalateResponse = {
    "id": "chatcmpl-smoke-escalate",
    "object": "chat.completion",
    "created": 1788525836,
    "model": "smoke-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_escalate_1",
                        "type": "function",
                        "function": {
                            "name": "escalate",
                            "arguments": json.dumps(
                                {"report": "Smoke Sender wants a human"}
                            ),
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

#: Canned image bytes "downloaded" from WAHA for the album check. A
#: real PNG header so ``first_frame_png``-style consumers would accept
#: them too; the wire assertion only needs the data URL to carry them.
SMOKE_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c626001000000ffff030000060005"
    "57bfabd40000000049454e44ae426082"  # pyright: ignore[reportImplicitStringConcatenation]
)


@contextlib.contextmanager
def serving(app: FastAPI) -> Iterator[int]:
    """Run *app* on an ephemeral localhost port; yields the port."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    # Uvicorn binds port 0 → find the actual port from its bound socket.
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def mock_transport() -> httpx.MockTransport:
    """A transport that answers every request with an empty 200 list."""
    return httpx.MockTransport(lambda _request: httpx.Response(200, json=[]))


def tool_call_response(
    name: str,
    arguments: dict[str, Any],
    *,
    call_id: str,
    response_id: str = "chatcmpl-smoke-tool",
    created: int = 1788525831,
    usage: tuple[int, int] = (10, 5),
) -> dict[str, Any]:
    """A ``chat.completion`` answer whose assistant turn emits one tool call.

    Shared by the operator-command, message-id-fence and foreign-chat
    running tests — the inline copies they used to rebuild drifted.
    """
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


def chat_event(
    *,
    body: str,
    mid: str,
    chat: str = CHAT_ID,
    from_me: bool = False,
    include_serialized: bool = False,
) -> dict[str, Any]:
    """A text ``message`` event for *mid* on *chat*, distinct from the shared fixtures.

    Used by the persistence reservations that must address a specific
    private chat (``PC``…). ``from_me`` marks the event as the bot's own;
    ``include_serialized`` writes the ``_data.id`` block the
    message-id handlers read. Replaces the per-test ``pc_event`` closures.
    """
    ev = waha_event()
    prefix = "true" if from_me else "false"
    ev["id"] = f"evt-smoke-pc-{mid}"
    ev["payload"]["id"] = f"{prefix}_{chat}_{mid}"
    ev["payload"]["from"] = chat
    ev["payload"]["fromMe"] = from_me
    ev["payload"]["body"] = body
    if include_serialized:
        ev["payload"]["_data"] = dict(ev["payload"]["_data"])
        ev["payload"]["_data"]["id"] = {"_serialized": f"{prefix}_{chat}_{mid}"}
    return ev


def is_caption_request(body: dict[str, Any]) -> bool:
    """True when a captured request is the image captioner, not an agent run.

    The captioner sends a single user message with an image part and no
    tool schemas; agent runs always advertise tools.
    """
    return "tools" not in body and image_part_count(body) > 0


class FakeLlm:
    """The scripted OpenAI-compatible chat completions server.

    Stateful, instance-scoped: ``requests`` records every body,
    ``override`` scripts the first agent-turn answer of a scenario, and
    ``selector`` routes every agent request through a callable when the
    round counter cannot tell interleaved runs apart (the concurrency
    checks). The app closure reads these attributes, never module
    globals, so a test sets ``fake_llm.override = …`` directly and the
    ``global`` declarations of the old script disappear.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.override: dict[str, Any] | None = None
        self.selector: Any = None

    def is_caption_request(self, body: dict[str, Any]) -> bool:
        """True when a captured request is the image captioner, not an agent run."""
        return is_caption_request(body)

    def app(self) -> FastAPI:
        """An OpenAI-compatible chat completions server with scripted answers."""
        return _make_llm_app(self)

    def clear(self) -> None:
        """Drop recorded requests and any per-scenario scripting."""
        self.requests.clear()
        self.override = None
        self.selector = None


def _make_llm_app(llm: FakeLlm) -> FastAPI:
    """An OpenAI-compatible chat completions server bound to *llm*."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def create(body: dict[str, Any]) -> dict[str, Any]:
        llm.requests.append(body)
        if llm.is_caption_request(body):
            return CaptionResponse
        if llm.selector is not None:
            selected = llm.selector(body)
            if inspect.isawaitable(selected):
                selected = await selected
            return cast(dict[str, Any], selected)
        agent_round = sum(
            1 for request in llm.requests if not llm.is_caption_request(request)
        )
        first = llm.override if llm.override else FirstResponse
        return first if agent_round == 1 else SecondResponse

    return app


class RecordingWaha(WahaClient):
    """WAHA client whose sends are recorded instead of hitting HTTP.

    The parent's HTTP client is created with a mock transport (the
    smoke run never touches WAHA HTTP). ``video_bytes`` holds the
    generated smoke MP4 served for video media URLs.
    """

    def __init__(self) -> None:
        super().__init__(
            base_url="http://waha.invalid", api_key="waha-key", transport=mock_transport()
        )
        self.sent: list[tuple[str, str, str, list[str] | None]] = []
        self.sent_files: list[tuple[str, str, dict[str, Any], str | None]] = []
        self.reactions: list[tuple[str, str]] = []
        self.video_bytes = b""

    @override
    def get_chat_overview(self, session: str, chat_id: str) -> Any:
        """Group roster lookup: two participants so the sender tag renders."""
        return {
            "id": chat_id,
            "name": "smoke group",
            "participants": [
                {"id": "491555000001@c.us", "name": "Smoke Sender"},
                {"id": "491555000000@c.us", "name": "kai"},
            ],
        }

    @override
    def fetch_chat_messages(
        self, session: str, chat_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """One recent message per sender, carrying notifyName for enrichment."""
        return [
            {
                "participant": "491555000001@c.us",
                "_data": {"notifyName": "Smoke Sender"},
            }
        ]

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

    @override
    def send_file(
        self,
        session: str,
        chat_id: str,
        file: dict[str, Any],
        caption: str | None = None,
    ) -> str:
        self.sent_files.append((session, chat_id, file, caption))
        return f"true_{chat_id}_SENTFILE{len(self.sent_files)}"

    @override
    def send_reaction(self, session: str, message_id: str, reaction: str) -> None:
        self.reactions.append((message_id, reaction))

    @override
    def get_message(self, session: str, chat_id: str, message_id: str) -> dict[str, Any]:
        """One canned message per id: fromMe + preview body for reactions."""
        body = "smoke reply one" if "SMOKEREPLY" in message_id else "someone else"
        return {
            "id": message_id,
            "from": chat_id,
            "fromMe": message_id.startswith("true_"),
            "body": body,
        }

    @override
    def get_session(self, session: str) -> dict[str, Any]:
        """Session info for the health seed: live and WORKING."""
        return {"name": session, "status": "WORKING"}

    @override
    def get_me(self, session: str) -> dict[str, Any]:
        """Own identity for the operator-notification target."""
        return {"id": ME_JID, "pushname": "kai"}

    @override
    def download_media(self, url: str, max_bytes: int | None = None) -> bytes:
        """Serve the canned PNG (or MP4 for video URLs) for any media fetch."""
        if ".mp4" in url:
            return self.video_bytes
        return SMOKE_PNG


def smoke_settings(data_dir: Path, llm_base: str) -> Settings:
    """Settings for a run; `_env_file=None` never touches the real .env."""
    return Settings(
        webhook_hmac_key="smoke-hmac-key",
        waha_url="http://waha.invalid",
        waha_api_key="waha-key",
        llm_api_base=llm_base,
        llm_api_key="sk-smoke",
        llm_model="smoke-model",
        session=SESSION,
        data_dir=data_dir,
        transcribe_url="http://whisper.invalid",
        _env_file=None,
    )


def waha_event() -> dict[str, Any]:
    """A realistic WAHA message event: group chat, bot mentioned.

    The timestamp is one second in the future so it deterministically
    postdates the handler's ``started_at`` (an integer second could
    truncate below the registration time and read as startup backlog).
    """
    now = int(time.time()) + 1
    return {
        "id": "evt-smoke-1",
        "timestamp": now,
        "event": "message",
        "session": SESSION,
        "me": {"id": ME_JID, "lid": "491555000000@lid"},
        "payload": {
            "id": f"false_{CHAT_ID}_ABCDEF",
            "timestamp": now,
            "from": CHAT_ID,
            "fromMe": False,
            "participant": "491555000001@c.us",
            "body": "kai hola, smoke check",
            "_data": {
                "type": "text",
                "id": {"_serialized": f"false_{CHAT_ID}_ABCDEF"},
                "notifyName": "Smoke Sender",
            },
        },
    }


def voice_event() -> dict[str, Any]:
    """A realistic WEBJS voice-note event: type `ptt`, empty body, audio media.

    Sent to a 1:1 chat id so ``is_group_addressed`` passes without a
    typed @mention (a bare note in a `mentioned` group is intentionally
    skipped before transcription).
    """
    chat = "491555000001@c.us"
    now = int(time.time()) + 1
    return {
        "id": "evt-smoke-voice",
        "timestamp": now,
        "event": "message",
        "session": SESSION,
        "me": {"id": ME_JID, "lid": "491555000000@lid"},
        "payload": {
            "id": f"false_{chat}_VOICE",
            "timestamp": now,
            "from": chat,
            "fromMe": False,
            "body": "",
            "hasMedia": True,
            "media": {
                "url": "http://waha.invalid/api/files/default/note.oga",
                "mimetype": "audio/ogg; codecs=opus",
            },
            "_data": {
                "type": "ptt",
                "duration": "5",
                "notifyName": "Smoke Sender",
            },
        },
    }


def image_event() -> dict[str, Any]:
    """A single addressed image message, distinct from an album member."""
    now = int(time.time()) + 1
    return {
        "id": "evt-smoke-image",
        "timestamp": now,
        "event": "message",
        "session": SESSION,
        "me": {"id": ME_JID, "lid": "491555000000@lid"},
        "payload": {
            "id": f"false_{CHAT_ID}_IMAGE",
            "timestamp": now,
            "from": CHAT_ID,
            "fromMe": False,
            "participant": "491555000001@c.us",
            "body": "kai look at this",
            "hasMedia": True,
            "media": {
                "url": "http://waha.invalid/api/files/default/smoke-image.png",
                "mimetype": "image/png",
            },
            "_data": {"type": "image", "notifyName": "Smoke Sender"},
        },
    }


def video_event(chat: str = "491555000001@c.us", mid: str = "VID") -> dict[str, Any]:
    """A realistic video message event: kind ``video``, empty body, media.

    Sent to a 1:1 chat id so ``is_group_addressed`` passes without a
    typed @mention (a bare video in a `mentioned` group is intentionally
    skipped, same as the voice fixture). ``RecordingWaha.download_media``
    serves the smoke MP4 (generated once, below) for its URL, so the
    frame path runs on real ffmpeg.
    """
    now = int(time.time()) + 1
    return {
        "id": f"evt-smoke-video-{mid}",
        "timestamp": now,
        "event": "message",
        "session": SESSION,
        "me": {"id": ME_JID, "lid": "491555000000@lid"},
        "payload": {
            "id": f"false_{chat}_{mid}",
            "timestamp": now,
            "from": chat,
            "fromMe": False,
            "body": "",
            "hasMedia": True,
            "media": {
                "url": f"http://waha.invalid/api/files/default/smoke-{mid}.mp4",
                "mimetype": "video/mp4",
            },
            "_data": {
                "type": "video",
                "id": {"_serialized": f"false_{chat}_{mid}"},
                "notifyName": "Smoke Sender",
            },
        },
    }


def album_events() -> list[dict[str, Any]]:
    """Container + two image events of one album, as WAHA NOWEB delivers.

    The images carry ``media`` blobs whose URLs the handler "downloads"
    via ``RecordingWaha.download_media``; the container declares
    ``expectedImageCount: 2`` so the buffer completes immediately.
    """
    now = int(time.time()) + 1
    base = {
        "timestamp": now,
        "event": "message",
        "session": SESSION,
        "me": {"id": ME_JID, "lid": "491555000000@lid"},
        "payload": {
            "timestamp": now,
            "from": CHAT_ID,
            "fromMe": False,
            "participant": "491555000001@c.us",
            "source": "app",
            "_data": {"notifyName": "Smoke Sender"},
        },
    }
    container = json.loads(json.dumps(base))
    container["id"] = "evt-smoke-album"
    container["payload"]["id"] = f"false_{CHAT_ID}_ALBUM"
    container["payload"]["body"] = ""
    container["payload"]["_data"].update({"type": "album", "expectedImageCount": 2})
    events = [container]
    for i in (1, 2):
        image = json.loads(json.dumps(base))
        image["id"] = f"evt-smoke-album-img{i}"
        image["payload"]["id"] = f"false_{CHAT_ID}_IMG{i}"
        image["payload"]["hasMedia"] = True
        image["payload"]["media"] = {
            "url": f"http://waha.invalid/api/files/default/smoke-img{i}.png",
            "mimetype": "image/png",
        }
        image["payload"]["_data"]["type"] = "image"
        events.append(image)
    return events


def write_session_config(settings: Settings) -> None:
    """A minimal working session config, as `sessions init` would write."""
    path = settings.access_config
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "whitelist": [],
        "blacklist": [],
        "goal": "",
        "system_prompt": "You are {{bot_name}} texting on WhatsApp.",
        "bot_name": "kai",
        "bot_mention_regex": "@?kai",
        "group_participation": "mentioned",
    }
    path.write_text(json.dumps(config, indent=2))


def sign(body: bytes, key: str) -> str:
    """Real production HMAC: sha512 hex digest, as WAHA sends it."""
    return hmac.new(key.encode(), body, hashlib.sha512).hexdigest()


def reset_handlers() -> None:
    """Drop handlers from any previous register_* call in this process."""
    for registry in webhook_module._registries.values():  # pyright: ignore[reportPrivateUsage]
        registry.clear()


def image_part_count(body: dict[str, Any]) -> int:
    """`image_url` parts across all messages of one captured LLM request."""
    count = 0
    messages = cast(list[dict[str, Any]], body.get("messages", []))
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts = cast(list[dict[str, Any]], content)
            count += sum(1 for part in parts if part.get("type") == "image_url")
    return count


def data_url_payload(body: dict[str, Any]) -> list[str]:
    """The data-URL payloads of a request's image parts, for prefix checks."""
    urls: list[str] = []
    messages = cast(list[dict[str, Any]], body.get("messages", []))
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in cast(list[dict[str, Any]], content):
            if part.get("type") == "image_url":
                urls.append(str(part["image_url"]["url"]))
    return urls


def smoke_video_bytes() -> bytes:
    """The generated smoke MP4, created once and cached for the process."""
    global _smoke_video_cache
    if _smoke_video_cache is None:
        with tempfile.TemporaryDirectory(prefix="wahabot-smoke-video-") as vdir:
            _smoke_video_cache = _smoke_video_mp4(vdir)
    return _smoke_video_cache


def _smoke_video_mp4(tmpdir: str) -> bytes:
    """A tiny real MP4 (color clip + sine audio), generated via ffmpeg.

    The frame path and the transcript stub both consume it; ffmpeg is
    a documented runtime dependency of the feature (Docker ships it).
    """
    out = Path(tmpdir) / "smoke-video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=320x240:rate=10:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(out),
        ],
        check=True,
    )
    return out.read_bytes()


_smoke_video_cache: bytes | None = None
