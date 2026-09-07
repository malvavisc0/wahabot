"""Frame extraction and the text anchor for video messages.

The LLM endpoint accepts images, not videos (see
``docs/plans/video-understanding.md``), so a video becomes: a handful
of evenly spaced JPEG frames for one caption call, plus the spoken
track from the WhisperX service, folded into the user message as the
durable text anchor — ``(video shows: …)`` and ``[audio: "…"]`` — the
same trick ``ai/vision.py`` uses for photos. The frames ride the first
LLM call of the run only (``workflow.with_image``); the anchor rides
every later turn for free.
"""

import asyncio
import json
import subprocess
import tempfile
from pathlib import Path

from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from loguru import logger

from wahabot.ai.vision import clamp_caption_line

__all__ = [
    "MAX_TRANSCRIPT_CHARS",
    "VIDEO_CAPTION_PROMPT",
    "caption_video",
    "extract_frames",
    "join_anchor",
    "probe_duration",
    "video_marker",
]

#: Longest edge of an extracted frame. Vision detail beyond this is
#: token cost, not information: 512 px recognizes subject, action and
#: setting (measured against real clips in the plan's evidence table).
FRAME_MAX_SIDE = 512

#: Transcript chars kept in the anchor; a long clip must not inflate
#: every later turn. Mirrors ``vision.MAX_CAPTION_CHARS``'s role for
#: captions.
MAX_TRANSCRIPT_CHARS = 1000

VIDEO_CAPTION_PROMPT = (
    "These are frames, in order, from one video. Describe in one short "
    "sentence what happens: the subject, the action, the setting, and any "
    "visible text. No preamble, no disclaimers — just the description."
)


def probe_duration(data: bytes) -> float:
    """Video duration in seconds via ffprobe; raises on any failure."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
        handle.write(data)
        handle.flush()
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                handle.name,
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def extract_frames(data: bytes, count: int, timeout: float = 30.0) -> list[bytes]:
    """*count* evenly spaced frames as JPEG bytes, or [] on any failure.

    One ffmpeg pass: duration from :func:`probe_duration`, then
    ``-vf fps=<count>/<duration>,scale=min(512,iw):-2`` writing
    ``frame_%d.jpg`` into a temp dir. Short clips yield fewer frames
    than requested — the caption call takes what exists. A dead
    ffmpeg or an unreadable file returns [] (fail-soft: the marker
    degrades to the transcript part alone).
    """
    try:
        duration = probe_duration(data)
    except Exception as exc:
        logger.warning("Video probe failed (frames dropped): {exc}", exc=exc)
        return []
    with (
        tempfile.NamedTemporaryFile(suffix=".mp4") as handle,
        tempfile.TemporaryDirectory() as outdir,
    ):
        handle.write(data)
        handle.flush()
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    handle.name,
                    "-vf",
                    f"fps={count}/{duration},scale='min({FRAME_MAX_SIDE},iw)':-2",
                    str(Path(outdir) / "frame_%d.jpg"),
                ],
                capture_output=True,
                check=True,
                timeout=timeout,
            )
        except Exception as exc:
            logger.warning("Frame extraction failed (frames dropped): {exc}", exc=exc)
            return []
        return [path.read_bytes() for path in sorted(Path(outdir).glob("frame_*.jpg"))]


async def caption_video(
    llm: FunctionCallingLLM, frames: list[bytes], timeout: float = 30.0
) -> str:
    """One-sentence description of the video from *frames*; "" on failure.

    Same shape as ``vision.caption_image``: one user message, N
    ImageBlocks + the prompt, first line clamped to
    ``vision.MAX_CAPTION_CHARS``. A failed caption degrades the marker
    to the bare ``(video)`` form while the frames still ride the first
    LLM call.
    """
    message = ChatMessage(
        role=MessageRole.USER,
        blocks=[ImageBlock(image=frame, image_mimetype="image/jpeg") for frame in frames]
        + [TextBlock(text=VIDEO_CAPTION_PROMPT)],
    )
    try:
        response = await asyncio.wait_for(llm.achat([message]), timeout=timeout)
    except Exception as exc:
        logger.warning("Video caption failed (turn stays text-anchored): {exc}", exc=exc)
        return ""
    caption = clamp_caption_line(str(response.message.content or ""))
    logger.info("Video caption: {caption!r}", caption=caption)
    return caption


def video_marker(caption: str, transcript: str) -> str:
    """The turn's anchor: ``(video shows: …)`` and/or ``[audio: "…"]``.

    Empty inputs drop their part; both empty yields ``(video)`` — the
    model at least knows a video arrived, which already beats today's
    silent drop. The ``shows`` framing keeps an instruction-like
    caption reading as a description of pixels, not as sender text
    (same convention as ``vision.image_noun``).
    """
    parts: list[str] = []
    if caption:
        parts.append(f"(video shows: {caption})")
    if transcript:
        trimmed = transcript[:MAX_TRANSCRIPT_CHARS].rstrip()
        suffix = "…" if len(transcript) > MAX_TRANSCRIPT_CHARS else ""
        parts.append(f'[audio: "{trimmed}{suffix}"]')
    if not parts:
        return "(video)"
    return " ".join(parts)


def join_anchor(body: str, marker: str) -> str:
    """Sender text plus the anchor, space-joined (empty body → marker)."""
    return f"{body} {marker}".strip()
