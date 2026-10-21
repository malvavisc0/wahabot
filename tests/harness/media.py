import subprocess
import tempfile
from functools import cache
from pathlib import Path
from typing import Any, cast


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


@cache
def smoke_video_bytes() -> bytes:
    """The generated smoke MP4, created once and cached for the process."""
    with tempfile.TemporaryDirectory(prefix="wahabot-smoke-video-") as vdir:
        return _smoke_video_mp4(vdir)


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
