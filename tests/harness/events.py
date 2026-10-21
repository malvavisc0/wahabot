import json
import time
from typing import Any

from tests.harness.constants import CHAT_ID, ME_JID, OWN_LID, SESSION
from tests.harness.media import image_part_count


def chat_event(
    *,
    body: str,
    mid: str,
    chat: str = CHAT_ID,
    from_me: bool = False,
) -> dict[str, Any]:
    """A text ``message`` event for *mid* on *chat*, distinct from the shared fixtures.

    Used by the persistence reservations that must address a specific
    private chat (``PC``…). ``from_me`` marks the event as the bot's own.
    The ``_data.id`` block always carries the same serialized id as
    ``payload.id`` (real WAHA keeps the two in sync). Replaces the
    per-test ``pc_event`` closures.
    """
    ev = waha_event()
    prefix = "true" if from_me else "false"
    ev["id"] = f"evt-smoke-pc-{mid}"
    ev["payload"]["id"] = f"{prefix}_{chat}_{mid}"
    ev["payload"]["from"] = chat
    ev["payload"]["fromMe"] = from_me
    ev["payload"]["body"] = body
    if from_me:
        make_operator_typed(ev, chat)
    # Real WAHA keeps payload.id and _data.id._serialized in sync (both
    # are the same serialized id); the inbound-note fallback reads
    # _data first, so a stale base fixture id here would make every
    # event share one message id — the redelivery dedup keys on it.
    ev["payload"]["_data"] = dict(ev["payload"]["_data"])
    ev["payload"]["_data"]["id"] = {"_serialized": f"{prefix}_{chat}_{mid}"}
    return ev


def make_operator_typed(ev: dict[str, Any], chat: str) -> dict[str, Any]:
    """Reshape *ev* into what WAHA (WEBJS) sends when the human types.

    Captured from production webhooks: a ``fromMe`` message carries the
    account itself in ``from``, the chat in ``to``, the account's LID as
    ``participant`` and ``source="app"`` (``"api"`` marks the bot's own
    sends through WAHA).
    """
    payload = ev["payload"]
    payload["fromMe"] = True
    payload["from"] = OWN_LID
    payload["to"] = chat
    payload["participant"] = OWN_LID
    payload["source"] = "app"
    payload["_data"] = dict(payload["_data"])
    payload["_data"]["notifyName"] = "Operator Human"
    return ev


def is_caption_request(body: dict[str, Any]) -> bool:
    """True when a captured request is the image captioner, not an agent run.

    The captioner sends a single user message with an image part and no
    tool schemas; agent runs always advertise tools.
    """
    return "tools" not in body and image_part_count(body) > 0


def waha_event(mid: str = "ABCDEF") -> dict[str, Any]:
    """A realistic WAHA message event: group chat, bot mentioned.

    The timestamp is one second in the future so it deterministically
    postdates the handler's ``started_at`` (an integer second could
    truncate below the registration time and read as startup backlog).

    *mid* rides both id shapes (``payload.id`` and
    ``_data.id._serialized``) — real WAHA keeps them in sync, and the
    inbound-note fallback prefers ``_data``, so a test posting two
    events into one chat must give each its own *mid* or the
    redelivery dedup keys collide.
    """
    now = int(time.time()) + 1
    return {
        "id": f"evt-smoke-{mid}",
        "timestamp": now,
        "event": "message",
        "session": SESSION,
        "me": {"id": ME_JID, "lid": "491555000000@lid"},
        "payload": {
            "id": f"false_{CHAT_ID}_{mid}",
            "timestamp": now,
            "from": CHAT_ID,
            "fromMe": False,
            "participant": "491555000001@c.us",
            "body": "kai hola, smoke check",
            "_data": {
                "type": "text",
                "id": {"_serialized": f"false_{CHAT_ID}_{mid}"},
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
