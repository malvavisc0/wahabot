from typing import Any, override

from tests.harness.constants import ME_JID, SMOKE_PNG
from tests.harness.llm import mock_transport
from wahabot.core.waha import WahaClient


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
        self.sent_images: list[tuple[str, str, dict[str, Any], str | None]] = []
        self.sent_files: list[tuple[str, str, dict[str, Any], str | None]] = []
        self.sent_videos: list[tuple[str, str, dict[str, Any], str | None, bool]] = []
        self.sent_voices: list[tuple[str, str, dict[str, Any], bool]] = []
        self.sent_stickers: list[tuple[str, str, dict[str, Any]]] = []
        self.typing_calls: list[tuple[str, str, bool]] = []
        self.seen_chats: list[tuple[str, str]] = []
        self.reactions: list[tuple[str, str]] = []
        self.video_bytes = b""
        #: What ``get_session`` answers — the probe path's WAHA view.
        self.session_status = "WORKING"

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
    def send_image(
        self,
        session: str,
        chat_id: str,
        file: dict[str, Any],
        caption: str | None = None,
    ) -> str:
        self.sent_images.append((session, chat_id, file, caption))
        return f"true_{chat_id}_SENTIMAGE{len(self.sent_images)}"

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
    def send_video(
        self,
        session: str,
        chat_id: str,
        file: dict[str, Any],
        caption: str | None = None,
        convert: bool = True,
    ) -> str:
        self.sent_videos.append((session, chat_id, file, caption, convert))
        return f"true_{chat_id}_SENTVIDEO{len(self.sent_videos)}"

    @override
    def send_voice(
        self,
        session: str,
        chat_id: str,
        file: dict[str, Any],
        reply_to: str | None = None,
        convert: bool = True,
    ) -> str:
        self.sent_voices.append((session, chat_id, file, convert))
        return f"true_{chat_id}_SENTVOICE{len(self.sent_voices)}"

    @override
    def send_sticker(
        self,
        session: str,
        chat_id: str,
        file: dict[str, Any],
        reply_to: str | None = None,
    ) -> str:
        self.sent_stickers.append((session, chat_id, file))
        return f"true_{chat_id}_SENTSTICKER{len(self.sent_stickers)}"

    @override
    def set_typing(self, session: str, chat_id: str, typing: bool) -> None:
        self.typing_calls.append((session, chat_id, typing))

    @override
    def send_seen(self, session: str, chat_id: str) -> None:
        self.seen_chats.append((session, chat_id))

    @override
    def send_reaction(
        self, session: str, chat_id: str, message_id: str, reaction: str
    ) -> None:
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
        """Session info for health seeds and recovery probes.

        ``session_status`` is the live answer (WORKING by default) —
        a test flips it to simulate WAHA hanging in a dead state and
        flipping back without ever emitting the status event, which is
        exactly what the recovery poller exists for.
        """
        return {"name": session, "status": self.session_status}

    @override
    def get_me(self, session: str) -> dict[str, Any]:
        """Own identity for the operator-notification target."""
        return {"id": ME_JID, "lid": "491555000000@lid", "pushname": "kai"}

    @override
    def download_media(self, url: str, max_bytes: int | None = None) -> bytes:
        """Serve the canned PNG (or MP4 for video URLs) for any media fetch."""
        if ".mp4" in url:
            return self.video_bytes
        return SMOKE_PNG
