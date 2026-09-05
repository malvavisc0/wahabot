"""HTTP client for the WAHA WhatsApp HTTP API."""

from typing import Any
from urllib.parse import quote

import httpx
from loguru import logger

API_PREFIX = "/api"

__all__ = ["MediaTooLargeError", "WahaClient"]


def message_media(message: dict[str, Any]) -> dict[str, Any]:
    """The media dict of a message, from either payload location."""
    return message.get("media") or message.get("_data", {}).get("media") or {}


def message_fields(message: dict[str, Any]) -> tuple[str, str, str]:
    """Body text, media filename and mimetype of a message."""
    media = message_media(message)
    return (
        str(message.get("body", "") or ""),
        str(media.get("filename") or ""),
        str(media.get("mimetype") or ""),
    )


def message_matches(message: dict[str, Any], needle: str) -> bool:
    """Whether a message's body, media filename or mimetype contains ``needle``."""
    return any(needle in field.casefold() for field in message_fields(message))


class WahaClient:
    """Minimal WAHA API client (see https://waha.devlike.pro/docs/how-to/send-messages)."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-Api-Key"] = api_key
        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=10)

    def send_text(
        self,
        session: str,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        mentions: list[str] | None = None,
    ) -> None:
        """Send a text message, raising for HTTP errors.

        ``reply_to`` (a serialized message id) sends the text as a native
        quote-reply to that message — the WAHA ``reply_to`` field, which
        replaced the deprecated ``POST /api/reply`` endpoint.
        ``mentions`` is a list of JIDs whose display names appear in
        *text* as ``@<name>``; WhatsApp highlights them and notifies the
        mentioned people.
        """
        body: dict[str, Any] = {"session": session, "chatId": chat_id, "text": text}
        if reply_to:
            body["reply_to"] = reply_to
        if mentions:
            body["mentions"] = mentions
        response = self._client.post(f"{API_PREFIX}/sendText", json=body)
        response.raise_for_status()
        logger.info(
            "Sent text to {chat_id} in session {session}",
            chat_id=chat_id,
            session=session,
        )

    def get_me(self, session: str) -> dict[str, Any]:
        """Fetch the logged-in user's info; 404s when the session is dead."""
        response = self._client.get(f"{API_PREFIX}/sessions/{session}/me")
        response.raise_for_status()
        return response.json()

    def get_session(self, session: str) -> dict[str, Any]:
        """Fetch session info including its status — GET /api/sessions/{session}.

        The status field is the WAHA ``SessionStatus`` enum (``STOPPED``,
        ``STARTING``, ``SCAN_QR_CODE``, ``PASSKEY_REQUIRED``,
        ``PASSKEY_CONFIRMATION_REQUIRED``, ``WORKING``, ``FAILED``); only
        ``WORKING`` means the session can send and receive.
        """
        response = self._client.get(f"{API_PREFIX}/sessions/{session}")
        response.raise_for_status()
        return response.json()

    def list_chats(self, session: str, limit: int = 200) -> list[dict[str, Any]]:
        """All chats (id + name), newest conversation first.

        WAHA ``GET /api/{session}/chats``; sorted by conversation
        timestamp descending (the endpoint default), capped at *limit*.
        """
        response = self._client.get(
            f"{API_PREFIX}/{session}/chats", params={"limit": limit}
        )
        response.raise_for_status()
        return response.json()

    def list_contacts(self, session: str, limit: int = 500) -> list[dict[str, Any]]:
        """All contacts (id + name) — WAHA ``GET /api/contacts/all``."""
        response = self._client.get(
            f"{API_PREFIX}/contacts/all", params={"session": session, "limit": limit}
        )
        response.raise_for_status()
        return response.json()

    def get_message(self, session: str, chat_id: str, message_id: str) -> dict[str, Any]:
        """Fetch a single message by its serialized id, raising for HTTP errors."""
        segment = quote(chat_id, safe="")
        message_segment = quote(message_id, safe="")
        response = self._client.get(
            f"{API_PREFIX}/{session}/chats/{segment}/messages/{message_segment}",
            params={"downloadMedia": False},
        )
        response.raise_for_status()
        return response.json()

    def download_media(self, url: str, max_bytes: int | None = None) -> bytes:
        """Download a message's media file, raising for HTTP errors.

        When ``max_bytes`` is set, the body is streamed and a
        ``MediaTooLargeError`` is raised (fail fast) instead of
        buffering a file the vision API would reject anyway.
        """
        if max_bytes is None:
            response = self._client.get(url)
            response.raise_for_status()
            return response.content
        chunks: list[bytes] = []
        size = 0
        with self._client.stream("GET", url) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise MediaTooLargeError(url, max_bytes)
                chunks.append(chunk)
        return b"".join(chunks)

    def send_reaction(self, session: str, message_id: str, reaction: str) -> None:
        """React to a message (empty reaction removes it), raising for HTTP errors."""
        self._client.put(
            f"{API_PREFIX}/reaction",
            json={
                "session": session,
                "messageId": message_id,
                "reaction": reaction,
            },
        ).raise_for_status()

    def send_image(
        self,
        session: str,
        chat_id: str,
        file: dict[str, Any],
        caption: str | None = None,
    ) -> None:
        """Send an image from a url or base64 payload, raising for HTTP errors."""
        body: dict[str, Any] = {"session": session, "chatId": chat_id, "file": file}
        if caption:
            body["caption"] = caption
        self._client.post(f"{API_PREFIX}/sendImage", json=body).raise_for_status()

    def fetch_chat_messages(
        self,
        session: str,
        chat_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch recent messages of a chat, raising for HTTP errors."""
        segment = quote(chat_id, safe="")
        response = self._client.get(
            f"{API_PREFIX}/{session}/chats/{segment}/messages",
            params={"limit": limit, "downloadMedia": False},
        )
        response.raise_for_status()
        return response.json()

    def get_chat_overview(
        self,
        session: str,
        chat_id: str,
    ) -> Any:
        """Return a chat's metadata from the chats overview, raising for HTTP errors."""
        body = {
            "filter": {"ids": [chat_id]},
            "pagination": {"limit": 1, "offset": 0},
        }
        response = self._client.post(f"{API_PREFIX}/{session}/chats/overview", json=body)
        response.raise_for_status()
        items = response.json()
        return items[0] if isinstance(items, list) and items else {}

    def search_messages(
        self,
        session: str,
        query: str,
        chat_id: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Search a chat's recent messages for a text substring, filtering locally.

        Fetches up to ``limit`` recent messages of ``chat_id`` and keeps
        those whose body, media filename or media mimetype contains
        ``query`` (case-insensitive).
        """
        params: dict[str, Any] = {
            "session": session,
            "chatId": chat_id,
            "limit": limit,
            "merge": True,
        }
        response = self._client.get(f"{API_PREFIX}/messages", params=params)
        response.raise_for_status()
        messages = response.json()
        needle = query.casefold()
        return [m for m in messages if message_matches(m, needle)]

    def forward_message(
        self,
        session: str,
        chat_id: str,
        message_id: str,
    ) -> None:
        """Forward a message to another chat, raising for HTTP errors."""
        self._client.post(
            f"{API_PREFIX}/forwardMessage",
            json={
                "session": session,
                "chatId": chat_id,
                "messageId": message_id,
            },
        ).raise_for_status()


class MediaTooLargeError(Exception):
    """A media file exceeded the configured maximum size."""

    def __init__(self, url: str, max_bytes: int) -> None:
        super().__init__(f"media at {url} exceeds {max_bytes} bytes")
        self.url = url
        self.max_bytes = max_bytes
