"""HTTP client for the WAHA WhatsApp HTTP API."""

from typing import Any
from urllib.parse import quote

import httpx
from loguru import logger

API_PREFIX = "/api"


def message_matches(message: dict[str, Any], needle: str) -> bool:
    """Whether a message's body, media filename or mimetype contains ``needle``."""
    body = str(message.get("body", "") or "").casefold()
    media = message.get("media") or message.get("_data", {}).get("media") or {}
    filename = str(media.get("filename") or "").casefold()
    mimetype = str(media.get("mimetype") or "").casefold()
    return needle in body or needle in filename or needle in mimetype


class WahaClient:
    """Minimal WAHA API client (see https://waha.devlike.pro/docs/how-to/send-messages)."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-Api-Key"] = api_key
        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=10)

    def send_text(self, session: str, chat_id: str, text: str) -> None:
        """Send a text message, raising for HTTP errors."""
        response = self._client.post(
            f"{API_PREFIX}/sendText",
            json={"session": session, "chatId": chat_id, "text": text},
        )
        response.raise_for_status()
        logger.info(
            "Sent text to {chat_id} in session {session}",
            chat_id=chat_id,
            session=session,
        )

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

    def send_seen(self, session: str, chat_id: str) -> None:
        """Mark the chat as seen/read, raising for HTTP errors."""
        self._client.post(
            f"{API_PREFIX}/sendSeen",
            json={"session": session, "chatId": chat_id},
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

    def get_contact(self, session: str, contact_id: str) -> dict[str, Any]:
        """Fetch a single contact by id, raising for HTTP errors."""
        segment = quote(contact_id, safe="")
        response = self._client.get(
            f"{API_PREFIX}/{session}/contacts/{segment}",
        )
        response.raise_for_status()
        return response.json()

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

    def set_typing(
        self,
        session: str,
        chat_id: str,
        typing: bool,
    ) -> None:
        """Start or stop the typing indicator, raising for HTTP errors."""
        endpoint = f"{API_PREFIX}/startTyping" if typing else f"{API_PREFIX}/stopTyping"
        self._client.post(
            endpoint, json={"session": session, "chatId": chat_id}
        ).raise_for_status()
