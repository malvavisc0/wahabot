"""HTTP client for the WAHA WhatsApp HTTP API."""

import httpx
from loguru import logger

API_PREFIX = "/api"


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
