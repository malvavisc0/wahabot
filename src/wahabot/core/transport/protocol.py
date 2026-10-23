"""The outbound operations the turn, delivery and status paths use."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Transport(Protocol):
    """Structural interface every WhatsApp transport implements."""

    def send_text(
        self,
        session: str,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        mentions: list[str] | None = None,
    ) -> str: ...

    def send_image(
        self,
        session: str,
        chat_id: str,
        file: dict[str, Any],
        caption: str | None = None,
    ) -> str: ...

    def send_file(
        self,
        session: str,
        chat_id: str,
        file: dict[str, Any],
        caption: str | None = None,
    ) -> str: ...

    def send_video(
        self,
        session: str,
        chat_id: str,
        file: dict[str, Any],
        caption: str | None = None,
        convert: bool = True,
    ) -> str: ...

    def send_voice(
        self,
        session: str,
        chat_id: str,
        file: dict[str, Any],
        reply_to: str | None = None,
        convert: bool = True,
    ) -> str: ...

    def send_sticker(
        self,
        session: str,
        chat_id: str,
        file: dict[str, Any],
        reply_to: str | None = None,
    ) -> str: ...

    def forward_message(self, session: str, chat_id: str, message_id: str) -> str: ...

    def send_reaction(
        self, session: str, chat_id: str, message_id: str, reaction: str
    ) -> None: ...

    def send_seen(self, session: str, chat_id: str) -> None: ...

    def set_typing(self, session: str, chat_id: str, typing: bool) -> None: ...

    def get_session(self, session: str) -> dict[str, Any]: ...

    def get_me(self, session: str) -> dict[str, Any]: ...

    def get_message(
        self, session: str, chat_id: str, message_id: str
    ) -> dict[str, Any]: ...

    def fetch_chat_messages(
        self, session: str, chat_id: str, limit: int = 50
    ) -> list[dict[str, Any]]: ...

    def list_chats(self, session: str, limit: int = 200) -> list[dict[str, Any]]: ...

    def list_contacts(self, session: str, limit: int = 500) -> list[dict[str, Any]]: ...

    def get_chat_overview(self, session: str, chat_id: str) -> Any: ...

    def get_profile_picture(self, session: str, jid: str) -> str: ...

    def lid_phone(self, session: str, lid: str) -> str: ...

    def search_messages(
        self, session: str, query: str, chat_id: str, limit: int = 200
    ) -> list[dict[str, Any]]: ...

    def download_media(self, url: str, max_bytes: int | None = None) -> bytes: ...
