"""The WAHA-backed transport, recording an intent around every send."""

import contextlib
import time
from collections.abc import Callable
from typing import Any

from wahabot.core.delivery.intents import (
    SENDING,
    DeliveryIntent,
    IntentStore,
)
from wahabot.core.delivery.send import (
    AMBIGUOUS,
    RETRYABLE,
    backoff_seconds,
    classify_send_error,
)
from wahabot.core.transport.context import current_intake_ids
from wahabot.core.waha import WahaClient


class WahaTransport:
    """Delegate to ``WahaClient`` while persisting each outbound intent."""

    def __init__(self, client: WahaClient, intents: IntentStore | None = None) -> None:
        self.client = client
        self.intents = intents

    def _deliver(
        self,
        *,
        session: str,
        chat_id: str,
        kind: str,
        payload: dict[str, Any],
        call: Callable[[], Any],
    ) -> Any:
        intents = self.intents
        if intents is None:
            return call()
        intent = intents.prepare(
            session=session,
            chat_id=chat_id,
            kind=kind,
            payload=payload,
            intake_ids=list(current_intake_ids()),
        )
        intents.sending(intent.id)
        try:
            result = call()
        except Exception as exc:
            self.record_failure(intent, exc)
            raise
        try:
            intents.sent(intent.id, result if isinstance(result, str) else "")
        except Exception as exc:
            # The remote side accepted it but the local receipt did not
            # commit: this is an uncertain outcome, not an unsent message.
            with contextlib.suppress(Exception):
                intents.unknown(intent.id, f"local receipt failed: {exc}")
            raise
        return result

    def record_failure(self, intent: DeliveryIntent, exc: Exception) -> None:
        """Persist the failed or unknown state for one intent."""
        intents = self.intents
        if intents is None:
            return
        disposition = classify_send_error(exc)
        if disposition == RETRYABLE:
            attempts = intent.attempts + 1
            if attempts >= intents.max_attempts:
                intents.failed(
                    intent.id,
                    str(exc),
                    next_attempt_at=None,
                    operator_required=True,
                )
            else:
                intents.failed(
                    intent.id,
                    str(exc),
                    next_attempt_at=time.time() + backoff_seconds(attempts),
                )
        elif disposition == AMBIGUOUS:
            if intent.kind == "operator_notice":
                # A notice is idempotent and losing it is worse than a
                # duplicate: retry it with the same bounded backoff.
                attempts = intent.attempts + 1
                if attempts >= intents.max_attempts:
                    intents.failed(
                        intent.id,
                        str(exc),
                        next_attempt_at=None,
                        operator_required=True,
                    )
                else:
                    intents.failed(
                        intent.id,
                        str(exc),
                        next_attempt_at=time.time() + backoff_seconds(attempts),
                    )
            else:
                intents.unknown(intent.id, str(exc))
        else:
            intents.failed(
                intent.id, str(exc), next_attempt_at=None, operator_required=True
            )

    def send_text(
        self,
        session: str,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        mentions: list[str] | None = None,
    ) -> str:
        payload = {
            "text": text,
            "reply_to": reply_to,
            "mentions": mentions,
        }
        return self._deliver(
            session=session,
            chat_id=chat_id,
            kind="text",
            payload=payload,
            call=lambda: self.client.send_text(
                session, chat_id, text, reply_to=reply_to, mentions=mentions
            ),
        )

    def send_image(
        self,
        session: str,
        chat_id: str,
        file: dict[str, Any],
        caption: str | None = None,
    ) -> str:
        return self._deliver(
            session=session,
            chat_id=chat_id,
            kind="image",
            payload={"file": file, "caption": caption},
            call=lambda: self.client.send_image(session, chat_id, file, caption),
        )

    def send_file(
        self,
        session: str,
        chat_id: str,
        file: dict[str, Any],
        caption: str | None = None,
    ) -> str:
        return self._deliver(
            session=session,
            chat_id=chat_id,
            kind="file",
            payload={"file": file, "caption": caption},
            call=lambda: self.client.send_file(session, chat_id, file, caption),
        )

    def send_video(
        self,
        session: str,
        chat_id: str,
        file: dict[str, Any],
        caption: str | None = None,
        convert: bool = True,
    ) -> str:
        return self._deliver(
            session=session,
            chat_id=chat_id,
            kind="video",
            payload={"file": file, "caption": caption, "convert": convert},
            call=lambda: self.client.send_video(
                session, chat_id, file, caption=caption, convert=convert
            ),
        )

    def send_voice(
        self,
        session: str,
        chat_id: str,
        file: dict[str, Any],
        reply_to: str | None = None,
        convert: bool = True,
    ) -> str:
        return self._deliver(
            session=session,
            chat_id=chat_id,
            kind="voice",
            payload={"file": file, "reply_to": reply_to, "convert": convert},
            call=lambda: self.client.send_voice(
                session, chat_id, file, reply_to=reply_to, convert=convert
            ),
        )

    def send_sticker(
        self,
        session: str,
        chat_id: str,
        file: dict[str, Any],
        reply_to: str | None = None,
    ) -> str:
        return self._deliver(
            session=session,
            chat_id=chat_id,
            kind="sticker",
            payload={"file": file, "reply_to": reply_to},
            call=lambda: self.client.send_sticker(
                session, chat_id, file, reply_to=reply_to
            ),
        )

    def forward_message(self, session: str, chat_id: str, message_id: str) -> str:
        return self._deliver(
            session=session,
            chat_id=chat_id,
            kind="forward",
            payload={"message_id": message_id},
            call=lambda: self.client.forward_message(session, chat_id, message_id),
        )

    def send_reaction(
        self, session: str, chat_id: str, message_id: str, reaction: str
    ) -> None:
        self._deliver(
            session=session,
            chat_id=chat_id,
            kind="reaction",
            payload={"message_id": message_id, "reaction": reaction},
            call=lambda: self.client.send_reaction(
                session, chat_id, message_id, reaction
            ),
        )

    def recover_intents(
        self, now: float | None = None, *, recover_sending: bool = True
    ) -> list[str]:
        """Resume prepared intents, schedule due failures, settle sendings."""
        intents = self.intents
        if intents is None:
            return []
        moment = time.time() if now is None else now
        replayed: list[str] = []
        for intent in intents.recoverable(moment):
            recovered = self._recover_one(intents, intent, moment, recover_sending)
            if recovered is not None:
                replayed.append(recovered)
        return replayed

    def recover_intent(
        self,
        intent: DeliveryIntent,
        now: float | None = None,
        *,
        recover_sending: bool = False,
    ) -> str | None:
        """Resume one already-loaded intent (used by the recovery tick)."""
        intents = self.intents
        if intents is None:
            return None
        moment = time.time() if now is None else now
        return self._recover_one(intents, intent, moment, recover_sending)

    def _recover_one(
        self,
        intents: IntentStore,
        intent: DeliveryIntent,
        moment: float,
        recover_sending: bool,
    ) -> str | None:
        if intent.state == SENDING:
            if recover_sending:
                intents.unknown(intent.id, "process stopped while the send was in flight")
            return None
        if intent.state == "failed" and intents.exhausted(intent):
            intents.failed(
                intent.id,
                intent.last_error,
                next_attempt_at=None,
                operator_required=True,
            )
            return None
        if intent.state == "failed" and (
            intent.next_attempt_at is None or intent.next_attempt_at > moment
        ):
            return None
        intents.sending(intent.id)
        try:
            result = self.replay(intent)
        except Exception as exc:
            self.record_failure(intent, exc)
            return None
        intents.sent(intent.id, result if isinstance(result, str) else "")
        return intent.id

    def replay(self, intent: DeliveryIntent) -> Any:
        """Re-invoke the transport call described by one stored intent."""
        payload = intent.payload
        chat_id = intent.chat_id
        session = intent.session
        if intent.kind in ("text", "operator_notice"):
            return self.client.send_text(
                session,
                chat_id,
                str(payload.get("text", "")),
                reply_to=payload.get("reply_to"),
                mentions=payload.get("mentions"),
            )
        if intent.kind == "image":
            return self.client.send_image(
                session, chat_id, payload["file"], caption=payload.get("caption")
            )
        if intent.kind == "file":
            return self.client.send_file(
                session, chat_id, payload["file"], caption=payload.get("caption")
            )
        if intent.kind == "video":
            return self.client.send_video(
                session,
                chat_id,
                payload["file"],
                caption=payload.get("caption"),
                convert=bool(payload.get("convert", True)),
            )
        if intent.kind == "voice":
            return self.client.send_voice(
                session,
                chat_id,
                payload["file"],
                reply_to=payload.get("reply_to"),
                convert=bool(payload.get("convert", True)),
            )
        if intent.kind == "sticker":
            return self.client.send_sticker(
                session, chat_id, payload["file"], reply_to=payload.get("reply_to")
            )
        if intent.kind == "forward":
            return self.client.forward_message(session, chat_id, payload["message_id"])
        if intent.kind == "reaction":
            return self.client.send_reaction(
                session, chat_id, payload["message_id"], payload["reaction"]
            )
        raise ValueError(f"cannot replay intent kind {intent.kind!r}")

    def send_seen(self, session: str, chat_id: str) -> None:
        self.client.send_seen(session, chat_id)

    def set_typing(self, session: str, chat_id: str, typing: bool) -> None:
        self.client.set_typing(session, chat_id, typing)

    def get_session(self, session: str) -> dict[str, Any]:
        return self.client.get_session(session)

    def get_me(self, session: str) -> dict[str, Any]:
        return self.client.get_me(session)

    def get_message(self, session: str, chat_id: str, message_id: str) -> dict[str, Any]:
        return self.client.get_message(session, chat_id, message_id)

    def fetch_chat_messages(
        self, session: str, chat_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        return self.client.fetch_chat_messages(session, chat_id, limit)

    def list_chats(self, session: str, limit: int = 200) -> list[dict[str, Any]]:
        return self.client.list_chats(session, limit)

    def list_contacts(self, session: str, limit: int = 500) -> list[dict[str, Any]]:
        return self.client.list_contacts(session, limit)

    def get_chat_overview(self, session: str, chat_id: str) -> Any:
        return self.client.get_chat_overview(session, chat_id)

    def get_profile_picture(self, session: str, jid: str) -> str:
        return self.client.get_profile_picture(session, jid)

    def lid_phone(self, session: str, lid: str) -> str:
        return self.client.lid_phone(session, lid)

    def search_messages(
        self, session: str, query: str, chat_id: str, limit: int = 200
    ) -> list[dict[str, Any]]:
        return self.client.search_messages(session, query, chat_id, limit)

    def download_media(self, url: str, max_bytes: int | None = None) -> bytes:
        return self.client.download_media(url, max_bytes)
