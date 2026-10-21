from typing import Any
from urllib.parse import urlsplit

import httpx
from llama_index.core.tools import BaseTool, FunctionTool
from loguru import logger

from wahabot.ai.serials import SerialEntry, serial_store
from wahabot.ai.tools.envelope import error, ok
from wahabot.ai.tools.runcontext import (
    RunTarget,
    current_target,
    delivery_target_error,
    fenced_chat,
    fenced_message_id,
    log_action_reason,
    operator_run,
)
from wahabot.ai.tools.schemas import (
    ProfilePictureSchema,
)
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings

_URL_PROBE_TIMEOUT_S = 10.0

_PROBE_USER_AGENT = "wahabot/0.6"


def probe_media_url(url: str) -> str | None:
    """Reject malformed URLs and confirmed HTTP 404/410; otherwise return None.

    A best-effort pre-send gate for ``send_media``'s URL source.
    Falls back to GET for HTTP 400 through 405 from HEAD. None is not
    proof that fetching or decoding the media will succeed.

    Soft-fail by design: connection/timeout errors only warn — WAHA
    may still fetch the URL fine (different network path, transient
    DNS) — while a definitive "not found" (404/410) refuses the send.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return f"not a valid URL: {url}"
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return f"not an http(s) URL: {url}"
    try:
        response = httpx.head(
            url,
            timeout=_URL_PROBE_TIMEOUT_S,
            follow_redirects=True,
            headers={"User-Agent": _PROBE_USER_AGENT},
        )
        if 400 <= response.status_code <= 405:
            # Some servers refuse HEAD — verify with the real verb.
            response = httpx.get(
                url,
                timeout=_URL_PROBE_TIMEOUT_S,
                follow_redirects=True,
                headers={"User-Agent": _PROBE_USER_AGENT},
            )
        if response.status_code in (404, 410):
            return (
                f"URL does not exist (HTTP {response.status_code}); "
                "do not guess media URLs"
            )
        if response.status_code >= 400:
            logger.warning(
                "Media URL probe got HTTP {code} for {url}; leaving the send to WAHA",
                code=response.status_code,
                url=url,
            )
    except httpx.HTTPError as exc:
        logger.warning(
            "Media URL probe failed for {url} ({exc}); leaving the send to WAHA",
            url=url,
            exc=exc,
        )
    return None


def profile_picture_builder(waha: WahaClient) -> BaseTool:
    """Build a tool that fetches a chat's or contact's profile picture URL."""

    def get_profile_picture_fn(chat: str | None = None, reason: str = "") -> str:
        """Fetch the profile picture URL for a chat (current by default)."""
        target = current_target()
        chat_id, fence_error = fenced_chat(chat, target)
        if fence_error:
            return error(fence_error)
        if not chat_id or not target.session:
            return error("no active conversation context")
        log_action_reason("get_profile_picture", reason, chat=chat_id)
        url = waha.get_profile_picture(target.session, chat_id)
        if not url:
            return ok(chat=chat_id, url="", available=False)
        return ok(
            chat=chat_id,
            url=url,
            available=True,
            hint="send it with send_media kind=image url=<url>",
        )

    return FunctionTool.from_defaults(
        fn=get_profile_picture_fn,
        fn_schema=ProfilePictureSchema,
        name="get_profile_picture",
        description=(
            "Fetch the profile picture URL for the current conversation or an "
            "exact `chat` JID (operator commands). Returns the URL, or "
            "available=false when WhatsApp hides it. Send it with send_media "
            "kind=image url=…."
        ),
    )


def resolve_message_token(
    token: str,
    *,
    session: str,
    chat_id: str,
    settings: Settings | None,
    allow_search: bool = False,
    allowed_chats: set[str] | None = None,
) -> tuple[str, str | None]:
    """Resolve a ``#m<n>`` token to its raw id; raw ids pass through.

    Resolution is against the *target* chat's registry: an operator
    command that names a chat and then replies by token acts in that
    chat, never in the console's synthetic context. With no single
    target chat, the command's resolved chats are tried (a unique match
    wins); operator reactions may finally search the session for a
    unique match. Failure is a soft error string, never an exception.
    """
    value = token.strip()
    if not value.startswith("#m"):
        return value, None
    if settings is None:
        return "", f"unknown message token {value}"
    store = serial_store(str(settings.data_dir))
    entry = lookup_token(store, session, chat_id, allowed_chats, value)
    if entry is None and allow_search:
        entry = store.find(session, value)
    if entry is None:
        return "", f"unknown message token {value} in this chat"
    return entry.message_id, None


def lookup_token(
    store: Any,
    session: str,
    chat_id: str,
    allowed_chats: set[str] | None,
    value: str,
) -> SerialEntry | None:
    """The token in the target chat, else a unique match across allowed chats."""
    if chat_id:
        entry = store.resolve(session, chat_id, value)
        if entry is not None:
            return entry
    if not allowed_chats:
        return None
    hits = {
        found
        for chat in allowed_chats
        if chat
        for found in (store.resolve(session, chat, value),)
        if found is not None
    }
    return next(iter(hits)) if len({found.message_id for found in hits}) == 1 else None


def resolve_mention_tokens(
    mentions: list[str] | None,
    *,
    session: str,
    chat_id: str,
    settings: Settings | None,
    allow_search: bool,
) -> list[str] | None:
    """Resolve ``#m<n>`` mention entries to their sender JID; names pass through.

    The prompt's mention handles are serials, not JIDs; the tool is where
    the id boundary may be crossed. An unresolved serial is dropped (it
    tags nobody) rather than sent to WAHA as a literal token.
    """
    if not mentions or settings is None:
        return mentions
    out: list[str] = []
    for mention in mentions:
        value = mention.strip()
        if not value.startswith("#m"):
            out.append(mention)
            continue
        store = serial_store(str(settings.data_dir))
        entry = store.resolve(session, chat_id, value)
        if entry is None and allow_search:
            entry = store.find(session, value)
        if entry is not None and entry.jid:
            out.append(entry.jid)
    return out


def prepare_send_target(
    chat: str | None,
    reply_to: str | None,
    target: RunTarget,
    settings: Settings | None,
) -> tuple[str, str | None, str | None]:
    """Validate a send's chat and quote; ``(chat_id, reply_id, error)``."""
    chat_id, fence_error = fenced_chat(chat, target)
    if fence_error:
        return "", None, fence_error
    if not target.session or not chat_id:
        return "", None, "no active conversation context"
    if reply_to:
        reply_to, token_error = resolve_message_token(
            reply_to,
            session=target.session,
            chat_id=chat_id,
            settings=settings,
            allow_search=operator_run(target),
            allowed_chats=target.allowed_chats,
        )
        if token_error:
            return chat_id, None, token_error
    delivery_error = delivery_target_error(chat_id, reply_to, target)
    if delivery_error:
        return chat_id, reply_to, delivery_error
    if reply_to:
        _, id_error = fenced_message_id(reply_to, target)
        if id_error:
            return chat_id, None, id_error
    return chat_id, reply_to, None
