"""WhatsApp tools for the function calling agent.

Each run binds its own :class:`RunTarget` (session, chat, delivery
latches, operator flag) through ``bind_target`` before the workflow
starts, so the shared agent's tools always speak for the message being
handled and concurrent runs never see each other's state. The WAHA
client raises on HTTP errors. Delivery tools catch send failures here;
the workflow wraps uncaught tool failures in the same JSON error envelope.

Cross-chat reach is fenced (see :func:`fenced_chat`): only operator
commands (CLI or self-chat) may select other chats. Ordinary runs may
act only in their current chat, except for ``escalate``'s fixed
operator destination.
"""

import base64
import io
import mimetypes
import re
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import httpx
from llama_index.core.tools import BaseTool, FunctionTool
from loguru import logger
from PIL import Image

from wahabot.ai.messages import jid_string
from wahabot.ai.serials import serial_store
from wahabot.ai.tools.envelope import error, ok
from wahabot.ai.tools.outfile import write_json_output
from wahabot.ai.tools.runcontext import (
    FENCE_ERROR,
    OPERATOR_ARMED,
    OPERATOR_KEY,
    RunTarget,
    bind_target,
    chat_jid,
    current_image_path,
    current_prefix,
    current_target,
    delivered_to_self,
    delivery_message_error,
    delivery_target_error,
    fenced_chat,
    fenced_message_id,
    jids_in_text,
    log_action_reason,
    operator_run,
    reset_target,
    send_failed_envelope,
    track_self_echo,
)
from wahabot.ai.tools.schemas import (
    EscalateSchema,
    ForwardMessageSchema,
    ProfilePictureSchema,
    ReactToMessageSchema,
    ReadChatSchema,
    SendMediaSchema,
    SendMessageSchema,
    StaySilentSchema,
)
from wahabot.ai.turn_records import read_record
from wahabot.core.audit import save_action
from wahabot.core.identity import AUTHOR_BOT, AUTHOR_OPERATOR, authors, display_name
from wahabot.core.identity import names as name_book
from wahabot.core.jid import roster_entries, same_chat
from wahabot.core.journal import chat_messages
from wahabot.core.presence import clear_typing, typing_pause
from wahabot.core.tts import synthesize
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings
from wahabot.status import state as _status_state

__all__ = [
    "OPERATOR_ARMED",
    "OPERATOR_KEY",
    "EscalationChannel",
    "RunTarget",
    "bind_target",
    "chat_jid",
    "current_image_path",
    "current_prefix",
    "current_target",
    "deliver_chat_text",
    "delivered_to_self",
    "dispatch_export",
    "dispatch_media",
    "escalate",
    "export_counts",
    "export_row",
    "fenced_chat",
    "fenced_message_id",
    "fit_messages",
    "forward_escalation",
    "forward_message",
    "image_file",
    "infer_mimetype",
    "jids_in_text",
    "journal_author",
    "local_file",
    "log_action_reason",
    "media_argument_error",
    "media_source_error",
    "normalize_media_source",
    "operator_run",
    "participant_jid",
    "probe_media_url",
    "react_to_message",
    "read_chat",
    "read_chat_directory",
    "read_export_journal",
    "remote_file",
    "reset_target",
    "resolve_mentions",
    "roster_entries",
    "same_chat",
    "search_matches",
    "send_media",
    "send_message",
    "send_voice_text",
    "sender_names",
    "sent_message_envelope",
    "slim_message",
    "squared_sticker_payload",
    "stay_silent",
    "sticker_file",
    "summarize_chat",
    "video_file",
    "voice_file",
]


#: Extension → MIME mapping for image URLs. Required because naive
#: ``image/{ext}`` synthesis (or hardcoding image/jpeg) sends unregistered
#: types such as ``image/jpg`` and ``image/tif`` which WAHA/WhatsApp reject.
_IMAGE_MIME_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".svg": "image/svg+xml",
    ".avif": "image/avif",
}

#: Extension → MIME mapping for documents. A wrong/absent mimetype makes
#: the receiving client render a generic blob instead of a PDF preview.
_DOC_MIME_BY_EXT: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".zip": "application/zip",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

#: Extension → MIME mapping for video sources. WAHA's ``convert`` path
#: transcodes them; the declared MIME describes the source, not a claim
#: that its container is directly playable in WhatsApp.
_VIDEO_MIME_BY_EXT: dict[str, str] = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".3gp": "video/3gpp",
    ".3g2": "video/3gpp2",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".ogv": "video/ogg",
}

#: Extension → MIME mapping for audio sent as voice notes. WAHA's ffmpeg
#: pass (``convert: true``) transcodes to opus, but the declared mimetype
#: must still be a real audio type.
_AUDIO_MIME_BY_EXT: dict[str, str] = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}

#: Per-request timeout for the best-effort media URL probe. It rejects
#: malformed links and confirmed HTTP 404/410, not every fetch failure.
_URL_PROBE_TIMEOUT_S = 10.0

#: Who we claim to be when probing; some CDNs refuse empty defaults.
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
) -> tuple[str, str | None]:
    """Resolve a ``#m<n>`` token to its raw id; raw ids pass through.

    Resolution is against the *target* chat's registry: an operator
    command that names a chat and then replies by token acts in that
    chat, never in the console's synthetic context. Operator reactions
    have no target chat, so they may search the session for a unique
    match. Failure is a soft error string, never an exception.
    """
    value = token.strip()
    if not value.startswith("#m"):
        return value, None
    if settings is None:
        return "", f"unknown message token {value}"
    store = serial_store(str(settings.data_dir))
    entry = store.resolve(session, chat_id, value)
    if entry is None and allow_search:
        entry = store.find(session, value)
    if entry is None:
        return "", f"unknown message token {value} in this chat"
    return entry.message_id, None


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


def send_message(waha: WahaClient, settings: Settings | None = None) -> BaseTool:
    """Build a tool that sends a WhatsApp text message.

    Sends to the current chat by default; an operator-command run may
    pass an explicit ``chat`` (a group or a person's JID) to reach a
    different target — a normal chat run cannot (the fence refuses,
    see :func:`fenced_chat`).

    One successful send per run, shared with media and forwarding.
    Further send calls return an error; reactions have a separate latch.

    ``@<number>`` tokens in *text* that name a roster member become
    real mentions automatically (WhatsApp shows people this way, e.g.
    "Para @111222333444555" — the model copying that shape into its
    reply must still use a numeric token). Explicit ``mentions`` JIDs
    are merged in, but neither a send acknowledgment nor this tool's
    warnings establish whether WhatsApp notified a recipient.
    """

    def send_message_fn(
        chat: str | None = None,
        text: str = "",
        reply_to: str | None = None,
        mentions: list[str] | None = None,
        reason: str = "",
    ) -> str:
        if not text.strip():
            return error("empty message text")
        target = current_target()
        if target.sent:
            return error(
                f"message already sent this run (to {target.sent}); do not send again"
            )
        chat_id, reply_to, target_error = prepare_send_target(
            chat, reply_to, target, settings
        )
        if target_error:
            return error(target_error)
        session = target.session
        log_action_reason("send_message", reason, chat=chat_id)
        try:
            sent_id, merged, dangling = deliver_chat_text(
                waha,
                session,
                chat_id,
                text,
                reply_to=reply_to,
                mentions=mentions,
                typing=target.typing,
                prefix=target.prefix,
            )
        except Exception as exc:
            return send_failed_envelope("send_message", chat_id, exc)
        delivered_to_self(chat_id, sent_id)
        target.sent = chat_id
        return sent_message_envelope(
            chat_id, apply_chat_prefix(text, target.prefix), merged, dangling
        )

    return FunctionTool.from_defaults(
        fn=send_message_fn,
        fn_schema=SendMessageSchema,
        name="send_message",
        description=(
            "Send text to the current chat, or an explicit operator target. "
            "reply_to quotes an exact message id from context or read_chat. "
            "Numeric @-tokens are auto-resolved against the roster; "
            "warnings identify unresolved tokens, not notification receipts. "
            "Shares one successful send per run with media and forwarding."
        ),
    )


def sent_message_envelope(
    chat_id: str, text: str, merged: list[str], dangling: list[str]
) -> str:
    """Describe a confirmed text send, retaining any mention-resolution warning."""
    fields: dict[str, Any] = {
        "chat": chat_id,
        "text": text,
        "mentions": merged,
    }
    if merged and not mention_tokens(text):
        fields["warning"] = (
            "mention JIDs were supplied without @<user-part> tokens; "
            "the text was sent, but mention rendering and notification "
            "are not confirmed"
        )
    elif dangling:
        fields["warning"] = (
            f"{' and '.join(f'`@{t}`' for t in dangling)} could not be "
            "auto-resolved against the roster; other explicit or resolved "
            "mentions may still be present. Resolve JIDs before future sends."
        )
    return ok(**fields)


def stay_silent() -> BaseTool:
    """Build the explicit silence tool: choose to not reply at all.

    Models follow a tool call far more reliably than the "reply with
    an empty string" instruction — without this, small models narrate
    their silence ("I'll stay silent here — ...") and the narration is
    sent to the chat as a normal reply. The workflow intercepts this
    call before executing any tools in its batch, logs its reason, and
    stops the run. Calling the function directly only returns ``ok``.
    """

    def stay_silent_fn(reason: str = "") -> str:
        log_action_reason("stay_silent", reason)
        return ok()

    return FunctionTool.from_defaults(
        fn=stay_silent_fn,
        fn_schema=StaySilentSchema,
        name="stay_silent",
        description=(
            "End the run without replying when no response is warranted. "
            "Call alone: any other tools in the same batch are canceled. "
            "Never send text to narrate silence."
        ),
    )


#: Per-chat cooldown (seconds) for escalate: one forwarded report per
#: chat per window, so a group (or an injected instruction in a fetched
#: page) cannot spam the operator's self-chat through the bot.
_ESCALATE_COOLDOWN_S = 3600


class EscalationChannel:
    """Per-agent escalation state: the cooldowns.

    Instance-scoped (one per ``build_default_tools`` call) rather than
    module-global: two agents built in one process — the smoke suite
    does exactly this — must never share a cooldown window. The
    operator JID is *not* cached here: it is read from
    ``status.state`` at call time, so a startup WAHA hiccup or a
    re-linked account can never wedge the lifeline (or aim it at a
    stale identity) for the process lifetime.
    """

    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    @property
    def operator_jid(self) -> str:
        """The current operator JID (single-sourced on ``status.state``)."""
        return _status_state.operator_jid

    def cooldown_refusal(self, chat_id: str) -> str | None:
        """The cooldown error text for *chat_id*, or None when clear."""
        last = self._last.get(chat_id)
        if last is None:
            return None
        elapsed = time.monotonic() - last
        if elapsed >= _ESCALATE_COOLDOWN_S:
            return None
        remaining = int(_ESCALATE_COOLDOWN_S - elapsed)
        message = f"already escalated from this chat; cooldown {remaining}s left"
        return (
            f"{message} — an earlier report from this chat was forwarded; "
            "this new report was not forwarded; do not claim it was"
        )

    def stamp(self, chat_id: str) -> None:
        """Record a successful escalation from *chat_id*."""
        self._last[chat_id] = time.monotonic()


def forward_escalation(
    waha: WahaClient,
    channel: EscalationChannel,
    settings: Settings,
    session: str,
    chat_id: str,
    report: str,
) -> str:
    """Forward a code- or model-written report from *chat_id* to the operator.

    Shared by the ``escalate`` tool and the triage gate. The cooldown is
    per chat, so a gate ping and an agent escalation in the same run can
    never both reach the operator. The report text is passed through;
    callers that do not have member text (the gate) write their own.
    """
    if not report.strip():
        return error("empty report text")
    operator_jid = channel.operator_jid
    if not operator_jid:
        return error("operator contact unknown; escalation unavailable")
    refusal = channel.cooldown_refusal(chat_id)
    if refusal is not None:
        return error(refusal)
    try:
        sent_id = waha.send_text(
            session,
            operator_jid,
            f"wahabot escalation from {chat_id}:\n{report.strip()}",
        )
    except Exception as exc:
        # No confirmed acknowledgment: leave the cooldown unstamped
        # and never claim the operator was notified.
        logger.warning(
            "Escalation from {chat_id} failed to send: {exc}",
            chat_id=chat_id,
            exc=exc,
        )
        return error(
            f"could not confirm forwarding to the operator: {exc}"
            + " — delivery was not confirmed; do not claim it went through "
            + "or repeat the report automatically"
        )
    track_self_echo(sent_id, "Escalation")
    channel.stamp(chat_id)
    # Durable record after the confirmed send — the audit journal's
    # fail-soft write never breaks the escalation.
    save_action(
        settings.data_dir,
        session,
        "escalation",
        chat_id=chat_id,
        report=report,
        sent_id=sent_id,
        status="open",
    )
    logger.info(
        "Escalated from {chat_id} to operator: {report}",
        chat_id=chat_id,
        report=report.strip()[:200],
    )
    return ok(chat=chat_id, escalated=True, sent_id=sent_id)


def escalate(
    waha: WahaClient, channel: EscalationChannel, settings: Settings
) -> BaseTool:
    """Build a tool that forwards a report from the chat to the operator.

    The one sanctioned way a chat run reaches the operator: the target
    JID is not a parameter but the bot's own self-chat (the same one
    the operator reads up/down notifications in), carried by *channel*.
    Only the supplied report and current chat id are sent. Report text
    is passed through, not sanitized: summarizing without pasted messages
    or secrets is a model instruction, not an enforced tool guarantee.
    This send does not consume the normal reply allowance.

    In-memory cooldown per chat: a second escalate after a successful
    send inside the window fails with an error naming remaining seconds.
    A new channel (including a process restart) starts with no cooldowns.
    The send itself is fail-soft — an unreachable session must not
    crash the run, and the cooldown is stamped only after a confirmed
    delivery: a send error says forwarding was not confirmed. A
    cooldown error distinguishes the earlier delivery from this
    request, which was not forwarded.

    After a confirmed send, the tool attempts a fail-soft audit write
    (``kind: "escalation"``). Audit failure does not undo the delivery.
    """

    def escalate_fn(report: str, reason: str = "") -> str:
        if not report.strip():
            return error("empty report text")
        target = current_target()
        chat_id = target.chat_id
        if not chat_id or not target.session:
            return error("no active conversation context")
        if operator_run(target):
            # An operator command already talks to the operator; the tool
            # would only loop the report back to its author.
            refusal = "this run is already an operator command"
            return error(f"{refusal} — report directly in the instruction instead")
        return forward_escalation(
            waha, channel, settings, target.session, chat_id, report
        )

    return FunctionTool.from_defaults(
        fn=escalate_fn,
        fn_schema=EscalateSchema,
        name="escalate",
        description=(
            "Before replying, forward a participant's report to the fixed human operator "
            "for human help, bot problems, complaints, or refusals. "
            "Not for operator commands. Does not consume the reply allowance. "
            "One successful forward per chat per hour. "
            "On success, confirm this report was forwarded. On "
            "cooldown, say an earlier report from this chat was "
            "forwarded but this new report was not. On send failure, "
            "say forwarding was not confirmed; do not automatically retry. "
            "Other errors mean the tool refused the forward. Do not "
            "claim it was read or promise a response time."
        ),
    )


def react_to_message(waha: WahaClient, settings: Settings | None = None) -> BaseTool:
    """Build a tool that reacts to a WhatsApp message.

    One successful reaction change (including removal) per run.
    Further reaction calls fail; the send latch is independent.
    ``message_id`` may be a ``#m<n>`` token; operator commands resolve it
    across the session when their synthetic context has no chat.
    """

    def react_to_message_fn(message_id: str, reaction: str = "", reason: str = "") -> str:
        target = current_target()
        if target.reacted:
            return error("already reacted this run; do not react again")
        session = target.session
        if not session:
            return error("no active conversation context")
        if not message_id:
            return error("message_id is required")
        message_id, token_error = resolve_message_token(
            message_id,
            session=session,
            chat_id=target.chat_id,
            settings=settings,
            allow_search=operator_run(target),
        )
        if token_error:
            return error(token_error)
        _, id_error = fenced_message_id(message_id, target)
        if id_error:
            return error(id_error)
        delivery_error = delivery_message_error(message_id, target)
        if delivery_error:
            return error(delivery_error)
        log_action_reason("react_to_message", reason, message_id=message_id)
        try:
            waha.send_reaction(session, message_id, reaction)
        except Exception as exc:
            return send_failed_envelope("react_to_message", message_id, exc)
        target.reacted = message_id
        return ok(message_id=message_id, reaction=reaction, removed=not reaction)

    return FunctionTool.from_defaults(
        fn=react_to_message_fn,
        fn_schema=ReactToMessageSchema,
        name="react_to_message",
        description=(
            "Add or remove the bot's reaction on an exact message id. "
            "Use an emoji to add it or an empty reaction to remove it. "
            "One successful change per run, separate from the send allowance."
        ),
    )


def send_media(waha: WahaClient, settings: Settings) -> BaseTool:
    """Build the media delivery tool: image, video, file, voice, sticker.

    One tool replaces the five old senders: a single ``kind`` argument
    plus one source per call —
    ``url`` XOR ``path`` XOR (for ``kind=voice``) ``text`` — no more
    odd-one-out schema drift (bug 1). Byte caps apply to local files and
    synthesized audio; remote downloads are delegated to WAHA. Local
    stickers are square-padded; remote stickers are not inspected or
    padded. One successful send is shared with text and forwarding.
    """

    def send_media_fn(
        kind: str = "",
        url: str | None = None,
        path: str | None = None,
        text: str | None = None,
        caption: str = "",
        language: str | None = None,
        filename: str | None = None,
        chat: str | None = None,
        reason: str = "",
    ) -> str:
        target = current_target()
        if target.sent:
            return error(
                f"message already sent this run (to {target.sent}); do not send again"
            )
        handler = _SEND_MEDIA_HANDLERS.get(kind)
        if handler is None:
            return error(f"unknown kind: {kind!r}")
        # Blank sources count as absent (a whitespace-only url must not
        # outrank a real path below) — normalize before the XOR check
        # and the dispatch, so both see the same source.
        url = normalize_media_source(url)
        path = normalize_media_source(path)
        text = normalize_media_source(text)
        arg_error = media_argument_error(
            url, path, text, kind, caption, filename, language
        )
        if arg_error:
            return error(arg_error)
        chat_id, fence_error = fenced_chat(chat, target)
        if fence_error:
            return error(fence_error)
        session = target.session
        if not session or not chat_id:
            return error("no active conversation context")
        delivery_error = delivery_target_error(chat_id, None, target)
        if delivery_error:
            return error(delivery_error)
        if url:
            probe_error = probe_media_url(url)
            if probe_error:
                return error(probe_error)
        log_action_reason("send_media", reason, chat=chat_id, kind=kind)
        return dispatch_media(
            waha,
            settings,
            handler,
            kind,
            session,
            chat_id,
            target,
            url,
            path,
            text,
            caption,
            language,
            filename,
        )

    return FunctionTool.from_defaults(
        fn=send_media_fn,
        fn_schema=SendMediaSchema,
        name="send_media",
        description=(
            "Send media using exactly one nonblank source: url, path, or "
            "voice text. Never invent an artifact or source. URL probes "
            "reject malformed/404/410 links, but other failures may be left "
            "to WAHA. Local files and synthesized audio have byte caps. "
            "Local stickers are square-padded; remote stickers are not. "
            "Voice text requires configured TTS; set language explicitly "
            "because chat language is not detected. Shares one successful "
            "send per run with text and forwarding."
        ),
    )


def normalize_media_source(source: str | None) -> str | None:
    """Strip a media source, treating missing or blank values as absent."""
    if source is None:
        return None
    return str(source).strip() or None


def media_source_error(
    url: str | None, path: str | None, text: str | None, kind: str
) -> str | None:
    """The argument-error message for send_media's source, or None.

    Exactly one source (url/path/text) per call, none blank (the fn
    normalizes blank to None first); `text` is only valid for
    ``kind=voice`` — a uniform rule with no odd-one-out (bug 1's
    lesson).
    """
    if sum(source is not None for source in (url, path, text)) != 1:
        return "pass exactly one of url, path or text"
    if text is not None and kind != "voice":
        return "text is only valid for kind=voice"
    return None


def media_argument_error(
    url: str | None,
    path: str | None,
    text: str | None,
    kind: str,
    caption: str,
    filename: str | None,
    language: str | None,
) -> str | None:
    """Validate normalized media sources and kind-specific options in order."""
    source_error = media_source_error(url, path, text, kind)
    if source_error:
        return source_error
    if caption and kind in _UNCAPTIONABLE_KINDS:
        return f"caption is not supported for kind={kind}"
    if filename and kind != "file":
        return "filename is only valid for kind=file"
    if language and (kind != "voice" or text is None):
        return "language is only valid for kind=voice with text"
    return None


def dispatch_media(
    waha: WahaClient,
    settings: Settings,
    handler: Callable[..., str],
    kind: str,
    session: str,
    chat_id: str,
    target: RunTarget,
    url: str | None,
    path: str | None,
    text: str | None,
    caption: str,
    language: str | None,
    filename: str | None,
) -> str:
    """Pass one validated media source to its existing kind-specific sender."""
    from_url = url is not None
    source = text if text is not None else (url if from_url else path)
    if kind == "voice":
        return handler(
            waha, settings, session, chat_id, target, source, text is not None, language
        )
    if kind in ("image", "video"):
        return handler(waha, settings, session, chat_id, target, source, caption)
    if kind == "file":
        return handler(
            waha, settings, session, chat_id, target, source, from_url, caption, filename
        )
    return handler(waha, settings, session, chat_id, target, source, from_url)


def send_media_image(
    waha: WahaClient,
    settings: Settings,
    session: str,
    chat_id: str,
    target: RunTarget,
    source: str,
    caption: str,
) -> str:
    """Send the single `image` source: a WAHA image payload, latch held."""
    file = image_file(source, settings.max_image_bytes)
    if isinstance(file, str):
        return error(file)
    try:
        sent_id = waha.send_image(session, chat_id, file=file, caption=caption or None)
    except Exception as exc:
        return send_failed_envelope("send_media", chat_id, exc)
    delivered_to_self(chat_id, sent_id)
    target.sent = chat_id
    return ok(
        chat=chat_id,
        kind="image",
        mimetype=file["mimetype"],
        filename=file.get("filename") or "",
        caption=caption,
    )


def send_media_video(
    waha: WahaClient,
    settings: Settings,
    session: str,
    chat_id: str,
    target: RunTarget,
    source: str,
    caption: str,
) -> str:
    """Send the single `video` source: WAHA transcodes it, latch held."""
    file = video_file(source, settings.max_video_upload_bytes)
    if isinstance(file, str):
        return error(file)
    try:
        sent_id = waha.send_video(session, chat_id, file=file, caption=caption or None)
    except Exception as exc:
        return send_failed_envelope("send_media", chat_id, exc)
    delivered_to_self(chat_id, sent_id)
    target.sent = chat_id
    return ok(
        chat=chat_id,
        kind="video",
        mimetype=file["mimetype"],
        filename=file.get("filename") or "",
        caption=caption,
    )


def send_media_file(
    waha: WahaClient,
    settings: Settings,
    session: str,
    chat_id: str,
    target: RunTarget,
    source: str,
    from_url: bool,
    caption: str,
    filename: str | None,
) -> str:
    """Send the single `file` source (a document), latch held."""
    file = (
        remote_file(source) if from_url else local_file(source, settings.max_file_bytes)
    )
    if isinstance(file, str):
        return error(file)
    if filename:
        file["filename"] = filename
    try:
        sent_id = waha.send_file(session, chat_id, file=file, caption=caption or None)
    except Exception as exc:
        return send_failed_envelope("send_media", chat_id, exc)
    delivered_to_self(chat_id, sent_id)
    target.sent = chat_id
    return ok(
        chat=chat_id,
        kind="file",
        mimetype=file["mimetype"],
        filename=file.get("filename") or "",
        caption=caption,
    )


def send_media_voice(
    waha: WahaClient,
    settings: Settings,
    session: str,
    chat_id: str,
    target: RunTarget,
    source: str,
    spoken: bool,
    language: str | None,
) -> str:
    """Send the single `voice` source: spoken text is TTS'd, else relayed."""
    if spoken:
        return send_voice_text(waha, settings, session, chat_id, target, source, language)
    file = voice_file(source, settings.max_voice_upload_bytes)
    if isinstance(file, str):
        return error(file)
    try:
        sent_id = waha.send_voice(session, chat_id, file=file)
    except Exception as exc:
        return send_failed_envelope("send_media", chat_id, exc)
    delivered_to_self(chat_id, sent_id)
    target.sent = chat_id
    return ok(
        chat=chat_id,
        kind="voice",
        mimetype=file["mimetype"],
        filename=file.get("filename") or "",
    )


def send_voice_text(
    waha: WahaClient,
    settings: Settings,
    session: str,
    chat_id: str,
    target: RunTarget,
    text: str,
    language: str | None,
) -> str:
    """Synthesize *text* with the configured TTS voice and send it (latch held)."""
    spoken = str(text).strip()
    if not spoken:
        return error("text is empty")
    if not settings.tts_url:
        return error("voice synthesis is not configured — pass url or path instead")
    audio = synthesize(settings, spoken, (language or "").strip().lower())
    if audio is None:
        return error("voice synthesis failed — send your reply as text instead")
    if len(audio) > settings.max_voice_upload_bytes:
        return error(f"voice note exceeds the {settings.max_voice_upload_bytes} B cap")
    file = voice_payload(audio)
    try:
        sent_id = waha.send_voice(session, chat_id, file=file)
    except Exception as exc:
        return send_failed_envelope("send_media", chat_id, exc)
    delivered_to_self(chat_id, sent_id)
    target.sent = chat_id
    return ok(
        chat=chat_id,
        kind="voice",
        mimetype=file["mimetype"],
        filename=file.get("filename") or "",
    )


def send_media_sticker(
    waha: WahaClient,
    settings: Settings,
    session: str,
    chat_id: str,
    target: RunTarget,
    source: str,
    from_url: bool,
) -> str:
    """Send the single `sticker` source, padding non-square locals to square."""
    file = (
        sticker_file(source, settings.max_sticker_bytes)
        if from_url
        else squared_sticker_payload(source, settings.max_sticker_bytes)
    )
    if isinstance(file, str):
        return error(file)
    try:
        sent_id = waha.send_sticker(session, chat_id, file=file)
    except Exception as exc:
        return send_failed_envelope("send_media", chat_id, exc)
    delivered_to_self(chat_id, sent_id)
    target.sent = chat_id
    return ok(
        chat=chat_id,
        kind="sticker",
        mimetype=file["mimetype"],
        **({"square": True} if not from_url else {}),
    )


_SEND_MEDIA_HANDLERS: dict[str, Callable[..., str]] = {
    "image": send_media_image,
    "video": send_media_video,
    "file": send_media_file,
    "voice": send_media_voice,
    "sticker": send_media_sticker,
}

#: Kinds whose WAHA send call has no caption parameter — the merged
#: schema advertises ``caption`` for all kinds, so the fn must refuse it
#: here rather than let it ride silently ignored.
_UNCAPTIONABLE_KINDS = frozenset({"voice", "sticker"})


def image_file(name_or_url: str, max_file_bytes: int) -> dict[str, Any] | str:
    """A WAHA image payload for a URL or local path, or an error string.

    Same shape as :func:`video_file` but typed with the image MIME
    map; a shell-tool render (``.png``, ``.webp``) must not ride the
    wire stamped ``application/octet-stream``.
    """
    if "://" in name_or_url:
        return remote_file(name_or_url, _IMAGE_MIME_BY_EXT, "image/jpeg")
    loaded = local_file(name_or_url, max_file_bytes)
    if isinstance(loaded, str):
        return loaded
    return loaded | {
        "mimetype": infer_mimetype(name_or_url, _IMAGE_MIME_BY_EXT, "image/jpeg"),
    }


def infer_mimetype(name_or_url: str, curated: dict[str, str], default: str) -> str:
    """Best-effort mimetype from a filename's or URL's extension.

    Prefers the curated extension→MIME map (avoids unregistered types
    such as ``image/jpg`` and ``image/tif``, which WAHA/WhatsApp
    reject), then ``mimetypes``, then *default* when nothing is known.
    """
    path = PurePosixPath(urlsplit(name_or_url).path)
    ext = path.suffix.lower()
    mapped = curated.get(ext)
    if mapped:
        return mapped
    guessed = mimetypes.guess_type(path.name or "")[0]
    if guessed and default.split("/")[0] == "image" and not guessed.startswith("image/"):
        return default
    return guessed or default


def remote_file(
    url: str,
    curated: dict[str, str] | None = None,
    default: str = "application/octet-stream",
) -> dict[str, Any]:
    """A WAHA RemoteFile for a URL, typed by the extension of its path.

    *curated*/*default* select the MIME map — documents by default,
    videos pass the video map. This declares a best-effort MIME type;
    it neither downloads nor validates the remote content or size.
    """
    file: dict[str, Any] = {
        "mimetype": infer_mimetype(url, curated or _DOC_MIME_BY_EXT, default),
        "url": url,
    }
    name = PurePosixPath(urlsplit(url).path).name
    if name:
        file["filename"] = name
    return file


def local_file(path: str, max_file_bytes: int) -> dict[str, Any] | str:
    """A WAHA BinaryFile for a local path, or an error message string."""
    local = Path(path)
    try:
        data = local.read_bytes()
    except OSError as exc:
        return f"cannot read {path}: {exc}"
    if len(data) > max_file_bytes:
        return f"file is {len(data)} B, over the {max_file_bytes} B cap"
    return {
        "mimetype": infer_mimetype(
            local.name, _DOC_MIME_BY_EXT, "application/octet-stream"
        ),
        "filename": local.name,
        "data": base64.b64encode(data).decode(),
    }


def voice_payload(audio: bytes) -> dict[str, Any]:
    """A WAHA ``VoiceBinaryFile`` for synthesized mp3 bytes.

    No disk touch: the bytes the TTS service returned ride straight to
    the send as base64. The synthesized form is always mp3
    (``response_format``), so the mimetype is fixed; the filename is
    required by the wire schema (WAHA's convert path names its temp
    file from it).
    """
    return {
        "mimetype": "audio/mpeg",
        "filename": "voice-note.mp3",
        "data": base64.b64encode(audio).decode(),
    }


def voice_file(name_or_url: str, max_file_bytes: int) -> dict[str, Any] | str:
    """A WAHA voice payload for a URL or local path, or an error string.

    Same shape as :func:`video_file` but typed with the audio MIME map;
    a shell-tool render (``.mp3``, ``.wav``) must not ride the wire
    stamped ``application/octet-stream``.
    """
    if "://" in name_or_url:
        return remote_file(name_or_url, _AUDIO_MIME_BY_EXT, "audio/mpeg")
    loaded = local_file(name_or_url, max_file_bytes)
    if isinstance(loaded, str):
        return loaded
    return loaded | {
        "mimetype": infer_mimetype(name_or_url, _AUDIO_MIME_BY_EXT, "audio/mpeg")
    }


def squared_sticker_payload(path: str, max_sticker_bytes: int) -> dict[str, Any] | str:
    """A square WAHA sticker payload for a local path, or an error string.

    WhatsApp renders stickers on a square canvas: a 1080x1360 meme
    sent as-is arrives squashed, and the model had no way to know
    A non-square image is
    letterboxed onto a square canvas (white bars where the image does
    not reach) and re-encoded as WebP; a square one rides the wire
    unchanged. Unreadable/corrupt images surface PIL's error as the
    error string — the model gets a diagnosis, not a crash.
    """
    loaded = local_file(path, max_sticker_bytes)
    if isinstance(loaded, str):
        return loaded
    try:
        with Image.open(io.BytesIO(base64.b64decode(loaded["data"]))) as im:
            width, height = im.size
            if width == height:
                return loaded | {
                    "mimetype": infer_mimetype(path, _IMAGE_MIME_BY_EXT, "image/webp")
                }
            side = max(width, height)
            canvas = Image.new(im.mode, (side, side), "white")
            canvas.paste(im, ((side - width) // 2, (side - height) // 2))
            buf = io.BytesIO()
            canvas.save(buf, "WEBP", lossless=True)
    except Exception as exc:
        return f"cannot decode {path} as an image: {exc}"
    data = buf.getvalue()
    if len(data) > max_sticker_bytes:
        return (
            f"padded sticker is {len(data)} B, over the "
            f"{max_sticker_bytes} B cap; downscale before sending"
        )
    return {
        "mimetype": "image/webp",
        "filename": PurePosixPath(path).name,
        "data": base64.b64encode(data).decode(),
    }


def sticker_file(name_or_url: str, max_file_bytes: int) -> dict[str, Any] | str:
    """A WAHA sticker payload for a URL or local path, or an error string.

    Typed with the image MIME map. This helper does not decode, square,
    or convert the source to WebP; local tool sends use
    :func:`squared_sticker_payload` for square-padding.
    """
    if "://" in name_or_url:
        return remote_file(name_or_url, _IMAGE_MIME_BY_EXT, "image/webp")
    loaded = local_file(name_or_url, max_file_bytes)
    if isinstance(loaded, str):
        return loaded
    return loaded | {
        "mimetype": infer_mimetype(name_or_url, _IMAGE_MIME_BY_EXT, "image/webp")
    }


def video_file(name_or_url: str, max_file_bytes: int) -> dict[str, Any] | str:
    """A WAHA video payload for a URL or local path, or an error string.

    A URL becomes a ``RemoteFile`` (WAHA downloads and transcodes it);
    a local path a ``BinaryFile`` via :func:`local_file`, re-typed to
    the video MIME map — a ``.mp4`` produced by the shell tool must
    not ride the wire stamped ``application/octet-stream``.
    """
    if "://" in name_or_url:
        return remote_file(name_or_url, _VIDEO_MIME_BY_EXT, "video/mp4")
    loaded = local_file(name_or_url, max_file_bytes)
    if isinstance(loaded, str):
        return loaded
    return loaded | {
        "mimetype": infer_mimetype(name_or_url, _VIDEO_MIME_BY_EXT, "video/mp4")
    }


def slim_message(message: dict[str, Any], max_body: int = 200) -> dict[str, Any]:
    """The model-relevant fields of a WAHA message, without the noise.

    WAHA messages carry a raw ``_data`` blob (messageSecret,
    reportingToken, engine flags — ~90% of the payload) that is useless
    to the model and inflates every tool result. Slimmed messages keep
    valid JSON and stay small enough for the inline preview. Message
    bodies are capped at *max_body* chars for the inline slice;
    ``body_truncated`` marks a cut body, whose full text lives in the
    spill file :func:`fit_messages` writes — nothing requested is
    silently dropped.
    """
    keys = ("id", "timestamp", "from", "fromMe", "participant", "body", "hasMedia", "ack")
    slimmed = {key: message[key] for key in keys if message.get(key) is not None}
    slimmed.update(sender_fields(message))
    body = slimmed.get("body")
    if isinstance(body, str) and len(body) > max_body:
        slimmed["body"] = body[:max_body] + "…"
        slimmed["body_truncated"] = True
    return slimmed


def sender_fields(message: dict[str, Any]) -> dict[str, str]:
    """Who wrote a fetched message: normalized JID, name, and account author.

    WAHA's history API carries no ``notifyName`` in LID groups, so the
    name comes from the webhook-learned name book. The bot and the
    human operator share the account, so ``fromMe`` alone cannot say
    who wrote a message: ``author`` is the recorded teammate
    (``bot``/``operator``) or ``unknown`` for messages from before the
    record started.
    """
    data = message.get("_data", {})
    sender = jid_string(message.get("participant") or data.get("author"))
    fields: dict[str, str] = {}
    if sender:
        fields["participant"] = sender
    name = str(data.get("notifyName") or "").strip() or display_name(sender)
    if name:
        fields["name"] = name
    if message.get("fromMe"):
        fields["author"] = authors.author(str(message.get("id") or "")) or "unknown"
    return fields


def fit_messages(
    messages: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
    session: str = "",
    chat_id: str = "",
) -> dict[str, Any]:
    """Slim every fetched message inline and spill the raw window to a file.

    Nothing requested is silently dropped: every message appears in
    the inline list (bodies capped by :func:`slim_message`), and the
    full raw window — complete ``_data``, uncapped bodies — goes to the
    spill file, whose metadata rides along as ``file``.
    ``returned``/``message_count`` agree; each message's
    ``body_truncated`` flags cut text whose full form is in the file.
    Empty windows omit count/file; spill failure omits ``file`` (and
    the inline list is still complete).

    With the records grammar, the inline view is the same shape the prompt
    reads (``records`` instead of ``messages``), so the console and the
    model agree and every row carries its ``#m<n>``. The raw spill is
    unchanged.
    """
    kept = [slim_message(message) for message in messages]
    out: dict[str, Any] = {"returned": len(kept)}
    if settings is not None:
        registry = serial_store(str(settings.data_dir)).for_chat(session, chat_id)
        out["records"] = [
            read_record(message, registry, settings.timezone) for message in messages
        ]
    else:
        out["messages"] = kept
    if messages:
        out["message_count"] = len(messages)
        try:
            out["file"] = write_json_output(f"chat_{int(time.time())}", messages)
        except Exception as exc:
            logger.warning("could not spill chat messages to file: {exc}", exc=exc)
    return out


#: Journal events of the shared account, by WAHA's ``source`` field:
#: ``api`` is a send through WAHA's HTTP API (the bot), ``app`` is a
#: human typing on the phone or WhatsApp Web (the operator).
_JOURNAL_AUTHORS_BY_SOURCE: dict[str, str] = {"api": AUTHOR_BOT, "app": AUTHOR_OPERATOR}

#: Row window for ``list``/``search`` when the call omits ``limit``.
_LIST_WINDOW = 20


def journal_author(payload: dict[str, Any], recorded: Any) -> str:
    """Who wrote an account's journal row: live record first, else source.

    ``recorded`` is the author book's entry for the id (``"unknown"``
    when absent): the book is the recorded truth for ids seen live,
    since WAHA history and later quote lookups lose ``source``. The
    journal keeps ``source`` on every event, so it authorizes the whole
    recorded span — including rows from before the book began.
    """
    if recorded not in (None, "", "unknown"):
        return str(recorded)
    return _JOURNAL_AUTHORS_BY_SOURCE.get(str(payload.get("source") or ""), "unknown")


def export_row(payload: dict[str, Any]) -> dict[str, Any]:
    """One slim, identity-annotated row of exported history; full body.

    Same shape as the ``list`` preview but with an uncapped body, so a
    script over the spill file greps what ``list`` would have shown.
    Account rows carry ``author`` (see :func:`journal_author`); bot rows
    drop ``name`` — the shared account's display name belongs to the
    operator, and on a bot row it would read as his words.
    """
    row = {
        key: payload[key]
        for key in ("id", "timestamp", "from", "fromMe", "hasMedia")
        if payload.get(key) is not None
    }
    row.update(sender_fields(payload))
    row["body"] = str(payload.get("body") or "")
    if row.get("fromMe"):
        row["author"] = journal_author(payload, row.get("author"))
        if row["author"] == AUTHOR_BOT:
            row.pop("name", None)
    return row


def export_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The inline export summary: author and member counts plus the spill.

    The envelope states the shared account's presence itself — one
    count per teammate — so figures built from the file cannot "lose"
    either of them; the rows, full bodies included, go to the file,
    not the context.
    """
    if not rows:
        return {}
    account: dict[str, int] = {}
    members: dict[str, int] = {}
    for row in rows:
        if row.get("fromMe"):
            author = str(row.get("author") or "unknown")
            account[author] = account.get(author, 0) + 1
        else:
            member = str(row.get("name") or row.get("participant") or "?")
            members[member] = members.get(member, 0) + 1
    out: dict[str, Any] = {
        "account_messages": account,
        "members": members,
        "span": (rows[0].get("timestamp"), rows[-1].get("timestamp")),
    }
    try:
        out["file"] = write_json_output("chat_export", rows)
    except Exception as exc:
        logger.warning("could not spill exported rows to file: {exc}", exc=exc)
    return out


def read_export_journal(
    settings: Settings, session: str, chat_id: str, limit: int | None
) -> str:
    """The ``export`` mode: the chat's recorded history in one file.

    Month-scale material for figures and registers: every journal row
    of the chat, slimmed, identity-annotated, full bodies — bounded by
    ``limit`` (newest kept) when the call asks for less.
    """
    payloads = chat_messages(settings.journal_dir, session, chat_id)
    recorded = len(payloads)
    if limit is not None and limit < recorded:
        payloads = payloads[-limit:]
    rows = [export_row(payload) for payload in payloads]
    return ok(
        chat=chat_id,
        mode="export",
        returned=len(rows),
        recorded=recorded,
        **export_counts(rows),
    )


def dispatch_export(
    settings: Settings, chat: str | None, target: RunTarget, limit: int | None
) -> str:
    """Fence then export: only operator runs may select another chat."""
    chat_id, fence_error = fenced_chat(chat, target)
    if fence_error:
        return error(fence_error)
    if not chat_id:
        return error("no active conversation context")
    if limit is not None and limit <= 0:
        return error("limit must be positive")
    target.allowed_chats.add(chat_id)
    return read_export_journal(settings, target.session, chat_id, limit)


def resolve_read_chat(
    chat: str | None, target: RunTarget, settings: Settings
) -> tuple[str, str | None]:
    """The fenced chat for a read, resolving a ``#m<n>`` chat argument."""
    if chat and chat.strip().startswith("#m"):
        raw, token_error = resolve_message_token(
            chat,
            session=target.session,
            chat_id=target.chat_id,
            settings=settings,
            allow_search=operator_run(target),
        )
        if token_error:
            return "", token_error
        chat = chat_jid(raw, target)
    chat_id, fence_error = fenced_chat(chat, target)
    if fence_error:
        return "", fence_error
    if not chat_id:
        return "", "no active conversation context"
    return chat_id, None


def read_chat(waha: WahaClient, settings: Settings) -> BaseTool:
    """Build the chat-reading tool: list/search/metadata/recent/resolve.

    ``mode`` is required in model calls. List/search inspect a bounded
    recent window; recent is operator-only; resolve searches available
    names without accepting ``chat``. Metadata is best-effort.
    The shell setting controls whether the model can read raw spills.
    """
    deref = (
        "Read full uncut bodies via run_shell_command and file.path."
        if settings.shell_tool
        else "Only the inline preview is yours; the operator can open file.path."
    )

    def read_chat_fn(
        mode: str = "list",
        chat: str | None = None,
        query: str = "",
        name: str = "",
        limit: int | None = None,
        reason: str = "",
    ) -> str:
        target = current_target()
        if not target.session:
            return error("no active conversation context")
        if mode in ("recent", "resolve"):
            return read_chat_directory(waha, target, mode, chat, name, limit)
        if mode == "export":
            return dispatch_export(settings, chat, target, limit)
        handler = _READ_CHAT_HANDLERS.get(mode)
        if handler is None:
            return error(f"unknown mode: {mode!r}")
        if mode in ("list", "search") and limit is not None and limit <= 0:
            return error("limit must be positive")
        chat_id, chat_error = resolve_read_chat(chat, target, settings)
        if chat_error:
            return error(chat_error)
        target.allowed_chats.add(chat_id)
        return handler(waha, target.session, chat_id, query, name, limit, settings)

    return FunctionTool.from_defaults(
        fn=read_chat_fn,
        fn_schema=ReadChatSchema,
        name="read_chat",
        description=(
            "Read available chat context. list/search return every fetched "
            "message slimmed (bodies capped at 200 chars) and spill the "
            "full raw window to a JSON file whose metadata comes back as "
            "file. returned/message_count count fetched/matched messages, "
            "not all history. body_truncated marks cut text whose full form "
            "is in the file. Never quote unseen content. "
            f"{deref} A search miss applies only to its recent window. "
            "Metadata returns available fields/counts; missing fields are "
            "unknown, names are display labels, and participant_list appears "
            "only for up to 20 known JIDs. resolve returns up to five available "
            "name matches: current roster, or operator's first 200 chats then "
            "first 500 contacts only if no chat matches. Misses are not "
            "exhaustive. export spills the chat's deep recorded history to one "
            "annotated file, teammates and members counted inline."
        ),
    )


def read_chat_directory(
    waha: WahaClient,
    target: RunTarget,
    mode: str,
    chat: str | None,
    name: str,
    limit: int | None,
) -> str:
    """Handle recent/resolve lookups, which do not accept a chat argument."""
    if chat:
        return error(f"`chat` is not valid for mode={mode}")
    if mode == "recent":
        return read_recent_chats(waha, target.session, target, limit)
    return read_resolve_chat(waha, target.session, target, name)


def read_recent_chats(
    waha: WahaClient, session: str, target: RunTarget, limit: int | None
) -> str:
    """Newest conversations for operator commands, clamped to at most 30."""
    if not operator_run(target):
        return error(FENCE_ERROR)
    if limit is None:
        limit = _RECENT_CHATS_CAP
    if limit <= 0:
        return error("limit must be positive")
    try:
        chats = waha.list_chats(session, limit=min(limit, _RECENT_CHATS_CAP))
    except Exception as exc:
        return error(f"could not list chats: {exc}")
    entries = [
        {"id": jid_string(c.get("id", "")), "name": str(c.get("name", ""))}
        for c in chats
        if c.get("id")
    ]
    target.allowed_chats.update(entry["id"] for entry in entries)
    return ok(chats=entries)


def read_resolve_chat(
    waha: WahaClient, session: str, target: RunTarget, name: str
) -> str:
    """Available-name matching: current roster, or chats then contacts.

    Operator scope is one page of 200 chats, then 500 contacts only
    if no chat matches; any partial chat match suppresses contact lookup.
    At most five exact/substring candidates are returned, without paging.
    """
    name = name.strip()
    if not name:
        return error("name is required")
    if not target.chat_id:
        return error("no active conversation context")
    if not operator_run(target):
        return resolve_in_current_chat(waha, session, target.chat_id, name)
    matches: list[dict[str, Any]] = []
    try:
        chats = waha.list_chats(session)
        matches = search_matches(chats, name)
        if not matches:
            contacts = waha.list_contacts(session)
            matches = search_matches(contacts, name)
    except Exception as exc:
        return error(f"could not search chats: {exc}")
    if not matches:
        return error(
            f"no match for {name!r} in the checked chats/contacts; not exhaustive"
        )
    target.allowed_chats.update(match["id"] for match in matches)
    return ok(name=name, matches=matches)


def read_chat_messages(
    waha: WahaClient,
    session: str,
    chat_id: str,
    query: str,
    name: str,
    limit: int | None,
    settings: Settings | None = None,
) -> str:
    """The `list` mode: recent messages from a chat, newest first.

    Every nonempty fetched window attempts a raw JSON spill alongside
    the slim inline preview (see :func:`fit_messages`). No paging.
    """
    messages = waha.fetch_chat_messages(session, chat_id, limit=limit or _LIST_WINDOW)
    return ok(
        chat=chat_id,
        **fit_messages(messages, settings=settings, session=session, chat_id=chat_id),
    )


def read_search_messages(
    waha: WahaClient,
    session: str,
    chat_id: str,
    query: str,
    name: str,
    limit: int | None,
    settings: Settings | None = None,
) -> str:
    """Substring-filter the latest *limit* messages, not the full history.

    All nonempty matched windows attempt a raw spill like ``list``.
    """
    if not query.strip():
        return error("query is required")
    messages = waha.search_messages(session, query, chat_id, limit=limit or _LIST_WINDOW)
    return ok(
        chat=chat_id,
        query=query,
        **fit_messages(messages, settings=settings, session=session, chat_id=chat_id),
    )


def read_chat_metadata(
    waha: WahaClient,
    session: str,
    chat_id: str,
    query: str,
    name: str,
    limit: int,
    settings: Settings | None = None,
) -> str:
    """Available overview fields and roster summary; names may be missing."""
    overview = waha.get_chat_overview(session, chat_id)
    if not overview:
        return error(f"no metadata found for {chat_id}")
    names = sender_names(waha, session, chat_id)
    return ok(**summarize_chat(chat_id, overview, names))


_READ_CHAT_HANDLERS: dict[str, Any] = {
    "list": read_chat_messages,
    "search": read_search_messages,
    "metadata": read_chat_metadata,
}


def summarize_chat(
    chat_id: str,
    overview: dict[str, Any],
    names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a compact, model-friendly summary of a chat overview dict.

    Extracts the stable scalar fields and the participant roster,
    normally skipping nested blobs. If only an id or no stable fields
    remain, ``_raw`` is a possibly truncated 1000-character string
    fallback, not structured or complete metadata. Recent display names
    enrich roster entries but do not verify identity.
    """
    scalar = chat_scalars(overview)
    add_participant_summary(scalar, overview, names or {})
    if not scalar or all(key == "id" for key in scalar):
        scalar["chat_id"] = chat_id
        scalar["_raw"] = str(overview)[:1000]
    return scalar


def sender_names(
    waha: WahaClient, session: str, chat_id: str, limit: int = 100
) -> dict[str, str]:
    """JID → recent display label, from at most *limit* messages (default 100).

    The group roster carries only JIDs and admin flags; display names
    ride the messages themselves (``_data.notifyName`` per sender).
    Fails soft — an unreadable chat yields no names and the summary
    falls back to bare JIDs.
    """
    try:
        messages = waha.fetch_chat_messages(session, chat_id, limit=limit)
    except Exception as exc:
        logger.debug(
            "sender names unavailable for {chat_id}: {exc}", chat_id=chat_id, exc=exc
        )
        return {}
    return names_from_messages(messages)


def names_from_messages(messages: list[dict[str, Any]]) -> dict[str, str]:
    """JID → display name from chat messages' ``notifyName`` fields.

    Shared by the roster enrichment (``participant_names``) and the
    chat-summary path (``sender_names``): the first name a sender's
    message carries wins, JID objects are normalized via
    :func:`jid_string` (LID groups report participants as objects), and
    entries without a name are dropped.
    """
    names: dict[str, str] = {}
    for message in messages:
        jid = jid_string(message.get("participant"))
        name = str(message.get("_data", {}).get("notifyName") or "").strip()
        if jid and name and jid not in names:
            names[jid] = name
            name_book.learn(jid, name)
    return names


def chat_scalars(overview: dict[str, Any]) -> dict[str, Any]:
    """Available scalar chat fields, preserving known false/zero values."""
    return {
        key: overview[key]
        for key in (
            "id",
            "name",
            "isReadOnly",
            "isGroup",
            "muted",
            "archived",
            "pinned",
            "unreadCount",
        )
        if overview.get(key) is not None and overview.get(key) != ""
    }


def add_participant_summary(
    scalar: dict[str, Any],
    overview: dict[str, Any],
    names: dict[str, str],
) -> None:
    """Add a nonempty roster count and up to 20 extractable id/name pairs.

    Names come from *names* (recent senders); roster entries without a
    known name keep the bare JID. An absent/empty roster omits both
    fields; omission does not establish that the chat has no members.
    """
    participants = roster_entries(overview)
    if not participants:
        return
    scalar["participants"] = len(participants)
    jids = [jid for jid in map(participant_jid, participants) if jid]
    pairs = [roster_pair(jid, names) for jid in jids]
    if 0 < len(pairs) <= 20:
        scalar["participant_list"] = pairs


def roster_pair(jid: str, names: dict[str, str]) -> dict[str, str]:
    """One ``participant_list`` entry: the JID, plus its name when known."""
    name = display_name(jid, names)
    return {"id": jid, "name": name} if name else {"id": jid}


def participant_jid(participant: Any) -> str:
    """The JID of a string or roster entry wrapping a string/JID object in id."""
    if isinstance(participant, dict):
        entry: dict[str, Any] = participant
        return jid_string(entry.get("id"))
    return jid_string(participant)


#: One ``@<token>`` in message text: a bare user part or a full JID.
#: Display names are deliberately not matched — a roster name like
#: "Ana" could collide with unrelated words, while a bare LID/phone
#: user part (what the chat itself shows, as in "Para @111222333444555")
#: identifies exactly one person. The JID alternative must come first:
#: regex alternation is ordered, and the digits-only branch would
#: otherwise truncate a full-JID token to its user part.
_MENTION_RE = re.compile(r"@([\dA-Za-z.-]+@[\da-z.-]+|\d{6,})")


def mention_tokens(text: str) -> list[str]:
    """The ``@``-tokens of *text* that can name a chat member.

    Each is a bare user part (``@111222333444555``) or a full JID
    (``@111222333444555@lid``); ``@`` followed by anything else is
    ordinary prose and never a mention.
    """
    return _MENTION_RE.findall(text)


def resolve_mentions(text: str, roster: list[str]) -> list[str]:
    """Roster JIDs named by *text*'s ``@``-tokens, in token order.

    A token matches a roster JID when their user parts are equal: the
    chat writes mentions as bare LID/phone user parts, so
    ``@111222333444555`` resolves against ``111222333444555@lid``.
    Unresolved tokens are left out — inventing a mention for a
    non-member would tag nobody (or worse, the wrong person in another
    chat's namespace).
    """
    by_user = {jid.split("@", 1)[0]: jid for jid in roster}
    resolved: list[str] = []
    for token in dict.fromkeys(mention_tokens(text)):
        jid = by_user.get(token.split("@", 1)[0])
        if jid is not None and jid not in resolved:
            resolved.append(jid)
    return resolved


def dangling_mentions(text: str, roster: list[str]) -> list[str]:
    """*text*'s ``@``-tokens naming nobody on *roster*, in token order.

    The counterpart of :func:`resolve_mentions`: the tokens it drops.
    Only numeric or full-JID-shaped tokens are recognized, including a
    stylized ``@L@s``. Bare ``@Lorenzo`` is not a token. This checks
    auto-resolution against the roster, not explicit mention JIDs or
    notification delivery.
    """
    by_user = {jid.split("@", 1)[0] for jid in roster}
    return [
        token
        for token in dict.fromkeys(mention_tokens(text))
        if token.split("@", 1)[0] not in by_user
    ]


def chat_roster(waha: WahaClient, session: str, chat_id: str) -> list[str]:
    """Best-effort participant JIDs; may be empty for DMs or unreadable chats.

    Two sources, both fail-soft — the roster only enriches mentions, so
    an unavailable one must not break the send itself. The overview's
    ``groupMetadata.participants`` carries phone JIDs, but LID groups
    speak ``@lid`` in message ``participant`` fields (verified against
    WAHA: a 34-member LID group's overview roster shows zero ``@lid``
    entries, while its messages carry only ``@lid`` participants), so
    recent senders are merged in to bridge the two namespaces.
    """
    roster: list[str] = []
    try:
        overview = waha.get_chat_overview(session, chat_id)
    except Exception:
        overview = None
    if overview:
        roster.extend(
            jid for jid in map(participant_jid, roster_entries(overview)) if jid
        )
    try:
        messages = waha.fetch_chat_messages(session, chat_id, limit=50)
    except Exception:
        messages = []
    for message in messages:
        jid = jid_string(message.get("participant"))
        if jid:
            roster.append(jid)
    return list(dict.fromkeys(roster))


#: A full-JID mention token (``@123@lid``): WhatsApp renders only the
#: ``@<user>`` part as a tag, so the ``@server`` tail would stay visible.
_FULL_JID_TOKEN_RE = re.compile(r"@(\d{6,})@(?:lid|c\.us|s\.whatsapp\.net)\b")

#: A name-shaped mention (``@Ana``, ``@Ángel Blanco``): letters only, not
#: glued to a preceding word, slash or @ (``L@s``, a URL's ``/@user``
#: are not mentions), one or two words.
_NAME_TOKEN_RE = re.compile(r"(?<![\w@/])@([^\W\d_]{2,}(?:[ \u00a0][^\W\d_]{2,})?)")


def fold_name(name: str) -> str:
    """Case- and accent-insensitive comparison key for a display name."""
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def roster_name_index(roster: list[str], own: set[str]) -> dict[str, set[str]]:
    """Folded full name / first name → roster JIDs, other members only.

    The shared account's own JIDs are excluded: tagging "@kAI" or the
    operator would tag the account itself, which notifies nobody.
    """
    index: dict[str, set[str]] = {}
    for jid in roster:
        if jid in own:
            continue
        name = name_book.name(jid)
        if not name:
            continue
        folded = fold_name(name)
        index.setdefault(folded, set()).add(jid)
        first = folded.split()[0] if folded.split() else ""
        if first and first != folded:
            index.setdefault(first, set()).add(jid)
    return index


def prepare_mentions(
    waha: WahaClient, session: str, text: str, roster: list[str]
) -> tuple[str, list[str]]:
    """Rewrite *text*'s mention tokens into the shape WhatsApp tags.

    Three repairs, all conservative:

    1. ``@123@lid`` → ``@123``: the JID tail would stay visible text.
    2. ``@Ana`` → ``@<digits>`` when exactly one other roster member is
       known by that full or first name; ambiguous or unknown names are
       left untouched (they stay plain text, as before).
    3. A numeric token missing from the roster is checked against
       WAHA's LID↔phone map, so a member who has not spoken recently
       (LID groups list members by phone number, messages by LID) is
       still taggable.

    Returns the rewritten text and the roster extended with any JID
    the LID map confirmed.
    """
    text = _FULL_JID_TOKEN_RE.sub(lambda m: f"@{m.group(1)}", text)
    own = {jid for jid in (_status_state.operator_jid, _status_state.operator_lid) if jid}
    index = roster_name_index(roster, own)
    if index:

        def by_name(match: re.Match[str]) -> str:
            words = match.group(1)
            for candidate in (words, words.split()[0]):
                jids = index.get(fold_name(candidate))
                if jids and len(jids) == 1:
                    (jid,) = jids
                    rest = words[len(candidate) :]
                    return f"@{jid.split('@', 1)[0]}{rest}"
            return match.group(0)

        text = _NAME_TOKEN_RE.sub(by_name, text)
    extended = list(roster)
    phones = {jid.split("@", 1)[0] for jid in roster if jid.endswith("@c.us")}
    for token in dangling_mentions(text, roster):
        user = token.split("@", 1)[0]
        if not user.isdigit():
            continue
        phone = waha.lid_phone(session, f"{user}@lid")
        if phone and phone.split("@", 1)[0] in phones:
            extended.append(f"{user}@lid")
    return text, extended


def ordered_merge(explicit: list[str], resolved: list[str]) -> list[str]:
    """Explicit mention JIDs first, then newly resolved ones; no dupes.

    The model's own ``mentions`` list wins on ordering so the envelope
    reports what it asked for first; resolver additions follow.
    """
    return list(dict.fromkeys([*explicit, *resolved]))


def apply_chat_prefix(text: str, prefix: str) -> str:
    """Prepend a chat's text prefix, once, to non-empty chat text.

    Idempotent: text already starting with the prefix — exactly, or
    with the prefix's trailing whitespace dropped, as a model copying
    the prefix from its own prefixed history may write it — is left
    alone, so the prefix is never doubled.
    """
    if not prefix.strip() or not text.strip():
        return text
    lead = text.lstrip()
    if lead.startswith(prefix) or lead.startswith(prefix.rstrip()):
        return text
    return f"{prefix}{text}"


def deliver_chat_text(
    waha: WahaClient,
    session: str,
    chat_id: str,
    text: str,
    reply_to: str | None = None,
    mentions: list[str] | None = None,
    typing: tuple[float, float] | None = None,
    prefix: str = "",
) -> tuple[str, list[str], list[str]]:
    """Send text with the full delivery treatment.

    Every text delivery — the ``send_message`` tool and the handler's
    final-reply send alike — goes through here, so mention resolution
    is a property of sending, not of which path fired: ``@``-tokens in
    *text* that name roster members become mention JIDs in both.
    Unresolved tokens remain literal text; explicit *mentions* JIDs
    merge in ahead of resolved ones. The send response does not confirm
    mention notification or rendering.

    *prefix* is the chat's ``chat-message-prefix`` rule: applied to the
    outgoing text before mention resolution, so the sent text and the
    stored turn (``visible_message``) carry the same prefix.

    *typing* — a ``(min_s, max_s)`` window from settings — adds the
    human-presence prelude: show "typing…", wait a length-scaled
    random moment (``None`` or ``min <= 0`` skips it). It lives here
    because both delivery paths must type alike; the caller passes the
    settings values, keeping this module settings-free.

    Returns ``(sent_id, merged_mentions, dangling)`` — the id of the
    sent message ("" when the response carried none), the merged JID
    list, and the ``@``-tokens that named no roster member, so a
    caller with a use for any of them (the tool's envelope, the echo
    guard's id) gets them without a second API call.
    """
    text = apply_chat_prefix(text, prefix)
    indicator = (
        typing_pause(waha, session, chat_id, text, typing[0], typing[1])
        if typing is not None
        else False
    )
    try:
        roster = chat_roster(waha, session, chat_id)
        text, roster = prepare_mentions(waha, session, text, roster)
        explicit, unresolved = resolve_explicit_mentions(
            waha, session, chat_id, mentions or [], roster
        )
        merged = ordered_merge(explicit, resolve_mentions(text, roster))
        dangling = dangling_mentions(text, roster)
        sent_id = waha.send_text(
            session, chat_id, text, reply_to=reply_to, mentions=merged or None
        )
    except Exception:
        # The send that would have cleared the indicator never landed;
        # best-effort clear it here, never masking the original error.
        if indicator:
            clear_typing(waha, session, chat_id)
        raise
    return sent_id, merged, [*dangling, *unresolved]


def resolve_explicit_mentions(
    waha: WahaClient,
    session: str,
    chat_id: str,
    mentions: list[str],
    roster: list[str],
) -> tuple[list[str], list[str]]:
    """Resolve explicit ``mentions`` names to JIDs; return the unresolved.

    A JID or numeric id passes through. A name resolves only when exactly
    one named member matches — two members sharing a name tag neither, and
    the name rides back as a warning so the model can quote the ``#m<n>``
    record instead. Roster reads fail soft: entries stay unresolved.
    """
    if not mentions:
        return [], []
    try:
        names = sender_names(waha, session, chat_id)
    except Exception:
        return [], [mention for mention in mentions if "@" not in mention]
    members = [{"id": jid, "name": display_name(jid, names)} for jid in roster]
    resolved: list[str] = []
    unresolved: list[str] = []
    for mention in mentions:
        jid, missing = resolve_one_mention(mention, members)
        if jid:
            resolved.append(jid)
        elif missing:
            unresolved.append(missing)
    return resolved, unresolved


def resolve_one_mention(mention: str, members: list[dict[str, str]]) -> tuple[str, str]:
    """One explicit mention as ``(jid, unresolved_name)``; one side is empty."""
    value = mention.strip()
    if not value:
        return "", ""
    if "@" in value or value.isdigit():
        return value, ""
    needle = value.casefold()
    hits = [m for m in members if m["name"] and m["name"].casefold() == needle]
    return (hits[0]["id"], "") if len(hits) == 1 else ("", value)


def resolve_in_current_chat(
    waha: WahaClient, session: str, chat_id: str, name: str
) -> str:
    """Match a name — or digits/JID — against the current chat's members.

    Members come from the overview roster plus recent senders (LID
    groups list the roster by phone JID but speak by LID), named by
    the overview, recent messages and the webhook-learned name book.
    Digits or a JID match the member's id directly, so the model can
    ask who a bare ``@<digits>`` is. Each match carries the ``tag`` to write in a
    reply. A miss can mean an unnamed member, not absence.
    """
    try:
        roster = chat_roster(waha, session, chat_id)
        names = sender_names(waha, session, chat_id)
    except Exception as exc:
        return error(f"could not read the chat's participants: {exc}")
    members = [{"id": jid, "name": display_name(jid, names)} for jid in roster]
    query = name.strip().lstrip("@")
    digits = query.split("@", 1)[0]
    hits = (
        members_by_id(members, digits)
        if digits.isdigit()
        else [m for m in search_matches(members, query) if m["name"]]
    )
    if not hits:
        return error(
            f"no match for {name!r} among this chat's named members"
            + " (resolving other chats is reserved to the operator)"
        )
    return ok(name=name, matches=[tagged(m) for m in hits[:_RESOLVE_CHAT_CANDIDATES]])


def members_by_id(members: list[dict[str, str]], digits: str) -> list[dict[str, str]]:
    """Members whose JID user part is *digits*, else a name-book hit."""
    hits = [m for m in members if m["id"].split("@", 1)[0] == digits]
    if hits:
        return hits
    jid = f"{digits}@lid"
    learned = name_book.name(jid)
    return [{"id": jid, "name": learned}] if learned else []


def tagged(member: dict[str, str]) -> dict[str, str]:
    """A resolve match with the ``@<digits>`` tag to write in a reply."""
    return {**member, "tag": "@" + member["id"].split("@", 1)[0]}


#: Cap on read_chat's `recent` output: one line per conversation, newest first.
_RECENT_CHATS_CAP = 30


#: How many name candidates the resolve mode returns, to keep the envelope small.
_RESOLVE_CHAT_CANDIDATES = 5


def search_matches(entries: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """Chat/contact entries whose name matches *name*, best first.

    Exact (case-insensitive) name matches come first, then substring
    matches; each keeps only its ``id`` and ``name``. The list is capped
    at ``_RESOLVE_CHAT_CANDIDATES``.
    """
    needle = name.casefold()
    pairs = [
        {"id": jid_string(e.get("id", "")), "name": str(e.get("name", ""))}
        for e in entries
        if e.get("id")
    ]
    exact = [p for p in pairs if p["name"].casefold() == needle]
    partial = [p for p in pairs if needle in p["name"].casefold() and p not in exact]
    return (exact + partial)[:_RESOLVE_CHAT_CANDIDATES]


def forward_message(waha: WahaClient) -> BaseTool:
    """Build native forwarding from a source message id to a destination JID.

    Ordinary runs fence both destination and recognizable source ids;
    operator runs can cross chats. Success echoes the source message_id,
    not the new forwarded id. Rendering/attribution is not verified here.
    """

    def forward_message_fn(
        message_id: str, chat: str | None = None, reason: str = ""
    ) -> str:
        target = current_target()
        if target.sent:
            return error(
                f"message already sent this run (to {target.sent}); do not send again"
            )
        if not message_id.strip():
            return error("message_id is required")
        chat_id, fence_error = fenced_chat(chat, target)
        if fence_error:
            return error(fence_error)
        session = target.session
        if not session or not chat_id:
            return error("no active conversation context")
        delivery_error = delivery_target_error(chat_id, None, target)
        if delivery_error:
            return error(delivery_error)
        _, id_error = fenced_message_id(message_id, target)
        if id_error:
            return error(id_error)
        delivery_error = delivery_message_error(message_id, target)
        if delivery_error:
            return error(delivery_error)
        log_action_reason("forward_message", reason, chat=chat_id)
        try:
            sent_id = waha.forward_message(session, chat_id, message_id)
        except Exception as exc:
            return send_failed_envelope("forward_message", chat_id, exc)
        delivered_to_self(chat_id, sent_id)
        target.sent = chat_id
        return ok(message_id=message_id, chat=chat_id)

    return FunctionTool.from_defaults(
        fn=forward_message_fn,
        fn_schema=ForwardMessageSchema,
        name="forward_message",
        description=(
            "Native WAHA forwarding without retyping: message_id is the "
            "existing source id, chat is the destination JID. Ordinary runs "
            "must keep both in the current chat; operator commands may cross "
            "chats. Success echoes the source id, not the new forwarded id. "
            "Media/rendering depends on WAHA; original-author attribution "
            "is not guaranteed. Shares the send allowance with text/media."
        ),
    )
