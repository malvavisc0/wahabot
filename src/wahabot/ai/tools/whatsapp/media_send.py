from collections.abc import Callable

from llama_index.core.tools import BaseTool, FunctionTool

from wahabot.ai.tools.envelope import error, ok
from wahabot.ai.tools.runcontext import (
    RunTarget,
    current_target,
    delivered_to_self,
    delivery_target_error,
    fenced_chat,
    log_action_reason,
    send_failed_envelope,
    send_guard,
)
from wahabot.ai.tools.schemas import (
    SendMediaSchema,
)
from wahabot.ai.tools.whatsapp.media_files import (
    image_file,
    local_file,
    media_argument_error,
    normalize_media_source,
    remote_file,
    squared_sticker_payload,
    sticker_file,
    video_file,
    voice_file,
    voice_payload,
)
from wahabot.ai.tools.whatsapp.targets import probe_media_url
from wahabot.ai.tools.whatsapp.text_send import blame_delivery_error
from wahabot.core.tts import synthesize
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings


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
        if refusal := send_guard(target):
            return refusal

        def preflight() -> tuple[Callable[..., str] | None, str | None]:
            """The handler and the argument error, with sources normalized."""
            nonlocal url, path, text
            if blame := blame_delivery_error(f"{caption}\n{text or ''}"):
                return None, blame
            handler = _SEND_MEDIA_HANDLERS.get(kind)
            if handler is None:
                return None, f"unknown kind: {kind!r}"
            # Blank sources count as absent (a whitespace-only url must not
            # outrank a real path below) — normalize before the XOR check
            # and the dispatch, so both see the same source.
            url = normalize_media_source(url)
            path = normalize_media_source(path)
            text = normalize_media_source(text)
            return handler, media_argument_error(
                url, path, text, kind, caption, filename, language
            )

        handler, preflight_error = preflight()
        if preflight_error:
            return error(preflight_error)
        assert handler is not None
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
    target.sent_id = sent_id
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
    target.sent_id = sent_id
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
    target.sent_id = sent_id
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
    target.sent_id = sent_id
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
    target.sent_id = sent_id
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
    target.sent_id = sent_id
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
