"""Message handler connecting the webhook to the LlamaIndex agent."""

import asyncio
import io
import time
from collections.abc import Callable
from typing import Any, cast

import openai
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Context
from loguru import logger
from PIL import Image

from wahabot.ai.albums import (
    AlbumBuffer,
    add_album_image,
    is_album_container,
    set_completion_handler,
    start_album,
)
from wahabot.ai.context import handle_message, render_system_prompt
from wahabot.ai.messages import (
    bot_jids,
    extract_text,
    image_media,
    is_group_addressed,
    is_replyable,
    jid_string,
    message_kind,
    self_command_instruction,
    video_media,
)
from wahabot.ai.observability import chat_trace_attributes, enable_langfuse
from wahabot.ai.tools import build_default_tools
from wahabot.ai.tools.url_videos import fetch_url_video, video_urls
from wahabot.ai.tools.whatsapp import EscalationChannel
from wahabot.ai.video import caption_video, extract_frames, join_anchor, video_marker
from wahabot.ai.vision import caption_images
from wahabot.ai.workflow import FunctionCallingAgentWorkflow, build_agent
from wahabot.core.access import SessionConfigReloader, load_session_config
from wahabot.core.cache import TtlCache
from wahabot.core.echoes import is_self_echo, remember_self_echo
from wahabot.core.filters import chat_allowed, jid_alias_lookup
from wahabot.core.models import WahaEvent
from wahabot.core.persistence import forget_memory
from wahabot.core.transcribe import fetch_transcript, transcribe_voice_note
from wahabot.core.waha import MediaTooLargeError, WahaClient
from wahabot.settings import Settings
from wahabot.status import session_healthy
from wahabot.webhook import on_forget, on_message

#: Messages older than this many seconds are stale backlog, not live turns.
_MAX_MESSAGE_AGE_S = 300
#: Dedup window for redelivered message ids. Must meet or exceed
#: ``_MAX_MESSAGE_AGE_S`` so the two guards overlap seamlessly: a
#: redelivery inside this window is dropped by the seen cache, past it
#: by the staleness check — between a shorter TTL and the age bound
#: (120 < age < 300) a redelivered message would slip through both
#: and be answered twice.
_SEE_TTL_S = _MAX_MESSAGE_AGE_S
#: Seen-id cache hard cap; past it the oldest entries are evicted.
_MAX_SEEN_IDS = 10_000
_seen_ids: TtlCache[str, bool] = TtlCache(_SEE_TTL_S, _MAX_SEEN_IDS)

#: Per-chat context table, run locks and persistence moved to
#: ``wahabot.core.runs`` (imported by the command path too); the names
#: are re-exported here so every existing import keeps working.
from wahabot.core.runs import (  # noqa: E402
    chat_lock,
    context_for,
    contexts,
    persist_memory,
)


def seen_recently(message_id: str) -> bool:
    """Return True if this message id was already handled in the last TTL.

    The id is marked seen on first sight, so a duplicate redelivery
    racing the in-flight run is also deduplicated. A run that later
    fails must call :func:`forget_seen` to allow a retry.
    """
    if not message_id or message_id in _seen_ids:
        return bool(message_id)
    _seen_ids.put(message_id, True)
    return False


def forget_seen(message_id: str) -> None:
    """Drop a message id's seen marker so a redelivery is reprocessed."""
    _seen_ids.drop(message_id)


async def append_to_memory(
    session: str,
    chat_id: str,
    agent: FunctionCallingAgentWorkflow,
    settings: Settings,
    message: ChatMessage,
) -> Context:
    """Append one message to a chat's memory buffer, no agent run.

    Shared fold-in path for out-of-band turns (operator-typed ``fromMe``
    messages, reaction notes): the model's self-history should reflect
    them, but a fold must never wake the LLM. Returns the chat's context
    so the caller can persist the fold at its save point.
    """
    ctx = await context_for(session, chat_id, agent, settings)
    memory = await ctx.store.get("memory", default=None)
    if memory is None:
        from_defaults = cast(
            Callable[..., ChatMemoryBuffer], ChatMemoryBuffer.from_defaults
        )
        memory = from_defaults(token_limit=agent.memory_token_limit, llm=agent.llm)
    await memory.aput(message)
    await ctx.store.set("memory", memory)
    return ctx


def log_final_text(chat_id: str, reply: str) -> None:
    """Log a dropped post-delivery final text, when there is one.

    After a delivery tool fired, the run's final text is dropped (the
    reply already went out); it is usually the model narrating what it
    just did, but it can carry real content — log it so nothing the
    model "said" vanishes without a trace.
    """
    if reply and reply.strip():
        logger.info(
            "Dropped post-delivery final text in {chat_id}: {reply}",
            chat_id=chat_id,
            reply=reply[:500],
        )


async def send_self_reply(waha: WahaClient, command: WahaEvent, reply: str) -> None:
    """Deliver a self-chat command's final text back to the operator.

    The command ran on a fresh context with no chat of its own; its
    final reply (the part not delivered via a tool) lands in the same
    "message yourself" chat the command came from, quote-replying the
    command message. The sent id is remembered so the echo of our own
    reply — which arrives as a ``fromMe`` event matching the mention
    regex when the reply quotes the trigger — cannot re-trigger the
    command path.
    """
    if not reply or not reply.strip():
        return
    chat_id = str(command.payload.get("reply_chat_id", ""))
    if not chat_id:
        return
    sent_id = await asyncio.to_thread(
        waha.send_text,
        command.session,
        chat_id,
        reply,
        str(command.payload.get("reply_to", "")) or None,
    )
    remember_self_echo(sent_id)


#: Prefix marking an assistant turn typed by the human operator in the
#: WhatsApp app, not generated by the model (see the session prompt's
#: "How messages reach you" section). Without it the two are
#: indistinguishable in the self-history; with it the model can tell
#: its own words from the operator's while still standing behind both —
#: both went out under the account's name.
OPERATOR_NOTE_PREFIX = "[operator message] "


async def remember_own_message(
    event: WahaEvent,
    agent: FunctionCallingAgentWorkflow,
    settings: Settings,
    body: str | None,
) -> Context | None:
    """Fold a ``fromMe`` message into the chat's memory as an assistant turn.

    Messages sent from the bot account by its human operator (typing in
    the WhatsApp app) are the bot's own voice as far as the chat is
    concerned — storing them as assistant messages keeps the model's
    self-history coherent (it "said" them). The ``[operator message]``
    prefix tells the model *who typed it* without changing *whose voice
    it carries*: the note is guidance for the model, bracketed so it is
    never repeated in replies (the same convention as every other
    bracketed marker). Memory-only: no agent run, so the bot can never
    wake on its own output and loop on itself.

    Returns the chat's context (or None when there was nothing to fold)
    so the caller can persist at the fold's save point.
    """
    if not body:
        return None
    chat_id = str(event.payload.get("from", ""))
    if not chat_id:
        return None
    if (event.session, chat_id) not in contexts:
        # No conversation to attach the words to: an assistant-only
        # buffer is invalid for the chat API (history must start with a
        # user turn), so sanitize would drop it on the next run — fold
        # nothing, say why, and skip the pointless memory-file write.
        logger.info(
            "Skipping fromMe message {id} in {chat_id}: no conversation yet",
            id=event.payload.get("id"),
            chat_id=chat_id,
        )
        return None
    ctx = await append_to_memory(
        event.session,
        chat_id,
        agent,
        settings,
        ChatMessage(role=MessageRole.ASSISTANT, content=f"{OPERATOR_NOTE_PREFIX}{body}"),
    )
    logger.debug("Remembered own outbound message {id}", id=event.payload.get("id"))
    return ctx


def is_stale(event: WahaEvent, started_at: float) -> bool:
    """True when the message is replayed backlog rather than a live turn.

    WhatsApp redelivers undelivered messages when the WAHA session or the
    phone reconnects, and WAHA forwards them as fresh ``message`` events.
    Two guards: anything sent before this process started is definitionally
    backlog, and anything older than ``_MAX_MESSAGE_AGE_S`` is stale even
    mid-run (phone reconnect flush). Unknown timestamps pass — better one
    late reply than silence.
    """
    ts = event.payload.get("timestamp")
    if not isinstance(ts, (int, float)):
        return False
    return ts < started_at or time.time() - ts > _MAX_MESSAGE_AGE_S


def download_image(
    waha: WahaClient, media: dict[str, Any], message_id: str, max_bytes: int
) -> dict[str, Any] | None:
    """Download an image's bytes; None keeps the turn text-only on failure."""
    url = str(media.get("url", ""))
    if not url:
        return None
    try:
        data = waha.download_media(url, max_bytes=max_bytes)
    except MediaTooLargeError:
        logger.info(
            "Skipping image over {max_bytes} B in message {id}",
            max_bytes=max_bytes,
            id=message_id,
        )
        return None
    except Exception as exc:
        logger.warning(
            "Image download failed for message {id}: {exc}", id=message_id, exc=exc
        )
        return None
    mimetype = str(media.get("mimetype") or "image/jpeg")
    logger.info("Downloaded image ({mime}, {size} B)", mime=mimetype, size=len(data))
    if mimetype == "image/webp":
        try:
            data = first_frame_png(data)
            mimetype = "image/png"
        except Exception as exc:
            logger.warning(
                "WebP first-frame conversion failed for {id}: {exc}",
                id=message_id,
                exc=exc,
            )
            return None
    return {"data": data, "mimetype": mimetype}


def first_frame_png(data: bytes) -> bytes:
    """The first frame of an animated image (webp/gif) as PNG bytes.

    Vision models accept stills, not animations — a 23-frame sticker
    webp would be rejected or misread. Callers decide by mimetype
    whether to convert; static webp converts losslessly too.
    """
    with Image.open(io.BytesIO(data)) as img:
        img.seek(0)
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()


async def video_transcript(settings: Settings, data: bytes, filename: str) -> str:
    """The video's spoken track via WhisperX, or "" when unavailable.

    Posts the whole video file — the service demuxes audio itself
    (verified in the plan's evidence table), so this deliberately
    bypasses ``is_transcribable_mimetype``: that gate exists for the
    voice-note path. Audio-less videos (loops, screen recordings) get
    a 500 from the service every time — expected traffic, not an
    error, hence the debug log — and any other failure drops the
    transcript part of the marker.
    """
    if not settings.transcribe_url:
        return ""
    try:
        transcript = await asyncio.to_thread(fetch_transcript, settings, data, filename)
    except Exception as exc:
        logger.debug("Video transcript unavailable: {exc}", exc=exc)
        return ""
    return transcript.strip()


async def prepare_video(
    event: WahaEvent,
    waha: WahaClient,
    settings: Settings,
    llm: FunctionCallingLLM,
    semaphore: asyncio.Semaphore | None = None,
) -> dict[str, Any] | None:
    """Download + frames + caption + transcript for a video turn.

    Returns ``{"frames": [...], "marker": "(video shows: …) …"}``, or
    None when the download failed (the turn degrades exactly like a
    failed image download: sender text runs, a bare video stays
    silent). Every stage is fail-soft — a dead ffmpeg, a failed
    caption call or a failed transcription each drop their marker
    part and the turn still runs. Runs before the chat's run lock so
    no network or vision call extends the serialized agent-run section.
    """
    media = video_media(event)
    if media is None:
        return None
    url = str(media["url"])
    message_id = str(event.payload.get("id", ""))
    try:
        data = await asyncio.to_thread(waha.download_media, url, settings.max_video_bytes)
    except MediaTooLargeError:
        logger.info(
            "Skipping video over {max} B in message {id}",
            max=settings.max_video_bytes,
            id=message_id,
        )
        return None
    except Exception as exc:
        logger.warning(
            "Video download failed for message {id}: {exc}", id=message_id, exc=exc
        )
        return None
    logger.info(
        "Downloaded video ({size} B) from message {id}", size=len(data), id=message_id
    )
    filename = str(media.get("filename") or "") or "video.mp4"
    return await build_video(settings, llm, semaphore, data, filename)


async def build_video(
    settings: Settings,
    llm: FunctionCallingLLM,
    semaphore: asyncio.Semaphore | None,
    data: bytes,
    filename: str,
) -> dict[str, Any]:
    """Frames + caption + transcript for already-downloaded video bytes.

    Shared by the WAHA video path (:func:`prepare_video`) and the URL
    path (:func:`prepare_url_video`): identical fail-soft behavior —
    a dead ffmpeg, caption call or transcription each drop their marker
    part and still return a (possibly bare) result.
    """
    frames = await asyncio.to_thread(extract_frames, data, settings.video_frames)
    caption = await caption_video(llm, frames, semaphore=semaphore) if frames else ""
    transcript = await video_transcript(settings, data, filename)
    return {
        "frames": [{"data": frame, "mimetype": "image/jpeg"} for frame in frames],
        "marker": video_marker(caption, transcript),
    }


async def prepare_url_video(
    settings: Settings,
    llm: FunctionCallingLLM,
    semaphore: asyncio.Semaphore | None,
    body: str,
) -> dict[str, Any] | None:
    """Resolve + download video URLs in *body*, then frames + transcript.

    Returns the same ``{"frames": [...], "marker": ...}`` shape as
    :func:`prepare_video`, or None when no URL resolved to a video. Only
    the first ``settings.max_url_videos`` links are tried; each failure
    is a log line and the link stays ordinary text. Runs before the
    chat's run lock (like ``prepare_video``) so the yt-dlp download and
    the vision/transcription calls don't extend the serialized
    agent-run section.
    """
    urls = video_urls(body, settings.max_url_videos)
    for url in urls:
        result = await asyncio.to_thread(fetch_url_video, settings, url)
        if result is None:
            continue
        return await build_video(
            settings, llm, semaphore, result["data"], result["filename"]
        )
    return None


def video_frames(
    video: dict[str, Any] | None, url_video: dict[str, Any] | None
) -> list[dict[str, Any]] | None:
    """The frames of whichever video source resolved, or None.

    The URL path only runs when no WAHA video was attached, so at most
    one of the two is ever non-None — this is a pick, not a merge.
    """
    for source in (video, url_video):
        if source is not None:
            return cast(list[dict[str, Any]], source["frames"]) or None
    return None


def register_forget_handler(settings: Settings) -> None:
    """Register the ``forget`` webhook handler that wipes a chat's memory.

    The ``wahabot forget`` CLI posts a signed ``forget`` event; the wipe
    runs here, in the bot process, under the chat's run lock so it
    serializes with that chat's runs: an in-flight run finishes and
    saves first, then the live context and the memory file are dropped
    — no resurrection window.
    """

    @on_forget
    async def handle_forget(event: WahaEvent) -> None:
        chat_id = str(event.payload.get("chat_id", ""))
        if not chat_id:
            logger.warning("Ignoring forget event without a chat_id")
            return
        async with chat_lock(event.session, chat_id):
            live = contexts.pop((event.session, chat_id), None) is not None
            removed = forget_memory(settings.data_dir, event.session, chat_id)
            logger.info(
                "Forgot {chat_id}: context {live}, file {removed}",
                chat_id=chat_id,
                live="present" if live else "absent",
                removed="deleted" if removed else "absent",
            )


def register_agent_handler(
    settings: Settings, waha: WahaClient
) -> tuple[FunctionCallingAgentWorkflow, SessionConfigReloader]:
    """Build the agent and register the reply handler.

    Returns the built agent and the config reloader so other handlers
    (the command channel) share the exact same agent and hot-reloaded
    config instead of building their own.
    """
    config = load_session_config(settings.access_config)
    config_reloader = SessionConfigReloader(settings.access_config)
    enable_langfuse(settings)

    async def run_album(buffer: AlbumBuffer) -> None:
        """Run the agent once over a completed album, all images attached.

        The container event drives the turn (sender tag, gating already
        done at arrival); each buffered image contributes its bytes.
        Runs under the same per-chat lock as single messages so the
        chat's memory and timeline stay consistent.

        Fire-and-forget from the album buffer, so failures must be
        caught here or they would die silently in an unretrieved task:
        the exception is logged with traceback and every buffered
        message's seen marker is dropped so WAHA's redelivery can
        retry the album, mirroring the single-message path.
        """
        try:
            await deliver_album_reply(buffer)
        except Exception:
            for image_event in buffer.images:
                forget_seen(str(image_event.payload.get("id", "")))
            forget_seen(str(buffer.container.payload.get("id", "")))
            logger.exception(
                "Failed to handle album in {chat_id}",
                chat_id=str(buffer.container.payload.get("from", "")),
            )

    async def deliver_album_reply(buffer: AlbumBuffer) -> None:
        """Download an album's images, run the agent once, send its reply."""
        event = buffer.container
        chat_id = str(event.payload.get("from", ""))
        downloaded: list[dict[str, Any]] = []
        for image_event in buffer.images:
            media = image_media(image_event)
            if media is None:
                continue
            image = await asyncio.to_thread(
                download_image,
                waha,
                media,
                str(image_event.payload.get("id", "")),
                settings.max_image_bytes,
            )
            if image is not None:
                downloaded.append(image)
        if not downloaded:
            logger.debug("Album in {chat_id} yielded no usable images", chat_id=chat_id)
            return
        # Caption before the lock: the vision call must not extend the
        # serialized agent-run section.
        captions = await caption_images(
            agent.llm, downloaded, semaphore=agent.llm_semaphore
        )
        for image, caption in zip(downloaded, captions, strict=True):
            image["caption"] = caption
        async with chat_lock(event.session, chat_id):
            ctx = await context_for(event.session, chat_id, agent, settings)
            with chat_trace_attributes(chat_id):
                reply, target = await handle_message(
                    event, agent, ctx=ctx, images=downloaded, settings=settings, waha=waha
                )
            await persist_memory(settings, event.session, chat_id, ctx)
            if target.sent or target.reacted:
                log_final_text(chat_id, reply)
                return
        if reply and reply.strip():
            logger.info(
                "Replying to album in {chat_id}: {reply}",
                chat_id=chat_id,
                reply=reply[:500],
            )
            await asyncio.to_thread(
                waha.send_text,
                event.session,
                chat_id,
                reply,
                str(event.payload.get("id", "")),
            )

    set_completion_handler(run_album)

    def render_prompt() -> str:
        """Re-render ``{{date}}``/``{{time}}`` and pick up config edits.

        The current config's prompt always wins over the startup
        snapshot, so prompt changes apply without a restart too.
        """
        current = config_reloader.current_config()
        return render_system_prompt(
            current.system_prompt, settings.timezone, current.bot_name, current.goal
        )

    # Per-agent escalation state (per-chat cooldowns; the operator JID
    # is read from status.state at call time, so a session recovery
    # needs no re-capture wiring here).
    escalation_channel = EscalationChannel()
    agent = build_agent(
        settings,
        tools=build_default_tools(waha, escalation_channel=escalation_channel),
        system_prompt=config.system_prompt,
        prompt_renderer=render_prompt,
    )
    logger.info(
        "Agent ready: {tools}",
        tools=", ".join(sorted(tool.metadata.get_name() for tool in agent.tools)),
    )
    started_at = time.time()

    @on_message
    async def reply_with_agent(event: WahaEvent) -> None:
        """Run the agent over an incoming message and send its reply via WAHA."""
        message_id = str(event.payload.get("id", ""))
        if not message_id:
            logger.debug(
                "Skipping message without an id from {chat_id}",
                chat_id=event.payload.get("from"),
            )
            return
        if not session_healthy():
            # Before seen_recently: a muted message must not be marked
            # seen, so WAHA's redelivery retries it after recovery.
            logger.info(
                "Muting message {id} while WAHA session is not WORKING", id=message_id
            )
            return
        # Hot-path config: picks up whitelist/prompt/mode edits per event.
        config = config_reloader.current_config()
        if seen_recently(message_id):
            logger.debug("Skipping duplicate event for message {id}", id=message_id)
            return
        if is_stale(event, started_at):
            logger.info(
                "Skipping stale message {id} (ts={ts})",
                id=message_id,
                ts=event.payload.get("timestamp"),
            )
            return
        if not is_replyable(event):
            logger.debug(
                "Ignoring non-replyable message from {sender}",
                sender=event.payload.get("from"),
            )
            return
        body = extract_text(event)
        if (
            body is None
            and message_kind(event) == "audio"
            and settings.transcribe_url
            and event.payload.get("fromMe")
            and jid_string(event.payload.get("from")) in bot_jids(event)
            and not is_self_echo(message_id)
        ):
            # The self-chat is the operator console: a voice note there
            # can only come from the operator's own devices, so it is
            # always worth transcribing early — it may be a spoken
            # command ("kai do x"). Everywhere else transcription stays
            # behind the chat gates. The transcript becomes the body;
            # the mention check below runs on it like typed text.
            transcript = await transcribe_voice_note(event, waha, settings)
            if transcript:
                event.payload["body"] = f"[voice note] {transcript}"
                body = event.payload["body"]
        instruction = (
            None
            if is_self_echo(message_id)
            else self_command_instruction(
                event,
                bot_name=config.bot_name,
                bot_mention_regex=config.bot_mention_regex,
            )
        )
        if instruction:
            # A self-chat mention is the WhatsApp equivalent of `wahabot
            # tell`: shared operator context, chat gates bypassed. The
            # reply lands in the same self-chat so the operator sees it
            # on their phone, and its id is marked as run output so the
            # echo event cannot re-trigger the command path.
            from wahabot.commands import build_command_event, run_command

            command = WahaEvent.model_validate(
                build_command_event(event.session, instruction)
            )
            command.payload["reply_chat_id"] = str(event.payload.get("from", ""))
            command.payload["reply_to"] = message_id
            try:
                reply = await run_command(command, agent, settings, waha)
            except Exception:
                # Same contract as the chat path: drop the seen marker
                # so WAHA's redelivery retries the command. Accepted
                # trade-off: a command that crashed *after* a tool
                # delivery already went out will deliver it again on
                # the retry (the at-most-once send latch is per-run) —
                # the alternative, never retrying, loses the command
                # outright.
                forget_seen(message_id)
                logger.exception("Failed to handle self-chat command {id}", id=message_id)
                return
            try:
                await send_self_reply(waha, command, reply)
            except Exception:
                # The command ran — its deliveries may already be out.
                # A redelivery would re-run it and duplicate them, so
                # the seen marker stays; only the reply is lost.
                logger.exception(
                    "Failed to deliver self-chat reply for command {id}", id=message_id
                )
            return
        if not chat_allowed(
            event,
            config.whitelist,
            config.blacklist,
            jid_aliases=jid_alias_lookup(event),
        ):
            return
        if event.payload.get("fromMe"):
            async with chat_lock(event.session, str(event.payload.get("from", ""))):
                ctx = await remember_own_message(event, agent, settings, body)
                if ctx is not None:
                    await persist_memory(
                        settings, event.session, str(event.payload.get("from", "")), ctx
                    )
            return
        if is_album_container(event):
            start_album(event)
            return
        if message_kind(event) in ("image", "sticker") and add_album_image(event):
            return
        image = image_media(event) if settings.vision else None
        if not is_group_addressed(
            event,
            bot_name=config.bot_name,
            bot_mention_regex=config.bot_mention_regex,
            participation=config.group_participation,
        ):
            logger.debug(
                "Ignoring unaddressed group message {id}", id=event.payload.get("id")
            )
            return
        if message_kind(event) == "audio" and settings.transcribe_url:
            transcript = await transcribe_voice_note(event, waha, settings)
            if transcript:
                event.payload["body"] = f"[voice note] {transcript}"
                body = event.payload["body"]
        video = None
        if message_kind(event) in ("video", "ptv") and settings.video:
            # Frames ride the vision path, so an endpoint that rejects
            # image inputs (settings.vision=false) gets no video prep
            # either — the whole download+ffmpeg+caption pass would be
            # wasted work before a doomed caption call.
            if settings.vision:
                video = await prepare_video(
                    event, waha, settings, agent.llm, agent.llm_semaphore
                )
            if video is not None:
                event.payload["body"] = join_anchor(body or "", video["marker"])
                body = event.payload["body"]
        url_video = None
        if (
            settings.video
            and settings.vision
            and settings.max_url_videos > 0
            and video is None
            and body
        ):
            # A media URL in text ("watch this reel link") gets the same
            # frames + transcript treatment as a forwarded video. Runs
            # before the lock like prepare_video; a miss keeps the link
            # as ordinary text.
            url_video = await prepare_url_video(
                settings, agent.llm, agent.llm_semaphore, body
            )
            if url_video is not None:
                body = join_anchor(body, url_video["marker"])
                event.payload["body"] = body
        if body is None and image is None and video is None and url_video is None:
            logger.debug("Skipping media/album message {id}", id=event.payload.get("id"))
            return
        chat_id = str(event.payload["from"])
        try:
            if image is not None:
                image = await asyncio.to_thread(
                    download_image, waha, image, message_id, settings.max_image_bytes
                )
                if image is not None:
                    # Caption before the lock: the vision call must not
                    # extend the serialized agent-run section.
                    captions = await caption_images(
                        agent.llm, [image], semaphore=agent.llm_semaphore
                    )
                    image["caption"] = captions[0]
            async with chat_lock(event.session, chat_id):
                # Runs in the same chat serialize on its memory and
                # timeline; runs in different chats proceed in parallel
                # (each binds its own run target inside the workflow).
                ctx = await context_for(event.session, chat_id, agent, settings)
                with chat_trace_attributes(chat_id):
                    reply, target = await handle_message(
                        event,
                        agent,
                        ctx=ctx,
                        image=image,
                        images=video_frames(video, url_video),
                        settings=settings,
                        waha=waha,
                    )
                await persist_memory(settings, event.session, chat_id, ctx)
                if target.sent or target.reacted:
                    logger.info(
                        "Agent decision for {chat_id}: delivered via tool",
                        chat_id=chat_id,
                    )
                    log_final_text(chat_id, reply)
                    return
            if not reply or not reply.strip():
                logger.info("Agent decision for {chat_id}: stay silent", chat_id=chat_id)
                return
            logger.info(
                "Replying to {chat_id}: {reply}", chat_id=chat_id, reply=reply[:500]
            )
            await asyncio.to_thread(
                waha.send_text, event.session, chat_id, reply, message_id
            )
        except openai.APIConnectionError as exc:
            # Provider unreachable — a transient outage, not a bug. The
            # seen marker is dropped so WAHA's redelivery retries, but
            # there's nothing to debug: log a one-line warning.
            forget_seen(message_id)
            logger.warning(
                "LLM endpoint unreachable for {id} in {chat_id}; dropped marker: {exc}",
                id=message_id,
                chat_id=chat_id,
                exc=exc,
            )
        except Exception:
            # Allow WAHA's redelivery of this message to be reprocessed.
            forget_seen(message_id)
            logger.exception(
                "Failed to handle message {id} in {chat_id} {detail}",
                id=message_id,
                chat_id=chat_id,
                detail="(seen marker dropped; a WAHA redelivery will retry)",
            )
            return

    return agent, config_reloader
