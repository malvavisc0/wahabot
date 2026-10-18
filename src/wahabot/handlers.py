"""Message handler connecting the webhook to the LlamaIndex agent."""

import asyncio
import io
import random
import time
from collections.abc import Callable
from typing import Any, cast

import httpx
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Context
from loguru import logger
from PIL import Image

from wahabot.ai import bursts
from wahabot.ai.albums import (
    AlbumBuffer,
    set_completion_handler,
)
from wahabot.ai.context import (
    handle_message,
    is_single_emoji,
    message_id_note,
    render_system_prompt,
    sender_tag,
)
from wahabot.ai.history import ContextBudgetError
from wahabot.ai.messages import (
    OPERATOR_FOLD_KWARG,
    OPERATOR_NOTE_PREFIX,
    OPERATOR_PAYLOAD_FLAG,
    conversation_jid,
    extract_text,
    image_media,
    is_group_addressed,
    is_operator_event,
    is_replyable,
    jid_string,
    message_kind,
    name_mentioned,
    replies_to_bot,
    self_command_instruction,
    sent_by_bot_api,
    typed_by_operator,
    video_media,
)
from wahabot.ai.observability import chat_trace_attributes, enable_langfuse
from wahabot.ai.scrub import strip_spoofed_markers
from wahabot.ai.tools import build_default_tools
from wahabot.ai.tools.url_videos import fetch_url_video, video_urls
from wahabot.ai.tools.whatsapp import EscalationChannel, RunTarget, deliver_chat_text
from wahabot.ai.triage import TriageGate
from wahabot.ai.video import caption_video, extract_frames, join_anchor, video_marker
from wahabot.ai.vision import caption_images
from wahabot.ai.workflow import FunctionCallingAgentWorkflow, build_agent
from wahabot.core import identity
from wahabot.core.access import SessionConfig, SessionConfigReloader, load_session_config
from wahabot.core.audit import save_action
from wahabot.core.cache import TtlCache
from wahabot.core.echoes import is_self_echo, remember_self_echo
from wahabot.core.filters import chat_allowed, jid_alias_lookup
from wahabot.core.identity import AUTHOR_BOT, AUTHOR_OPERATOR
from wahabot.core.inbound import (
    buffer_album_message,
    can_prepare_video,
    self_chat_needs_transcript,
    url_video_enabled,
)
from wahabot.core.learned import forget_learned
from wahabot.core.models import WahaEvent
from wahabot.core.notebook import Notebook, notebook_operation
from wahabot.core.persistence import forget_memory, memory_file
from wahabot.core.presence import mark_seen
from wahabot.core.thread_index import forget_thread_index
from wahabot.core.transcribe import fetch_transcript, transcribe_voice_note
from wahabot.core.waha import MediaTooLargeError, WahaClient
from wahabot.settings import Settings
from wahabot.status import (
    classify_llm_failure,
    llm_call_timed_out,
    mark_llm_recovered,
    probe_session_recovery,
    session_healthy,
)
from wahabot.status import state as status_state
from wahabot.webhook import on_forget, on_message, on_shutdown

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


def journal_send_failure(
    settings: Settings,
    session: str,
    chat_id: str,
    action: str,
    message_id: str,
    exc: Exception,
) -> None:
    """Journal a failed final-reply send.

    The single, burst and album reply paths share this: a send that
    raised left nothing in the chat, so the audit row must say the
    send *failed* — never a ``reply``/``reaction`` record the chat
    never saw (the same standard the durable escalation record set:
    the record follows the confirmed send).
    """
    save_action(
        settings.data_dir,
        session,
        "send_failed",
        chat_id=chat_id,
        action=action,
        message_id=message_id,
        error=str(exc),
    )
    logger.warning(
        "Failed to deliver reply to {chat_id} via {action}: {exc}",
        chat_id=chat_id,
        action=action,
        exc=exc,
    )


def album_message_id(event: WahaEvent) -> str:
    """The album container's message id — the anchor its reply quotes."""
    return str(event.payload.get("id", ""))


async def finish_agent_reply(
    waha: WahaClient,
    session: str,
    chat_id: str,
    message_id: str,
    reply: str,
    delivered: bool,
    settings: Settings,
    emoji_reaction: bool = False,
) -> None:
    """Deliver a finished run's reply text, or record why not.

    Shared post-run decision of the single-message, burst and album
    paths: tool-delivered runs log their dropped final text; silence
    stays silent; anything else is quote-replied to *message_id* with
    the typing presence. A lone emoji is not an answer — *emoji_reaction*
    converts it into a reaction on *message_id* with ~50 % probability,
    otherwise it is dropped as silence. The burst path opts in, so a
    burst's lone-emoji answer reacts like a single message's.

    Every branch journals its decision
    — the audit timeline shows *why* a chat saw nothing: delivered via
    tool, silence, or a dropped emoji. Records for actions that send
    (``reply``, ``reaction``) are written only after the send
    succeeded — the same standard the durable escalation record set —
    so a failed send never leaves an audit row claiming the chat was
    answered; its failure is journaled as ``send_failed`` instead.
    """
    if delivered:
        logger.info(
            "Agent decision for {chat_id}: delivered via tool",
            chat_id=chat_id,
        )
        save_action(settings.data_dir, session, "delivered_via_tool", chat_id=chat_id)
        log_final_text(chat_id, reply)
        return
    if not reply or not reply.strip():
        save_action(settings.data_dir, session, "silence", chat_id=chat_id)
        logger.info("Agent decision for {chat_id}: stay silent", chat_id=chat_id)
        return
    if is_single_emoji(reply):
        emoji = reply.strip()
        if emoji_reaction and message_id and random.random() < 0.5:
            logger.info(
                "Converting lone emoji to reaction on {mid}: {emoji}",
                mid=message_id,
                emoji=emoji,
            )
            try:
                await asyncio.to_thread(waha.send_reaction, session, message_id, emoji)
            except Exception as exc:
                journal_send_failure(
                    settings, session, chat_id, "send_reaction", message_id, exc
                )
                return
            save_action(
                settings.data_dir,
                session,
                "reaction",
                chat_id=chat_id,
                reaction=emoji,
                message_id=message_id,
            )
            return
        save_action(
            settings.data_dir, session, "silence", chat_id=chat_id, dropped_emoji=emoji
        )
        logger.info(
            "Dropping lone emoji as silence for {chat_id}: {reply!r}",
            chat_id=chat_id,
            reply=reply,
        )
        return
    logger.info("Replying to {chat_id}: {reply}", chat_id=chat_id, reply=reply[:500])
    try:
        await asyncio.to_thread(
            deliver_chat_text,
            waha,
            session,
            chat_id,
            reply,
            message_id,
            typing=(
                settings.typing_presence_min_s,
                settings.typing_presence_max_s,
            ),
        )
    except Exception as exc:
        journal_send_failure(settings, session, chat_id, "send_text", message_id, exc)
        return
    save_action(
        settings.data_dir,
        session,
        "reply",
        chat_id=chat_id,
        message_id=message_id,
        reply=reply,
    )


#: Waits between self-chat reply attempts on a connection that never
#: opened; the last entry is never slept (it marks the final attempt).
_SELF_REPLY_RETRY_DELAYS_S = (2.0, 5.0, 0.0)


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
    reply_to = str(command.payload.get("reply_to", "")) or None
    sent_id = ""
    for attempt, delay in enumerate(_SELF_REPLY_RETRY_DELAYS_S):
        try:
            sent_id, _, _ = await asyncio.to_thread(
                deliver_chat_text,
                waha,
                command.session,
                chat_id,
                reply,
                reply_to,
                typing=None,  # self-chat: no one is watching for "typing…"
            )
            break
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # WAHA restarting refuses connections for a few seconds; the
            # answer is worth a short wait rather than being lost. Only
            # connection-establishment failures retry: a read/write
            # timeout may have delivered the send already, and a retry
            # would duplicate it.
            if attempt == len(_SELF_REPLY_RETRY_DELAYS_S) - 1:
                raise
            logger.warning(
                "Self-chat reply not delivered ({exc}); retrying in {delay}s",
                exc=exc,
                delay=delay,
            )
            await asyncio.sleep(delay)
    remember_self_echo(sent_id)


async def remember_own_message(
    event: WahaEvent,
    agent: FunctionCallingAgentWorkflow,
    settings: Settings,
    body: str | None,
) -> Context | None:
    """Fold an operator-typed message into the chat's memory, no run.

    The human operator writes from the bot's own account. Those words
    are a teammate's, not the bot's: they are stored as a *user* turn
    tagged ``[Name <jid>] [operator message] …`` — the same shape a
    waking operator turn renders — so the model always knows who said
    what: it must never take the operator's lines for its own, nor
    answer them as if a stranger wrote them. A mention of the bot
    wakes a real run instead (``reply_with_agent``).

    *event* is already normalized (``from`` is the chat). Chats with no
    conversation yet — neither in memory nor on disk — fold nothing.
    Returns the chat's context so the caller can persist the fold.
    """
    if not body:
        logger.debug(
            "Skipping operator message {id}: no text to fold",
            id=event.payload.get("id"),
        )
        return None
    chat_id = str(event.payload.get("from", ""))
    if not chat_id:
        return None
    if (event.session, chat_id) not in contexts and not memory_file(
        settings.data_dir, event.session, chat_id
    ).exists():
        logger.info(
            "Skipping operator message {id} in {chat_id}: no conversation yet",
            id=event.payload.get("id"),
            chat_id=chat_id,
        )
        return None
    # Tag first, exactly like the waking turn renders it
    # (``[Name <jid>] [operator message] text``), so the model sees one
    # shape for operator lines whether they wake it or not.
    tag = sender_tag(event)
    content = f"{tag} {OPERATOR_NOTE_PREFIX}{strip_spoofed_markers(body)}".strip()
    ctx = await append_to_memory(
        event.session,
        chat_id,
        agent,
        settings,
        ChatMessage(
            role=MessageRole.USER,
            content=content,
            additional_kwargs={OPERATOR_FOLD_KWARG: True},
        ),
    )
    logger.info(
        "Remembered operator message {id} in {chat}",
        id=event.payload.get("id"),
        chat=chat_id,
    )
    return ctx


def operator_wake(event: WahaEvent, config: SessionConfig) -> bool | None:
    """Whether an operator message wakes the bot; None means skip it.

    A ``fromMe`` message with no explicit source cannot be attributed
    safely: folding the bot's own echo as the operator's would make it
    answer itself. A known operator message is normalized to the chat
    it belongs to, then only a mention of the bot wakes the agent.
    """
    if not event.payload.get("fromMe"):
        return False
    if not typed_by_operator(event):
        logger.debug(
            "Skipping own message {id} (source={source})",
            id=str(event.payload.get("id", "")),
            source=event.payload.get("source"),
        )
        return None
    normalize_operator_event(event)
    return name_mentioned(event, config.bot_name, config.bot_mention_regex)


async def fold_operator_message(
    event: WahaEvent,
    agent: FunctionCallingAgentWorkflow,
    settings: Settings,
    body: str | None,
) -> None:
    """Store an operator-typed message in the chat's memory, no run."""
    chat = str(event.payload.get("from", ""))
    async with chat_lock(event.session, chat):
        ctx = await remember_own_message(event, agent, settings, body)
        if ctx is not None:
            await persist_memory(settings, event.session, chat, ctx)


def normalize_operator_event(event: WahaEvent) -> None:
    """Reshape an operator-typed ``fromMe`` message like any member's.

    WAHA reports a ``fromMe`` message with ``from`` = the account and
    ``to`` = the chat. Everything downstream (gates, memory, sender
    tags, runs) keys on ``from`` as the chat, so the chat moves into
    ``from`` and the account's own JID becomes the ``participant`` —
    exactly the shape a member's group message has. The flag marks the
    turn as the operator's for rendering.
    """
    payload = event.payload
    me = event.me or {}
    chat = conversation_jid(event)
    own = jid_string(me.get("lid")) or jid_string(me.get("id"))
    payload["from"] = chat
    if own:
        payload["participant"] = own
    payload[OPERATOR_PAYLOAD_FLAG] = True


def learn_identity(event: WahaEvent) -> None:
    """Record who wrote *event* and its sender's display name.

    The display name (``notifyName``) and WAHA's ``source`` exist only
    in the live webhook — history fetches carry neither — so both are
    captured here, for quotes, reactions and ``read_chat`` later.
    """
    payload = event.payload
    data = cast(dict[str, Any], payload.get("_data") or {})
    message_id = str(payload.get("id", ""))
    if payload.get("fromMe"):
        if sent_by_bot_api(event):
            identity.authors.record(message_id, AUTHOR_BOT)
        elif typed_by_operator(event):
            identity.authors.record(message_id, AUTHOR_OPERATOR)
        return
    sender = jid_string(payload.get("participant") or data.get("author"))
    if not sender and not str(payload.get("from", "")).endswith("@g.us"):
        sender = jid_string(payload.get("from"))
    name = str(data.get("notifyName") or payload.get("notifyName") or "")
    identity.names.learn(sender, name)


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

    Posts the whole video file — the service demuxes audio itself, so
    this deliberately
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
    if settings.vision_max_images > 0:
        frames = frames[: settings.vision_max_images]
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


def register_forget_handler(
    settings: Settings,
    agent: FunctionCallingAgentWorkflow | None = None,
) -> None:
    """Register the ``forget`` webhook handler that wipes a chat's memory.

    The ``wahabot forget`` CLI posts a signed ``forget`` event; the wipe
    runs here, in the bot process, under the chat's run lock so it
    serializes with that chat's runs: an in-flight run finishes and
    saves first, then the live context and the memory file are dropped
    along with the chat's notebook, recap and thread index, with no
    resurrection window. When *agent* is supplied, its in-memory index
    queue is cleared too.
    """
    notebook = Notebook(settings.data_dir)

    @on_forget
    async def handle_forget(event: WahaEvent) -> None:
        chat_id = str(event.payload.get("chat_id", ""))
        if not chat_id:
            logger.warning("Ignoring forget event without a chat_id")
            return
        async with chat_lock(event.session, chat_id):
            live = contexts.pop((event.session, chat_id), None) is not None
            removed = forget_memory(settings.data_dir, event.session, chat_id)
            forgot_learned = forget_learned(settings.data_dir, event.session, chat_id)
            forgot_index = forget_thread_index(settings.data_dir, event.session, chat_id)
            if agent is not None and agent.thread_index is not None:
                agent.thread_index.forget(event.session, chat_id)
            await notebook_operation(notebook.forget, event.session, chat_id)
            logger.info(
                "Forgot {chat_id}: context {live}, file {removed}, "
                + "recap {recap}, index {index}",
                chat_id=chat_id,
                live="present" if live else "absent",
                removed="deleted" if removed else "absent",
                recap="deleted" if forgot_learned else "absent",
                index="deleted" if forgot_index else "absent",
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
    identity.configure(settings.data_dir, settings.session)
    enable_langfuse(settings)
    if settings.burst_enabled:
        bursts.configure(settings.burst_inactivity_s, settings.burst_hold_cap_s)
    else:
        bursts.set_completion_handler(None)

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
        except ContextBudgetError:
            # The album's images can never fit the budget; keep every
            # seen marker or redelivery retries the album forever.
            logger.warning(
                "Album in {chat_id} exceeds the context budget ({limit} tokens)",
                chat_id=str(buffer.container.payload.get("from", "")),
                limit=settings.memory_token_limit,
            )
        except Exception:
            for image_event in buffer.images:
                forget_seen(str(image_event.payload.get("id", "")))
            forget_seen(str(buffer.container.payload.get("id", "")))
            logger.exception(
                "Failed to handle album in {chat_id}",
                chat_id=str(buffer.container.payload.get("from", "")),
            )

    async def download_album_images(buffer: AlbumBuffer) -> list[dict[str, Any]]:
        """An album's images, downloaded and capped for the vision model.

        The vision cap applies at collection time: images past
        ``vision_max_images`` can never ride the prompt (handle_message
        trims to the same cap), so downloading them would be pure
        waste.
        """
        downloaded: list[dict[str, Any]] = []
        for image_event in buffer.images:
            if (
                settings.vision_max_images > 0
                and len(downloaded) >= settings.vision_max_images
            ):
                break
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
        return downloaded

    async def deliver_album_reply(buffer: AlbumBuffer) -> None:
        """Download an album's images, run the agent once, send its reply."""
        event = buffer.container
        chat_id = str(event.payload.get("from", ""))
        downloaded = await download_album_images(buffer)
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
        current = config_reloader.current_config()
        addressed = any(
            group_message_addressed(member, current)
            for member in (buffer.container, *buffer.images)
        )
        gate_first = not addressed and current.group_participation == "judicious"
        async with chat_lock(event.session, chat_id):
            ctx = await context_for(event.session, chat_id, agent, settings)
            with chat_trace_attributes(chat_id):
                reply, target = await handle_message(
                    event,
                    agent,
                    ctx=ctx,
                    images=downloaded,
                    settings=settings,
                    waha=waha,
                    bot_name=current.bot_name,
                    bot_mention_regex=current.bot_mention_regex,
                    addressed=addressed,
                    gate_first=gate_first,
                )
            await persist_memory(settings, event.session, chat_id, ctx)
        delivered = bool(target.sent or target.reacted)
        await finish_agent_reply(
            waha,
            event.session,
            chat_id,
            album_message_id(event),
            reply,
            delivered,
            settings,
            emoji_reaction=not delivered,
        )
        # Post-reply memory jobs outside the chat lock: the recap merge
        # and the embedding drain must never hold the chat's next run.
        await agent.merge_learned()

    set_completion_handler(run_album)

    async def run_burst(buffer: bursts.BurstBuffer) -> None:
        """Run the agent once over a completed burst, all messages joined.

        Fire-and-forget from the burst buffer, so failures must be
        caught here or they would die silently in an unretrieved task:
        the exception is logged with traceback and every buffered
        message's seen marker is dropped so WAHA's redelivery
        reprocesses them through the normal path — a burst never
        disappears, mirroring the single-message and album paths.
        """
        try:
            await deliver_burst_reply(buffer)
        except Exception as exc:
            # Same failure classes as the chat path: a timeout or an
            # outage notifies cleanly (never the false "unreachable"
            # narrative for a mere slow generation); anything else is
            # a bug with the full traceback. An oversized burst is
            # permanent — the same messages can never fit — so its
            # seen markers stay and no retry loop starts. The dropped
            # seen markers let WAHA's redelivery retry the whole burst
            # in every other case.
            if isinstance(exc, ContextBudgetError):
                logger.warning(
                    "Burst in {chat_id} exceeds the context budget ({limit} tokens)",
                    chat_id=buffer.key[1],
                    limit=settings.memory_token_limit,
                )
                return
            for burst_id in buffer.ids():
                forget_seen(burst_id)
            failure_class = await classify_llm_failure(
                waha, buffer.key[0], exc, settings.llm_timeout, settings.run_timeout
            )
            if failure_class == "bug":
                logger.exception(
                    "Failed to handle burst in {chat_id}",
                    chat_id=buffer.key[1],
                )

    async def transcribe_message(event: WahaEvent, body: str | None) -> str | None:
        """Replace an audio turn's body with its available voice transcript."""
        if message_kind(event) == "audio" and settings.transcribe_url:
            transcript = await transcribe_voice_note(event, waha, settings)
            if transcript:
                body = f"[voice note] {transcript}"
                event.payload["body"] = body
        return body

    async def download_captioned_image(
        media: dict[str, Any] | None, message_id: str
    ) -> dict[str, Any] | None:
        """Download and caption an image before taking the chat's run lock."""
        if media is None:
            return None
        image = await asyncio.to_thread(
            download_image, waha, media, message_id, settings.max_image_bytes
        )
        if image is not None:
            captions = await caption_images(
                agent.llm, [image], semaphore=agent.llm_semaphore
            )
            image["caption"] = captions[0]
        return image

    async def prepare_message_videos(
        event: WahaEvent, body: str | None, *, burst: bool = False
    ) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None]:
        """Prepare attached or linked video and stamp its marker on the turn."""
        video = None
        if can_prepare_video(event, settings):
            video = await prepare_video(
                event, waha, settings, agent.llm, agent.llm_semaphore
            )
            if video is not None:
                body = join_anchor(body or "", video["marker"])
                event.payload["body"] = body
        # Bursts exclude attached videos even after a failed download;
        # single turns try body URLs whenever no attached video resolved.
        excluded = message_kind(event) in ("video", "ptv") if burst else video is not None
        url_video = None
        if url_video_enabled(settings) and not excluded and body:
            url_video = await prepare_url_video(
                settings, agent.llm, agent.llm_semaphore, body
            )
            if url_video is not None:
                body = join_anchor(body, url_video["marker"])
                event.payload["body"] = body
        return body, video, url_video

    async def prepare_burst_media(
        burst_event: WahaEvent,
        images: list[dict[str, Any]],
    ) -> str | None:
        """Prepare one buffered turn's text and append its captioned media."""
        await transcribe_message(burst_event, None)
        body = extract_text(burst_event)
        # The vision cap applies at collection time: once the shared
        # list holds ``vision_max_images`` entries, a further image can
        # never ride the prompt (the run trims to the same cap), so
        # downloading and captioning it would be pure waste.
        capped = (
            settings.vision_max_images > 0 and len(images) >= settings.vision_max_images
        )
        if (
            message_kind(burst_event) in ("image", "sticker")
            and settings.vision
            and not capped
        ):
            image = await download_captioned_image(
                image_media(burst_event), str(burst_event.payload.get("id", ""))
            )
            if image is not None:
                images.append(image)
        body, video, url_video = await prepare_message_videos(
            burst_event, body, burst=True
        )
        for source in (video, url_video):
            if source is not None:
                images.extend(source["frames"])
        return body

    async def merge_burst(
        buffer: bursts.BurstBuffer,
    ) -> tuple[WahaEvent, list[dict[str, Any]]] | None:
        """Merge a burst into one turn; None when it has no usable content.

        Each part keeps its original message id so quoting, audit and
        evidence links survive the merge. The anchor's own id rides
        ``handle_message``'s message-id note on the merged event; a
        per-part note would duplicate it. Member words are scrubbed here,
        the last point the part is pure member text; the code-stamped id
        notes must survive unscrubbed, so the merged turn rides with
        ``body_scrubbed=True``.
        """
        if not buffer.messages:
            return None
        last = buffer.messages[-1]
        images: list[dict[str, Any]] = []
        parts: list[str] = []
        for burst_event in buffer.messages:
            body = await prepare_burst_media(burst_event, images)
            if body is None:
                continue
            if burst_event is last:
                parts.append(body)
                continue
            part_id = message_id_note(burst_event).strip()
            clean = strip_spoofed_markers(body)
            parts.append(f"{clean}\n{part_id}" if part_id else clean)
        if not parts and not images:
            logger.debug(
                "Burst in {chat_id} yielded no usable content",
                chat_id=buffer.key[1],
            )
            return None
        merged = last.model_copy(deep=True)
        merged.payload["id"] = str(last.payload.get("id", ""))
        merged.payload["body"] = "\n".join(parts)
        merged.payload["hasMedia"] = bool(images)
        # A burst turn must not inherit one member's quote context.
        merged.payload.pop("replyTo", None)
        merged.payload.get("_data", {}).pop("quotedMsg", None)
        return merged, images

    async def run_burst_turn(
        buffer: bursts.BurstBuffer,
        merged: WahaEvent,
        images: list[dict[str, Any]],
    ) -> tuple[str, RunTarget]:
        """Run one merged burst turn under the chat lock."""
        chat_id = buffer.key[1]
        async with chat_lock(merged.session, chat_id):
            ctx = await context_for(merged.session, chat_id, agent, settings)
            with chat_trace_attributes(chat_id):
                current = config_reloader.current_config()
                addressed = any(
                    group_message_addressed(member, current) for member in buffer.messages
                )
                gate_first = not addressed and current.group_participation == "judicious"
                reply, target = await handle_message(
                    merged,
                    agent,
                    ctx=ctx,
                    images=images or None,
                    settings=settings,
                    waha=waha,
                    bot_name=current.bot_name,
                    bot_mention_regex=current.bot_mention_regex,
                    body_scrubbed=True,
                    addressed=addressed,
                    gate_first=gate_first,
                )
            await persist_memory(settings, merged.session, chat_id, ctx)
        return reply, target

    async def deliver_burst_reply(buffer: bursts.BurstBuffer) -> None:
        """Prepare each buffered message's media, run the agent once.

        One ``handle_message`` call over a merged turn whose text joins
        every message's body. Media rides as image blocks; quote/reply
        targets the LAST buffered message (the natural reply anchor).
        """
        merged_burst = await merge_burst(buffer)
        if merged_burst is None:
            return
        merged, images = merged_burst
        chat_id = buffer.key[1]
        message_id = str(merged.payload.get("id", ""))
        reply, target = await run_burst_turn(buffer, merged, images)
        # The provider answered: flip the LLM health flag back up
        # (once-per-transition notify) outside the chat lock — the
        # burst path shares the single-message path's contract.
        await mark_llm_recovered(waha, merged.session)
        await finish_agent_reply(
            waha,
            merged.session,
            chat_id,
            message_id,
            reply,
            bool(target.sent or target.reacted),
            settings,
            emoji_reaction=True,
        )
        # Post-reply memory jobs outside the chat lock: the recap merge
        # and the embedding drain must never hold the chat's next run.
        await agent.merge_learned()

    if settings.burst_enabled:
        bursts.set_completion_handler(run_burst)

    def render_prompt() -> str:
        """Re-render ``{{date}}``/``{{time}}`` and pick up config edits.

        The current config's prompt always wins over the startup
        snapshot, so prompt changes apply without a restart too. The
        bot's own ids (``{{own_jid}}``/``{{own_lid}}``/``{{own_identities}}``)
        come from ``status.state`` — captured at startup and re-captured
        on every session recovery — so identity in the prompt is
        knowledge, not inference; an unknown identity drops the
        prompt's identity lines instead of rendering a stale JID.
        """
        current = config_reloader.current_config()
        return render_system_prompt(
            current.system_prompt,
            settings.timezone,
            current.bot_name,
            current.goal,
            current.memory,
            own_jid=status_state.operator_jid,
            own_lid=status_state.operator_lid,
            operator_name=current.operator_name,
        )

    # Per-agent escalation state (per-chat cooldowns; the operator JID
    # is read from status.state at call time, so a session recovery
    # needs no re-capture wiring here).
    escalation_channel = EscalationChannel()
    triage = (
        TriageGate(settings, waha, channel=escalation_channel)
        if settings.triage_gate and settings.gate_api_base
        else None
    )
    if triage is not None:
        # Detached escalation sends outlive the run: drain them at app
        # shutdown so a clean exit never drops a flagged escalation.
        on_shutdown(triage.close)

    def group_message_addressed(event: WahaEvent, current: SessionConfig) -> bool:
        """True for DMs and for group messages that mention/reply to the bot."""
        if not str(event.payload.get("from", "")).endswith("@g.us"):
            return True
        return replies_to_bot(event, current.bot_name, current.bot_mention_regex)

    def render_session_rules() -> str:
        """The session rules that gate behaviour, as one current line.

        Hot-reloaded like the prompt: the participation mode comes
        from the live config, so a config edit changes the next run's
        pinned state block without a restart. Support hours and reply
        windows are the v1.2.0 support product's settings; the engine
        states what it actually enforces today.
        """
        current = config_reloader.current_config()
        participation = {
            "never": "In groups, never reply.",
            "mentioned": "In groups, reply only when addressed or replied to.",
            "judicious": "In groups, judge each message for whether to reply.",
        }.get(current.group_participation, "")
        return " ".join(
            part for part in (f"Timezone: {settings.timezone}.", participation) if part
        )

    agent = build_agent(
        settings,
        tools=build_default_tools(waha, settings, escalation_channel=escalation_channel),
        system_prompt=config.system_prompt,
        prompt_renderer=render_prompt,
        rules_renderer=render_session_rules,
        triage=triage,
    )
    logger.info(
        "Agent ready: {tools}",
        tools=", ".join(sorted(tool.metadata.get_name() for tool in agent.tools)),
    )
    started_at = time.time()

    async def accept_message(event: WahaEvent) -> SessionConfig | None:
        """Check health, dedup and freshness before learning the sender's identity."""
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
            # The delivery itself is evidence the session may be back
            # (a hung WAHA emits nothing) — one cheap probe against
            # GET /api/sessions decides; the poller arms only if the
            # status event never arrives (register_session_status_handler).
            logger.info(
                "Muting message {id} while WAHA session is not WORKING", id=message_id
            )
            await probe_session_recovery(waha, event.session)
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
        # Off the event loop: a rate-limited identity flush writes disk.
        await asyncio.to_thread(learn_identity, event)
        return config

    async def handle_self_command(
        event: WahaEvent, config: SessionConfig
    ) -> tuple[bool, str | None]:
        """Run a self-chat command before chat gates, preserving retry and echo rules."""
        message_id = str(event.payload.get("id", ""))
        body = extract_text(event)
        if self_chat_needs_transcript(event, body, settings):
            # The self-chat is the operator console: a voice note there
            # can only come from the operator's own devices, so it is
            # always worth transcribing early — it may be a spoken
            # command ("kai do x"). Everywhere else transcription stays
            # behind the chat gates. The transcript becomes the body;
            # the mention check below runs on it like typed text.
            body = await transcribe_message(event, body)
        instruction = (
            None
            if is_self_echo(message_id)
            else self_command_instruction(
                event,
                bot_name=config.bot_name,
                bot_mention_regex=config.bot_mention_regex,
            )
        )
        if not instruction:
            return False, body
        # Commands share the operator context and bypass chat gates;
        # send_self_reply remembers the reply id to suppress its echo.
        from wahabot.commands import build_command_event, run_command

        command = WahaEvent.model_validate(
            build_command_event(event.session, instruction)
        )
        command.payload["reply_chat_id"] = str(event.payload.get("from", ""))
        command.payload["reply_to"] = message_id
        try:
            reply = await run_command(command, agent, settings, waha)
        except Exception as exc:
            # Retry failed runs, even if a tool delivered before the crash;
            # the delivery latch is per-run, not a durable command receipt.
            # A permanent ContextBudgetError never reaches here: the
            # command path reports it as the reply (journaled, delivered
            # back to this self-chat) instead of raising.
            forget_seen(message_id)
            if llm_call_timed_out(exc):
                detail = (
                    f"LLM call timed out after {settings.llm_timeout}s on"
                    f" self-chat command {message_id} ({settings.llm_model} via"
                    f" {settings.llm_api_base}); dropped for redelivery"
                )
                logger.warning("{detail}", detail=detail)
            else:
                logger.exception("Failed to handle self-chat command {id}", id=message_id)
            return True, body
        try:
            await send_self_reply(waha, command, reply)
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            # The command completed: keep its seen marker even when only
            # the final reply failed, or redelivery duplicates its actions.
            journal_send_failure(
                settings,
                event.session,
                str(command.payload.get("reply_chat_id", "")),
                "send_text",
                message_id,
                exc,
            )
        except Exception:
            logger.exception(
                "Failed to deliver self-chat reply for command {id}", id=message_id
            )
        return True, body

    async def prepare_chat_message(
        event: WahaEvent, config: SessionConfig, body: str | None
    ) -> bool | None:
        """Apply access gates and fold or buffer turns that need no immediate run."""
        operator_wakes = operator_wake(event, config)
        if operator_wakes is None:
            return
        if not chat_allowed(
            event,
            config.whitelist,
            config.blacklist,
            jid_aliases=jid_alias_lookup(event),
        ):
            return
        if settings.send_seen and not operator_wakes and not is_operator_event(event):
            # The read receipt answers *their* message, not the bot's
            # reply. It goes out before any participation decision and
            # before burst/album buffering, so holding or ignoring a
            # message never withholds it.
            await asyncio.to_thread(
                mark_seen, waha, event.session, str(event.payload["from"])
            )
        if is_operator_event(event) and not operator_wakes:
            await fold_operator_message(event, agent, settings, body)
            return
        if buffer_album_message(event, operator_wakes):
            return
        return operator_wakes

    async def skip_or_buffer_message(
        event: WahaEvent, config: SessionConfig, operator_wakes: bool
    ) -> bool:
        """Apply group addressing and queue eligible bursts."""
        if not operator_wakes and not is_group_addressed(
            event,
            bot_name=config.bot_name,
            bot_mention_regex=config.bot_mention_regex,
            participation=config.group_participation,
        ):
            logger.debug(
                "Ignoring unaddressed group message {id}", id=event.payload.get("id")
            )
            return True
        if settings.burst_enabled and not operator_wakes and bursts.is_bufferable(event):
            # Burst assembly: hold this message for its sender's window
            # instead of running the agent now. Only messages that
            # already passed every gate above reach here (the doc's
            # "only messages that would wake the bot enter the
            # buffer"); albums never reach this line (the album
            # branches above), and unknown kinds run the pre-burst path.
            bursts.add_message(event)
            return True
        return False

    async def deliver_single_reply(
        event: WahaEvent,
        config: SessionConfig,
        image: dict[str, Any] | None,
        video: dict[str, Any] | None,
        url_video: dict[str, Any] | None,
    ) -> None:
        """Run and persist one turn under its chat lock, then deliver its reply."""
        message_id = str(event.payload.get("id", ""))
        # jid_string, not the raw payload value: a `from` carrying a
        # linked-device suffix must key the same memory (and lock) as
        # the operator fold's device-stripped chat id, or one DM splits
        # in two.
        chat_id = jid_string(event.payload["from"])
        addressed = group_message_addressed(event, config)
        gate_first = not addressed and config.group_participation == "judicious"
        try:
            image = await download_captioned_image(image, message_id)
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
                        bot_name=config.bot_name,
                        bot_mention_regex=config.bot_mention_regex,
                        addressed=addressed,
                        gate_first=gate_first,
                    )
                await persist_memory(settings, event.session, chat_id, ctx)
                delivered = bool(target.sent or target.reacted)
            # The provider answered: flip the LLM health flag back up
            # (once-per-transition notify) outside the chat lock, so
            # the notify send never serializes this chat's next run.
            await mark_llm_recovered(waha, event.session)
            await finish_agent_reply(
                waha,
                event.session,
                chat_id,
                message_id,
                reply,
                delivered,
                settings,
                emoji_reaction=True,
            )
            # Post-reply memory jobs outside the chat lock: the recap
            # merge and the embedding drain must never hold this
            # chat's next run.
            await agent.merge_learned()
        except Exception as exc:
            # Clean failure classes first: a timeout gets one concise
            # line (the chained httpx/openai traceback is noise for a
            # mere slow generation), an outage the operator notify —
            # both with the seen-marker drop so WAHA's redelivery
            # retries; the run is just as lost either way. An
            # oversized turn is permanent — the same message can
            # never fit — so its seen marker stays and no retry loop
            # starts. Anything else is a bug: full traceback, also
            # redeliverable.
            if isinstance(exc, ContextBudgetError):
                logger.warning(
                    "Message {id} in {chat_id} exceeds the budget ({limit} tokens)",
                    id=message_id,
                    chat_id=chat_id,
                    limit=settings.memory_token_limit,
                )
                return
            failure_class = await classify_llm_failure(
                waha, event.session, exc, settings.llm_timeout, settings.run_timeout
            )
            forget_seen(message_id)
            if failure_class != "bug":
                return
            logger.exception(
                "Failed to handle message {id} in {chat_id} {detail}",
                id=message_id,
                chat_id=chat_id,
                detail="(seen marker dropped; a WAHA redelivery will retry)",
            )
            return

    @on_message
    async def reply_with_agent(event: WahaEvent) -> None:
        """Run the agent over an incoming message and send its reply via WAHA."""
        config = await accept_message(event)
        if config is None:
            return
        handled, body = await handle_self_command(event, config)
        if handled:
            return
        operator_wakes = await prepare_chat_message(event, config, body)
        if operator_wakes is None:
            return
        image = image_media(event) if settings.vision else None
        if await skip_or_buffer_message(event, config, operator_wakes):
            return
        body = await transcribe_message(event, body)
        body, video, url_video = await prepare_message_videos(event, body)
        if body is None and image is None and video is None and url_video is None:
            logger.debug("Skipping media/album message {id}", id=event.payload.get("id"))
            return
        await deliver_single_reply(event, config, image, video, url_video)

    return agent, config_reloader
