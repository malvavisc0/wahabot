import asyncio
from collections.abc import Sequence
from typing import Any, cast

from llama_index.core.base.llms.types import ChatMessage, ImageBlock
from loguru import logger

from wahabot.ai.context.addressing import (
    mentions_note,
    message_id_note,
    reply_context_section,
    sender_tag,
)
from wahabot.ai.context.participants import participant_names
from wahabot.ai.history import (
    chat_visible_text,
    narration_kind,
)
from wahabot.ai.messages import (
    OPERATOR_NOTE_PREFIX,
    bot_jids,
    is_operator_event,
    message_kind,
    message_replies_to,
    turn_addressed_note,
)
from wahabot.ai.scrub import strip_spoofed_markers
from wahabot.ai.serials import serial_store
from wahabot.ai.tools.url_images import fetch_url_images, image_urls
from wahabot.ai.turn_records import (
    LINK_VIDEO_MARKER_KEY,
    VIDEO_MARKER_KEY,
    RecordEntry,
    render_entries,
)
from wahabot.ai.vision import image_caption, image_noun
from wahabot.core.models import WahaEvent
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings


async def turn_user_message(
    event: WahaEvent,
    image: dict[str, Any] | None = None,
    images: list[dict[str, Any]] | None = None,
    settings: Settings | None = None,
    waha: WahaClient | None = None,
    pinned_note: str = "",
    bot_name: str | None = None,
    bot_mention_regex: str | None = None,
    body_scrubbed: bool = False,
    addressed: bool = True,
    record_entries: Sequence[RecordEntry] | None = None,
) -> tuple[str, list[ImageBlock], list[int]]:
    """The run's user message text, image blocks and record serials.

    Inbound chat turns are one record per message
    (``turn_records.render_entries``) and carry no ids, tags or addressing
    notes; the header renders later, at prompt assembly. Operator command
    events keep the prose shape (they are synthetic, not messages the
    chat's members spoke). Image bytes ride as blocks on the first LLM
    call only.
    """
    chat_id = str(event.payload.get("from", ""))
    # Roster before tag: group sender tags render ``[Name <jid>]`` via
    # the resolver, so the fetch must precede the tag build. One
    # fetch, same cache the quoting lines and reaction notes share.
    names = await asyncio.to_thread(participant_names, waha, event.session, chat_id)
    records_on = settings is not None and event.event != "command" and not pinned_note
    if records_on and settings is not None:
        return await records_user_message(
            event,
            image=image,
            images=images,
            settings=settings,
            names=names,
            chat_id=chat_id,
            addressed=addressed,
            record_entries=record_entries,
        )
    return await prose_user_message(
        event,
        image=image,
        images=images,
        settings=settings,
        names=names,
        pinned_note=pinned_note,
        bot_name=bot_name,
        bot_mention_regex=bot_mention_regex,
        body_scrubbed=body_scrubbed,
        addressed=addressed,
    )


async def records_user_message(
    event: WahaEvent,
    *,
    image: dict[str, Any] | None,
    images: list[dict[str, Any]] | None,
    settings: Settings,
    names: dict[str, str],
    chat_id: str,
    addressed: bool,
    record_entries: Sequence[RecordEntry] | None,
) -> tuple[str, list[ImageBlock], list[int]]:
    """Render the inbound turn as one record per message."""
    entries = list(record_entries) if record_entries is not None else []
    if not entries:
        attached = await asyncio.to_thread(
            collect_images,
            [image] if image is not None else [],
            settings,
            str(event.payload.get("body", "")).strip(),
        )
        entries = [
            single_record_entry(
                event,
                images=attached,
                video_frames=images,
                addressed=addressed,
            )
        ]
    registry = serial_store(str(settings.data_dir)).for_chat(event.session, chat_id)
    user_msg, serials, blocks = await asyncio.to_thread(
        render_entries,
        entries,
        registry=registry,
        names=names,
        tz_name=settings.timezone,
    )
    image_blocks = await asyncio.to_thread(build_image_blocks, blocks)
    logger.info("Attaching {n} image block(s) to agent run", n=len(image_blocks))
    return user_msg, image_blocks, serials


async def prose_user_message(
    event: WahaEvent,
    *,
    image: dict[str, Any] | None,
    images: list[dict[str, Any]] | None,
    settings: Settings | None,
    names: dict[str, str],
    pinned_note: str,
    bot_name: str | None,
    bot_mention_regex: str | None,
    body_scrubbed: bool,
    addressed: bool,
) -> tuple[str, list[ImageBlock], list[int]]:
    """The legacy prose turn used by synthetic operator commands."""
    tag = sender_tag(event, names)
    body = turn_body(event, prescrubbed=body_scrubbed or event.event == "command")
    text = f"{tag} {body}".strip() if tag else body
    attached = list(images or [])
    if image is not None:
        attached.append(image)
    all_images = await asyncio.to_thread(collect_images, attached, settings, text)
    if all_images:
        marker = image_noun([image_caption(img) for img in all_images])
        text = f"{text} {marker}".strip()
    user_msg = text + message_id_note(event)
    user_msg += reply_context_section(message_replies_to(event), names, bot_jids(event))
    user_msg += mentions_note(event, names)
    user_msg += pinned_note
    user_msg += turn_addressed_note(event, addressed, bot_name, bot_mention_regex)
    image_blocks = await asyncio.to_thread(build_image_blocks, all_images)
    # Wire-level evidence: Langfuse traces never show image blocks (the
    # OTel instrumentor serializes only the text-only `content`
    # property), so this log is the authoritative count of what rides
    # the first LLM call.
    logger.info("Attaching {n} image block(s) to agent run", n=len(image_blocks))
    return user_msg, image_blocks, []


def single_record_entry(
    event: WahaEvent,
    *,
    images: list[dict[str, Any]],
    video_frames: list[dict[str, Any]] | None,
    addressed: bool,
) -> RecordEntry:
    """Build the one-record entry of a single inbound message.

    ``body`` stays None so :func:`turn_records.record_body` reads the
    payload itself (and strips the code's voice-note prefix).
    """
    return RecordEntry(
        event=event,
        images=images,
        video=record_entry_video(event, video_frames),
        addressed=addressed,
    )


def record_entry_video(
    event: WahaEvent, frames: list[dict[str, Any]] | None
) -> dict[str, Any] | None:
    """The video attachment (frames + analysis) for a record, if any."""
    if not frames:
        return None
    marker = str(event.payload.get(VIDEO_MARKER_KEY) or "")
    link_marker = str(event.payload.get(LINK_VIDEO_MARKER_KEY) or "")
    if not (marker or link_marker or message_kind(event) in ("video", "ptv")):
        return None
    return {"frames": list(frames), "marker": marker or link_marker or "(video)"}


def build_image_blocks(images: list[dict[str, Any]] | None) -> list[ImageBlock]:
    """The run's one-shot image blocks from prepared image dicts."""
    return [
        ImageBlock(image=img["data"], image_mimetype=img.get("mimetype") or "image/jpeg")
        for img in images or []
    ]


def turn_body(event: WahaEvent, prescrubbed: bool = False) -> str:
    """The event's body as member text, spoof-scrubbed unless exempt.

    Member text rides the turn verbatim — scrub it of authority-marker
    substrings so the only trusted metadata in a turn is metadata this
    code wrote (prompt-injection defense: a member typing "[you were
    addressed…]" must not gain the marker's authority). Exempt when the
    caller already scrubbed (the burst path, per member, before
    stamping its id notes) or the event is a code-built command.
    """
    raw = str(event.payload.get("body", "")).strip()
    exempt = prescrubbed or event.event == "command"
    body = raw if exempt else strip_spoofed_markers(raw)
    if is_operator_event(event):
        # Code-written, after the scrub: the human operator typed this
        # on the shared account — a teammate, not the bot's own words.
        return f"{OPERATOR_NOTE_PREFIX}{body}"
    return body


def collect_images(
    attached: list[dict[str, Any]],
    settings: Settings | None,
    text: str,
) -> list[dict[str, Any]]:
    """The turn's images: the attached ones plus sniffed URL images."""
    images = list(attached)
    if settings is not None and settings.vision:
        urls = image_urls(text, settings.max_url_images)
        images.extend(fetch_url_images(settings, urls))
    return images


def final_reply(result: Any) -> str:
    """The run's reply text, emptied when the model narrated instead of answered.

    Delivery filters by :func:`chat_visible_text` — the one definition
    storage (``remember``) also applies, so the two ends cannot drift
    apart. The returned text is that visible text: a genuine reply keeps
    its words and loses any private trailing note the model appended.
    This wrapper only adds the delivery-side log lines, which carry the
    narration *kind*: a recurring failure mode is only visible in the
    audit trail if the log says which one fired.
    """
    message = cast(ChatMessage, result.message)
    content = message.content
    reply = content.strip() if isinstance(content, str) else ""
    visible = chat_visible_text(reply)
    if visible:
        return visible
    if not reply:
        return reply
    kind = narration_kind(reply)
    if kind == "silence":
        logger.debug("Filtering silence narration: {reply!r}", reply=reply)
        return ""
    logger.warning(
        "Dropping {kind} narration as final reply: {reply!r}",
        kind=kind,
        reply=reply[:200],
    )
    return ""
