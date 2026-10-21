"""Compose prompt-time records from inbound WAHA events (Phase 1).

The pure grammar lives in :mod:`wahabot.ai.records`; this is the thin
WAHA-aware layer that maps parsed events onto it — the record's kind, its
speaker token, presence flags, media slots and the renderer-written payload
lines. Callers pass what only they can resolve (the ``#m<n>`` registry, the
roster names, the prepared media), so the module stays free of I/O and the
roster.

Media slots are numbered per turn in arrival order: an image record carries
``image#n``, a video ``video#n``, and an album ``album#a-b`` over the image
slots it owns. The attachment list handed to the model is built by the same
pass, in the same order, so slot *n* is attachment *n*.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from wahabot.ai.messages import (
    bot_jids,
    conversation_jid,
    extract_text,
    is_operator_event,
    is_reaction_only,
    jid_string,
    message_kind,
    message_replies_to,
    quoted_account_author,
    reply_description,
)
from wahabot.ai.records import format_record
from wahabot.ai.serials import (
    SerialEntry,
    SerialRegistry,
    SerialStore,
    format_serial_token,
)
from wahabot.ai.video import MAX_TRANSCRIPT_CHARS
from wahabot.ai.vision import image_caption, image_noun
from wahabot.core.identity import AUTHOR_BOT, AUTHOR_OPERATOR, authors, display_name
from wahabot.core.identity import names as name_book
from wahabot.core.jid import chat_from_message_id, is_own_message_id
from wahabot.core.models import WahaEvent

#: Payload keys the media pipeline stamps on an event before rendering.
TRANSCRIPT_KEY = "_wahabot_transcript"
VIDEO_MARKER_KEY = "_wahabot_video_marker"
LINK_VIDEO_MARKER_KEY = "_wahabot_link_video_marker"

#: A body that is nothing but one URL is a ``link`` record.
BARE_URL_RE = re.compile(r"^https?://\S+$")


@dataclass
class RecordEntry:
    """One record to render: the event plus what only the caller resolved."""

    event: WahaEvent
    body: str | None = None
    images: list[dict[str, Any]] = field(default_factory=list)
    video: dict[str, Any] | None = None
    omitted: int = 0
    album: bool = False
    extra_payloads: list[str] = field(default_factory=list)
    addressed: bool = False


@dataclass
class Slots:
    """Per-turn attachment counters, shared by every record in the turn."""

    images: int = 0
    videos: int = 0

    def take_images(self, count: int) -> int:
        """Reserve *count* image slots and return the first one."""
        start = self.images + 1
        self.images += count
        return start

    def take_video(self) -> int:
        """Reserve one video slot and return it."""
        self.videos += 1
        return self.videos


def record_kind(event: WahaEvent) -> str:
    """The record's ``kind``, derived from the message's own shape.

    A text body that is a bare URL is a ``link`` whether or not the
    download/caption later succeeds, so the kind is stable across runs.
    A text body of emoji/punctuation only is a ``reaction``; a WAHA
    reaction event is one too. Everything else is the WAHA kind unchanged.
    """
    if event.event == "message.reaction" or isinstance(
        event.payload.get("reaction"), dict
    ):
        return "reaction"
    kind = message_kind(event)
    if kind == "ptv":
        return "video"
    if kind != "text":
        return kind
    if is_reaction_only(event):
        return "reaction"
    return "link" if BARE_URL_RE.match((extract_text(event) or "").strip()) else "text"


def speaker_token(event: WahaEvent, *, name: str = "", ordinal: int | None = None) -> str:
    """The record's speaker: the shared account's human, a name, or an ordinal.

    ``<operator>`` for the human on the account; the learned display *name*
    when there is one; otherwise ``<member-n>``, the per-chat ordinal — never
    the JID (the id boundary). ``<member-?>`` only when the caller resolved
    neither, which should not happen.
    """
    if is_operator_event(event):
        return "<operator>"
    if name:
        return name
    return f"<member-{ordinal}>" if ordinal is not None else "<member-?>"


def event_sender(event: WahaEvent) -> str:
    """The author JID of *event* (the chat itself in a DM)."""
    data = event.payload.get("_data", {})
    sender = jid_string(event.payload.get("participant") or data.get("author"))
    if not sender and not str(event.payload.get("from", "")).endswith("@g.us"):
        sender = jid_string(event.payload.get("from"))
    return sender


def event_display_name(event: WahaEvent) -> str:
    """The sender's learnable display name, "" when none was learned."""
    data = event.payload.get("_data", {})
    name = str(data.get("notifyName") or event.payload.get("notifyName") or "").strip()
    return name or name_book.name(event_sender(event))


def assign_event_serial(store: SerialStore, event: WahaEvent) -> SerialEntry:
    """Assign this inbound event its chat-global ``#m<n>`` at ingest.

    Runs before any gate, so messages that never wake a run still get a
    serial (a later quote or reaction can anchor to them). An unknown
    sender reserves a stable ``<member-n>`` ordinal in the same write.
    """
    chat_id = conversation_jid(event)
    sender = event_sender(event)
    from_me = bool(event.payload.get("fromMe"))
    name = "" if from_me or is_operator_event(event) else event_display_name(event)
    registry = store.for_chat(event.session, chat_id)
    ordinal = 0
    if not name and not from_me and not is_operator_event(event) and sender:
        ordinal = registry.ordinal_for(sender)
    return store.assign(
        event.session,
        chat_id,
        message_id=str(event.payload.get("id", "")),
        jid=sender,
        name=name,
        ordinal=ordinal,
    )


def time_label(event: WahaEvent, tz_name: str = "UTC") -> str:
    """Chat-local ``hh:mm``, or ``dd.mm hh:mm`` for an earlier day."""
    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError, ValueError, OSError:
        tz = ZoneInfo("UTC")
    now = datetime.now(tz)
    timestamp = event.payload.get("timestamp", event.timestamp)
    try:
        moment = datetime.fromtimestamp(float(timestamp), tz)
    except TypeError, ValueError, OSError, OverflowError:
        moment = now
    pattern = "%H:%M" if moment.date() == now.date() else "%d.%m %H:%M"
    return moment.strftime(pattern)


def quote_parts(
    event: WahaEvent, registry: SerialRegistry
) -> tuple[list[str], list[str]]:
    """The ``quote=…`` flag and its context payload for a reply."""
    quoted = message_replies_to(event)
    if not quoted:
        return [], []
    flags: list[str] = []
    account_author = quoted_account_author(event)
    if account_author == AUTHOR_BOT:
        flags.append("quote=you")
    elif account_author == AUTHOR_OPERATOR:
        flags.append("quote=operator")
    elif account_author == "":
        # An account message without an author record: the classifier
        # already proved it belongs to the shared account; keep the old
        # bot fallback for the record flag.
        flags.append("quote=you")
    else:
        serial = registry.serial_for(str(quoted.get("id") or ""))
        if serial:
            flags.append(f"quote={format_serial_token(serial)}")
    preview = reply_description(quoted)[:120].replace('"', "'")
    return flags, [f'quoted (context): "{preview}"'] if preview else []


def reaction_parts(event: WahaEvent, registry: SerialRegistry) -> list[str]:
    """The ``to: #m<n>`` / ``to: you`` payload of a reaction event."""
    reaction = event.payload.get("reaction")
    if not isinstance(reaction, dict):
        return []
    reaction = cast(dict[str, Any], reaction)
    target = str(reaction.get("messageId") or "")
    if not target:
        return []
    if is_own_message_id(target) or authors.author(target) == AUTHOR_BOT:
        label = "you"
    else:
        serial = registry.serial_for(target)
        label = format_serial_token(serial) if serial else ""
    if not label:
        return []
    preview = str(reaction.get("preview") or "").strip()[:60].replace('"', "'")
    return [f'to: {label} "{preview}"' if preview else f"to: {label}"]


def mentions_parts(
    event: WahaEvent,
    names: dict[str, str] | None,
    registry: SerialRegistry,
) -> list[str]:
    """The ``mentions: @<typed> is <speaker>`` payload, JID-free.

    A mentioned JID with no roster name renders its stored ordinal when
    one exists (persisted with an ingested message); otherwise
    ``<member-?>`` — this path never allocates an ordinal, since an
    unpersisted allocation would be reassigned to someone else after a
    restart.
    """
    mentioned = event.payload.get("_data", {}).get("mentionedJidList") or []
    jids = list(dict.fromkeys(j for j in map(jid_string, mentioned) if j))
    if not jids:
        return []
    own = bot_jids(event)
    parts: list[str] = []
    for jid in jids[:10]:
        user = jid.split("@", 1)[0]
        if jid in own:
            who = "you or your operator"
        else:
            name = display_name(jid, names)
            ordinal = registry.known_ordinal(jid)
            who = name or (f"<member-{ordinal}>" if ordinal else "<member-?>")
        parts.append(f"@{user} is {who}")
    return ["mentions: " + "; ".join(parts)]


def audio_payload(event: WahaEvent) -> list[str]:
    """The ``[audio: "…"]`` payload of a transcribed voice note."""
    transcript = str(event.payload.get(TRANSCRIPT_KEY) or "").strip()
    if not transcript:
        return []
    trimmed = transcript[:MAX_TRANSCRIPT_CHARS].rstrip().replace('"', "'")
    suffix = "…" if len(transcript) > MAX_TRANSCRIPT_CHARS else ""
    return [f'[audio: "{trimmed}{suffix}"]']


def image_payloads(images: Sequence[dict[str, Any]]) -> list[str]:
    """The media-analysis payload line for an entry's images."""
    if not images:
        return []
    return [image_noun([image_caption(image) for image in images])]


def record_body(entry: RecordEntry) -> str:
    """Member text for *entry*.

    No denylist: a member may type any marker, including the legacy
    ``[operator message]`` prefix. Only the code's own voice-note prefix
    is removed, and only when the transcript key proves the code wrote it.
    """
    body = (
        entry.body
        if entry.body is not None
        else str(entry.event.payload.get("body", "")).strip()
    )
    if str(entry.event.payload.get(TRANSCRIPT_KEY) or "").strip():
        body = body.removeprefix("[voice note]").strip()
    return body.strip()


def media_parts(
    entry: RecordEntry, slots: Slots
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Flags, payloads and image blocks for one entry's attachments."""
    flags: list[str] = []
    payloads: list[str] = []
    blocks: list[dict[str, Any]] = []
    if entry.video is not None:
        flags.append(f"video#{slots.take_video()}")
        payloads.append(str(entry.video.get("marker") or "(video)"))
        blocks.extend(entry.video.get("frames") or [])
    if entry.album:
        total = len(entry.images) + entry.omitted
        if total:
            start = slots.take_images(total)
            flags.append(
                f"album#{start}" if total == 1 else f"album#{start}-{start + total - 1}"
            )
        payloads.extend(image_payloads(entry.images))
        payloads.extend(["(image omitted: not shown)"] * entry.omitted)
        blocks.extend(entry.images)
        return flags, payloads, blocks
    for _ in entry.images:
        flags.append(f"image#{slots.take_images(1)}")
    for _ in range(entry.omitted):
        flags.append(f"image#{slots.take_images(1)}")
    payloads.extend(image_payloads(entry.images))
    payloads.extend(["(image omitted: not shown)"] * entry.omitted)
    blocks.extend(entry.images)
    return flags, payloads, blocks


def resolve_identity(
    event: WahaEvent,
    registry: SerialRegistry,
    names: dict[str, str] | None,
) -> tuple[str, int | None, int]:
    """Resolve ``(name, ordinal, serial)`` for a record.

    The registry's stored name is the fallback after the event and the
    roster: a reaction or an operator fold may have learned a name at
    ingest without the name book learning it. A stored ordinal wins over
    a new allocation, so ``<member-n>`` survives every later render.
    """
    sender = event_sender(event)
    message_id = str(event.payload.get("id", ""))
    serial = registry.serial_for(message_id)
    stored = registry.entries.get(serial) if serial else None
    name = stream_name(event, sender, names, stored)
    ordinal = stream_ordinal(event, sender, name, stored, registry)
    if not serial:
        serial = registry.next(
            message_id=message_id,
            jid=sender,
            name=name,
            ordinal=ordinal or 0,
        )
    return name, ordinal, serial


def stream_name(
    event: WahaEvent,
    sender: str,
    names: dict[str, str] | None,
    stored: SerialEntry | None,
) -> str:
    """The event name, then the roster's, then the name stored at ingest."""
    return (
        event_display_name(event)
        or (names or {}).get(sender, "")
        or (stored.name if stored else "")
    )


def stream_ordinal(
    event: WahaEvent,
    sender: str,
    name: str,
    stored: SerialEntry | None,
    registry: SerialRegistry,
) -> int | None:
    """The stored ``<member-n>`` ordinal, allocating only when none exists."""
    if is_operator_event(event) or name:
        return None
    if stored is not None and stored.ordinal:
        return stored.ordinal
    return registry.ordinal_for(sender)


def render_entry(
    entry: RecordEntry,
    *,
    registry: SerialRegistry,
    names: dict[str, str] | None,
    tz_name: str,
    slots: Slots,
) -> tuple[str, int, list[dict[str, Any]]]:
    """Render one record and return it with its serial and image blocks."""
    event = entry.event
    name, ordinal, serial = resolve_identity(event, registry, names)
    media_flags, media_payloads, blocks = media_parts(entry, slots)
    quote_flags, quote_payloads = quote_parts(event, registry)
    payloads = [
        *media_payloads,
        *audio_payload(event),
        *quote_payloads,
        *reaction_parts(event, registry),
        *mentions_parts(event, names, registry),
        *entry.extra_payloads,
    ]
    line = format_record(
        time_label=time_label(event, tz_name),
        speaker=speaker_token(event, name=name, ordinal=ordinal),
        serial=serial,
        kind=record_kind(event),
        flags=(
            *media_flags,
            *(["addressed"] if entry.addressed else []),
            *quote_flags,
        ),
        body=record_body(entry),
        payloads=payloads,
    )
    return line, serial, blocks


def render_entries(
    entries: Sequence[RecordEntry],
    *,
    registry: SerialRegistry,
    names: dict[str, str] | None = None,
    tz_name: str = "UTC",
) -> tuple[str, list[int], list[dict[str, Any]]]:
    """Render a turn's entries in order: text, serials and image blocks."""
    slots = Slots()
    records: list[str] = []
    serials: list[int] = []
    blocks: list[dict[str, Any]] = []
    for entry in entries:
        line, serial, entry_blocks = render_entry(
            entry, registry=registry, names=names, tz_name=tz_name, slots=slots
        )
        records.append(line)
        serials.append(serial)
        blocks.extend(entry_blocks)
    return "\n".join(records), serials, blocks


def read_record(
    message: dict[str, Any],
    registry: SerialRegistry,
    tz_name: str = "UTC",
) -> str:
    """Render one fetched message as a read view record.

    ``read_chat`` and the prompt must agree on what a message looks like,
    and the console needs the ``#m<n>`` to act on. The registry is written
    at ingest only: a fetched message the bot never saw renders without a
    serial (a raw id still addresses it). The account's own ``fromMe``
    rows render without one on purpose — the model cannot quote or react
    to its own or the operator's output by token, only by the raw id it
    holds when it has one; ``quote=you`` / ``to: you`` remain the prompt's
    sentinels for them.
    """
    payload = dict(message)
    message_id = str(payload.get("id", ""))
    if not payload.get("from"):
        payload["from"] = chat_from_message_id(message_id)
    event = WahaEvent(event="message", session="", payload=payload)
    sender = event_sender(event)
    serial = 0 if payload.get("fromMe") else registry.serial_for(message_id)
    if payload.get("fromMe"):
        speaker = (
            "<operator>" if authors.author(message_id) == AUTHOR_OPERATOR else "<bot>"
        )
    else:
        name = display_name(sender)
        ordinal = registry.known_ordinal(sender)
        speaker = speaker_token(event, name=name, ordinal=ordinal if ordinal else None)
    return format_record(
        time_label=time_label(event, tz_name),
        speaker=speaker,
        serial=serial,
        kind=record_kind(event),
        body=str(payload.get("body", "") or "").strip(),
    )
