"""Compose prompt-time records from inbound WAHA events (Phase 1).

The pure grammar lives in :mod:`wahabot.ai.records`; this is the thin
WAHA-aware layer that maps a parsed event onto it — the record's kind, its
speaker token, and its presence flags. Callers pass what only they can
resolve (the display name or ``<member-n>`` ordinal, the ``#m<n>`` serial,
the time label, and any media/quote payloads), so the module stays free of
I/O and the roster.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from wahabot.ai.messages import extract_text, is_operator_event, message_kind
from wahabot.ai.records import format_record
from wahabot.core.models import WahaEvent

#: A body that is nothing but one URL is a ``link`` record.
BARE_URL_RE = re.compile(r"^https?://\S+$")


def record_kind(event: WahaEvent) -> str:
    """The record's ``kind``, derived from the message's own shape.

    A text body that is a bare URL is a ``link`` whether or not the
    download/caption later succeeds, so the kind is stable across runs;
    everything else is the WAHA kind unchanged.
    """
    kind = message_kind(event)
    if kind != "text":
        return kind
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


def build_record(
    event: WahaEvent,
    *,
    speaker: str,
    serial: int,
    time_label: str,
    addressed: bool = False,
    skipped: bool = False,
    body: str | None = None,
    media_slots: Sequence[str] = (),
    links: Sequence[str] = (),
    payloads: Sequence[str] = (),
) -> str:
    """One record line for *event*, in the §10.2 flag order.

    Flags assemble as media slots, then state (``addressed``, ``skipped``),
    then links (``quote=#m<n>`` / ``to: #m<n>``). ``body`` overrides the
    event's own text (an operator fold, a reaction note, a collapsed
    narration); ``payloads`` are the already-formatted machine lines only
    the caller can build (media analysis, a quote or reaction preview).
    """
    state = tuple(
        flag for flag, on in (("addressed", addressed), ("skipped", skipped)) if on
    )
    flags = (*media_slots, *state, *links)
    text = extract_text(event) if body is None else body
    return format_record(
        time_label=time_label,
        speaker=speaker,
        serial=serial,
        kind=record_kind(event),
        flags=flags,
        body=text or "",
        payloads=payloads,
    )
