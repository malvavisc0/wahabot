"""Render inbound messages as records — the Phase 1 turn grammar.

A pure module: callers pass already-resolved primitives (a formatted time
label, a speaker token, a kind, a body, flag strings) and get text back.
Nothing here reads a WAHA event, the roster, the clock or the disk, so the
grammar is testable in isolation and the same renderer serves the stored
turn, the ``read_chat`` view and the golden suite.

The grammar (one line per message, plus renderer-written payload lines)::

    [[<hh:mm> <speaker> #m<n>]] <kind> <flags> :: <body>
    > <body continuation, escaped>
      · <machine payload>

Only the renderer writes the three line-anchored shapes — ``[[`` at column 0,
``> `` continuations, ``  · `` payloads — so member text can contain those
characters but never that shape (there is no denylist to maintain).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

#: The addressing verdict, policy line and attach-slot vocabulary live in the
#: session prompt; the renderer only assembles the line.
CONTINUATION_PREFIX = "> "
PAYLOAD_PREFIX = "  · "


def escape_body(body: str) -> str:
    """Escape a member body so it can never start a line that looks like a record.

    The record's first body line rides after the renderer's own `` :: ``, so
    it cannot fake an envelope. Every continuation line is prefixed
    ``> `` unconditionally — a line that already starts ``> `` becomes
    ``> > ...`` — so a body can carry the characters but never the shape.
    """
    lines = body.split("\n")
    escaped = [lines[0], *(f"{CONTINUATION_PREFIX}{line}" for line in lines[1:])]
    return "\n".join(escaped)


def format_record(
    *,
    time_label: str,
    speaker: str,
    serial: int,
    kind: str,
    flags: Sequence[str] = (),
    body: str = "",
    payloads: Sequence[str] = (),
) -> str:
    """One record line (plus any payload lines) for one inbound message.

    ``flags`` are the presence-only tokens — media slots (``image#1``), then
    state (``addressed``, ``skipped``, ``edited``, ``deleted``), then
    ``quote=…`` / ``to: …`` — joined in the order given. ``body`` is escaped;
    an empty body drops the `` :: `` tail entirely (a voice note or a bare
    media record). ``payloads`` are renderer-written machine lines, already
    formatted, one per line under the record.
    """
    head = f"[[{time_label} {speaker} #m{serial}]] {kind}"
    if flags:
        head = f"{head} {' '.join(flags)}"
    line = f"{head} :: {escape_body(body)}" if body else head
    if not payloads:
        return line
    return "\n".join([line, *(f"{PAYLOAD_PREFIX}{payload}" for payload in payloads)])


def format_header(
    *,
    count: int,
    newest_serial: int,
    addressed: bool,
    backlog: bool,
    policy: str = "",
) -> str:
    """The prompt-time turn header, naming the newest record and the verdict.

    ``count`` is the number of records in the turn and ``newest_serial`` their
    highest ``#m<n>``. ``policy`` — the one-line answering rule for an
    unaddressed group turn — renders on its own line below the header.
    """
    plural = "s" if count != 1 else ""
    line = (
        f"[turn: {count} message{plural} | newest #m{newest_serial} | "
        f"addressed: {'yes' if addressed else 'no'} | "
        f"backlog: {'yes' if backlog else 'no'}]"
    )
    return f"{line}\n{policy}" if policy else line


def render_turn(records: Iterable[str], *, header: str = "") -> str:
    """Join a turn's records, with the optional header on top."""
    parts = [header, *records] if header else list(records)
    return "\n".join(parts)
