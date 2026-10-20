"""Render inbound messages as records — the message-construction grammar.

A pure module: callers pass already-resolved primitives (a formatted time
label, a speaker token, a kind, a body, flag strings) and get text back.
Nothing here reads a WAHA event, the roster, the clock or the disk, so the
grammar is testable in isolation and the same renderer serves the stored
turn, the ``read_chat`` view and the golden suite.

The grammar (one line per live message, plus renderer-written payload
lines)::

    [[<hh:mm> <speaker> #m<n>]] <kind> <flags> :: <body>
    > <body continuation, escaped>
      · <machine payload>

Quiet messages the bot read and did not answer collapse to one ledger line
instead, so an old backlog cannot recreate a wall of text::

    <hh:mm> <speaker> <gist> — #m<n>

Only the renderer writes the line-anchored shapes — ``[[`` at column 0,
``> `` continuations, ``  · `` payloads, and ledger lines — so member text
can contain those characters but never that shape (there is no denylist).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence

#: The addressing verdict, policy line and attach-slot vocabulary live in the
#: session prompt; the renderer only assembles the line.
CONTINUATION_PREFIX = "> "
PAYLOAD_PREFIX = "  · "
LEDGER_SEPARATOR = " — "

#: A record head: ``[[hh:mm speaker #m<n>]] kind flags...``. The serial is
#: anchored to the head's end: the renderer always writes it last, so a
#: speaker name carrying an inline ``[[… #m9]]`` cannot shift the parse.
RECORD_HEAD_RE = re.compile(r"^(\[\[.*#m(\d+)\]\])\s+(\S+)(.*)$")

#: A ledger line: ``hh:mm speaker <gist> — #m<n>`` (also ``dd.mm hh:mm``).
LEDGER_RE = re.compile(
    r"^(?P<time>\d{2}:\d{2}(?:\.\d{2})?|\d{2}\.\d{2} \d{2}:\d{2}) "
    + r"(?P<speaker>.+) — #m(?P<serial>\d+)$"
)

#: A reference to an earlier record from a flag or payload.
REFERENCE_RE = re.compile(r"(?:quote=#m|to: #m)(\d+)")

#: How many body words a mechanical ledger gist keeps.
GIST_WORDS = 8
GIST_CHARS = 120


def record_serials(text: str) -> list[int]:
    """Every live record serial in *text*, in order (ledger lines excluded)."""
    return [
        int(match.group(2))
        for line in text.splitlines()
        if (match := RECORD_HEAD_RE.match(line))
    ]


def ledger_serial(line: str) -> int:
    """The serial named by a ledger line, or 0."""
    match = LEDGER_RE.match(line)
    return int(match.group("serial")) if match else 0


def record_blocks(text: str) -> list[tuple[int, str]]:
    """Split *text* into signed blocks.

    A live record owns its continuation and payload lines and carries its
    positive serial; a ledger line carries its serial negated so the cap can
    tell a gist line from a live record. Anything unclassified is serial 0.
    """
    blocks: list[tuple[int, str]] = []
    for line in text.splitlines():
        match = RECORD_HEAD_RE.match(line)
        if match is not None:
            blocks.append((int(match.group(2)), line))
            continue
        ledger = ledger_serial(line)
        if ledger:
            blocks.append((-ledger, line))
            continue
        if blocks and (
            line.startswith(CONTINUATION_PREFIX) or line.startswith(PAYLOAD_PREFIX)
        ):
            serial, block = blocks[-1]
            blocks[-1] = (serial, f"{block}\n{line}")
            continue
        blocks.append((0, line))
    return blocks


def referenced_serials(text: str) -> set[int]:
    """Serials named by ``quote=#m<n>`` / ``to: #m<n>`` anywhere in *text*."""
    return {int(serial) for serial in REFERENCE_RE.findall(text)}


def cap_turn(
    text: str,
    *,
    keep_serials: Sequence[int] = (),
    window: int,
    max_tokens: int,
    token_counter: Callable[[str], int],
) -> str:
    """Keep the current turn, the newest *window* older records and its ledger.

    The current burst is always kept; older live records are selected newest
    first up to *window*, and any record an older selected record quotes or
    reacts to is kept too. Ledger lines are kept unless a live record for the
    same serial supersedes them. Then the oldest block — ledger first, because
    it is older — is dropped one at a time until the result fits
    *max_tokens*; the newest live record always stays.
    """
    blocks = record_blocks(text)
    live = [serial for serial, _ in blocks if serial > 0]
    if not live:
        return text
    current = {int(serial) for serial in keep_serials}
    kept = select_blocks(blocks, text, current, window)
    kept = trim_blocks(kept, max_tokens, token_counter)
    return "\n".join(block for _, block in kept)


def select_blocks(
    blocks: list[tuple[int, str]], text: str, current: set[int], window: int
) -> list[tuple[int, str]]:
    """Select the current burst, the newest *window* records, quotes and ledger."""
    live = [serial for serial, _ in blocks if serial > 0]
    selected: set[int] = set(current)
    older = [serial for serial in live if serial not in current]
    selected.update(older[-max(window, 0) :])
    selected.update(referenced_serials(text) & set(live))
    selected.add(live[-1])
    return [(serial, block) for serial, block in blocks if keep_block(serial, selected)]


def keep_block(serial: int, selected: set[int]) -> bool:
    """Whether a block survives selection.

    A ledger line is dropped only when a *kept* live record supersedes it;
    an unselected live record older than the window must not hide it.
    """
    if serial == 0:
        return True
    if serial > 0:
        return serial in selected
    return -serial not in selected


def trim_blocks(
    blocks: list[tuple[int, str]],
    max_tokens: int,
    token_counter: Callable[[str], int],
) -> list[tuple[int, str]]:
    """Drop the oldest selected blocks until the turn fits the token ceiling."""
    kept = blocks
    while (
        token_counter("\n".join(block for _, block in kept)) > max_tokens
        and len([serial for serial, _ in kept if serial]) > 1
    ):
        kept = drop_oldest(kept)
    return kept


def drop_oldest(blocks: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop the oldest block, never the newest live record.

    Ledger serials are older than the live window, so their absolute serial
    wins the comparison; the final live record is never a candidate.
    """
    serials = [serial for serial, _ in blocks if serial]
    if len(serials) <= 1:
        return blocks
    newest = max((serial for serial in serials if serial > 0), default=0)
    candidates = [
        index for index, (serial, _) in enumerate(blocks) if serial and serial != newest
    ]
    if not candidates:
        return blocks
    drop = min(candidates, key=lambda index: abs(blocks[index][0]))
    return blocks[:drop] + blocks[drop + 1 :]


def record_to_ledger(block: str, gist: str = "") -> str | None:
    """One record block as a ledger line, dropping its payload and body detail.

    The gist is flattened: a ledger line is one line, so a multi-line
    gist (however it arrived) cannot break the shape.
    """
    match = RECORD_HEAD_RE.match(block.splitlines()[0])
    if match is None:
        return None
    time_label, speaker = split_head(match.group(1))
    kind = match.group(3)
    body = record_body_text(block, match.group(4))
    words = escape_oneline(gist) or mechanical_gist(body, kind, block)
    return f"{time_label} {speaker} {words}{LEDGER_SEPARATOR}#m{match.group(2)}"


def split_head(head: str) -> tuple[str, str]:
    """``[[hh:mm Name #m1]]`` -> ``(hh:mm, Name)``; speaker spaces stay."""
    inner = head[2:-2]
    left, _, _ = inner.rpartition(" #m")
    time_label, _, speaker = left.partition(" ")
    return time_label, speaker


def record_body_text(block: str, rest: str) -> str:
    """The member text of a record block, including escaped continuation lines."""
    _, separator, first = rest.partition(" :: ")
    if not separator:
        return ""
    parts = [first, *continuation_text(block)]
    return " ".join(" ".join(parts).split())


def continuation_text(block: str) -> list[str]:
    """The body continuation lines of a block (payload lines excluded)."""
    return [
        line[len(CONTINUATION_PREFIX) :]
        for line in block.splitlines()[1:]
        if line.startswith(CONTINUATION_PREFIX)
    ]


def mechanical_gist(body: str, kind: str, block: str = "") -> str:
    """A deterministic one-line gist when no reader supplied one."""
    text = " ".join(body.split())
    if kind == "reaction":
        target = reaction_target(block)
        words = f"reacted {text or 'to a message'}"
        return f"{words} {target}".strip()[:GIST_CHARS]
    if text.startswith(("http://", "https://")):
        words = "shared a link"
    elif text:
        words = " ".join(text.split()[:GIST_WORDS])
        if len(text.split()) > GIST_WORDS:
            words = f"{words}…"
    else:
        words = dict(KIND_GISTS).get(kind, "sent a message")
    return words[:GIST_CHARS]


def reaction_target(block: str) -> str:
    """The ``to: …`` payload of a reaction record, if it has one."""
    for line in block.splitlines():
        if line.startswith(PAYLOAD_PREFIX):
            payload = line[len(PAYLOAD_PREFIX) :]
            if payload.startswith("to: "):
                return payload
    return ""


#: Immutable kind -> mechanical gist; a tuple keeps the module free of a
#: mutable global.
KIND_GISTS = (
    ("image", "sent an image"),
    ("video", "sent a video"),
    ("audio", "sent a voice note"),
    ("sticker", "sent a sticker"),
    ("document", "sent a document"),
    ("album", "sent photos"),
    ("link", "shared a link"),
    ("reaction", "sent a reaction"),
)


def ledger_turn(text: str, *, gist: str = "", newest_serial: int = 0) -> str:
    """Rewrite a quiet turn's records as ledger lines, keeping existing ones."""
    out: list[str] = []
    for serial, block in record_blocks(text):
        if serial > 0:
            line = record_to_ledger(block, gist if serial == newest_serial else "")
            if line is not None:
                out.append(line)
        elif block.strip():
            out.append(block)
    return "\n".join(out)


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
    state (``addressed``, ``edited``, ``deleted``), then ``quote=…`` /
    ``to: …`` — joined in the order given. ``body`` is escaped; an empty body
    drops the `` :: `` tail entirely (a voice note or a bare media record).
    ``payloads`` are renderer-written machine lines, one per line under the
    record. Speakers, flags and payloads are flattened to one line
    (:func:`escape_oneline`), so member-derived text inside them can never
    fake a record head or ledger line. ``serial=0`` omits the ``#m<n>``
    token: the account's own output has no arrival serial, so a
    ``read_chat`` view of a bot message renders ``[[hh:mm <bot>]]``.
    """
    token = f" #m{serial}" if serial else ""
    head = f"[[{time_label} {escape_oneline(speaker)}{token}]] {kind}"
    if flags:
        head = f"{head} {' '.join(escape_oneline(flag) for flag in flags)}"
    line = f"{head} :: {escape_body(body)}" if body else head
    if not payloads:
        return line
    return "\n".join(
        [line, *(f"{PAYLOAD_PREFIX}{escape_oneline(payload)}" for payload in payloads)]
    )


def escape_oneline(text: str) -> str:
    """Flatten *text* so it can never break a line-anchored shape.

    One renderer-written string — a speaker, a flag, a payload — must
    stay on its own line: a newline inside it would put member-derived
    text at column 0, where it could fake a record head or a ledger
    line. Newlines collapse to spaces.
    """
    return " ".join(text.split())


def format_header(
    *,
    count: int,
    newest_serial: int,
    addressed: bool,
    backlog: bool,
    policy: str = "",
) -> str:
    """The prompt-time turn header, naming the newest record and the verdict.

    ``count`` is the number of live records in the turn and ``newest_serial``
    their highest ``#m<n>``. Ledger lines are not live records and do not
    count. ``policy`` — the one-line answering rule for an unaddressed group
    turn — renders on its own line below the header.
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
