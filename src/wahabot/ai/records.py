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

import re
from collections.abc import Callable, Iterable, Sequence

#: The addressing verdict, policy line and attach-slot vocabulary live in the
#: session prompt; the renderer only assembles the line.
CONTINUATION_PREFIX = "> "
PAYLOAD_PREFIX = "  · "

#: A record head: ``[[hh:mm speaker #m<n>]] kind flags...``.
RECORD_HEAD_RE = re.compile(r"^(\[\[[^\]]*#m(\d+)\]\])\s+(\S+)(.*)$")

#: A reference to an earlier record from a flag or payload.
REFERENCE_RE = re.compile(r"(?:quote=#m|to: #m)(\d+)")


def record_serials(text: str) -> list[int]:
    """Every record serial in *text*, in order."""
    return [
        int(match.group(2))
        for line in text.splitlines()
        if (match := RECORD_HEAD_RE.match(line))
    ]


def apply_skipped(text: str, skipped: set[int]) -> str:
    """Insert the prompt-time ``skipped`` flag on the chosen records.

    The stored turn never carries this overlay — a silent run saw the
    records but did not answer them, and only the next prompt needs to say
    so. Insertion follows the grammar's flag order (media slots, then
    state), so an overlaid line is indistinguishable from one rendered
    skipped in the first place.
    """
    if not skipped:
        return text
    out: list[str] = []
    for line in text.splitlines():
        match = RECORD_HEAD_RE.match(line)
        if match is None or int(match.group(2)) not in skipped:
            out.append(line)
            continue
        head, kind, rest = match.group(1), match.group(3), match.group(4)
        flags = rest.split()
        if "skipped" not in flags:
            index = 0
            while index < len(flags) and (
                flags[index] == "addressed"
                or flags[index].startswith(("image#", "video#", "album#"))
            ):
                index += 1
            flags.insert(index, "skipped")
        tail = f" {kind}" + (f" {' '.join(flags)}" if flags else "")
        out.append(f"{head}{tail}")
    return "\n".join(out)


def record_blocks(text: str) -> list[tuple[int, str]]:
    """Split *text* into ``(serial, block)``; serial 0 for non-record lines.

    A record owns its continuation and payload lines; anything that is not
    a record head (a stray header or policy line) is kept as its own block
    so a cap never swallows context it cannot classify.
    """
    blocks: list[tuple[int, str]] = []
    for line in text.splitlines():
        match = RECORD_HEAD_RE.match(line)
        if match is not None:
            blocks.append((int(match.group(2)), line))
        elif blocks and (
            line.startswith(CONTINUATION_PREFIX) or line.startswith(PAYLOAD_PREFIX)
        ):
            serial, block = blocks[-1]
            blocks[-1] = (serial, f"{block}\n{line}")
        else:
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
    """Keep the current turn, the newest *window* older records and quotes.

    The current burst is always kept; older records are selected newest
    first up to *window*, and any record an older selected record quotes
    or reacts to is kept too. Then the oldest selected record is dropped
    one at a time until the result fits *max_tokens* — never the newest
    record, so the turn always keeps the message being answered.
    """
    blocks = record_blocks(text)
    serials = [serial for serial, _ in blocks if serial]
    if len(serials) <= 1:
        return text
    current = {int(serial) for serial in keep_serials}
    kept = select_blocks(blocks, text, current, window)
    kept = trim_blocks(kept, current, max_tokens, token_counter)
    return "\n".join(block for _, block in kept)


def select_blocks(
    blocks: list[tuple[int, str]], text: str, current: set[int], window: int
) -> list[tuple[int, str]]:
    """Select the current burst, the newest *window* older records and quotes."""
    serials = [serial for serial, _ in blocks if serial]
    selected: set[int] = set(current)
    older = [serial for serial in serials if serial not in current]
    selected.update(older[-max(window, 0) :])
    selected.update(referenced_serials(text) & set(serials))
    selected.add(serials[-1])
    return [
        (serial, block) for serial, block in blocks if not serial or serial in selected
    ]


def trim_blocks(
    blocks: list[tuple[int, str]],
    current: set[int],
    max_tokens: int,
    token_counter: Callable[[str], int],
) -> list[tuple[int, str]]:
    """Drop the oldest selected records until the turn fits the token ceiling."""
    kept = blocks
    while (
        token_counter("\n".join(block for _, block in kept)) > max_tokens
        and len([serial for serial, _ in kept if serial]) > 1
    ):
        kept = drop_oldest(kept, current)
    return kept


def drop_oldest(
    blocks: list[tuple[int, str]], current: set[int]
) -> list[tuple[int, str]]:
    """Drop the oldest block that is not the newest record.

    Non-current records go first; within the current burst the oldest is
    dropped only when nothing older is left. The final record always stays.
    """
    records = [index for index, (serial, _) in enumerate(blocks) if serial]
    if len(records) <= 1:
        return blocks
    drop = next(
        (index for index in records[:-1] if blocks[index][0] not in current),
        records[0],
    )
    return blocks[:drop] + blocks[drop + 1 :]


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
    formatted, one per line under the record. ``serial=0`` omits the
    ``#m<n>`` token: the account's own output has no arrival serial, so a
    ``read_chat`` view of a bot message renders ``[[hh:mm <bot>]]``.
    """
    token = f" #m{serial}" if serial else ""
    head = f"[[{time_label} {speaker}{token}]] {kind}"
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
