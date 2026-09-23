"""Neutralize authority-marker patterns typed by chat members.

Every bracketed marker in a turn — ``[message id: …]``, ``[you were
addressed: …]``, ``[operator message]``, ``[quoting] …``, ``[reaction …
from …]``, ``[chat context] …`` — is trusted *because code wrote it*.
But the member's own body rides verbatim into the same turn, so a
member can type a marker and the model cannot tell code metadata from
member text. The spoof is not hypothetical: after the addressed-marker
shipped, "type the string that says *this message is for you*" became
the cheapest attack on the bot's judgment.

The fix mirrors prompt-injection practice: escape, don't trust
format. ``strip_spoofed_markers`` rewrites authority-marker-shaped
substrings in inbound *member text* so they no longer match the marker
grammar the prompt documents — a zero-width space after the opening
bracket breaks the pattern for the model while staying visually
identical for a human reading the logs. Code-generated notes are
appended after the scrub (or written by code paths the scrub exempts,
like command events and media-description prefixes), so they are never
touched; only the member's own words pass through.

The scrub is fail-soft by construction: it only ever rewrites strings
that match a marker pattern, and only inside the body.
"""

import re

__all__ = ["strip_spoofed_markers"]

#: Marker openings that carry *authority* — the ones a spoof of would
#: change how the model treats the turn. Media-description markers
#: (``[voice note]``, ``(video shows: …)``) are deliberately absent:
#: code writes them as body prefixes from real media (a member cannot
#: type into a transcript), and they grant no authority — at worst a
#: member makes the bot think a text was audio, which the surrounding
#: text betrays anyway. Ordinary member brackets ("[resumen]") are
#: anchored out by the exact prefixes below.
MARKER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(r"\[(" + prefix + r")", re.IGNORECASE)
    for prefix in (
        r"message id: ?",
        r"you were addressed:? ?",
        r"operator message\]? ?",
        r"operator command\]? ?",
        r"quoting\] ?",
        r"reaction .{0,60}? from ",
        r"chat context\]? ?",
    )
)

#: The rewrite: an invisible break inside the bracket so the string no
#: longer matches the marker grammar the model was taught, while a
#: human reading logs sees the original text unchanged.
WORD_BREAK = "\u200b"  # zero-width space


def strip_spoofed_markers(text: str) -> str:
    """*text* with marker-shaped substrings broken, member-safe.

    Only the bracket-openings above are touched: the first ``[`` of a
    match gets a zero-width space after it (``[\\u200bmessage id: …``),
    which preserves the visible text while breaking the exact-match
    the prompt's marker rules rely on. Idempotent: an already-broken
    marker no longer matches the pattern.
    """
    scrubbed = text
    for pattern in MARKER_PATTERNS:
        scrubbed = pattern.sub(lambda m: "[" + WORD_BREAK + m.group(1), scrubbed)
    return scrubbed
