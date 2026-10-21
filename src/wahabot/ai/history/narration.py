import json
import re
from typing import Any, cast

from wahabot.ai.messages import (
    EMOJI_CHARS,
)

INBOUND_ID_RE = re.compile(r"\[message id: ([^\]]+)\]")

SINGLE_EMOJI_RE = re.compile(rf"^\s*(?:[{EMOJI_CHARS}])\s*$")


def is_single_emoji(reply: str) -> bool:
    """True when *reply* is exactly one emoji and nothing else."""
    return bool(SINGLE_EMOJI_RE.match(reply))


EMOJI_NARRATION_RE = re.compile(
    "".join(
        (
            r"^\s*",
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF",
            r"[\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF",
            r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]",
            r"\U0000FE0F?[\U0001F3FB-\U0001F3FF]?",
            r"\s*(?:[\n·|—-]+\s*)?",
            r"(?:i\s+|ich\s+)?",
            r"(?:already\s+|bereits\s+|schon\s+)?",
            r"(?:habe\s+|hab\s+)?",
            r"(?:mit\s+\S+\s+)?",
            r"(?:",
            r"reacted|reaccion(?:é|o|amos)|reacted with|",
            r"he\s+reacted|he\s+reaccionado|",
            r"envié|envió|enviado|mandé|mandó|mandado|",
            r"sent|answered|respondí|respondió|",
            r"replied|replió|",
            r"reagiert|habe\s+reagiert|",
            r"geschickt|gesendet|",
            r"geantwortet|habe\s+geantwortet|",
            r"reagierte",
            r")\b",
            r".*",
        )
    ),
    re.IGNORECASE | re.DOTALL,
)


def is_emoji_narration(reply: str) -> bool:
    """True when *reply* is one emoji plus a report about that action.

    ``👍\\nReaccioné con 👍 al matiz de Bruno, sin escribir por el
    modo silencio.`` — the model narrated the reaction it meant to
    deliver instead of calling the tool (or after a tool round). The
    emoji half makes it *look* like a lone-emoji reaction; the words
    half is a self-report no chat member asked for, in any language.
    Delivery turns this into silence (the handler's lone-emoji path
    cannot run — the emoji rides narration, not a message), and
    storage filters it so the model's self-history cannot re-teach
    the pattern.
    """
    return bool(EMOJI_NARRATION_RE.match(reply.strip()))


SILENCE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^no response\b",
        r"^no reply\b",
        r"^nothing to (add|say)\b",
        r"^nothing (more|else) to (add|say|do)\b",
        r"^(i'?ll |i will |i'?m )?(stay|staying|remain|choosing to stay)[\s'-]*silent\b",
        r"^stay_silent\b",
        r"^(i'?ll |i will )?(stay|keep) (quiet|out of (this|it|the conversation))\b",
        r"^silence[.!…]?$",
        r"^\(silence\)$",
        r"^not (addressed|directed) (to|at) me\b",
        r"^(i'?ll |i will )?say nothing\b",
        r"^i (already )?(reacted|replied|sent|answered)\b[^.]*(so )?i'?m done\b",
        "".join(
            (
                r"^i (already )?(reacted|replied|sent|answered)\b[^.]*",
                r"(so )?(there'?s?|there is) nothing (more|else|left) (to )?(add|say|do)",
            )
        ),
        r"^i'?m done (here|with this)\b",
        r"^sin respuesta[.!…]?$",
        r"^(no tengo )?nada( más)? que (añadir|decir)[.!…]?$",
        r"^(me |prefiero )?(quedo|quedarme) (callado|callada|en silencio)[.!…]?$",
        r"^silencio[.!…]?$",
        r"^\(silencio\)$",
        r"^no diré nada\b",
        r"^no voy a (responder|contestar)\b",
        r"^keine antwort[.!…]?$",
        r"^keine (antwort|rückmeldung) hierzu[.!…]?$",
        "".join(
            (
                r"^(ich )?(habe )?nichts( (mehr|weiter))?",
                r"( zu (sagen|ergänzen|hinzuzufügen))?[.!…]?$",
            )
        ),
        r"^nichts hinzuzufügen[.!…]?$",
        r"^ich bleibe (still|stumm|leise)[.!…]?$",
        r"^bleib(?:e|t)? (still|stumm)[.!…]?$",
        r"^stille[.!…]?$",
        r"^\(stille\)$",
        r"^ich sage nichts[.!…]?$",
        r"^ich (werde |will )?nicht (antworten|antworten werde|reagieren)\b",
        r"^nicht an mich (gerichtet|addressiert)\b",
        r"^ich bin fertig (hier|damit)\b",
        "".join(
            (
                r"^ich habe (bereits )?(reagiert|geantwortet|geschickt)",
                r"[,.]?\s*(damit )?bin ich fertig\b",
            )
        ),
    )
)


def is_silence_narration(reply: str) -> bool:
    """True when *reply* narrates a silence instead of being one.

    Stripped of surrounding whitespace/quotes/parentheses and matched
    case-insensitively against the silence-meta patterns (English,
    Spanish and German — the chat's languages); anything the model
    actually wanted to say still goes through.
    """
    cleaned = reply.strip().strip("\"'`()").strip()
    return any(pattern.search(cleaned) for pattern in SILENCE_PATTERNS)


def is_error_narration(reply: str) -> bool:
    """True when *reply* is an error payload, not a chat answer.

    Small models sometimes *write* an API error as their reply — e.g.
    a made-up ``{"error": {"message": "resource exhausted …", "type":
    "upstream_error", "code": "resource_exhausted"}}`` naming a provider
    the bot never used. Whatever the model's reason (pattern-
    completing text it has seen), the result must never reach the
    chat. Only near-JSON bodies whose top level is an ``error`` object
    match; genuine prose answers never do.
    """
    stripped = reply.strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError, ValueError:
        return False
    if not isinstance(data, dict):
        return False
    error = cast(dict[str, Any], data).get("error")
    return isinstance(error, dict) and any(
        key in error for key in ("message", "code", "type", "status")
    )


def narration_kind(reply: str) -> str:
    """The narration class of *reply*: "error", "emoji", "silence", or "".

    The one classifier behind :func:`is_error_narration`,
    :func:`is_emoji_narration` and :func:`is_silence_narration` — a
    single dispatch whose *kind* survives to the caller, so delivery
    and storage not only agree that a reply is narration (the drop),
    they can log — and later act on — *which* failure mode the model
    produced. An empty string means the reply is chat-visible text.
    """
    text = str(reply or "").strip()
    if not text:
        return ""
    if is_error_narration(text):
        return "error"
    if is_emoji_narration(text):
        return "emoji"
    if is_silence_narration(text):
        return "silence"
    return ""


_NOTE_VERBS_EN = "".join(
    (
        r"(?:sent|replied|reacted|answered|responded|forwarded|shared|declined|",
        r"told|confirmed|acknowledged|delivered|handled|escalated|notified|",
        r"recorded|saved|wrote|drafted|posted|kept)",
    )
)

_NOTE_VERBS_DE = (
    r"(?:geantwortet|reagiert|geschickt|gesendet|hingewiesen|verwiesen|informiert)"
)

_NOTE_VERBS_ES = "".join(
    (
        r"(?:envié|envió|respondí|respondió|reaccioné|reaccionó|contesté|contestó|",
        r"mandé|mandó|compartí|compartió|confirmé|confirmó|informé|informó)",
    )
)

_NOTE_REPORT_RE = re.compile(
    "".join(
        (
            r"^(?:",
            _NOTE_VERBS_EN,
            r"\b[^.!?]*?\b",
            r"|auf\s+\S+?\s+(?:habe\s+|hab(?:e)?\s+)?[^.!?]*?\b",
            _NOTE_VERBS_DE,
            r"\b",
            r"|",
            _NOTE_VERBS_ES,
            r"\b[^.!?]*?\b",
            r")",
        )
    ),
    re.IGNORECASE,
)

_NOTE_CUE_RE = re.compile(
    "".join(
        (
            r"\b(?:the message|the reply|the request|the chat|the conversation|",
            r"instead of|als antwort statt|statt einer nachricht|als reaktion statt|",
            r"as requested|as asked|as he|as she|as they|",
            r"no further|anymore|private conversation|",
            r"a final|a short|a text(?:-based)?|a confirmation|a mini-lesson|",
            r"a lesson|a spanish mini-lesson|that i|",
            r"mit trockenem selbstkommentar|mit einem scherz|",
            r"scherz über|frage, ob|frage, wie)\b",
        )
    ),
    re.IGNORECASE,
)


def is_trailing_note_text(text: str) -> bool:
    """True when *text* reads as a private self-report (a wrap-up note).

    Only the wrap-up grammar qualifies: a third-person action report
    about the run ("Sent Tobias a final confirmation …", "Auf Roys
    Frage … geantwortet") that also carries a self-reference cue. The
    cue requirement and the third-person restriction are what keep a
    genuine reply line from being deleted — "I sent you the PDF" and
    "Sent the invoice and the receipt" are customer answers, not
    notes.
    """
    line = text.strip()
    return bool(_NOTE_REPORT_RE.match(line) and _NOTE_CUE_RE.search(line))


def strip_trailing_note(reply: str) -> str:
    """*reply* without an appended self-report line, or *reply* unchanged.

    Only a reply of two or more lines whose *last* line reads as a
    self-report is affected; the genuine reply above the note is kept
    (the note itself is dropped, never sent). A single line is never
    stripped — a chat message may legitimately open with "Sent".
    """
    lines = reply.rstrip().splitlines()
    if len(lines) < 2 or not is_trailing_note_text(lines[-1]):
        return reply
    return "\n".join(lines[:-1]).rstrip()


def chat_visible_text(content: Any) -> str:
    """The chat-visible text of *content*.

    Empty when the model narrated instead of answering:

    One definition of "this text would reach the chat" shared by both
    ends of a run — delivery (``final_reply``) and storage
    (``remember``) — so a leaked ``stay_silent`` token, an invented
    error payload, or a private wrap-up note appended to a reply is
    filtered at the source and can never drift back into the model's
    self-history (the exact string it pattern-completes on). A genuine
    reply keeps its words and loses only the trailing note. Thinking
    blocks are ignored: only the *text* blocks decide visibility.
    """
    reply = str(content or "").strip() if content is not None else ""
    if not reply:
        return ""
    if narration_kind(reply):
        return ""
    return strip_trailing_note(reply)


def inbound_message_id(incoming: str) -> str:
    """The serialized id of an inbound turn's ``[message id: …]`` note.

    Empty when the turn carries no note — operator commands have none,
    so they are never treated as redeliveries.
    """
    match = INBOUND_ID_RE.search(incoming)
    return match.group(1).strip() if match else ""
