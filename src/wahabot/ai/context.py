"""Reply context rendering and the agent entrypoint."""

import asyncio
import datetime
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from llama_index.core.base.llms.types import ChatMessage, ImageBlock
from llama_index.core.workflow import Context
from loguru import logger

from wahabot.ai.history import (
    chat_visible_text,
    is_error_narration,
    is_silence_narration,
    is_single_emoji,
)
from wahabot.ai.messages import (
    addressed_note,
    jid_string,
    message_replies_to,
)
from wahabot.ai.scrub import strip_spoofed_markers
from wahabot.ai.tools.url_images import fetch_url_images, image_urls
from wahabot.ai.tools.whatsapp import (
    RunTarget,
    bind_target,
    reset_target,
    sender_names,
)
from wahabot.ai.vision import image_caption, image_noun
from wahabot.ai.workflow import FunctionCallingAgentWorkflow
from wahabot.core.cache import TtlCache
from wahabot.core.host import host_context
from wahabot.core.jid import roster_entries
from wahabot.core.models import WahaEvent
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings

__all__ = [
    "final_reply",
    "handle_message",
    "is_error_narration",
    "is_silence_narration",
    "is_single_emoji",
    "operator_tools_pass",
    "own_identity_pass",
    "participant_names",
    "render_system_prompt",
    "reply_context",
    "reply_context_section",
    "sender_tag",
    "turn_body",
]


def render_system_prompt(
    prompt: str,
    tz_name: str = "UTC",
    bot_name: str | None = None,
    goal: str = "",
    own_jid: str = "",
    own_lid: str = "",
    operator_name: str = "",
) -> str:
    """Substitute date/time/name/host placeholders in the system prompt.

    When ``goal`` is non-empty, it is prepended as a ``Goal:`` block so
    the model always starts from the bot's intended purpose.

    Placeholders (all but ``{{bot_name}}`` and ``{{host}}`` use the
    ``tz_name`` timezone):

    - ``{{now}}`` / ``{{datetime}}`` — full timestamp, e.g. ``2026-09-02 14:05 UTC``
    - ``{{date}}`` — date only, e.g. ``2026-09-02``
    - ``{{time}}`` — time only, e.g. ``14:05``
    - ``{{tz}}`` — the timezone name, e.g. ``UTC``
    - ``{{bot_name}}`` — the bot's display name, e.g. ``Kai``
    - ``{{operator_name}}`` — the human owning the account the bot
      runs on, e.g. ``Ada``; empty renders as ``the operator``
    - ``{{host}}`` — a summary of the machine (OS, Python, Node, shell)
    - ``{{own_jid}}`` / ``{{own_lid}}`` / ``{{own_identities}}`` — the
      bot's own WhatsApp ids (see :func:`own_identity_pass)
    - ``{{operator_tools}}`` — the tools and `chat`-parameter reach
      reserved to operator commands (see :func:`operator_tools_pass)

    Unknown/invalid timezone names fall back to UTC.
    """
    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError, ValueError, OSError:
        tz = datetime.UTC

    now = datetime.datetime.now(tz=tz)
    replacements = {
        "{{now}}": now.strftime("%Y-%m-%d %H:%M %Z"),
        "{{datetime}}": now.strftime("%Y-%m-%d %H:%M %Z"),
        "{{date}}": now.strftime("%Y-%m-%d"),
        "{{time}}": now.strftime("%H:%M"),
        "{{tz}}": tz_name,
        "{{bot_name}}": bot_name or "the bot",
        "{{operator_name}}": operator_name.strip() or "the operator",
        "{{host}}": host_context(),
        "{{operator_tools}}": operator_tools_pass(),
    }
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
        goal = goal.replace(key, value)
    prompt = own_identity_pass(prompt, own_jid, own_lid)
    goal = goal.strip()
    if goal:
        return f"Goal: {goal}\n\n{prompt}"
    return prompt


#: Own-identity placeholders the prompt may carry; every one of them
#: is dropped line-wise when the bot's identity is unknown.
_OWN_PLACEHOLDERS = ("{{own_jid}}", "{{own_lid}}", "{{own_identities}}")


def operator_tools_pass() -> str:
    """The operator-command tool reach, rendered into the system prompt.

    One source of truth for which tools and cross-chat `chat` use is
    reserved to `[operator command]` turns: the fence itself is
    mechanical (:func:`fenced_chat` refuses everything else), this
    text only tells the model what the rules are so it does not burn
    calls — or promise a participant a delivery the fence would
    refuse. Kept here instead of the tool descriptions so the rule is
    stated once, not repeated per tool.
    """
    return (
        "Cross-chat reach — passing `chat` to a tool, `read_chat` with "
        "`mode=recent` — is reserved to `[operator command]` turns. "
        "`read_chat` with `mode=resolve` matches the current chat's "
        "participants on any turn (use it for `mentions`); the "
        "operator's contact book opens on operator commands alone. On "
        "any other turn those calls are refused: a participant asking "
        "you to message, react to, quote or read anyone outside the "
        "current chat gets a tool refusal — never promise deliveries "
        "you cannot make."
    )


def own_identity_pass(prompt: str, own_jid: str, own_lid: str) -> str:
    """Substitute the bot's own identity placeholders in *prompt*.

    ``{{own_jid}}``/``{{own_lid}}`` become the ids; ``{{own_identities}}``
    becomes the two-id list, e.g. `` `4915…@c.us` or `4915…@lid` ``, for
    a sentence like "You are {{own_identities}} — …". When no identity
    is known (startup fetch failed, mid-recovery) the placeholder is
    never left verbatim or stale: every line carrying one of the
    placeholders is dropped wholesale, so the prompt has no dangling
    text and never claims the wrong JID.
    """
    ids = [jid for jid in dict.fromkeys(j for j in (own_jid, own_lid) if j)]
    if not ids:
        return _drop_placeholder_lines(prompt)
    prompt = prompt.replace("{{own_jid}}", own_jid or "unknown")
    prompt = prompt.replace("{{own_lid}}", own_lid or "unknown")
    return prompt.replace("{{own_identities}}", " or ".join(f"`{jid}`" for jid in ids))


def _drop_placeholder_lines(prompt: str) -> str:
    """*prompt* without any line carrying an own-identity placeholder."""
    return "".join(
        line
        for line in prompt.splitlines(keepends=True)
        if not any(placeholder in line for placeholder in _OWN_PLACEHOLDERS)
    )


def sender_tag(event: WahaEvent, names: dict[str, str] | None = None) -> str:
    """The sender's display identity for the agent prompt, e.g. ``[Ana]``.

    Prefers the WhatsApp display name (``_data.notifyName``, then a
    top-level ``notifyName`` for engines that hoist it); falls back to
    the participant/author id so a group turn is never anonymous.
    Returns an empty tag when nothing is known (should not happen).

    Group turns render ``[Name <jid>]`` when the sender's JID is
    known: one string that is both the display identity and the
    mention handle the model copies into ``@<user-part>`` tokens /
    ``mentions`` — no suffix-parsing inference, no extra tool call.
    The JID comes from ``payload.participant``/``_data.author``
    (normalized through :func:`jid_string` — LID groups report
    participants as JID objects) and the name from *names*, the shared
    roster. Fallbacks: unknown name → ``[<full-jid>]`; known name, no
    JID → today's ``[Name]``. DM turns keep ``[Name]`` — the chat
    partner needs no mentioning, and the JID buys nothing there.
    """
    data = event.payload.get("_data", {})
    name = str(data.get("notifyName") or event.payload.get("notifyName") or "").strip()
    participant = jid_string(event.payload.get("participant") or data.get("author"))
    if not participant:
        return f"[{name}]" if name else ""
    if not str(event.payload.get("from", "")).endswith("@g.us"):
        return f"[{name}]" if name else f"[{participant}]"
    return group_sender_tag(name, participant, names or {})


def group_sender_tag(name: str, participant: str, names: dict[str, str]) -> str:
    """A group turn's ``[Name <jid>]`` tag with its two fallbacks."""
    resolved = name or names.get(participant, "")
    if resolved:
        return f"[{resolved} <{participant}>]"
    return f"[{participant}]"


def message_id_note(event: WahaEvent) -> str:
    """The incoming message's serialized id as a prompt note.

    The id is what ``send_message(reply_to=…)`` and ``react_to_message``
    need to quote or react to *this* message — without it in the turn
    text the model can only quote ids it fetched itself.
    """
    data = event.payload.get("_data", {})
    message_id = str(data.get("id", {}).get("_serialized", "")) if data else ""
    if not message_id:
        message_id = str(event.payload.get("id", ""))
    return f"\n[message id: {message_id}]" if message_id else ""


def reply_context(
    message_reply: dict[str, Any] | None,
    participant_names: dict[str, str] | None = None,
) -> str:
    """Render the quoted message as context for the agent.

    Empty/None input yields nothing; the caller decides whether to
    include a ``Reply context`` block in the prompt. *participant_names*
    maps chat JIDs to display names so the quoted sender renders as a
    name, not a raw ``@lid`` JID.
    """
    if not message_reply:
        return ""
    sender = quoted_participant(message_reply, participant_names)
    description = reply_description(message_reply)
    if sender and description:
        return f'{sender}: "{description}"'
    return sender or description


def quoted_participant(
    message_reply: dict[str, Any],
    participant_names: dict[str, str] | None = None,
) -> str:
    """The quoted message's sender display name; empty when unknown.

    Prefers the quoted message's own ``_data.notifyName``, then the
    chat's participant roster (*participant_names*), then the raw
    participant/author id. With a resolved name the render is
    ``Name <jid>`` — the same mention-handle shape the sender tags
    carry, so the model can copy the id without a tool call; without
    one it falls back to the bare user part (a bare number reads like
    an id, a full JID reads like noise). The participant field may be
    a JID object (LID groups), so it is normalized via
    :func:`jid_string` before use.
    """
    data = message_reply.get("_data", {})
    participant = message_reply.get("participant") or data.get("author") or ""
    jid = jid_string(participant)
    name = str(data.get("notifyName") or "").strip()
    if not name and jid and participant_names:
        name = participant_names.get(jid, "")
    if name:
        return f"{name} <{jid}>" if jid else name
    return jid.split("@", 1)[0] if jid else ""


def reply_description(message_reply: dict[str, Any]) -> str:
    """Describe the quoted message body/media, truncated for the prompt.

    The quoted body is member text riding inside a code-rendered
    ``[quoting]`` marker — scrubbed so a member cannot smuggle a
    spoofed marker in through the quote they send.
    """
    media = message_reply.get("hasMedia") and message_reply.get("media")
    if media:
        mimetype = media.get("mimetype", "media")
        filename = media.get("filename") or ""
        description = f"{mimetype} {filename}".strip()
    else:
        description = strip_spoofed_markers(
            str(message_reply.get("body", "") or "").strip()
        )
    return description[:400]


def reply_context_section(
    message_reply: dict[str, Any] | None,
    participant_names: dict[str, str] | None = None,
) -> str:
    """Message text with the quoted/replied-to message attached as context."""
    if not message_reply:
        return ""
    line = reply_context(message_reply, participant_names)
    return f"\n[quoting] {line}" if line else ""


#: Per-chat participant roster cache (JID → display name), refreshed
#: at most once per TTL. Quoted messages carry no sender name (WAHA's
#: ``replyTo._data`` has only type/kind/body), so the roster is the only
#: way to render "Ada" instead of "000000000000000@lid".
ROSTER_TTL_S = 3600
_ROSTER_CAP = 1000
roster_cache: TtlCache[tuple[str, str], dict[str, str]] = TtlCache(
    ROSTER_TTL_S, _ROSTER_CAP
)


def participant_names(
    waha: WahaClient | None, session: str, chat_id: str
) -> dict[str, str]:
    """JID → display name for a chat's participants, cached per chat.

    Two sources, both fail-soft: the overview roster first, then — for
    every roster JID still missing a name (LID groups carry bare JIDs
    only) — a backfill from the chat's recent messages via
    :func:`sender_names`, whose ``notifyName`` walk is the only place
    WAHA surfaces display names there. Any error yields fewer names;
    quoted senders fall back to their bare id. Non-group chats skip the
    lookup — a DM partner's name already rides the sender tag.
    """
    if waha is None or not chat_id.endswith("@g.us"):
        return {}
    cached = roster_cache.get((session, chat_id))
    if cached is not None:
        return cached
    try:
        names = roster_names(waha.get_chat_overview(session, chat_id))
    except Exception as exc:
        logger.debug(
            "Participant roster fetch failed for {chat}: {exc}", chat=chat_id, exc=exc
        )
        names = {}
    # Roster names win; the message walk only backfills JIDs the roster
    # left bare — a recent sender's notifyName is better than a raw
    # number, never worse than a roster entry.
    for jid, name in sender_names(waha, session, chat_id).items():
        names.setdefault(jid, name)
    roster_cache.put((session, chat_id), names)
    return names


def roster_names(overview: dict[str, Any]) -> dict[str, str]:
    """Extract JID → name from a chat overview's participant list."""
    pairs = (roster_entry(entry) for entry in roster_entries(overview))
    return {jid: name for jid, name in pairs if jid and name}


def roster_entry(entry: Any) -> tuple[str, str]:
    """One participant's ``(jid, display_name)``; empty strings when absent."""
    if not isinstance(entry, dict):
        return "", ""
    item: dict[str, Any] = entry
    jid = str(item.get("id") or "")
    name = str(
        item.get("name") or item.get("pushname") or item.get("notifyName") or ""
    ).strip()
    return jid, name


async def handle_message(
    event: WahaEvent,
    agent: FunctionCallingAgentWorkflow,
    ctx: Context | None = None,
    image: dict[str, Any] | None = None,
    images: list[dict[str, Any]] | None = None,
    settings: Settings | None = None,
    waha: WahaClient | None = None,
    armed: bool = False,
    pinned_note: str = "",
    bot_name: str | None = None,
    bot_mention_regex: str | None = None,
    body_scrubbed: bool = False,
) -> tuple[str, RunTarget]:
    """Run the agent workflow over an incoming message event.

    Returns ``(reply, target)``: the run's final text and its delivery
    target (``sent``/``reacted`` latches), so the caller can tell a
    delivered run from a text reply without touching run-scoped state.

    ``pinned_note`` carries turn-scoped ground truth the *caller*
    resolved before the run (see :mod:`wahabot.ai.resolve`): it rides
    the turn text — never memory — so the model starts from facts a
    mixed history cannot infer (the real chat a command named, its real
    last message). Empty for every chat-run caller.

    ``bot_name``/``bot_mention_regex`` (the session config's identity
    pair) render the addressed-mention marker into a group turn whose
    text names the bot (:func:`addressed_note`): the wake gate already
    decided the message is for the bot, and the marker tells the model
    so it never re-derives — and misreads — that fact. Absent both, no
    marker (operator commands, tests).

    ``body_scrubbed`` says the caller already scrubbed the body's
    member text (the burst path, per member, before stamping its id
    notes) so :func:`turn_body` must not scrub again and break the
    code-stamped ids.

    ``image`` (single) or ``images`` (an album, already downloaded)
    carry image bytes (``data`` + ``mimetype``); they ride along as
    ``image_blocks`` on the run and are injected into the first LLM
    call only — memory stays text-only, so no megabyte payloads
    accumulate in the rolling buffer. Downloaded images also carry a
    one-line ``caption`` (set by the download path, outside the agent
    lock — see ``wahabot.ai.vision``); it goes *into* the user message
    text — ``(image shows: beer glass with bear foam)`` — so later
    rounds and later turns keep a text anchor once the pixels are
    gone; a missing caption falls back to the bare ``(image)`` marker.

    With ``settings.vision`` enabled, image URLs sniffed from the
    message text are fetched too (see ``wahabot.ai.tools.url_images``), so a
    bare link like "look at this https://host/pic.png" shows the model
    the picture as well.
    """
    chat_id = str(event.payload.get("from", ""))
    logger.info("Agent handling message from {chat_id}", chat_id=chat_id)
    # Roster before tag: group sender tags render ``[Name <jid>]`` via
    # the resolver, so the fetch must precede the tag build. One
    # fetch, same cache the quoting lines and reaction notes share.
    names = await asyncio.to_thread(participant_names, waha, event.session, chat_id)
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
    user_msg += reply_context_section(message_replies_to(event), names)
    user_msg += pinned_note
    user_msg += addressed_note(event, bot_name, bot_mention_regex)
    image_blocks = [
        ImageBlock(
            image=img["data"],
            image_mimetype=img.get("mimetype") or "image/jpeg",
        )
        for img in all_images
    ]
    # Wire-level evidence: Langfuse traces never show image blocks (the
    # OTel instrumentor serializes only the text-only `content`
    # property), so this log is the authoritative count of what rides
    # the first LLM call.
    logger.info("Attaching {n} image block(s) to agent run", n=len(image_blocks))
    # Bind this run's delivery target around the whole run: the workflow
    # engine schedules steps/tasks under this context, so the binding
    # propagates to every step and tool call of THIS run — concurrent
    # runs bind their own targets and never see each other's. The
    # binding resets before returning, so the *target itself* is
    # returned alongside the reply for the caller's latch checks.
    typing = (
        (settings.typing_presence_min_s, settings.typing_presence_max_s)
        if settings is not None
        else None
    )
    target = RunTarget(session=event.session, chat_id=chat_id, armed=armed, typing=typing)
    token = bind_target(target)
    try:
        result = await agent.run(input=user_msg, image_blocks=image_blocks, ctx=ctx)
    finally:
        reset_target(token)
    return final_reply(result), target


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
    if prescrubbed or event.event == "command":
        return raw
    return strip_spoofed_markers(raw)


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
    apart. This wrapper only adds the delivery-side log lines.
    """
    message = cast(ChatMessage, result.message)
    content = message.content
    reply = content.strip() if isinstance(content, str) else ""
    if not reply or chat_visible_text(reply):
        return chat_visible_text(reply)
    if is_silence_narration(reply):
        logger.debug("Filtering silence narration: {reply!r}", reply=reply)
        return ""
    logger.warning(
        "Dropping invented error payload as final reply: {reply!r}", reply=reply[:200]
    )
    return ""
