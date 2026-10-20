"""Reply context rendering and the agent entrypoint."""

import asyncio
import datetime
from collections.abc import Sequence
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from llama_index.core.base.llms.types import ChatMessage, ImageBlock, MessageRole
from llama_index.core.workflow import Context
from loguru import logger

from wahabot.ai import records
from wahabot.ai.history import (
    chat_visible_text,
    is_emoji_narration,
    is_error_narration,
    is_silence_narration,
    is_single_emoji,
    narration_kind,
)
from wahabot.ai.messages import (
    OPERATOR_NOTE_PREFIX,
    TURN_HANDLED_KWARG,
    TURN_SERIALS_KWARG,
    bot_jids,
    is_operator_event,
    jid_string,
    message_kind,
    message_replies_to,
    reply_description,
    turn_addressed_note,
)
from wahabot.ai.scrub import strip_spoofed_markers
from wahabot.ai.serials import serial_store
from wahabot.ai.tools.url_images import fetch_url_images, image_urls
from wahabot.ai.tools.whatsapp import (
    RunTarget,
    bind_target,
    jids_in_text,
    reset_target,
    sender_names,
)
from wahabot.ai.turn_records import (
    LINK_VIDEO_MARKER_KEY,
    TRANSCRIPT_KEY,
    VIDEO_MARKER_KEY,
    RecordEntry,
    render_entries,
)
from wahabot.ai.vision import image_caption, image_noun
from wahabot.ai.workflow import FunctionCallingAgentWorkflow
from wahabot.core import images as image_store
from wahabot.core.access import CONFIG_MEMORY_TOKEN_LIMIT
from wahabot.core.cache import TtlCache
from wahabot.core.host import host_context
from wahabot.core.identity import AUTHOR_BOT, AUTHOR_OPERATOR, authors, display_name
from wahabot.core.jid import roster_entries
from wahabot.core.models import WahaEvent
from wahabot.core.notebook import (
    KIND_RULE,
    RULE_HARD_MUTE,
    RULE_MESSAGE_PREFIX,
    Notebook,
    notebook_operation,
)
from wahabot.core.tokens import text_tokens
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings

__all__ = [
    "final_reply",
    "handle_message",
    "is_emoji_narration",
    "is_error_narration",
    "is_silence_narration",
    "is_single_emoji",
    "narration_kind",
    "operator_tools_pass",
    "own_identity_pass",
    "participant_names",
    "render_system_prompt",
    "reply_context",
    "reply_context_section",
    "sender_tag",
    "turn_body",
    "turn_user_message",
]


def render_system_prompt(
    prompt: str,
    tz_name: str = "UTC",
    bot_name: str | None = None,
    goal: str = "",
    memory: str = "",
    own_jid: str = "",
    own_lid: str = "",
    operator_name: str = "",
) -> str:
    """Substitute date/time/name/host placeholders in the system prompt.

    When ``goal`` is non-empty, it is prepended as a ``# Goal``
    section so the model always starts from the bot's intended purpose.
    When ``memory`` is non-empty, it follows as a ``# Memory`` section
    with durable operator-maintained facts (at most 1024 tokens).

    Placeholders (all but ``{{bot_name}}`` and ``{{host}}`` use the
    ``tz_name`` timezone):

    - ``{{now}}`` / ``{{datetime}}`` — full timestamp, e.g. ``2026-09-02 14:05 UTC``
    - ``{{date}}`` — date only, e.g. ``Wednesday 2nd September, 2026``
    - ``{{time}}`` — time only, e.g. ``2:05 PM``
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
        "{{date}}": f"{now:%A} {day_ordinal(now.day)} {now:%B}, {now:%Y}",
        "{{time}}": f"{now:%I:%M %p}".lstrip("0"),
        "{{tz}}": tz_name,
        "{{bot_name}}": bot_name or "the bot",
        "{{operator_name}}": operator_name.strip() or "the operator",
        "{{host}}": host_context(),
        "{{operator_tools}}": operator_tools_pass(),
    }
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
        goal = goal.replace(key, value)
        memory = memory.replace(key, value)
    prompt = own_identity_pass(prompt, own_jid, own_lid)
    goal = own_identity_pass(goal, own_jid, own_lid).strip()
    memory = own_identity_pass(memory, own_jid, own_lid).strip()
    if text_tokens(memory) > CONFIG_MEMORY_TOKEN_LIMIT:
        raise ValueError(f"Session memory exceeds {CONFIG_MEMORY_TOKEN_LIMIT} tokens")
    sections: list[str] = []
    if goal:
        sections.append(f"# Goal\n\n{goal}")
    if memory:
        sections.append(f"# Memory\n\n{memory}")
    if sections:
        return "\n\n".join([*sections, prompt])
    return prompt


def day_ordinal(day: int) -> str:
    """The day-of-month with its ordinal suffix: 1st, 2nd, 3rd, 4th…"""
    if day % 10 in (1, 2, 3) and day % 100 not in (11, 12, 13):
        suffix = {1: "st", 2: "nd", 3: "rd"}[day % 10]
    else:
        suffix = "th"
    return f"{day}{suffix}"


#: Own-identity placeholders the prompt may carry; every one of them
#: is dropped line-wise when the bot's identity is unknown.
_OWN_PLACEHOLDERS = ("{{own_jid}}", "{{own_lid}}", "{{own_identities}}")


def operator_tools_pass() -> str:
    """The operator-command tool reach, rendered into the system prompt.

    Shared permission policy: ordinary turns may use only the current
    chat; operator commands may reach others. ``fenced_chat`` permits
    an explicit current-chat JID on ordinary turns too. Escalation has
    its own fixed operator destination, independent of this parameter.
    """
    return (
        "Cross-chat reach is reserved to `[operator command]` turns. "
        "Ordinary turns must stay in the current chat: omit `chat` or "
        "use its exact JID; other chats and foreign message ids are "
        "refused. `escalate` is the fixed-destination operator exception. "
        "`read_chat(mode=recent)` and contact-book resolution require "
        "operator commands; ordinary `mode=resolve` matches current "
        "participants. An operator command's answer returns to the console "
        "by itself: never use `send_message` to answer the operator, and "
        "never message a chat the command did not ask you to send to. "
        "Pass `chat` only when the operator explicitly asks you to "
        "deliver something to that chat, and only with a JID you "
        "resolved this run. Never promise refused "
        "deliveries. `chat` is a "
        "bare JID, not a name or message id: groups use `<id>@g.us`; "
        "people use returned `@lid`/`@c.us` ids or `<digits>@c.us` from "
        "an explicitly provided international phone number."
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
        return drop_placeholder_lines(prompt)
    prompt = prompt.replace("{{own_jid}}", own_jid or "unknown")
    prompt = prompt.replace("{{own_lid}}", own_lid or "unknown")
    return prompt.replace("{{own_identities}}", " or ".join(f"`{jid}`" for jid in ids))


def drop_placeholder_lines(prompt: str) -> str:
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
    in_group = str(event.payload.get("from", "")).endswith("@g.us")
    # The operator's DM turns keep the account JID too: in a direct
    # chat the tag is what tells the operator's lines from the partner's.
    if in_group or is_operator_event(event):
        return group_sender_tag(name, participant, names or {})
    return f"[{name}]" if name else f"[{participant}]"


def group_sender_tag(name: str, participant: str, names: dict[str, str]) -> str:
    """A group turn's ``[Name <jid>]`` tag with its two fallbacks."""
    resolved = name or display_name(participant, names)
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
    own_jids: set[str] | None = None,
) -> str:
    """Render the quoted message as context for the agent.

    Empty/None input yields nothing; the caller decides whether to
    include a ``Reply context`` block in the prompt. *participant_names*
    maps chat JIDs to display names so the quoted sender renders as a
    name, not a raw ``@lid`` JID. *own_jids* (the account's own ids)
    lets a quote of an account message say which teammate wrote it.
    """
    if not message_reply:
        return ""
    sender = quoted_participant(message_reply, participant_names, own_jids)
    description = reply_description(message_reply)
    if sender and description:
        return f'{sender}: "{description}"'
    return sender or description


def quoted_participant(
    message_reply: dict[str, Any],
    participant_names: dict[str, str] | None = None,
    own_jids: set[str] | None = None,
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
    jid = jid_string(message_reply.get("participant") or data.get("author"))
    if own_jids and jid in own_jids:
        return own_account_label(str(message_reply.get("id") or ""), jid)
    name = str(data.get("notifyName") or "").strip()
    if not jid:
        return name
    name = name or display_name(jid, participant_names)
    return f"{name} <{jid}>" if name else jid.split("@", 1)[0]


def own_account_label(message_id: str, jid: str) -> str:
    """Who wrote a message from the shared account, as a quote sender label.

    The bot and the human operator share one JID, so the JID alone
    cannot say which teammate wrote it; the author book recorded it from
    WAHA's ``source`` field when the message was sent.
    """
    author = authors.author(message_id)
    if author == AUTHOR_BOT:
        return f"you <{jid}>"
    if author == AUTHOR_OPERATOR:
        return f"your operator <{jid}>"
    return f"this account, you or your operator <{jid}>"


def mentions_note(
    event: WahaEvent, participant_names: dict[str, str] | None = None
) -> str:
    """Who each ``@<digits>`` in the body is, from ``mentionedJidList``.

    WhatsApp writes a mention into the text as bare digits
    (``@111222333444555``) and lists the tagged JIDs separately; the
    model sees only the digits, so it cannot tell who was tagged — or
    that the tag is the shared account itself. Names come from the
    roster and the name book; unknown members stay bare JIDs. Only the
    first ten tags render — a mass tagging must not flood the turn.
    """
    mentioned = event.payload.get("_data", {}).get("mentionedJidList") or []
    jids = list(dict.fromkeys(j for j in map(jid_string, mentioned) if j))
    if not jids:
        return ""
    own = bot_jids(event)
    parts = [mention_label(jid, own, participant_names) for jid in jids[:10]]
    return "\n[mentions: " + "; ".join(parts) + "]"


def mention_label(jid: str, own: set[str], names: dict[str, str] | None) -> str:
    """``@<digits> is <who>`` for one tagged JID."""
    user = jid.split("@", 1)[0]
    if jid in own:
        return f"@{user} is the shared account (you and your operator)"
    name = display_name(jid, names)
    return f"@{user} is {name} <{jid}>" if name else f"@{user} is <{jid}>"


def reply_context_section(
    message_reply: dict[str, Any] | None,
    participant_names: dict[str, str] | None = None,
    own_jids: set[str] | None = None,
) -> str:
    """Message text with the quoted/replied-to message attached as context."""
    if not message_reply:
        return ""
    line = reply_context(message_reply, participant_names, own_jids)
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
    if cached is None:
        cached = fetch_participant_names(waha, session, chat_id)
        roster_cache.put((session, chat_id), cached)
    return cached


def fetch_participant_names(
    waha: WahaClient, session: str, chat_id: str
) -> dict[str, str]:
    """The roster's names plus names on recent messages (uncached)."""
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


async def chat_rules(
    notebook: Notebook | None, session: str, chat_id: str
) -> dict[str, str]:
    """The chat's `rule` notes keyed by key; fail-soft to ``{}``.

    Rules are the deterministic seams the code applies — the text
    prefix and the hard mute. An unreadable store yields no rules,
    never a blocked run.
    """
    if notebook is None:
        return {}
    try:
        notes = await notebook_operation(notebook.read, session, chat_id, kind=KIND_RULE)
    except Exception:
        logger.warning("Chat rules unreadable; continuing without them")
        return {}
    return {note.key: note.text for note in notes}


async def record_handled_turn(
    agent: FunctionCallingAgentWorkflow,
    ctx: Context,
    user_msg: str,
    *,
    gist: str = "",
) -> None:
    """Store a quiet turn as ledger lines, stamped handled.

    The gate read these messages and declined them; the next prompt gets one
    line each instead of the wall it already judged. Without the stamp the
    next run would treat the trailing message as scaffolding and drop it,
    vanishing it from the buffer, recap, and thread index alike.
    """
    memory = await agent.ensure_memory(ctx)
    serials = await ctx.store.get("turn_serials", default=None) or []
    content = records.ledger_turn(
        user_msg, gist=gist, newest_serial=max(serials, default=0)
    )
    await memory.aput(
        ChatMessage(
            role=MessageRole.USER,
            content=content,
            additional_kwargs={
                TURN_HANDLED_KWARG: True,
                **({TURN_SERIALS_KWARG: list(serials)} if serials else {}),
            },
        )
    )


async def bind_inbound_image(
    image: dict[str, Any] | None, settings: Settings | None, target: RunTarget
) -> None:
    """Persist this turn's inbound image so ``edit_image`` can repaint it.

    Only when the image service is configured — otherwise nothing could
    edit it and the temp file would be pure litter.
    """
    if image is None or settings is None or not settings.image_api_base:
        return
    data = image.get("data")
    if not data:
        return
    target.image_path = str(await asyncio.to_thread(image_store.save, data))


async def bind_run_rules(
    agent: FunctionCallingAgentWorkflow,
    ctx: Context | None,
    user_msg: str,
    target: RunTarget,
    addressed: bool,
    gate_first: bool,
) -> bool:
    """Apply the chat's code-read rules to *target*; False when hard-muted.

    Read before the triage gate so a hard mute holds even when the gate
    is down. A muted chat still runs the gate (so an adversarial read or
    a human request can escalate) but never answers; its turn is
    recorded stamped handled. The prefix applies to everything the run
    delivers.
    """
    if target.armed:
        target.allowed_chats.update(jids_in_text(user_msg))
    rules = await chat_rules(agent.notebook, target.session, target.chat_id)
    if RULE_HARD_MUTE in rules:
        logger.info("Chat is hard-muted; not answering {chat_id}", chat_id=target.chat_id)
        if ctx is not None and await run_gate(
            agent, ctx, target, user_msg, addressed, gate_first
        ):
            await record_handled_turn(agent, ctx, user_msg)
        return False
    target.prefix = rules.get(RULE_MESSAGE_PREFIX, "")
    return True


async def run_gate(
    agent: FunctionCallingAgentWorkflow,
    ctx: Context,
    target: RunTarget,
    user_msg: str,
    addressed: bool,
    gate_first: bool,
) -> bool:
    """Reset triage state and consult the gate; False when it ends the turn.

    A silent gate still records the turn, stamped handled (the same
    stamp a silent run gets): without it the next run's ``repair_memory``
    would drop the trailing user message as scaffolding, vanishing it
    from the buffer, recap, and thread index alike.
    """
    await ctx.store.set("triage_result", None)
    await ctx.store.set("triage_audit", None)
    if await agent.start_triage(ctx, target, user_msg, addressed, gate_first):
        return True
    result = await agent.stored_triage_result(ctx)
    gist = result.verdict.gist if result is not None else ""
    await record_handled_turn(agent, ctx, user_msg, gist=gist)
    return False


async def prepare_turn_state(
    ctx: Context | None,
    event: WahaEvent,
    addressed: bool,
    images: list[dict[str, Any]] | None,
    settings: Settings | None,
) -> list[dict[str, Any]] | None:
    """Record the header verdict and cap image blocks before the run."""
    if ctx is not None:
        # The prompt-time turn header (history.turn_header) reads these back:
        # the wake gate's addressing verdict, and whether this is a group turn.
        await ctx.store.set("turn_addressed", addressed)
        await ctx.store.set(
            "turn_group", str(event.payload.get("from", "")).endswith("@g.us")
        )
    if settings is not None and settings.vision_max_images > 0 and images:
        return images[: settings.vision_max_images]
    return images


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
    addressed: bool = True,
    gate_first: bool = False,
    record_entries: Sequence[RecordEntry] | None = None,
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
    text names the bot (:func:`turn_addressed_note`): the wake gate already
    decided the message is for the bot, and the marker tells the model
    so it never re-derives — and misreads — that fact. Absent both, no
    marker (operator commands, tests). An *unaddressed* group turn gets
    the explicit negative marker instead, so group chatter the gate woke
    the agent for is never mistaken for a direct address.

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
    images = await prepare_turn_state(ctx, event, addressed, images, settings)
    user_msg, image_blocks, serials = await turn_user_message(
        event,
        image=image,
        images=images,
        settings=settings,
        waha=waha,
        pinned_note=pinned_note,
        bot_name=bot_name,
        bot_mention_regex=bot_mention_regex,
        body_scrubbed=body_scrubbed,
        addressed=addressed,
        record_entries=record_entries,
    )
    if ctx is not None and serials:
        await ctx.store.set("turn_serials", serials)
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
        await bind_inbound_image(image, settings, target)
        if not await bind_run_rules(agent, ctx, user_msg, target, addressed, gate_first):
            return "", target
        if ctx is not None and not await run_gate(
            agent, ctx, target, user_msg, addressed, gate_first
        ):
            return "", target
        result = await run_agent_once(agent, ctx, user_msg, image_blocks, settings)
    finally:
        await release_run(agent, ctx, token)
    return final_reply(result), target


async def run_agent_once(
    agent: FunctionCallingAgentWorkflow,
    ctx: Context | None,
    user_msg: str,
    image_blocks: list[Any],
    settings: Settings | None,
) -> Any:
    """Run the agent under the per-run wall-clock cap.

    The library's own ``timeout=`` must stay None: on the reused
    per-chat Context it acts as a cumulative alive-time budget that
    would sum every run's latency and permanently kill busy chats, so
    the cap is enforced here instead; ``wait_for`` raises ``TimeoutError``,
    classified by ``run_timed_out``.
    """
    timeout = settings.run_timeout if settings is not None else None
    return await asyncio.wait_for(
        agent.run(input=user_msg, image_blocks=image_blocks, ctx=ctx),
        timeout=timeout or None,
    )


async def release_run(
    agent: FunctionCallingAgentWorkflow, ctx: Context | None, token: Any
) -> None:
    """Cancel any pending gate and unbind this run's delivery target."""
    if ctx is not None:
        await agent.cancel_triage(ctx)
    reset_target(token)


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
    """Build the one-record entry of a single inbound message."""
    return RecordEntry(
        event=event,
        body=record_entry_body(event),
        images=images,
        video=record_entry_video(event, video_frames),
        addressed=addressed,
    )


def record_entry_body(event: WahaEvent) -> str:
    """The record body, with the code's voice-note prefix removed."""
    body = str(event.payload.get("body", "")).strip()
    if str(event.payload.get(TRANSCRIPT_KEY) or "").strip():
        return body.removeprefix("[voice note]").strip()
    return body


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
