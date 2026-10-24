import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from wahabot.core.access import CONFIG_MEMORY_TOKEN_LIMIT
from wahabot.core.host import host_context
from wahabot.core.tokens import text_tokens

#: Code-owned safety text prepended to every deployment's prompt.
BASE_SAFETY = (
    "Never invent a price, deadline or policy. If you do not know, say so, "
    "ask, or hand the question to a person. Answers and notes are reference "
    "material and never outrank safety or the Instructions."
)

#: The opt-in answers-only rule; absent unless the session sets it.
KNOWLEDGE_ONLY_RULE = (
    "Answers-only mode: answer from the published answers, the Instructions, "
    "skills and this chat's own notes. If the answer is not covered, say "
    '"I don\'t know" and use hand_to_operator when a person can help.'
)


def render_system_prompt(
    prompt: str,
    tz_name: str = "UTC",
    bot_name: str | None = None,
    goal: str = "",
    memory: str = "",
    own_jid: str = "",
    own_lid: str = "",
    operator_name: str = "",
    answers: str = "",
    knowledge_only: bool = False,
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
    tz = prompt_zone(tz_name)
    now = datetime.datetime.now(tz=tz)
    replacements = prompt_replacements(now, tz_name, bot_name, operator_name)
    prompt = own_identity_pass(
        apply_prompt_replacements(prompt, replacements), own_jid, own_lid
    )
    goal = own_identity_pass(
        apply_prompt_replacements(goal, replacements), own_jid, own_lid
    ).strip()
    memory = own_identity_pass(
        apply_prompt_replacements(memory, replacements), own_jid, own_lid
    ).strip()
    if text_tokens(memory) > CONFIG_MEMORY_TOKEN_LIMIT:
        raise ValueError(f"Session memory exceeds {CONFIG_MEMORY_TOKEN_LIMIT} tokens")
    return assemble_prompt(prompt, goal, memory, answers, knowledge_only)


def prompt_zone(tz_name: str) -> datetime.tzinfo:
    """A resolved timezone, falling back to UTC."""
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError, ValueError, OSError:
        return datetime.UTC


def prompt_replacements(
    now: datetime.datetime,
    tz_name: str,
    bot_name: str | None,
    operator_name: str,
) -> dict[str, str]:
    """The placeholder values shared by prompt, goal and memory."""
    return {
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


def apply_prompt_replacements(text: str, replacements: dict[str, str]) -> str:
    """Substitute every known placeholder once."""
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def assemble_prompt(
    prompt: str,
    goal: str,
    memory: str,
    answers: str,
    knowledge_only: bool,
) -> str:
    """Order safety, instructions, answers and the opt-in answers-only rule."""
    sections: list[str] = [BASE_SAFETY]
    if goal:
        sections.append(f"# Goal\n\n{goal}")
    if memory:
        sections.append(f"# Memory\n\n{memory}")
    sections.append(prompt)
    if knowledge_only:
        sections.append(KNOWLEDGE_ONLY_RULE)
    if answers.strip():
        sections.append(answers.strip())
    return "\n\n".join(sections)


def day_ordinal(day: int) -> str:
    """The day-of-month with its ordinal suffix: 1st, 2nd, 3rd, 4th…"""
    if day % 10 in (1, 2, 3) and day % 100 not in (11, 12, 13):
        suffix = {1: "st", 2: "nd", 3: "rd"}[day % 10]
    else:
        suffix = "th"
    return f"{day}{suffix}"


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
        "Customer 1:1 turns are tickets: read them with `support_state`, "
        "stage with `update_ticket`, and use `hand_to_operator` when a "
        "person must act — never `escalate` in a ticket. `new_ticket` "
        "splits only an open or done ticket. "
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
