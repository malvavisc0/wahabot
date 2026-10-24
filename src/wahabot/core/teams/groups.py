"""Team-group allowlist, addressing override and code-enforced fences."""

import re
from collections.abc import Sequence

from wahabot.ai.messages.addressing import (
    bot_mention_pattern as configured_mention_pattern,
)
from wahabot.ai.messages.addressing import (
    is_group_addressed,
)
from wahabot.core.access import SessionConfig
from wahabot.core.models import WahaEvent

#: Capabilities a team-group run may never exercise.
FORBIDDEN_TEAM_CAPABILITIES = frozenset(
    {
        "ticket_write",
        "customer_send",
        "customer_system_write",
        "tts",
        "greeting",
        "post_first",
    }
)

#: Ordinary prose that must never match a drafted mention pattern.
_MENTION_NEGATIVES = ("kayak", "email", "gmail", "waiting", "detail")


def is_team_group(team_groups: Sequence[str], chat_id: str) -> bool:
    """Whether a chat is in the enabled team-group allowlist."""
    return bool(chat_id) and chat_id in team_groups


def group_participation(config: SessionConfig, chat_id: str) -> str:
    """The additive override: a listed group runs mentioned-only."""
    if is_team_group(config.team_groups, chat_id):
        return "mentioned"
    return str(config.group_participation)


def surface_for(config: SessionConfig, chat_id: str) -> str:
    """Bind the run surface; only a listed group is the team surface."""
    if chat_id.endswith("@g.us") and is_team_group(config.team_groups, chat_id):
        return "team_group"
    return "customer"


def addressed_in_group(config: SessionConfig, event: WahaEvent) -> bool:
    """Group addressing with the allowlist override applied at the gate."""
    chat_id = str(event.payload.get("from", ""))
    return is_group_addressed(
        event,
        bot_name=getattr(config, "bot_name", None),
        bot_mention_regex=getattr(config, "bot_mention_regex", None),
        participation=group_participation(config, chat_id),
    )


def team_capability_error(target: object, capability: str) -> str | None:
    """Refuse a forbidden action in a team-group run.

    The fence is code, not prompt text: no tool call can opt out.
    """
    if getattr(target, "surface", "") != "team_group":
        return None
    if capability in FORBIDDEN_TEAM_CAPABILITIES:
        return (
            f"team-group runs cannot {capability.replace('_', ' ')}; "
            "answer internally or ask the operator to act"
        )
    return None


def candidate_patterns(name: str) -> tuple[str, ...]:
    """Regex variants tried for one assistant name."""
    quoted = re.escape(name.strip())
    if not quoted:
        return ()
    return (
        rf"(?iu)(?<![^\W_])@?{quoted}(?!\w)",
        rf"(?iu)@?{quoted}\b",
    )


def mention_pattern_ok(pattern: str, name: str) -> bool:
    """Compile a pattern and check it matches the name, not ordinary prose."""
    try:
        compiled = re.compile(pattern)
    except re.error:
        return False
    if not name.strip():
        return False
    if not compiled.search(name) or not compiled.search(f"@{name}"):
        return False
    return not any(compiled.search(word) for word in _MENTION_NEGATIVES)


def bot_mention_pattern(name: str) -> str:
    """Draft a safe mention regex, falling back to an escaped word match."""
    cleaned = name.strip()
    if not cleaned:
        return configured_mention_pattern(None, None).pattern
    for candidate in candidate_patterns(cleaned):
        if mention_pattern_ok(candidate, cleaned):
            return candidate
    return rf"(?iu)(?<![^\W_])@?{re.escape(cleaned)}(?!\w)"


def mention_match(pattern: str, text: str) -> bool:
    """Whether a compiled mention pattern matches *text*."""
    try:
        return bool(re.search(pattern, text))
    except re.error:
        return False
