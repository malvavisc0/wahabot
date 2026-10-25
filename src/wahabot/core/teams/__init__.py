"""Internal team-group surface."""

from wahabot.core.teams.groups import (
    FORBIDDEN_TEAM_CAPABILITIES,
    bot_mention_pattern,
    candidate_patterns,
    group_participation,
    is_team_group,
    mention_match,
    mention_pattern_ok,
    surface_for,
    team_capability_error,
)

__all__ = [
    "FORBIDDEN_TEAM_CAPABILITIES",
    "bot_mention_pattern",
    "candidate_patterns",
    "group_participation",
    "is_team_group",
    "mention_match",
    "mention_pattern_ok",
    "surface_for",
    "team_capability_error",
]
