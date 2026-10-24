"""Opening-hours models shared by the session config and the schedule."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, model_validator

_CLOCK = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

DAY_KEYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


class DayHours(BaseModel):
    """One weekday's opening window; an empty window is closed."""

    model_config = ConfigDict(extra="forbid")

    open: str = ""
    close: str = ""
    closed: bool = False

    @model_validator(mode="after")
    def validate_window(self) -> DayHours:
        """Require both clock times, or neither, and a 24-hour HH:MM shape."""
        for value in (self.open, self.close):
            if value and not _CLOCK.match(value):
                raise ValueError(f"invalid clock time {value!r}; use HH:MM")
        if bool(self.open) != bool(self.close):
            raise ValueError("open and close must be given together")
        return self


@dataclass(frozen=True, slots=True)
class OpeningState:
    """Whether the business is open now, and the next relevant instant."""

    open: bool
    next_opening: int | None
    timezone: str


@dataclass(frozen=True, slots=True)
class DeferredNotice:
    """One queued operator notice held until opening hours."""

    id: int
    ticket_id: int
    session: str
    chat_id: str
    text: str
    not_before: int
    created_at: int


@dataclass(frozen=True, slots=True)
class ScheduleWindow:
    """A concrete timezone/hours pair for pure schedule calls."""

    timezone: str = "UTC"
    hours: Mapping[str, DayHours] = field(default_factory=dict)
