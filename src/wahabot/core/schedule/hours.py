"""Opening hours, next opening and the away message."""

import datetime
from collections.abc import Mapping
from typing import Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from wahabot.core.schedule.models import (
    DAY_KEYS,
    DayHours,
    OpeningState,
)

DEFAULT_AWAY_MESSAGE = (
    "Thanks for your message — we are away right now. We will reply after {next_opening}."
)


class ScheduleConfig(Protocol):
    """The slice of session config the schedule needs."""

    @property
    def timezone(self) -> str: ...

    @property
    def hours(self) -> Mapping[str, DayHours]: ...


def effective_timezone(config_timezone: str, fallback_timezone: str = "UTC") -> str:
    """The session timezone, then the env timezone, then UTC."""
    return config_timezone.strip() or fallback_timezone.strip() or "UTC"


def session_zone(config_timezone: str, fallback_timezone: str = "UTC") -> ZoneInfo:
    """A usable zone for a configured timezone name."""
    name = effective_timezone(config_timezone, fallback_timezone)
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError, ValueError, OSError:
        return ZoneInfo("UTC")


def day_hours(config: ScheduleConfig, weekday: int) -> DayHours | None:
    """The configured window for a Python weekday (Monday is 0)."""
    index = weekday % 7
    return config.hours.get(DAY_KEYS[index])


def clock_moment(day: datetime.date, value: str, zone: ZoneInfo) -> datetime.datetime:
    """An aware local instant for a HH:MM clock value on *day*."""
    hour, minute = (int(part) for part in value.split(":", 1))
    return datetime.datetime.combine(
        day,
        datetime.time(hour=hour, minute=minute),
        tzinfo=zone,
    )


def window_bounds(
    day: datetime.date, hours: DayHours | None, zone: ZoneInfo
) -> tuple[datetime.datetime, datetime.datetime] | None:
    """The absolute open/close instants of one weekday's window."""
    if hours is None or hours.closed or not hours.open or not hours.close:
        return None
    opened = clock_moment(day, hours.open, zone)
    closed = clock_moment(day, hours.close, zone)
    if closed <= opened:
        closed += datetime.timedelta(days=1)
    return opened, closed


def current_window(
    local: datetime.datetime, zone: ZoneInfo, config: ScheduleConfig
) -> tuple[datetime.datetime, datetime.datetime] | None:
    """The window containing *local*, including last night's overnight window."""
    if not config.hours:
        return None
    today = window_bounds(local.date(), day_hours(config, local.weekday()), zone)
    if today is not None and today[0] <= local < today[1]:
        return today
    yesterday_date = local.date() - datetime.timedelta(days=1)
    yesterday = window_bounds(
        yesterday_date, day_hours(config, yesterday_date.weekday()), zone
    )
    if yesterday is not None and yesterday[0] <= local < yesterday[1]:
        return yesterday
    return None


def next_open_after(
    local: datetime.datetime, zone: ZoneInfo, config: ScheduleConfig
) -> int | None:
    """The next window opening at or after *local*, as epoch seconds."""
    if not config.hours:
        return None
    for offset in range(0, 8):
        day = local.date() + datetime.timedelta(days=offset)
        bounds = window_bounds(day, day_hours(config, day.weekday()), zone)
        if bounds is None:
            continue
        if bounds[0] > local:
            return int(bounds[0].timestamp())
    return None


def opening_state(
    config: ScheduleConfig, now: int, fallback_timezone: str = "UTC"
) -> OpeningState:
    """The current opening state for an epoch instant."""
    zone = session_zone(config.timezone, fallback_timezone)
    local = datetime.datetime.fromtimestamp(now, tz=zone)
    window = current_window(local, zone, config)
    if window is not None or not config.hours:
        closing = int(window[1].timestamp()) if window is not None else None
        return OpeningState(open=True, next_opening=closing, timezone=str(zone))
    return OpeningState(
        open=False,
        next_opening=next_open_after(local, zone, config),
        timezone=str(zone),
    )


def next_opening(
    config: ScheduleConfig, now: int, fallback_timezone: str = "UTC"
) -> int | None:
    """The next instant the business opens, or None when never configured."""
    return opening_state(config, now, fallback_timezone).next_opening


def format_opening(moment: int | None, timezone: str) -> str:
    """A short operator/customer-facing next-opening label."""
    if moment is None:
        return "our next opening"
    zone = session_zone(timezone)
    local = datetime.datetime.fromtimestamp(moment, tz=zone)
    return local.strftime("%A %H:%M")


def render_away_message(
    template: str, ticket_number: int, next_open: int | None, timezone: str
) -> str:
    """Fill the away message's ``{ticket}`` and ``{next_opening}`` slots."""
    text = (template or DEFAULT_AWAY_MESSAGE).strip() or DEFAULT_AWAY_MESSAGE
    label = format_opening(next_open, timezone)
    return text.replace("{ticket}", str(ticket_number)).replace("{next_opening}", label)
