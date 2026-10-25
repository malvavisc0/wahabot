"""Shared web service helpers."""

import datetime


def format_time(value: int | None) -> str:
    """Format a Unix timestamp as a short UTC date-time."""
    if not value:
        return ""
    moment = datetime.datetime.fromtimestamp(int(value), tz=datetime.UTC)
    return moment.strftime("%Y-%m-%d %H:%M")
