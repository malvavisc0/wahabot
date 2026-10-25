"""Settings form handling: session JSON validation and number resolution."""

import json
import re
from collections.abc import Mapping
from typing import Any

import httpx

from wahabot.core.access import SessionConfig, parse_session_config
from wahabot.core.schedule.hours import render_away_message
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings

DAY_LABELS = (
    ("mon", "Monday"),
    ("tue", "Tuesday"),
    ("wed", "Wednesday"),
    ("thu", "Thursday"),
    ("fri", "Friday"),
    ("sat", "Saturday"),
    ("sun", "Sunday"),
)


def read_raw(settings: Settings) -> dict[str, Any]:
    """The session JSON as an editable mapping."""
    path = settings.access_config
    if not path.exists():
        raise ValueError("the session config is missing")
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("the session config is not an object")
    return raw


def load_config(settings: Settings) -> SessionConfig:
    """The validated session config."""
    return parse_session_config(
        settings.access_config.read_text(), settings.access_config
    )


def commit(settings: Settings, raw: dict[str, Any]) -> SessionConfig:
    """Validate the whole config before replacing the file."""
    text = json.dumps(raw, indent=2, ensure_ascii=False) + "\n"
    config = parse_session_config(text, settings.access_config)
    settings.access_config.write_text(text)
    return config


def normalize_number(value: str) -> str:
    """Digits only, with the international ``00`` or trunk ``0`` dropped."""
    digits = re.sub(r"\D", "", value)
    if digits.startswith("00"):
        digits = digits[2:]
    elif digits.startswith("0"):
        digits = digits[1:]
    if not 8 <= len(digits) <= 15:
        raise ValueError(f"{value.strip()} is not an international number (8-15 digits)")
    return digits


def normalize_number_lines(text: str) -> list[str]:
    """Normalize one number per line; blank lines are ignored."""
    found: list[str] = []
    for line in text.splitlines():
        if line.strip():
            found.append(normalize_number(line))
    return found


def resolve_numbers(values: list[str], waha: WahaClient, session: str) -> list[str]:
    """Check every number on WhatsApp and store its resolved chat id."""
    resolved: list[str] = []
    for digits in values:
        chat_id = f"{digits}@c.us"
        try:
            overview = waha.get_chat_overview(session, chat_id)
        except httpx.HTTPError as exc:
            raise ValueError(f"{digits} could not be checked on WhatsApp") from exc
        if not overview:
            raise ValueError(f"{digits} is not on WhatsApp")
        if chat_id not in resolved:
            resolved.append(chat_id)
    return resolved


def display_numbers(values: list[str]) -> str:
    """Stored chat ids back to one number per line."""
    seen: list[str] = []
    for value in values:
        digits = value.split("@", 1)[0]
        if digits and digits not in seen:
            seen.append(digits)
    return "\n".join(seen)


def draft_mention(name: str) -> str:
    """Draft and sanity-check the mention pattern; fall back to a plain name."""
    clean = name.strip()
    if not clean:
        return ""
    pattern = rf"(?iu)(?<![^\W_])@?{re.escape(clean)}(?!\w)"
    try:
        compiled = re.compile(pattern)
    except re.error:
        return rf"(?iu)(?<![^\W_])@?{re.escape(clean)}(?!\w)"
    return pattern if compiled.search(clean) else rf"(?iu){re.escape(clean)}"


def save_assistant(settings: Settings, form: dict[str, str]) -> None:
    """Save the Assistant section and its invisible mention pattern."""
    raw = read_raw(settings)
    name = form.get("bot_name", "").strip()
    instructions = form.get("instructions", "").strip()
    if not instructions:
        raise ValueError("instructions cannot be empty")
    raw["bot_name"] = name or None
    raw["goal"] = form.get("goal", "").strip()
    raw["system_prompt"] = instructions
    raw["operator_name"] = form.get("operator_name", "").strip()
    raw["bot_mention_regex"] = draft_mention(name) or None
    commit(settings, raw)


def save_who(
    settings: Settings, form: Mapping[str, object], groups: list[str], waha: WahaClient
) -> None:
    """Save allowed/blocked numbers, resolving each one on WhatsApp."""
    raw = read_raw(settings)
    whitelist = resolve_numbers(
        normalize_number_lines(str(form.get("whitelist", ""))), waha, settings.session
    )
    blacklist = resolve_numbers(
        normalize_number_lines(str(form.get("blacklist", ""))), waha, settings.session
    )
    raw["whitelist"] = whitelist
    raw["blacklist"] = blacklist
    raw["team_groups"] = groups
    if groups:
        raw["group_participation"] = "mentioned"
    commit(settings, raw)


def save_hours(settings: Settings, form: dict[str, str]) -> None:
    """Save timezone, weekday hours, away message, window and night number."""
    raw = read_raw(settings)
    hours: dict[str, dict[str, Any]] = {}
    for day, _label in DAY_LABELS:
        opened = form.get(f"{day}_open", "").strip()
        closed = form.get(f"{day}_close", "").strip()
        is_closed = bool(form.get(f"{day}_closed"))
        if opened or closed or is_closed:
            hours[day] = {"open": opened, "close": closed, "closed": is_closed}
    raw["timezone"] = form.get("timezone", "").strip()
    raw["hours"] = hours
    raw["away_message"] = form.get("away_message", "")
    try:
        window = int(form.get("reply_window_days", "7"))
    except ValueError as exc:
        raise ValueError("the reply window must be a number of days") from exc
    if not 1 <= window <= 30:
        raise ValueError("the reply window must be between 1 and 30 days")
    raw["reply_window_days"] = window
    night = form.get("night_number", "").strip()
    raw["night_number"] = normalize_number(night) if night else ""
    commit(settings, raw)


def hours_values(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize the stored hours for the form."""
    hours = raw.get("hours")
    stored: dict[str, Any] = hours if isinstance(hours, dict) else {}
    values: dict[str, dict[str, Any]] = {}
    for day, _label in DAY_LABELS:
        found = stored.get(day)
        item: dict[str, Any] = found if isinstance(found, dict) else {}
        values[day] = {
            "open": item.get("open", ""),
            "close": item.get("close", ""),
            "closed": bool(item.get("closed")),
        }
    return values


def away_preview(raw: dict[str, Any]) -> str:
    """Render the away message with example placeholders."""
    return render_away_message(
        str(raw.get("away_message") or ""),
        142,
        None,
        str(raw.get("timezone") or "UTC"),
    )


def team_groups_view(
    waha: WahaClient, session: str, enabled: list[str]
) -> list[dict[str, Any]]:
    """The connected number's groups, with name, member count and choice."""
    try:
        chats = waha.list_chats(session)
    except httpx.HTTPError:
        return []
    rows: list[dict[str, Any]] = []
    for chat in chats:
        chat_id = str(chat.get("id") or "")
        if not chat_id.endswith("@g.us"):
            continue
        participants = chat.get("participants") or []
        rows.append(
            {
                "id": chat_id,
                "name": str(chat.get("name") or chat.get("subject") or "Group"),
                "count": len(participants) if isinstance(participants, list) else 0,
                "enabled": chat_id in enabled,
            }
        )
    return rows
