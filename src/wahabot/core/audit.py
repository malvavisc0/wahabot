"""Audit journal of bot actions: every tool call, reply and decision.

The roadmap's "journal everything" clause (docs/plans/
commercial-roadmap.md, Commercial Foundation) makes every bot action
auditable end to end. The raw event journal (``core/journal``) already
preserves what came *in* verbatim; this journal records what the bot
*did* — one JSON object per line, appended under
``<data_dir>/audit/<session>/<YYYY-MM-DD>.jsonl``.

Entries are fail-soft throughout: an audit write must never break a
run (the same contract as memory persistence), so any disk error is
logged and swallowed.

Truncation: ``report``/``result``/``reply`` fields cap at the same
300-char budget as the tool-call log lines — this is an audit trail,
not a transcript (the full tool output already lives in the chat's
memory and the tool result itself).
"""

import datetime
import json
from pathlib import Path
from typing import Any

from loguru import logger

#: Cap on the string fields of an audit entry (chars).
_CAP = 300


def audit_dir(data_dir: Path, session: str) -> Path:
    """Path of the session's audit directory."""
    return data_dir / "audit" / session


def capped(value: Any) -> Any:
    """A JSON-safe, length-capped copy of an audit field."""
    if isinstance(value, str) and len(value) > _CAP:
        return value[:_CAP]
    return value


def save_action(
    data_dir: Path,
    session: str,
    kind: str,
    chat_id: str = "",
    **fields: Any,
) -> None:
    """Append one bot action to the session's daily audit journal.

    *kind* names the action (``tool_call``, ``reply``, ``silence``,
    ``escalation``…); the keyword fields carry its context. Any write
    failure is logged and swallowed — the audited run already
    happened, and a broken audit file must not break the reply.
    """
    day = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
    entry = {
        "at": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "kind": kind,
        "session": session,
        "chat_id": chat_id,
        **{key: capped(value) for key, value in fields.items()},
    }
    try:
        directory = audit_dir(data_dir, session)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{day}.jsonl").open("a", encoding="utf-8") as audit:
            audit.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Audit journal write failed ({kind}): {exc}", kind=kind, exc=exc)
