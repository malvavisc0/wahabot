"""Journal of raw webhook events, one JSON object per line per day."""

import datetime
from pathlib import Path


def save_event(events_dir: Path, session: str, body: bytes) -> None:
    """Append the raw event body to the session's daily journal file.

    Files land at `<events_dir>/<session>/<YYYY-MM-DD>.jsonl`, one JSON
    object per line — greppable and replayable as-is.
    """
    day = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
    session_dir = events_dir / session
    session_dir.mkdir(parents=True, exist_ok=True)
    with (session_dir / f"{day}.jsonl").open("a", encoding="utf-8") as journal:
        journal.write(body.decode("utf-8") + "\n")
