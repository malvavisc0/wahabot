"""Audit helpers shared by the skill runtime."""

import json
import time
from pathlib import Path

from wahabot.core.envelope import error as envelope_error
from wahabot.core.skills.errors import SkillError
from wahabot.core.skills.execution import (
    CALL_PATH_CAP,
    REQUEST_JSON_CAP,
    append_call,
    note_failure,
    record_ticket,
)
from wahabot.core.skills.models import (
    PreparedRequest,
    SkillRuntime,
)
from wahabot.core.skills.secrets import SecretStore
from wahabot.core.skills.store import SkillStore
from wahabot.core.skills.support import request_json
from wahabot.core.transport import Transport


class SkillAuditMixin:
    """Record calls, refusals and consecutive failures."""

    store: SkillStore  # pyright: ignore[reportUninitializedInstanceVariable]
    data_dir: Path | None  # pyright: ignore[reportUninitializedInstanceVariable]
    session: str  # pyright: ignore[reportUninitializedInstanceVariable]
    secrets: SecretStore  # pyright: ignore[reportUninitializedInstanceVariable]
    transport: Transport  # pyright: ignore[reportUninitializedInstanceVariable]

    def record_refusal(
        self,
        name: str,
        run: SkillRuntime,
        method: str,
        path: str,
        envelope: str,
        started: float,
    ) -> None:
        """Audit a call refused before it was sent."""
        reason = envelope
        parsed: dict[str, object] | None
        try:
            loaded = json.loads(envelope)
            parsed = loaded if isinstance(loaded, dict) else None
        except TypeError, ValueError:
            parsed = None
        if parsed is not None:
            reason = str(parsed.get("error") or envelope)
        safe = self.safe_text(reason)
        safe_path = self.safe_text(path)
        self.record_call(
            name=name,
            run=run,
            method=method,
            path=safe_path,
            status=None,
            duration=int((time.monotonic() - started) * 1000),
            outcome="refused",
            error=safe,
            request="",
        )
        record_ticket(
            run, f"{name} {method} {safe_path[:CALL_PATH_CAP]} refused", "refused"
        )

    def record_call(
        self,
        *,
        name: str,
        run: SkillRuntime,
        method: str,
        path: str,
        status: int | None,
        duration: int,
        outcome: str,
        error: str,
        request: str,
    ) -> None:
        """Append one skill call and its audit entry with redacted text."""
        append_call(
            store=self.store,
            data_dir=self.data_dir,
            session=self.session,
            name=name,
            run=run,
            method=method,
            path=path,
            status=status,
            duration_ms=duration,
            outcome=outcome,
            error=error,
            request_json=request,
            redact=self.secrets.redact,
        )

    def count_failure(self, name: str, error: str) -> None:
        """Increment the failure streak and alert at the threshold."""
        note_failure(
            store=self.store,
            transport=self.transport,
            session=self.session,
            name=name,
            error=error,
            redact=self.secrets.redact,
        )

    def safe_text(self, text: str) -> str:
        """Redact known secret values, returning the text on redaction failure."""
        try:
            return self.secrets.redact(text)
        except SkillError:
            return text

    def fail_event(
        self, name: str, run: SkillRuntime, prepared: PreparedRequest, error: str
    ) -> None:
        """Audit a failed event delivery and count it."""
        safe = self.safe_text(error)
        self.record_call(
            name=name,
            run=run,
            method="POST",
            path=prepared.path,
            status=None,
            duration=0,
            outcome="failed",
            error=safe,
            request=request_json(prepared),
        )
        self.count_failure(name, safe)

    def fail_sql(
        self, name: str, run: SkillRuntime, query: str, error: str, started: float
    ) -> str:
        """Audit one failed SQL call with its query, and count the failure."""
        safe = self.safe_text(error)
        safe_query = self.safe_text(query)
        self.record_call(
            name=name,
            run=run,
            method="SQL",
            path=safe_query[:CALL_PATH_CAP],
            status=None,
            duration=int((time.monotonic() - started) * 1000),
            outcome="failed",
            error=safe,
            request=safe_query[:REQUEST_JSON_CAP],
        )
        record_ticket(run, f"{name} SQL failed", "failed")
        self.count_failure(name, safe)
        return envelope_error(safe, skill=name)
