"""Audit helpers shared by the skill runtime."""

import json
import time
from pathlib import Path

from wahabot.core.skills.errors import SkillError
from wahabot.core.skills.execution import (
    CALL_PATH_CAP,
    REQUEST_JSON_CAP,
    note_failure,
    record_call,
    record_ticket,
)
from wahabot.core.skills.models import (
    PreparedRequest,
    SkillRuntime,
    envelope_error,
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

    def _record_refusal(
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
        safe = self._safe(reason)
        self._record(
            name=name,
            run=run,
            method=method,
            path=path,
            status=None,
            duration=int((time.monotonic() - started) * 1000),
            outcome="refused",
            error=safe,
            request="",
        )
        record_ticket(run, f"{name} {method} {path[:CALL_PATH_CAP]} refused", "refused")

    def _record(
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
        record_call(
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

    def _note_failure(self, name: str, error: str) -> None:
        note_failure(
            store=self.store,
            transport=self.transport,
            session=self.session,
            name=name,
            error=error,
            redact=self.secrets.redact,
        )

    def _safe(self, text: str) -> str:
        try:
            return self.secrets.redact(text)
        except SkillError:
            return text

    def _fail_event(
        self, name: str, run: SkillRuntime, prepared: PreparedRequest, error: str
    ) -> None:
        safe = self._safe(error)
        self._record(
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
        self._note_failure(name, safe)

    def _fail_sql(self, name: str, run: SkillRuntime, error: str, started: float) -> str:
        safe = self._safe(error)
        self._record(
            name=name,
            run=run,
            method="SQL",
            path=safe[:CALL_PATH_CAP],
            status=None,
            duration=int((time.monotonic() - started) * 1000),
            outcome="failed",
            error=safe,
            request=safe[:REQUEST_JSON_CAP],
        )
        record_ticket(run, f"{name} SQL failed", "failed")
        self._note_failure(name, safe)
        return envelope_error(safe, skill=name)
