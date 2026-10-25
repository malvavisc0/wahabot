"""Ticket-event skills: signed, bounded, receiver-deduplicable delivery."""

import hashlib
import hmac
import json
import queue
import threading
import time

from loguru import logger

from wahabot.core.skills.errors import SkillError
from wahabot.core.skills.loader import load_skill
from wahabot.core.skills.service import SkillService
from wahabot.core.tickets.models import Ticket

MAX_ATTEMPTS = 3


class EventDispatcher:
    """Deliver ticket lifecycle events to every enabled event skill.

    Production wiring uses ``background=True`` so webhook latency never blocks
    the ticket transaction path. Tests can keep the default synchronous mode.
    """

    def __init__(self, service: SkillService, *, background: bool = False) -> None:
        self.service = service
        self._background = background
        self._queue: queue.Queue[tuple[str, Ticket] | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        if background:
            self._worker = threading.Thread(
                target=self._work, name="skill-events", daemon=True
            )
            self._worker.start()

    def emit(self, event_kind: str, ticket: Ticket) -> None:
        """Queue or run the configured event skills; never raise into tickets."""
        if self._background:
            self._queue.put((event_kind, ticket))
            return
        try:
            self._emit(event_kind, ticket)
        except Exception:
            logger.exception("Skill event dispatch failed for {kind}", kind=event_kind)

    def flush(self, timeout: float = 30.0) -> bool:
        """Wait for queued deliveries; useful for tests and graceful shutdown."""
        if not self._background:
            return True
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                event_kind, ticket = item
                try:
                    self._emit(event_kind, ticket)
                except Exception:
                    logger.exception(
                        "Skill event dispatch failed for {kind}", kind=event_kind
                    )
            finally:
                self._queue.task_done()

    def _emit(self, event_kind: str, ticket: Ticket) -> None:
        for skill in self.service.store.enabled():
            try:
                config = load_skill(skill.name, self.service.skills_dir)
            except SkillError:
                continue
            if not config.events.get(event_kind):
                continue
            payload = event_payload(event_kind, ticket)
            body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            headers = {
                "Content-Type": "application/json",
                "X-Wahabot-Event-Id": str(payload["event_id"]),
                "X-Wahabot-Timestamp": str(payload["timestamp"]),
            }
            if config.sign_events:
                try:
                    signing_secret = self.service.secrets.get(config.signing_secret)
                except SkillError as exc:
                    logger.warning(
                        "Event skill {name} has no signing secret: {error}",
                        name=skill.name,
                        error=type(exc).__name__,
                    )
                    continue
                digest = hmac.new(
                    signing_secret.encode(), body, hashlib.sha256
                ).hexdigest()
                headers["X-Wahabot-Signature"] = f"sha256={digest}"
            self._deliver(skill.name, body, headers)

    def _deliver(self, name: str, body: bytes, headers: dict[str, str]) -> None:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                self.service.send_event(name, body, headers)
                return
            except SkillError as exc:
                logger.warning(
                    "Event skill {name} attempt {attempt} failed: {error}",
                    name=name,
                    attempt=attempt,
                    error=type(exc).__name__,
                )


def event_payload(event_kind: str, ticket: Ticket) -> dict[str, object]:
    """The stable payload: no message bodies, no phone numbers."""
    at = ticket.updated_at or int(time.time())
    return {
        "event": event_kind,
        "event_id": event_id(ticket, event_kind, at),
        "ticket": ticket.number,
        "state": ticket.state,
        "done_by": ticket.done_by or "",
        "contact_ref": ticket.external_user_ref,
        "timestamp": at,
    }


def event_id(ticket: Ticket, event_kind: str, at: int) -> str:
    """A deterministic id the receiver can deduplicate retries with."""
    material = f"{ticket.number}:{event_kind}:{at}:{ticket.state}".encode()
    return hashlib.sha256(material).hexdigest()[:32]
