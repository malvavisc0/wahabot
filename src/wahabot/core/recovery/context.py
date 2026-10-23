"""Context flags for a run started by durable recovery."""

from contextvars import ContextVar

recovering: ContextVar[bool] = ContextVar("wahabot_recovering", default=False)
apology: ContextVar[str] = ContextVar("wahabot_recovery_apology", default="")
