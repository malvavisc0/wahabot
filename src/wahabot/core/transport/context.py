"""Accepted intake ids for the delivery currently in flight."""

from collections.abc import Iterable, Sequence
from contextvars import ContextVar, Token

_intake_ids: ContextVar[tuple[str, ...]] = ContextVar(
    "wahabot_delivery_intake_ids", default=()
)


def bind_intake_ids(ids: Iterable[str]) -> Token[tuple[str, ...]]:
    """Bind the intake ids a send answers for this run."""
    return _intake_ids.set(tuple(ids))


def reset_intake_ids(token: Token[tuple[str, ...]]) -> None:
    """Restore the previous delivery binding."""
    _intake_ids.reset(token)


def current_intake_ids() -> Sequence[str]:
    """The intake ids associated with the current delivery."""
    return _intake_ids.get()
