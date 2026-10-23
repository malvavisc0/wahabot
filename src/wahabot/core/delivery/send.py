"""Classify a transport failure before deciding whether it may be retried."""

import httpx

RETRYABLE = "retryable"
DEFINITE = "definite"
AMBIGUOUS = "ambiguous"

BACKOFF_BASE_S = 60
BACKOFF_CAP_S = 3600


def classify_send_error(exc: Exception) -> str:
    """A definite retryable failure, a definite terminal failure, or unknown.

    Connection establishment fails before the request reaches WAHA, so
    a retry cannot duplicate a message. A 5xx is also a definite failure
    before acceptance. Timeouts and other transport errors after the
    request may have reached WAHA, so their outcome is unknown.
    """
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return RETRYABLE
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code >= 500:
            return RETRYABLE
        return DEFINITE
    return AMBIGUOUS


def backoff_seconds(attempts: int) -> int:
    """Bounded exponential backoff for a definite retryable failure."""
    return min(BACKOFF_BASE_S * (2 ** max(attempts - 1, 0)), BACKOFF_CAP_S)
