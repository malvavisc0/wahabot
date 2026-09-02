"""HMAC verification for incoming WAHA webhooks."""

import hashlib
import hmac

from fastapi import HTTPException

from whabot.settings import Settings


def verify_hmac(
    settings: Settings,
    body: bytes,
    received_hmac: str | None,
    algorithm_header: str | None,
) -> None:
    """Verify the X-Webhook-Hmac header against the raw request body."""
    if received_hmac is None:
        raise HTTPException(status_code=401, detail="Missing X-Webhook-Hmac header")
    algorithm = hmac_algorithm(algorithm_header)
    digest = hmac.new(
        settings.webhook_hmac_key.encode(), body, getattr(hashlib, algorithm)
    ).hexdigest()
    if not hmac.compare_digest(digest, received_hmac):
        raise HTTPException(status_code=401, detail="HMAC verification failed")


def hmac_algorithm(algorithm_header: str | None) -> str:
    """Resolve the header-named hash algorithm, defaulting to sha512."""
    algorithm = (algorithm_header or "sha512").lower()
    if algorithm not in hashlib.algorithms_available:
        raise HTTPException(
            status_code=400, detail=f"Unsupported HMAC algorithm: {algorithm}"
        )
    return algorithm
