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
    secret = settings.webhook_hmac_key

    if received_hmac is None:
        raise HTTPException(status_code=401, detail="Missing X-Webhook-Hmac header")

    algorithm = (algorithm_header or "sha512").lower()
    if algorithm not in hashlib.algorithms_available:
        raise HTTPException(
            status_code=400, detail=f"Unsupported HMAC algorithm: {algorithm}"
        )

    digest = hmac.new(secret.encode(), body, getattr(hashlib, algorithm)).hexdigest()
    if not hmac.compare_digest(digest, received_hmac):
        raise HTTPException(status_code=401, detail="HMAC verification failed")
