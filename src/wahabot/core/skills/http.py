"""One guarded HTTP request: pinned DNS, redirects and response caps."""

import json
from urllib.parse import urljoin

import httpx
from loguru import logger

from wahabot.core.skills.errors import SkillError
from wahabot.core.skills.guardrails import (
    EndpointRefused,
    allows_private,
    ensure_public_endpoint,
    pinned_request,
    same_origin,
)
from wahabot.core.skills.models import HttpResult, SkillConfig

REQUEST_TIMEOUT_S = 15.0
MAX_REQUEST_BYTES = 32 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_REDIRECTS = 3
REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})


def http_call(
    config: SkillConfig,
    *,
    method: str,
    url: str,
    body: bytes | None,
    headers: dict[str, str],
) -> HttpResult:
    """Execute one request inside the base-URL lock and address guard."""
    if body is not None and len(body) > MAX_REQUEST_BYTES:
        raise SkillError(f"the request body exceeds {MAX_REQUEST_BYTES} bytes")
    allow_private = allows_private(config)
    current = url
    for redirect_count in range(MAX_REDIRECTS + 1):
        if not (_origin_allowed(current, config)):
            raise EndpointRefused("the request left the skill's configured base URL")
        address = ensure_public_endpoint(current, allow_private=allow_private)
        pinned, host_headers, extensions = pinned_request(current, address)
        result, location = _send(
            method,
            pinned,
            {**headers, **host_headers},
            body,
            extensions,
        )
        if result is None:
            raise SkillError("the skill request returned no response")
        if result.status not in REDIRECT_CODES or not location:
            return result
        current = _next_redirect(current, url, location, config, redirect_count)
    raise EndpointRefused("the skill request exceeded the redirect limit")


def _next_redirect(
    current: str, first: str, location: str, config: SkillConfig, count: int
) -> str:
    if count >= MAX_REDIRECTS:
        raise EndpointRefused("the skill request exceeded the redirect limit")
    next_url = urljoin(current, location)
    if not same_origin(next_url, first) or not _origin_allowed(next_url, config):
        raise EndpointRefused(
            "a redirect changed the host; the skill's base URL still owns the request"
        )
    return next_url


def _send(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    extensions: dict[str, object],
) -> tuple[HttpResult | None, str]:
    timeout = httpx.Timeout(REQUEST_TIMEOUT_S)
    try:
        with (
            httpx.Client(timeout=timeout, follow_redirects=False) as client,
            client.stream(
                method.upper(),
                url,
                content=body,
                headers=headers,
                extensions=extensions,
            ) as response,
        ):
            chunks: list[bytes] = []
            size = 0
            truncated = False
            for chunk in response.iter_bytes():
                if size + len(chunk) > MAX_RESPONSE_BYTES:
                    chunks.append(chunk[: MAX_RESPONSE_BYTES - size])
                    truncated = True
                    break
                chunks.append(chunk)
                size += len(chunk)
            text = b"".join(chunks).decode("utf-8", errors="replace")
            return (
                HttpResult(
                    status=response.status_code,
                    text=text,
                    truncated=truncated,
                ),
                response.headers.get("location", ""),
            )
    except httpx.TimeoutException as exc:
        raise SkillError("the skill request timed out") from exc
    except httpx.HTTPError as exc:
        logger.debug("Skill HTTP transport failed: {error}", error=type(exc).__name__)
        raise SkillError("the skill request failed before a response") from exc


def _origin_allowed(url: str, config: SkillConfig) -> bool:
    return any(same_origin(url, base) for base in config.effective_base_urls)


def decode_response(result: HttpResult) -> object:
    """Parse JSON when possible, otherwise return the text."""
    if not result.truncated:
        try:
            return json.loads(result.text)
        except json.JSONDecodeError:
            pass
    return result.text
