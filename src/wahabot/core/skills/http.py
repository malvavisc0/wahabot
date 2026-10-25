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
    url_under_base,
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
    current_method = method
    current_body = body
    for redirect_count in range(MAX_REDIRECTS + 1):
        if not origin_allowed(current, config) or not url_under_base(config, current):
            raise EndpointRefused("the request left the skill's configured base URL")
        address = ensure_public_endpoint(current, allow_private=allow_private)
        pinned, host_headers, extensions = pinned_request(current, address)
        result, location = send_pinned(
            current_method,
            pinned,
            {**headers, **host_headers},
            current_body,
            extensions,
        )
        if result is None:
            raise SkillError("the skill request returned no response")
        if result.status not in REDIRECT_CODES or not location:
            return result
        current = next_redirect(current, url, location, config, redirect_count)
        current_method, current_body = redirected_request(
            current_method, current_body, result.status
        )
        if current_body is None:
            headers.pop("Content-Type", None)
    raise EndpointRefused("the skill request exceeded the redirect limit")


def redirected_request(
    method: str, body: bytes | None, status: int
) -> tuple[str, bytes | None]:
    """HTTP redirect semantics: 303 and the 301/302 write fallback become GET."""
    upper = method.upper()
    if status == 303 and upper != "HEAD":
        return "GET", None
    if status in (301, 302) and upper not in ("GET", "HEAD"):
        return "GET", None
    return method, body


def next_redirect(
    current: str, first: str, location: str, config: SkillConfig, count: int
) -> str:
    """Resolve a Location, refusing an origin change or base-path escape."""
    if count >= MAX_REDIRECTS:
        raise EndpointRefused("the skill request exceeded the redirect limit")
    next_url = urljoin(current, location)
    if (
        not same_origin(next_url, first)
        or not origin_allowed(next_url, config)
        or not url_under_base(config, next_url)
    ):
        raise EndpointRefused(
            "a redirect changed the host or left the skill's configured base URL"
        )
    return next_url


def send_pinned(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    extensions: dict[str, object],
) -> tuple[HttpResult | None, str]:
    """Send one pinned request and read a bounded response."""
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
    except httpx.InvalidURL as exc:
        raise SkillError("the skill request URL is not valid") from exc
    except httpx.HTTPError as exc:
        logger.debug("Skill HTTP transport failed: {error}", error=type(exc).__name__)
        raise SkillError("the skill request failed before a response") from exc


def origin_allowed(url: str, config: SkillConfig) -> bool:
    """Whether a URL belongs to one of the skill's configured origins."""
    return any(same_origin(url, base) for base in config.effective_base_urls)


def decode_response(result: HttpResult) -> object:
    """Parse JSON when possible, otherwise return the text."""
    if not result.truncated:
        try:
            return json.loads(result.text)
        except json.JSONDecodeError:
            pass
    return result.text
