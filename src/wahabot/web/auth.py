"""Local passphrase auth, signed cookies, CSRF and failed-login limiting."""

import base64
import hashlib
import hmac
import ipaddress
import json
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any

from fastapi import Request, Response
from starlette.responses import HTMLResponse

from wahabot.settings import Settings

SESSION_COOKIE = "wahabot_web"
SESSION_HOURS = 12
CSRF_HOURS = 1
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32


@dataclass(frozen=True, slots=True)
class AuthSession:
    """The verified state of one browser cookie."""

    csrf: str
    expires: int
    authenticated: bool


def hash_passphrase(passphrase: str) -> str:
    """Return the ``scrypt$<salt>$<hash>`` line stored in ``.env``."""
    if not passphrase:
        raise ValueError("a passphrase is required")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        passphrase.encode(),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=SCRYPT_DKLEN,
    )
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_passphrase(passphrase: str, encoded: str) -> bool:
    """Verify *passphrase* against a stored scrypt line in constant time."""
    parts = encoded.split("$")
    if len(parts) != 3 or parts[0] != "scrypt":
        return False
    try:
        salt = bytes.fromhex(parts[1])
        expected = bytes.fromhex(parts[2])
    except ValueError:
        return False
    if not salt or not expected:
        return False
    digest = hashlib.scrypt(
        passphrase.encode(),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=len(expected),
    )
    return hmac.compare_digest(digest, expected)


def configured(settings: Settings) -> bool:
    """Both the hash and the session key must be present; otherwise fail closed."""
    return bool(settings.web_passphrase_hash.strip() and settings.web_session_key.strip())


def sign_payload(key: str, payload: dict[str, Any]) -> str:
    """Encode and HMAC-SHA256 sign one cookie payload."""
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    body = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    signature = hmac.new(key.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{signature}"


def verify_payload(key: str, value: str) -> dict[str, Any] | None:
    """Verify a cookie's signature and decode its JSON object."""
    if not key or "." not in value:
        return None
    body, _, signature = value.rpartition(".")
    try:
        expected = hmac.new(key.encode(), body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature.encode(), expected.encode()):
            return None
        padding = "=" * (-len(body) % 4)
        raw = base64.urlsafe_b64decode(body + padding)
        payload = json.loads(raw)
    except ValueError, TypeError:
        return None
    return payload if isinstance(payload, dict) else None


def read_session(request: Request, settings: Settings) -> AuthSession | None:
    """The request's valid cookie state, or None when missing/tampered/expired."""
    if not settings.web_session_key.strip():
        return None
    cookie = request.cookies.get(SESSION_COOKIE, "")
    payload = verify_payload(settings.web_session_key, cookie)
    if payload is None:
        return None
    csrf = str(payload.get("csrf") or "")
    expires = int(payload.get("exp") or 0)
    if not csrf or expires <= int(time.time()):
        return None
    return AuthSession(
        csrf=csrf,
        expires=expires,
        authenticated=bool(payload.get("auth")),
    )


def issue_session_cookie(
    request: Request,
    response: Response,
    settings: Settings,
    csrf: str,
) -> None:
    """Issue the authenticated HttpOnly cookie for this browser."""
    payload = {
        "v": 1,
        "auth": True,
        "exp": int(time.time()) + SESSION_HOURS * 3600,
        "csrf": csrf,
    }
    response.set_cookie(
        SESSION_COOKIE,
        sign_payload(settings.web_session_key, payload),
        max_age=SESSION_HOURS * 3600,
        httponly=True,
        samesite="lax",
        secure=cookie_secure(request, settings),
        path="/",
    )


def issue_csrf_cookie(
    request: Request,
    response: Response,
    settings: Settings,
    csrf: str,
) -> None:
    """Issue the unauthenticated cookie that only carries CSRF state."""
    payload = {
        "v": 1,
        "auth": False,
        "exp": int(time.time()) + CSRF_HOURS * 3600,
        "csrf": csrf,
    }
    response.set_cookie(
        SESSION_COOKIE,
        sign_payload(settings.web_session_key, payload),
        max_age=CSRF_HOURS * 3600,
        httponly=True,
        samesite="lax",
        secure=cookie_secure(request, settings),
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    """Remove the browser cookie."""
    response.delete_cookie(SESSION_COOKIE, path="/", samesite="lax")


def csrf_token(request: Request, settings: Settings) -> str:
    """The request's CSRF token, or a fresh one when it has no cookie."""
    session = read_session(request, settings)
    return session.csrf if session is not None else secrets.token_urlsafe(24)


def csrf_matches(request: Request, settings: Settings, submitted: str) -> bool:
    """Compare the submitted token with the signed cookie's token."""
    session = read_session(request, settings)
    if session is None or not submitted:
        return False
    return hmac.compare_digest(session.csrf.encode(), submitted.encode())


def peer_address(
    peer: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """The parsed peer address, with IPv4-mapped IPv6 unwrapped."""
    try:
        address = ipaddress.ip_address(peer)
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def peer_is_trusted_proxy(peer: str, settings: Settings) -> bool:
    """Whether the direct peer is a configured trusted proxy address."""
    address = peer_address(peer)
    if address is None:
        return False
    for entry in settings.web_trusted_proxy_hosts.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            if address in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue
    return False


def forwarded_scheme(request: Request, settings: Settings) -> str:
    """The X-Forwarded-Proto scheme, honored only from a trusted proxy."""
    if request.client is None or not peer_is_trusted_proxy(request.client.host, settings):
        return ""
    forwarded = request.headers.get("x-forwarded-proto", "")
    return forwarded.split(",", 1)[0].strip().lower()


def cookie_secure(request: Request, settings: Settings) -> bool:
    """The cookie's Secure flag follows the actual transport, not loopback policy."""
    if request.url.scheme == "https":
        return True
    return forwarded_scheme(request, settings) == "https"


def is_secure(request: Request, settings: Settings) -> bool:
    """Whether the request arrived over HTTPS, a trusted proxy or loopback.

    Loopback is decided by the parsed socket peer, never by the
    client's Host header, and the forwarded scheme counts only when
    the peer is a configured proxy.
    """
    forwarded = forwarded_scheme(request, settings)
    if forwarded:
        return forwarded == "https"
    if request.url.scheme == "https":
        return True
    peer = request.client.host if request.client is not None else ""
    address = peer_address(peer)
    return address is not None and address.is_loopback


def origin_ok(request: Request, settings: Settings) -> bool:
    """Reject cross-site Origin and Sec-Fetch-Site values."""
    site = request.headers.get("sec-fetch-site", "").strip().lower()
    if site and site not in {"same-origin", "none"}:
        return False
    origin = request.headers.get("origin", "").strip().rstrip("/")
    if not origin:
        return True
    allowed = {str(request.base_url).rstrip("/")}
    if settings.public_origin.strip():
        allowed.add(settings.public_origin.strip().rstrip("/"))
    return origin in allowed


def client_key(request: Request) -> str:
    """The direct client address; forwarded headers never change the key."""
    return request.client.host if request.client is not None else "unknown"


class RateLimiter:
    """In-memory failed-attempt window per client."""

    def __init__(self, limit: int, window_s: int) -> None:
        self.limit = limit
        self.window_s = window_s
        self.failures: dict[str, list[float]] = {}
        self.lock = threading.Lock()

    def retry_after(self, key: str, now: float | None = None) -> int | None:
        """Seconds to wait, or None when the client may try again."""
        moment = time.time() if now is None else now
        with self.lock:
            recent = [
                item
                for item in self.failures.get(key, [])
                if moment - item < self.window_s
            ]
            self.failures[key] = recent
            if len(recent) < self.limit:
                return None
            return max(1, int(self.window_s - (moment - recent[0])) + 1)

    def record_failure(self, key: str, now: float | None = None) -> None:
        """Record one failed attempt."""
        moment = time.time() if now is None else now
        with self.lock:
            self.failures.setdefault(key, []).append(moment)

    def reset(self, key: str | None = None) -> None:
        """Clear one client's failures, or every client's."""
        with self.lock:
            if key is None:
                self.failures.clear()
            else:
                self.failures.pop(key, None)


def forbidden() -> HTMLResponse:
    """A small non-leaking 403 body for failed CSRF checks."""
    return HTMLResponse("<p>This request was refused.</p>", status_code=403)
