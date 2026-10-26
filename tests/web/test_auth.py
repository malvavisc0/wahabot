"""Authentication, cookies, CSRF, origin, rate limiting and fail-closed."""

import json
import time

from fastapi.testclient import TestClient
from tests.web.conftest import PASSPHRASE, SESSION_KEY, csrf_of, sign_in

from wahabot.web.auth import (
    SESSION_COOKIE,
    hash_passphrase,
    sign_payload,
    verify_passphrase,
)


def test_hash_roundtrip() -> None:
    """The stored hash verifies the right passphrase and rejects another."""
    encoded = hash_passphrase(PASSPHRASE)
    assert encoded.startswith("scrypt$")
    assert verify_passphrase(PASSPHRASE, encoded)
    assert not verify_passphrase("not-it", encoded)
    assert not verify_passphrase(PASSPHRASE, "scrypt$$")


def test_fail_closed_without_hash_or_key(web_client: TestClient, bot) -> None:
    """With either web secret missing no page or login POST is served."""
    original_hash = bot.settings.web_passphrase_hash
    original_key = bot.settings.web_session_key
    try:
        bot.settings.web_passphrase_hash = ""
        bot.settings.web_session_key = ""
        assert web_client.get("/health").status_code == 200
        assert web_client.get("/login").status_code == 503
        assert web_client.get("/app/history").status_code == 503
        assert (
            web_client.post("/login", data={"passphrase": PASSPHRASE}).status_code == 503
        )
        bot.settings.web_passphrase_hash = original_hash
        bot.settings.web_session_key = ""
        assert web_client.get("/login").status_code == 503
    finally:
        bot.settings.web_passphrase_hash = original_hash
        bot.settings.web_session_key = original_key


def test_wrong_and_correct_passphrase(web_client: TestClient) -> None:
    """A wrong passphrase is refused quietly; the right one signs in."""
    token = csrf_of(web_client.get("/login").text)
    wrong = web_client.post("/login", data={"passphrase": "not-it", "csrf_token": token})
    assert wrong.status_code == 401
    assert "not-it" not in wrong.text
    good = web_client.post(
        "/login",
        data={"passphrase": PASSPHRASE, "csrf_token": token},
        follow_redirects=False,
    )
    assert good.status_code == 303
    assert good.headers["location"] == "/setup"


def test_protected_page_redirects_to_login(web_client: TestClient) -> None:
    """No cookie means a page redirect, not a rendered protected page."""
    response = web_client.get("/app/history", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_login_requires_csrf(web_client: TestClient) -> None:
    """A login POST without the signed token is always refused."""
    response = web_client.post(
        "/login", data={"passphrase": PASSPHRASE}, follow_redirects=False
    )
    assert response.status_code == 403


def test_cross_site_origin_and_fetch_metadata(web_client: TestClient) -> None:
    """Origin and Sec-Fetch-Site reject a cross-site POST even with a token."""
    token = csrf_of(web_client.get("/login").text)
    origin = web_client.post(
        "/login",
        data={"passphrase": PASSPHRASE, "csrf_token": token},
        headers={"Origin": "https://evil.invalid"},
        follow_redirects=False,
    )
    assert origin.status_code == 403
    fetch = web_client.post(
        "/login",
        data={"passphrase": PASSPHRASE, "csrf_token": token},
        headers={"Sec-Fetch-Site": "cross-site"},
        follow_redirects=False,
    )
    assert fetch.status_code == 403


def test_csrf_replay_and_mismatch(web_client: TestClient) -> None:
    """A token from another session or a mutated token is refused."""
    token = csrf_of(web_client.get("/login").text)
    mismatch = web_client.post(
        "/login",
        data={"passphrase": PASSPHRASE, "csrf_token": token + "x"},
        follow_redirects=False,
    )
    assert mismatch.status_code == 403
    assert sign_in(web_client).status_code == 303
    page = web_client.get("/app/teach")
    old_token = csrf_of(page.text)
    web_client.cookies.clear()
    replay = web_client.post(
        "/logout", data={"csrf_token": old_token}, follow_redirects=False
    )
    assert replay.status_code == 403


def test_tampered_and_expired_cookies(web_client: TestClient) -> None:
    """A forged signature and a past expiry both redirect to login."""
    web_client.cookies.clear()
    tampered = web_client.get(
        "/app/history",
        cookies={SESSION_COOKIE: "tampered.value"},
        follow_redirects=False,
    )
    assert tampered.status_code == 303
    expired = sign_payload(
        SESSION_KEY,
        {"v": 1, "auth": True, "exp": int(time.time()) - 60, "csrf": "x"},
    )
    web_client.cookies.clear()
    stale = web_client.get(
        "/app/history", cookies={SESSION_COOKIE: expired}, follow_redirects=False
    )
    assert stale.status_code == 303


def test_rate_limit_blocks_correct_passphrase(web_client: TestClient) -> None:
    """Three failures inside the window return 429 with a retry delay."""
    token = csrf_of(web_client.get("/login").text)
    for _ in range(3):
        web_client.post("/login", data={"passphrase": "wrong", "csrf_token": token})
    blocked = web_client.post(
        "/login", data={"passphrase": PASSPHRASE, "csrf_token": token}
    )
    assert blocked.status_code == 429
    assert "Retry-After" in blocked.headers
    assert "try again" in blocked.text.lower()


def test_rotating_the_session_key_invalidates_cookies(
    auth_client: TestClient, bot
) -> None:
    """Rotation is the recovery path: the old cookie stops working."""
    assert auth_client.get("/app/history").status_code == 200
    old_key = bot.settings.web_session_key
    try:
        bot.settings.web_session_key = old_key + "-rotated"
        response = auth_client.get("/app/history", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/login"
    finally:
        bot.settings.web_session_key = old_key


def test_logout_is_post_only(auth_client: TestClient) -> None:
    """GET /logout never mutates; POST clears the cookie."""
    still_valid = auth_client.get("/app/teach")
    assert still_valid.status_code == 200
    token = csrf_of(still_valid.text)
    get_response = auth_client.get("/logout", follow_redirects=False)
    assert get_response.status_code == 405
    assert auth_client.get("/app/history", follow_redirects=False).status_code == 200
    post_response = auth_client.post(
        "/logout", data={"csrf_token": token}, follow_redirects=False
    )
    assert post_response.status_code == 303
    after = auth_client.get("/app/history", follow_redirects=False)
    assert after.status_code == 303


def test_plain_http_serves_instructions_not_login(web_client: TestClient) -> None:
    """A non-loopback plain HTTP request cannot log in."""
    from wahabot.webhook import app

    plain = TestClient(app, base_url="http://testserver")
    try:
        assert plain.get("/login").status_code == 403
        assert plain.get("/app/history").status_code == 403
        assert plain.get("/health").status_code == 200
    finally:
        plain.close()


def test_no_secret_in_logs_or_pages(auth_client: TestClient, bot) -> None:
    """The passphrase, session key and hash never reach audit rows or HTML."""
    token = csrf_of(auth_client.get("/app/teach").text)
    auth_client.post(
        "/app/teach/add",
        data={"question": "Q", "answer": "A", "csrf_token": token},
        follow_redirects=True,
    )
    rows = [
        json.loads(line)
        for path in (bot.settings.data_dir / "audit").rglob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    text = json.dumps(rows)
    assert PASSPHRASE not in text
    assert SESSION_KEY not in text
    assert bot.settings.web_passphrase_hash not in text
    page = auth_client.get("/app/settings")
    assert PASSPHRASE not in page.text
    assert SESSION_KEY not in page.text
    assert bot.settings.web_passphrase_hash not in page.text


def test_host_spoof_does_not_unlock_plain_http(bot) -> None:
    """A non-loopback peer stays refused even with a loopback Host header."""
    from tests.conftest import webhook_app as build_app

    bot.settings.web_passphrase_hash = hash_passphrase(PASSPHRASE)
    bot.settings.web_session_key = SESSION_KEY
    app = build_app()
    from wahabot.web import mount_routes

    mount_routes(app)
    plain = TestClient(app, base_url="http://lan.testserver")
    try:
        spoofed = plain.get("/login", headers={"Host": "localhost"})
        forwarded = plain.get("/login", headers={"X-Forwarded-Proto": "https"})
    finally:
        plain.close()
    assert spoofed.status_code == 403
    assert forwarded.status_code == 403


def test_forwarded_scheme_trusted_only_from_configured_proxies(bot) -> None:
    """X-Forwarded-Proto counts only from a configured proxy address."""
    from starlette.requests import Request

    from wahabot.web.auth import cookie_secure, is_secure

    def request_with(peer: str, forwarded: str) -> Request:
        return Request(
            {
                "type": "http",
                "method": "GET",
                "scheme": "http",
                "path": "/login",
                "query_string": b"",
                "root_path": "",
                "http_version": "1.1",
                "client": (peer, 50000),
                "server": ("10.0.0.5", 8080),
                "headers": [(b"x-forwarded-proto", forwarded.encode())],
            }
        )

    proxied = request_with("172.18.0.5", "https")
    mapped = request_with("::ffff:172.18.0.5", "https")
    mapped_loopback = request_with("::ffff:127.0.0.1", "")
    direct = request_with("192.168.1.9", "https")
    loopback = request_with("127.0.0.1", "")
    bot.settings.web_trusted_proxy_hosts = ""
    try:
        assert is_secure(proxied, bot.settings) is False
        assert cookie_secure(proxied, bot.settings) is False
        bot.settings.web_trusted_proxy_hosts = "172.18.0.0/16"
        assert is_secure(proxied, bot.settings) is True
        assert cookie_secure(proxied, bot.settings) is True
        assert is_secure(mapped, bot.settings) is True
        assert is_secure(direct, bot.settings) is False
        assert is_secure(loopback, bot.settings) is True
        assert is_secure(mapped_loopback, bot.settings) is True
    finally:
        bot.settings.web_trusted_proxy_hosts = ""


def test_non_ascii_csrf_token_is_refused_not_crashed(web_client: TestClient) -> None:
    """A non-ASCII CSRF token never raises a 500."""
    web_client.get("/login")
    token = web_client.post("/login", data={"passphrase": "x", "csrf_token": "nö-ascii"})
    assert token.status_code == 403


def test_non_ascii_cookie_signature_is_rejected() -> None:
    """verify_payload rejects a non-ASCII signature instead of raising."""
    from wahabot.web.auth import verify_payload

    assert verify_payload(SESSION_KEY, "abc.sïgnature") is None
