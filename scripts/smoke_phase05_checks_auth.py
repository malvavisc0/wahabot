"""Authentication, cookie and logout smoke checks for phase 5 web."""

import time
from pathlib import Path

from smoke_phase05_support import (
    PASSPHRASE,
    SESSION_KEY,
    check,
    client_for,
    close,
    csrf_of,
    login,
    make_db,
    settings_for,
    write_session_config,
)


def check_hash_command() -> None:
    """The hash-passphrase command emits a verifiable scrypt line."""
    from typer.testing import CliRunner

    from wahabot.cli import app
    from wahabot.web.auth import verify_passphrase

    result = CliRunner().invoke(app, ["hash-passphrase"], input=PASSPHRASE + "\n")
    line = result.stdout.strip()
    check("hash-passphrase prints an scrypt line", line.startswith("scrypt$"))
    check("the printed hash verifies", verify_passphrase(PASSPHRASE, line))
    check("a wrong passphrase does not verify", not verify_passphrase("wrong", line))


def check_fail_closed(root: Path) -> None:
    """Without hash and key the app refuses access; health stays up."""
    settings = settings_for(root / "unconfigured", configured=False)
    write_session_config(settings)
    client = client_for(settings)
    try:
        health = client.get("/health")
        login_page = client.get("/login")
        protected = client.get("/app/history")
        post = client.post("/login", data={"passphrase": PASSPHRASE, "csrf_token": "x"})
        check("health still answers", health.status_code == 200)
        check("login is refused when unconfigured", login_page.status_code == 503)
        check(
            "protected pages are refused when unconfigured", protected.status_code == 503
        )
        check("login POST is refused when unconfigured", post.status_code == 503)
        check(
            "the failure page carries no passphrase",
            PASSPHRASE not in login_page.text,
        )
    finally:
        close(client)


def check_login(root: Path) -> None:
    """Correct/wrong passphrase, CSRF, origin, rate limit and logout."""
    settings = settings_for(root / "login")
    write_session_config(settings)
    make_db(settings)
    client = client_for(settings)
    try:
        page = client.get("/login")
        token = csrf_of(page.text)
        check("login page renders", page.status_code == 200 and bool(token))
        wrong = client.post("/login", data={"passphrase": "not-it", "csrf_token": token})
        check("a wrong passphrase is refused", wrong.status_code != 303)
        check("the wrong passphrase is not echoed", "not-it" not in wrong.text)
        good = client.post(
            "/login",
            data={"passphrase": PASSPHRASE, "csrf_token": token},
            follow_redirects=False,
        )
        check("a correct passphrase signs in", good.status_code == 303)
        check("sign-in redirects to Setup", good.headers.get("location") == "/setup")
    finally:
        close(client)

    parallel = client_for(settings)
    try:
        page = parallel.get("/login")
        token = csrf_of(page.text)
        no_token = parallel.post(
            "/login", data={"passphrase": PASSPHRASE}, follow_redirects=False
        )
        check("a login POST without CSRF is refused", no_token.status_code == 403)
        cross = parallel.post(
            "/login",
            data={"passphrase": PASSPHRASE, "csrf_token": token},
            headers={"Origin": "https://evil.invalid"},
            follow_redirects=False,
        )
        check("a cross-site login POST is refused", cross.status_code == 403)
        fetch_cross = parallel.post(
            "/login",
            data={"passphrase": PASSPHRASE, "csrf_token": token},
            headers={"Sec-Fetch-Site": "cross-site"},
            follow_redirects=False,
        )
        check("Sec-Fetch-Site cross-site is refused", fetch_cross.status_code == 403)
        replay = parallel.post(
            "/login",
            data={"passphrase": PASSPHRASE, "csrf_token": token + "x"},
            follow_redirects=False,
        )
        check("a mismatched CSRF token is refused", replay.status_code == 403)
    finally:
        close(parallel)

    limited = client_for(settings)
    try:
        page = limited.get("/login")
        token = csrf_of(page.text)
        for _ in range(4):
            limited.post("/login", data={"passphrase": "wrong", "csrf_token": token})
        blocked = limited.post(
            "/login", data={"passphrase": PASSPHRASE, "csrf_token": token}
        )
        check("repeated failures are rate-limited", blocked.status_code == 429)
        check(
            "the retry delay is stated",
            "try again" in blocked.text.lower() and "retry-after" in blocked.headers,
        )
    finally:
        close(limited)


def check_cookies(root: Path) -> None:
    """A tampered or expired cookie does not grant access."""
    from wahabot.web.auth import SESSION_COOKIE, sign_payload

    settings = settings_for(root / "cookies")
    write_session_config(settings)
    make_db(settings)
    client = client_for(settings)
    try:
        login(client)
        fresh = client.get("/setup", follow_redirects=False)
        check("a signed-in cookie reaches Setup", fresh.status_code == 200)
        client.cookies.clear()
        tampered = client.get(
            "/app/history",
            cookies={SESSION_COOKIE: "tampered.value"},
            follow_redirects=False,
        )
        check("a tampered cookie redirects to login", tampered.status_code == 303)
        expired = sign_payload(
            SESSION_KEY,
            {"v": 1, "auth": True, "exp": int(time.time()) - 60, "csrf": "x"},
        )
        client.cookies.clear()
        stale = client.get(
            "/app/history", cookies={SESSION_COOKIE: expired}, follow_redirects=False
        )
        check("an expired cookie redirects to login", stale.status_code == 303)
        check(
            "media without a cookie is 401",
            client.get("/app/media/1", cookies={SESSION_COOKIE: ""}).status_code
            in {401, 403},
        )
    finally:
        close(client)


def check_logout(root: Path) -> None:
    """POST logs out; GET does not mutate the session."""
    settings = settings_for(root / "logout")
    write_session_config(settings)
    make_db(settings)
    client = client_for(settings)
    try:
        login(client)
        page = client.get("/app/teach")
        token = csrf_of(page.text)
        get_logout = client.get("/logout", follow_redirects=False)
        still_in = client.get("/app/history", follow_redirects=False)
        check("GET /logout does not mutate", get_logout.status_code in {200, 405})
        check("GET /logout leaves the session valid", still_in.status_code == 200)
        posted = client.post(
            "/logout", data={"csrf_token": token}, follow_redirects=False
        )
        check("POST /logout redirects", posted.status_code == 303)
        after = client.get("/app/history", follow_redirects=False)
        check("POST /logout clears access", after.status_code == 303)
    finally:
        close(client)
