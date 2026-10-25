"""Playwright layout smoke for the phase 5 web app at five viewport widths.

Run after the core smoke:  uv run python scripts/smoke_phase05_layout.py
Requires the Playwright Chromium build:  uv run playwright install chromium
"""

import re
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.sync_api import Page, expect, sync_playwright
from tests.harness import (
    RecordingWaha,
    serving,
    smoke_settings,
    write_session_config,
)

from wahabot.core.db import Database
from wahabot.core.tickets.store import TicketStore
from wahabot.web import mount_routes
from wahabot.web.auth import hash_passphrase

PASSPHRASE = "bridge-club-open-sesame"
SESSION_KEY = "layout-session-key-32-bytes-long!!!!"
WIDTHS = (320, 390, 768, 1024, 1440)
PAGES = (
    ("/setup?reconnect=1", "Set up WhatsApp"),
    ("/app/history", "History"),
    ("/app/teach", "Teach"),
    ("/app/skills", "Skills"),
    ("/app/settings", "Settings"),
)
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    """Record one layout check."""
    mark = "PASS" if ok else "FAIL"
    suffix = f" — {detail}" if detail and not ok else ""
    print(f"[{mark}] {label}{suffix}")
    if not ok:
        failures.append(label)


def settings_for(data_dir: Path) -> Any:
    """Configured settings for the layout server."""
    settings = smoke_settings(data_dir, "http://llm.invalid/v1")
    settings.web_passphrase_hash = hash_passphrase(PASSPHRASE)
    settings.web_session_key = SESSION_KEY
    settings.web_login_attempts = 5
    settings.web_login_window_s = 60
    return settings


def seed(settings: Any) -> None:
    """One ticket so History has a real row with text and metadata."""
    database = Database(settings.data_dir / "wahabot.db")
    database.migrate()
    store = TicketStore(database.path)
    contact = store.upsert_contact(
        "491555000001@c.us",
        phone="491555000001@c.us",
        display_name="Bridge Club with a deliberately long operator-facing name",
        language="es",
    )
    ticket = store.create_ticket(
        contact.id, contact.chat_id, summary="Opening times", urgent=True
    )
    store.add_event(ticket.id, "customer", "customer_message", "hola, abren el lunes?")
    store.add_event(ticket.id, "assistant", "reply", "si, de 9 a 17")


def sign_in(page: Page, base: str) -> None:
    """Sign in and accept Setup or History as the destination."""
    page.goto(f"{base}/login")
    expect(page.locator("#passphrase")).to_be_visible()
    page.fill("#passphrase", PASSPHRASE)
    page.click("button[type=submit]")
    expect(page).to_have_url(re.compile(r".*/(setup|app/history)$"))


def no_overflow(page: Page, label: str) -> None:
    """The document must not scroll sideways."""
    width = page.evaluate("() => document.documentElement.scrollWidth")
    viewport = page.viewport_size["width"]
    check(label, width <= viewport + 1, f"{width}px document in {viewport}px viewport")


def check_widths(page: Page, base: str, screenshots: Path) -> None:
    """Every page fits at every accepted width."""
    for width in WIDTHS:
        page.set_viewport_size({"width": width, "height": 900})
        for path, heading in PAGES:
            page.goto(f"{base}{path}")
            expect(page.locator("h1.page-title")).to_have_text(heading)
            no_overflow(page, f"{path} at {width}px")
        if width == 390:
            page.goto(f"{base}/app/settings")
            page.screenshot(path=str(screenshots / "settings-390.png"), full_page=True)
            page.goto(f"{base}/app/history")
            page.screenshot(path=str(screenshots / "history-390.png"), full_page=True)


def check_login_controls(page: Page, base: str) -> None:
    """Tap targets and keyboard sign-in at 390, before the session exists."""
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"{base}/login")
    field = page.locator("#passphrase").bounding_box()
    button = page.locator("button[type=submit]").bounding_box()
    check("passphrase is at least 44px", bool(field and field["height"] >= 44))
    check("sign-in is at least 44px", bool(button and button["height"] >= 44))
    page.focus("#passphrase")
    page.keyboard.type(PASSPHRASE)
    page.keyboard.press("Enter")
    expect(page).to_have_url(re.compile(r".*/(setup|app/history)$"))


def check_controls(page: Page, base: str) -> None:
    """Dirty marker, hash focus and no JavaScript errors."""
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"{base}/app/settings")
    page.fill("#bot_name", "Bridge Club Desk")
    dirty = page.locator("#assistant form").evaluate(
        "el => el.classList.contains('is-dirty')"
    )
    check("a changed form is marked unsaved", dirty)
    page.goto(f"{base}/app/history")
    page.click("a[href='/app/history?show=needs_you#needs-you']")
    expect(page.locator("#needs-you")).to_be_focused()
    check("the needs-you shortcut focuses its heading", True)
    check("no JavaScript errors on the main pages", errors == [])


def main() -> int:
    """Serve the app, run Playwright and report."""
    print("wahabot phase 5 web layout smoke")
    import wahabot.webhook as webhook_module

    with tempfile.TemporaryDirectory(prefix="wahabot-layout-") as raw:
        root = Path(raw)
        settings = settings_for(root)
        write_session_config(settings)
        seed(settings)
        waha = RecordingWaha()
        waha.session_status = "SCAN_QR_CODE"
        app = webhook_module.app
        mount_routes(app)
        app.state.web_waha_client = waha
        app.state.web_limiter = None
        with (
            patch("wahabot.settings.get_settings", return_value=settings),
            serving(app, lifespan="off") as port,
        ):
            base = f"http://127.0.0.1:{port}"
            screenshots = root / "screenshots"
            screenshots.mkdir()
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page()
                try:
                    check_login_controls(page, base)
                    check_widths(page, base, screenshots)
                    check_controls(page, base)
                finally:
                    browser.close()
            print(f"screenshots: {screenshots}")
        app.state.web_limiter = None
        if hasattr(app.state, "web_waha_client"):
            delattr(app.state, "web_waha_client")
    if failures:
        print(f"\n{len(failures)} layout check(s) failed")
        return 1
    print("\nAll phase 5 web layout checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
