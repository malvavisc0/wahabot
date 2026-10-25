"""Smoke test: the v1.2.0 phase 5 web back office acceptance.

Run:  uv run python scripts/smoke_phase05_web.py
Exit: 0 = smoke passed, 1 = failed.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smoke_phase05_checks_app import (
    check_history,
    check_journal,
    check_settings,
    check_setup,
    check_skills,
    check_teach,
)
from smoke_phase05_checks_auth import (
    check_cookies,
    check_fail_closed,
    check_hash_command,
    check_login,
    check_logout,
)
from smoke_phase05_support import (
    API_FAILURE,
    check,
    failures,
    fake_rest,
    fake_waha,
    guarded,
)


def check_layout() -> None:
    """Run the Playwright layout smoke in its own process."""
    script = Path(__file__).with_name("smoke_phase05_layout.py")
    result = subprocess.run([sys.executable, str(script)], check=False)
    check("Playwright layout at 320/390/768/1024/1440", result.returncode == 0)


def main() -> int:
    """Run the phase checks and report."""
    print("wahabot phase 5 web smoke")
    try:
        import wahabot.web  # noqa: F401
    except Exception as exc:
        print(f"[FAIL] {API_FAILURE} — {type(exc).__name__}: {exc}")
        return 1
    from tests.harness import serving

    waha_app = fake_waha()
    rest_app = fake_rest()
    with tempfile.TemporaryDirectory(prefix="wahabot-smoke-web-") as raw:
        root = Path(raw)
        with serving(waha_app) as waha_port, serving(rest_app) as rest_port:
            guarded("hash command", check_hash_command)
            guarded("fail closed", lambda: check_fail_closed(root))
            guarded("login", lambda: check_login(root))
            guarded("cookies", lambda: check_cookies(root))
            guarded("logout", lambda: check_logout(root))
            guarded("setup", lambda: check_setup(root, waha_port))
            guarded("history", lambda: check_history(root))
            guarded("teach", lambda: check_teach(root))
            guarded("skills", lambda: check_skills(root, rest_port))
            guarded("settings", lambda: check_settings(root, waha_port))
            guarded("journal", lambda: check_journal(root))
    guarded("layout", check_layout)
    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("\nAll phase 5 web smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
