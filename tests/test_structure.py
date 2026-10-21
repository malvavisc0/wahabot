"""Repository structure checks."""

from pathlib import Path

MAX_LINES = 600
ROOT = Path(__file__).resolve().parents[1]


def test_python_files_stay_under_the_line_cap() -> None:
    offenders = [
        f"{path.relative_to(ROOT)}: {len(path.read_text().splitlines())} lines"
        for path in sorted((ROOT / "src").rglob("*.py"))
        + sorted((ROOT / "tests").rglob("*.py"))
        if len(path.read_text().splitlines()) > MAX_LINES
    ]
    assert not offenders, "Files over the 600-line cap:\n" + "\n".join(offenders)
