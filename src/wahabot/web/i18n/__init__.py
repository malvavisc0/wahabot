"""Gettext support for the operator interface."""

import gettext
from functools import lru_cache
from pathlib import Path

LOCALE_DIR = Path(__file__).resolve().parent
LANGUAGE = "en"


@lru_cache(maxsize=1)
def translator() -> gettext.NullTranslations:
    """Load the English catalog, falling back to message ids."""
    return gettext.translation(
        "messages",
        localedir=str(LOCALE_DIR),
        languages=[LANGUAGE],
        fallback=True,
    )


def _(message: str) -> str:
    """Translate one operator-facing string."""
    return translator().gettext(message)
