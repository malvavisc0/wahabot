"""The unified JSON envelope returned by every agent tool.

Every tool returns a compact JSON string so the model can parse the
result deterministically instead of scraping free-form status text::

    {"ok": true,  ...payload}        # success
    {"ok": false, "error": "..."}    # failure

``ok`` is the only guaranteed key. Success payloads add tool-specific
fields (``chat``, ``text``, ``messages``, ``results``, ``exit_code``, …);
failures always carry a human-readable ``error``. All tools build the
envelope through :func:`ok` / :func:`error` so no tool hand-rolls JSON.
"""

import json
from typing import Any

__all__ = ["error", "ok"]


def ok(**payload: Any) -> str:
    """The success envelope: ``{"ok": true, ...payload}`` as a JSON string."""
    return json.dumps({"ok": True, **payload}, ensure_ascii=False)


def error(message: str, **payload: Any) -> str:
    """The failure envelope: ``{"ok": false, "error": message, ...payload}``.

    *payload* is for context worth preserving even on failure — e.g. a
    shell command that timed out still reports its exit code, captured
    partial output and spill file so nothing already captured is lost.
    """
    return json.dumps({"ok": False, "error": message, **payload}, ensure_ascii=False)
