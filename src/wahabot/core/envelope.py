"""The unified JSON envelope returned by every agent tool.

Every tool returns a compact JSON string so the model can parse the
result deterministically instead of scraping free-form status text::

    {"ok": true,  ...payload}        # success
    {"ok": false, "error": "..."}    # failure

``ok`` is the only guaranteed key. Success payloads add tool-specific
fields; failures always carry a human-readable ``error``. ``ok`` describes
tool execution: a completed shell command's own success uses ``exit_code``.
"""

import json
from typing import Any

__all__ = ["error", "ok"]


def ok(**payload: Any) -> str:
    """The success envelope: ``{"ok": true, ...payload}`` as a JSON string."""
    return json.dumps({"ok": True, **payload}, ensure_ascii=False)


def error(message: str, **payload: Any) -> str:
    """The failure envelope: ``{"ok": false, "error": message, ...payload}``."""
    return json.dumps({"ok": False, "error": message, **payload}, ensure_ascii=False)
