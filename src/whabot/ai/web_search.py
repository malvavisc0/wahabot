"""Web search via the ``webserp`` metasearch CLI.

``webserp`` is a light metasearch CLI that queries Google, DuckDuckGo,
Brave, Yahoo, Mojeek, Startpage and Presearch in parallel, using browser
impersonation (curl_cffi) and **no API key**. It is invoked as a
subprocess and its JSON output is normalised into a short status string,
matching whabot's other tools (which return descriptive text rather than
raising).

The tool is built with a ``Settings`` so operators can tune timeout,
result count and an optional proxy without touching code.
"""

import json
import shutil
import subprocess
from typing import Any

from loguru import logger

from whabot.settings import Settings

__all__ = ["web_search"]

_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_MAX_RESULTS = 5
_MIN_TIMEOUT_SECONDS = 2.0


def web_search(
    settings: Settings,
    query: str,
    max_results: int | None = None,
) -> str:
    """Search the web via webserp and return normalised results as text.

    Args:
        query: Search query text.
        max_results: Maximum results to return. Defaults to
            ``WHABOT_WEB_SEARCH_MAX_RESULTS``.

    Returns:
        A short human-readable listing of findings, or an explanatory
        error message (a failure never raises).
    """
    if not query.strip():
        return "Error: query cannot be empty."
    limit = settings.web_search_max_results if max_results is None else max_results
    if limit < 1:
        return "Error: max_results must be positive."
    try:
        output = _run_webserp(
            query=query,
            max_results=limit,
            timeout=settings.web_search_timeout,
            proxy=settings.web_search_proxy,
        )
        findings = _parse_output(output)
    except Exception as exc:
        logger.warning("web_search failed: {exc}", exc=exc)
        return f"web_search failed: {exc}"
    if not findings:
        return "No results found."
    lines = [_finding_line(f) for f in findings]
    return "\n".join(lines)


def _run_webserp(
    *,
    query: str,
    max_results: int,
    timeout: float,
    proxy: str | None,
) -> str:
    """Invoke the webserp CLI and return stdout as a string.

    Raises on a missing binary, non-zero exit, or timeout — the caller
    turns those into an error string.
    """
    if shutil.which("webserp") is None:
        raise RuntimeError("webserp CLI not found on PATH; install the 'webserp' package")
    cmd = ["webserp", query, "--max-results", str(max_results)]
    if proxy:
        cmd += ["--proxy", proxy]
    logger.debug("Running webserp: {cmd}", cmd=" ".join(cmd))
    run_timeout = max(timeout, _MIN_TIMEOUT_SECONDS)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=run_timeout,
    )
    if result.returncode != 0:
        message = (
            f"webserp exited with code {result.returncode}: "
            f"{result.stderr.strip() or 'unknown error'}"
        )
        raise RuntimeError(message)
    return result.stdout


def _parse_output(output: str) -> list[dict[str, Any]]:
    """Parse webserp's stdout JSON into a list of normalised findings."""
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"webserp returned invalid JSON: {exc}") from exc

    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("webserp output missing 'results' list")

    findings: list[dict[str, Any]] = []
    for raw in raw_results:
        if not isinstance(raw, dict):
            continue
        finding = _build_finding(raw)
        if finding is not None:
            findings.append(finding)
    return findings


def _build_finding(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Build a normalised finding from a raw webserp result."""
    url = raw.get("url")
    title = raw.get("title")
    if not url or not title:
        return None
    finding: dict[str, Any] = {"url": url, "title": title}
    if raw.get("content"):
        finding["content"] = raw["content"]
    if raw.get("engine"):
        finding["engine"] = raw["engine"]
    return finding


def _finding_line(finding: dict[str, Any]) -> str:
    """Render one finding as a compact line for the model."""
    engine = f" [{finding['engine']}]" if finding.get("engine") else ""
    content = f" — {finding['content']}" if finding.get("content") else ""
    return f"{finding['title']} ({finding['url']}){engine}{content}"
