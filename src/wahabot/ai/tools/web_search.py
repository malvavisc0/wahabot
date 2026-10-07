"""Web search via the ``webserp`` metasearch CLI.

``webserp`` is a light metasearch CLI that queries Google, DuckDuckGo,
Brave, Yahoo, Mojeek, Startpage and Presearch in parallel, using browser
impersonation (curl_cffi) and **no API key**. It is invoked as a
subprocess and its JSON output is normalised into the shared JSON
envelope, matching wahabot's other tools (which return structured JSON
rather than raising).

The tool is built with a ``Settings`` so operators can tune timeout,
result count and an optional proxy without touching code.
"""

import json
import shutil
import subprocess
from typing import Any

from loguru import logger

from wahabot.ai.tools.envelope import error, ok
from wahabot.ai.tools.outfile import write_json_output
from wahabot.settings import Settings

__all__ = ["web_search"]

_MIN_TIMEOUT_SECONDS = 2.0

#: Cap on one search result's inline ``content`` snippet. webserp's
#: snippets are the only unbounded field across all tools — every other
#: payload self-limits at its source. The inline cap keeps the envelope
#: small, but the full findings ride a spill file when any snippet was
#: cut, so nothing is silently lost (see :func:`_spill_findings`).
_MAX_CONTENT_CHARS = 600


def web_search(
    settings: Settings,
    query: str,
    max_results: int | None = None,
) -> str:
    """Search the web via webserp and return normalised results as text.

    Args:
        query: Search query text.
        max_results: Maximum results to return. Defaults to
            ``WAHABOT_WEB_SEARCH_MAX_RESULTS``.

    Returns:
        A JSON envelope with a ``results`` list of findings (inline
        snippets capped at ``_MAX_CONTENT_CHARS``), or an ``error``
        envelope (a failure never raises). When any snippet was cut,
        the full findings also ride ``file`` (a temp JSON file) so the
        model can read the rest via the shell tool.
    """
    if not query.strip():
        return error("query cannot be empty")
    limit = settings.web_search_max_results if max_results is None else max_results
    if limit < 1:
        return error("max_results must be positive")
    try:
        output = _run_webserp(
            query=query,
            max_results=limit,
            timeout=settings.web_search_timeout,
            proxy=settings.web_search_proxy,
        )
        findings, raw_results = _parse_output(output)
    except Exception as exc:
        logger.warning("web_search failed: {exc}", exc=exc)
        return error(f"web_search failed: {exc}")
    payload: dict[str, Any] = {
        "query": query,
        "count": len(findings),
        "results": findings,
    }
    payload.update(_spill_findings(query, raw_results, findings))
    return ok(**payload)


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


def _parse_output(
    output: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse stdout once into inline findings and raw results for spilling."""
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"webserp returned invalid JSON: {exc}") from exc

    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("webserp output missing 'results' list")

    raw_results = [raw for raw in raw_results if isinstance(raw, dict)]
    findings: list[dict[str, Any]] = []
    for raw in raw_results:
        finding = _build_finding(raw, cap_content=True)
        if finding is not None:
            findings.append(finding)
    return findings, raw_results


def _spill_findings(
    query: str, raw_results: list[dict[str, Any]], findings: list[dict[str, Any]]
) -> dict[str, Any]:
    """Envelope fields preserving any snippet cut past the inline cap.

    :func:`_build_finding` caps each inline snippet at
    ``_MAX_CONTENT_CHARS`` and flags it ``content_truncated``, but the
    trimmed tails must not vanish: when any snippet was cut, the full
    findings (uncut content, rebuilt from the raw results) ride a temp
    JSON file (``file``) the model can dereference in parts. Fail-soft:
    a write error just leaves the envelope inline-only.
    """
    if not any(finding.get("content_truncated") for finding in findings):
        return {}
    full = [
        finding
        for raw in raw_results
        if (finding := _build_finding(raw, cap_content=False)) is not None
    ]
    try:
        return {"file": write_json_output("search", {"query": query, "results": full})}
    except Exception as exc:
        logger.warning("could not spill web_search results to file: {exc}", exc=exc)
        return {}


def _build_finding(
    raw: dict[str, Any], cap_content: bool = True
) -> dict[str, Any] | None:
    """Build a normalised finding from a raw webserp result.

    With *cap_content* the inline snippet is capped at
    ``_MAX_CONTENT_CHARS`` and flagged ``content_truncated`` when cut;
    the uncapped variant is what reaches the spill file
    (:func:`_spill_findings`), so the envelope itself never carries the
    bulk.
    """
    url = raw.get("url")
    title = raw.get("title")
    if not url or not title:
        return None
    finding: dict[str, Any] = {"url": url, "title": title}
    if content := raw.get("content"):
        if cap_content and len(content) > _MAX_CONTENT_CHARS:
            finding["content"] = content[:_MAX_CONTENT_CHARS]
            finding["content_truncated"] = True
        else:
            finding["content"] = content
    if raw.get("engine"):
        finding["engine"] = raw["engine"]
    return finding
