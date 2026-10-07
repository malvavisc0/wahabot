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
from typing import Any, cast

from loguru import logger

from wahabot.ai.tools.envelope import error, ok
from wahabot.ai.tools.outfile import write_json_output
from wahabot.settings import Settings

__all__ = ["web_search"]

_MIN_TIMEOUT_SECONDS = 2.0

#: Cap on one search result's inline ``content`` snippet. webserp's
#: snippets are previewed inline. When one is cut, the selected findings
#: attempt a full-text spill; failed writes leave only the preview.
_MAX_CONTENT_CHARS = 600


def web_search(
    settings: Settings,
    query: str,
    max_results: int | None = None,
) -> str:
    """Search via webserp and return a normalized JSON envelope.

    Args:
        query: Search query text.
        max_results: Maximum total valid findings across all engines. Defaults to
            ``WAHABOT_WEB_SEARCH_MAX_RESULTS``.

    Returns:
        A JSON envelope with a ``results`` list of findings (inline
        snippets capped at ``_MAX_CONTENT_CHARS``), or an ``error``
        envelope (a failure never raises). When any snippet was cut,
        selected findings attempt a full-content ``file`` spill. The
        model can read it only if the shell tool is available; missing
        file metadata means the spill failed.
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
        findings, raw_results, failures = _parse_output(output, max_results=limit)
    except Exception as exc:
        logger.warning("web_search failed: {exc}", exc=exc)
        return error(f"web_search failed: {exc}")
    payload: dict[str, Any] = {
        "query": query,
        "count": len(findings),
        "results": findings,
    }
    if failures:
        payload.update(partial=True, failed_engines=failures)
        if not findings:
            return error(
                "no findings returned and search engines failed; search was incomplete",
                **payload,
            )
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

    webserp's count is per engine; the caller applies the total limit.
    Its default 10-second per-request timeout is independent of the
    configured subprocess deadline, allowing partial engines to finish.
    Missing binary, nonzero exit, or timeout raises to the caller's
    JSON error-envelope boundary.
    """
    if shutil.which("webserp") is None:
        raise RuntimeError("webserp CLI not found on PATH; install the 'webserp' package")
    run_timeout = max(timeout, _MIN_TIMEOUT_SECONDS)
    cmd = ["webserp", "--max-results", str(max_results)]
    if proxy:
        cmd += ["--proxy", proxy]
    cmd += ["--", query]
    logger.debug("Running webserp: {cmd}", cmd=" ".join(cmd))
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
    max_results: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    """Select at most max_results valid findings, preserving upstream order."""
    parsed = _webserp_payload(output)
    raw_results = [
        raw for raw in cast("list[Any]", parsed["results"]) if isinstance(raw, dict)
    ]
    findings, selected = _select_findings(raw_results, max_results)
    return findings, selected, _engine_failures(parsed)


def _webserp_payload(output: str) -> dict[str, Any]:
    """Parsed webserp stdout: a JSON object carrying a results list."""
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"webserp returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("webserp output must be a JSON object")
    parsed = cast(dict[str, Any], data)
    if not isinstance(parsed.get("results"), list):
        raise ValueError("webserp output missing 'results' list")
    return parsed


def _select_findings(
    raw_results: list[dict[str, Any]], max_results: int | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Capped findings plus the raw results they were built from."""
    findings: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for raw in raw_results:
        finding = _build_finding(raw, cap_content=True)
        if finding is None:
            continue
        findings.append(finding)
        selected.append(raw)
        if max_results is not None and len(findings) >= max_results:
            break
    return findings, selected


def _engine_failures(parsed: dict[str, Any]) -> list[dict[str, str]]:
    """Bounded engine/error pairs from webserp's unresponsive engines."""
    return [
        {"engine": str(item[0])[:80], "error": str(item[1])[:200]}
        for item in (parsed.get("unresponsive_engines") or [])
        if isinstance(item, list) and len(item) >= 2
    ][:20]


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
    if not isinstance(url, str) or not url or not isinstance(title, str) or not title:
        return None
    finding: dict[str, Any] = {"url": url, "title": title}
    content = raw.get("content")
    if isinstance(content, str) and content:
        if cap_content and len(content) > _MAX_CONTENT_CHARS:
            finding["content"] = content[:_MAX_CONTENT_CHARS]
            finding["content_truncated"] = True
        else:
            finding["content"] = content
    if raw.get("engine"):
        finding["engine"] = raw["engine"]
    return finding
