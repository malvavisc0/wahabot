"""Slim the OpenAI tool schemas the LLM sees.

Pydantic's ``model_json_schema()`` (what llama-index serializes into
every request) pads each parameter with fields the model has no use
for: auto-generated ``title`` strings (``"title": "Chat"`` — the name
already says it), ``anyOf: [{type: X}, {type: null}]`` unions for
every optional parameter (five tokens of structure where
``"type": "X"`` plus an absent value does the same job on the
non-strict wire), and a per-tool ``strict: false`` that the wire
format treats as the default anyway. Measured on the ten bundled
tools: ~360 tokens of every request, ~13% of the tool block, pure
JSON scaffolding (docs/bug-report-2c665d8.md sizing).

Slimming runs on the serialized dict, never on the Pydantic models,
so validation and the tool-call path are untouched: only what the LLM
reads changes.
"""

import copy
from typing import Any, cast

__all__ = ["slim_tool_spec", "slim_tool_specs"]


def slim_tool_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """One OpenAI tool spec without the padding the LLM does not read.

    ``title`` keys vanish at every depth; ``anyOf: [{type: X}, {type:
    null}]`` collapses to ``"type": "X"`` (the ``null`` arm only
    matters to strict mode, which the request does not use); a
    top-level ``strict: false`` goes (absent means false). Everything
    else — names, descriptions, defaults, ``required``, enums —
    passes through untouched.
    """
    function: dict[str, Any] | None = spec.get("function")
    if not isinstance(function, dict):
        return spec
    slimmed: dict[str, Any] = copy.deepcopy(spec)
    fn = slimmed["function"]
    if fn.get("strict") is False:
        del fn["strict"]
    fn["parameters"] = slim_schema(fn.get("parameters"))
    return slimmed


def slim_tool_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Slim every tool spec of a request's ``tools`` payload."""
    return [slim_tool_spec(spec) for spec in specs]


def slim_schema(node: Any) -> Any:
    """A JSON-schema node without ``title`` keys and nullable unions."""
    if isinstance(node, dict):
        return _slim_object(node)
    if isinstance(node, list):
        return [slim_schema(item) for item in node]
    return node


def nullable_union(union: Any) -> dict[str, Any] | None:
    """The single typed arm of ``anyOf: [X, null]``, or None.

    A two-member union whose second arm is exactly ``{"type": "null"}`
    and whose first arm carries a plain ``type`` collapses to that
    arm; every other union (a real multi-type choice) is returned as
    None and stays intact.
    """
    if not isinstance(union, list) or len(union) != 2:
        return None
    typed: Any = union[0]
    null_arm: Any = union[1]
    if not isinstance(typed, dict) or null_arm != {"type": "null"}:
        return None
    arm = cast(dict[str, Any], typed)
    if isinstance(arm.get("type"), str):
        return arm
    return None


def _slim_object(node: dict[str, Any]) -> dict[str, Any]:
    """One schema object: drop ``title``, collapse nullable ``anyOf``."""
    slimmed = {k: slim_schema(v) for k, v in node.items() if k != "title"}
    if (typed := nullable_union(slimmed.get("anyOf"))) is not None:
        slimmed = {k: v for k, v in slimmed.items() if k != "anyOf"} | typed
    return slimmed
