"""Slim the OpenAI tool schemas the LLM sees.

Remove auto-generated parameter titles and redundant ``strict: false``.
Nullable primitive/array unions use a compact type array, retaining
explicit null support; omission and null are not interchangeable in a
schema. Real unions, strict mode, and validation constraints are kept.

Slimming runs on the serialized dict, never on the Pydantic models,
so validation and the tool-call path are untouched: only what the LLM
reads changes.
"""

import copy
from typing import Any, cast

__all__ = ["slim_tool_spec", "slim_tool_specs"]


def slim_tool_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """One OpenAI tool spec without the padding the LLM does not read.

    ``title`` keys vanish at every depth; a simple nullable ``anyOf``
    becomes ``"type": [X, "null"]`` without changing allowed values. A
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
    """A JSON-schema node without titles and with compact nullable type arrays."""
    if isinstance(node, dict):
        return _slim_object(node)
    if isinstance(node, list):
        return [slim_schema(item) for item in node]
    return node


def nullable_union(union: Any) -> dict[str, Any] | None:
    """A compact, equivalent nullable type schema, or None.

    A two-member union whose second arm is exactly ``{"type": "null"}`
    and whose first arm has only type/items becomes a type array. Arms
    with enums or other constraints stay as unions so null remains valid.
    """
    if not isinstance(union, list) or len(union) != 2:
        return None
    typed: Any = union[0]
    null_arm: Any = union[1]
    if not isinstance(typed, dict) or null_arm != {"type": "null"}:
        return None
    arm = cast(dict[str, Any], typed)
    if isinstance(arm.get("type"), str) and set(arm) <= {"type", "items"}:
        return arm | {"type": [arm["type"], "null"]}
    return None


def _slim_object(node: dict[str, Any]) -> dict[str, Any]:
    """One schema object: drop titles, compact only equivalent nullable unions."""
    slimmed = {k: slim_schema(v) for k, v in node.items() if k != "title"}
    if (typed := nullable_union(slimmed.get("anyOf"))) is not None:
        slimmed = {k: v for k, v in slimmed.items() if k != "anyOf"} | typed
    return slimmed
