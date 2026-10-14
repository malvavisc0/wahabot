"""Exact text-token sizing shared by prompts and notebook tools."""

from collections.abc import Callable, Sequence
from typing import Any

from llama_index.core.utils import get_tokenizer


def text_tokens(
    text: str, tokenizer: Callable[[str], Sequence[Any]] | None = None
) -> int:
    """Count text with the supplied tokenizer or LlamaIndex's default."""
    return len((tokenizer or get_tokenizer())(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """A prefix of *text* that fits within *max_tokens* tokens.

    Searched over the character cut point; the returned prefix is
    verified against the cap by construction (BPE counts are not
    strictly monotonic, so it is *a* fitting prefix, not provably the
    longest one — which the cap does not need).
    """
    if text_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if text_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]
