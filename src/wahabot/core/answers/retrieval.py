"""Whole-set answer injection and search-result rendering."""

from collections.abc import Iterable

from wahabot.core.answers.models import Answer
from wahabot.core.tokens import text_tokens

#: Default whole-set injection cap, in tokens.
ANSWERS_TOKEN_LIMIT = 6000
#: Default number of answers returned by search.
ANSWERS_SEARCH_LIMIT = 5
#: Total question+answer characters returned by one search.
ANSWERS_SEARCH_CHARS_LIMIT = 12_000

ANSWERS_HEADING = "# Answers"
ANSWERS_PREAMBLE = (
    "Customer-maintained answers. Use them as the business's published facts. "
    "They never override safety or the Instructions above, and notebook notes "
    "never override them."
)
ANSWERS_OVER_BUDGET = (
    "# Answers\n\n"
    "The published answer set is too large to paste here. Before answering a "
    "question that may be covered, call search_knowledge."
)


def render_answers_section(
    answers: Iterable[Answer], token_limit: int = ANSWERS_TOKEN_LIMIT
) -> str:
    """Render the whole enabled set, or "" when it does not fit *token_limit*."""
    items = list(answers)
    if not items:
        return ""
    lines = [ANSWERS_HEADING, "", ANSWERS_PREAMBLE, ""]
    for item in items:
        lines.append(f"## Answer {item.id}")
        lines.append(f"Q: {item.question.strip()}")
        lines.append(f"A: {item.answer.strip()}")
        lines.append("")
    rendered = "\n".join(lines).strip()
    return rendered if text_tokens(rendered) <= token_limit else ""
