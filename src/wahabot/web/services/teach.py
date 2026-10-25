"""Teach view models over the shared answer service."""

from dataclasses import dataclass

from wahabot.core.answers.models import Answer
from wahabot.core.answers.service import AnswerService


@dataclass(frozen=True, slots=True)
class TeachView:
    """The answer list plus the one latest undo affordance."""

    answers: list[Answer]
    latest_deleted: Answer | None


def teach_view(service: AnswerService) -> TeachView:
    """Load the list and the latest deletion for the page."""
    return TeachView(
        answers=service.list(), latest_deleted=service.store.latest_deleted()
    )
