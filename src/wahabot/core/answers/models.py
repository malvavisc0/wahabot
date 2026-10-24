"""Answer domain models."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Answer:
    """One published question/answer pair."""

    id: int
    question: str
    answer: str
    enabled: bool
    created_at: int
    updated_at: int
    deleted_at: int | None = None


@dataclass(frozen=True, slots=True)
class AnswerHit:
    """One retrieval hit with its BM25 rank."""

    answer: Answer
    rank: float


@dataclass(frozen=True, slots=True)
class AnswerCandidate:
    """An operator answer offered for saving from a ticket reply."""

    id: int
    ticket_id: int
    question: str
    answer: str
    created_at: int
    saved_at: int | None = None
