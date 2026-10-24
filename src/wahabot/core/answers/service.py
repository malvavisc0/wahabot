"""High-level answer CRUD, search and candidate saving."""

from wahabot.core.answers.models import Answer, AnswerCandidate, AnswerHit
from wahabot.core.answers.retrieval import (
    ANSWERS_SEARCH_LIMIT,
    ANSWERS_TOKEN_LIMIT,
    render_answers_section,
)
from wahabot.core.answers.store import AnswerStore

MAX_QUESTION_CHARS = 500
MAX_ANSWER_CHARS = 8000


def validate_text(question: str, answer: str) -> tuple[str, str]:
    """Normalize and validate a QA pair; empty or oversized input raises."""
    clean_question = question.strip()
    clean_answer = answer.strip()
    if not clean_question:
        raise ValueError("an answer needs a question")
    if not clean_answer:
        raise ValueError("an answer needs an answer")
    if len(clean_question) > MAX_QUESTION_CHARS:
        raise ValueError(f"question exceeds {MAX_QUESTION_CHARS} characters")
    if len(clean_answer) > MAX_ANSWER_CHARS:
        raise ValueError(f"answer exceeds {MAX_ANSWER_CHARS} characters")
    return clean_question, clean_answer


class AnswerService:
    """CRUD for the Teach surface, retrieval and reply-candidate saving."""

    def __init__(
        self,
        store: AnswerStore,
        *,
        token_limit: int = ANSWERS_TOKEN_LIMIT,
    ) -> None:
        self.store = store
        self.token_limit = token_limit

    def add(self, question: str, answer: str) -> Answer:
        """Create one enabled answer."""
        clean_question, clean_answer = validate_text(question, answer)
        return self.store.add(clean_question, clean_answer)

    def edit(
        self,
        answer_id: int,
        *,
        question: str | None = None,
        answer: str | None = None,
    ) -> Answer:
        """Edit one answer's text."""
        current = self.store.get(answer_id)
        if current is None:
            raise KeyError(f"answer {answer_id} not found")
        clean_question, clean_answer = validate_text(
            question if question is not None else current.question,
            answer if answer is not None else current.answer,
        )
        return self.store.update(
            answer_id,
            question=clean_question,
            answer=clean_answer,
        )

    def enable(self, answer_id: int) -> Answer:
        """Enable one answer for retrieval and prompt injection."""
        return self.store.update(answer_id, enabled=True)

    def disable(self, answer_id: int) -> Answer:
        """Disable one answer without deleting it."""
        return self.store.update(answer_id, enabled=False)

    def delete(self, answer_id: int) -> Answer:
        """Soft-delete one answer."""
        return self.store.soft_delete(answer_id)

    def undo_delete(self, answer_id: int | None = None) -> Answer | None:
        """Undo one deletion, or the latest."""
        return self.store.undo_delete(answer_id)

    def list(self, *, include_deleted: bool = False) -> list[Answer]:
        """List answers for Teach."""
        return self.store.list(include_deleted=include_deleted)

    def search(self, query: str, limit: int = ANSWERS_SEARCH_LIMIT) -> list[AnswerHit]:
        """Search enabled answers and record a use for every returned hit."""
        hits = self.store.search(query, limit=limit)
        for hit in hits:
            self.store.record_use(hit.answer.id)
        return hits

    def prompt_section(self, token_limit: int | None = None) -> str:
        """The whole enabled set when it fits, else "" (search takes over)."""
        return render_answers_section(
            self.store.enabled(),
            token_limit=self.token_limit if token_limit is None else token_limit,
        )

    def offer_candidate(
        self, ticket_id: int, question: str, answer: str
    ) -> AnswerCandidate:
        """Store an unsaved candidate built from a ticket reply."""
        clean_question, clean_answer = validate_text(question, answer)
        return self.store.add_candidate(ticket_id, clean_question, clean_answer)

    def latest_candidate(self) -> AnswerCandidate | None:
        """The newest unsaved candidate."""
        return self.store.latest_candidate()

    def save_latest_candidate(self) -> Answer | None:
        """Save the newest candidate once; None when there is nothing to save."""
        candidate = self.store.latest_candidate()
        if candidate is None:
            return None
        return self.save_candidate(candidate.id)

    def save_candidate(self, candidate_id: int) -> Answer | None:
        """Save one named candidate; None when it was already saved."""
        candidate = self.store.candidate(candidate_id)
        if candidate is None or candidate.saved_at is not None:
            return None
        answer = self.add(candidate.question, candidate.answer)
        self.store.mark_candidate_saved(candidate.id)
        return answer
