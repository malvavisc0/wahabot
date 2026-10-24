"""Published answers: FTS, whole-set injection, uses and reply candidates."""

import json

import pytest

from wahabot.ai.context.prompt import render_system_prompt
from wahabot.ai.tools.knowledge import search_knowledge
from wahabot.core.answers.retrieval import (
    ANSWERS_SEARCH_LIMIT,
    render_answers_section,
)
from wahabot.core.answers.service import AnswerService
from wahabot.core.answers.store import AnswerStore
from wahabot.core.tickets.store import TicketStore
from wahabot.core.tokens import text_tokens


def service(store: TicketStore) -> AnswerService:
    """A service against the migrated ticket database."""
    return AnswerService(AnswerStore(store.path))


def test_answers_crud_fts_and_soft_delete(store: TicketStore) -> None:
    answers = service(store)
    first = answers.add("What is the refund window?", "Within thirty days.")
    second = answers.add("Do you ship at weekends?", "No weekend dispatch.")

    assert [hit.answer.id for hit in answers.search("refund")] == [first.id]
    assert "within thirty days" in answers.prompt_section().lower()
    answers.disable(second.id)
    assert answers.search("weekend") == []
    answers.enable(second.id)
    answers.delete(first.id)
    assert answers.search("refund") == []
    restored = answers.undo_delete()
    assert restored is not None and restored.id == first.id
    assert restored.deleted_at is None


def test_whole_set_injection_is_budgeted(store: TicketStore) -> None:
    answers = service(store)
    answers.add("Refund window", "Thirty days from delivery.")
    section = render_answers_section(answers.store.enabled(), token_limit=6000)
    assert section.startswith("# Answers")
    assert "never override safety" in section.lower()
    assert text_tokens(section) <= 6000
    assert render_answers_section(answers.store.enabled(), token_limit=1) == ""


def test_validation_and_use_ledger(store: TicketStore) -> None:
    answers = service(store)
    with pytest.raises(ValueError):
        answers.add("", "answer")
    with pytest.raises(ValueError):
        answers.add("question", "")
    first = answers.add("Refund", "Yes, within thirty days.")
    before = answers.store.uses_count(first.id)
    hits = answers.search("refund", limit=ANSWERS_SEARCH_LIMIT)
    assert hits
    assert answers.store.uses_count(first.id) == before + 1


def test_candidates_save_once(store: TicketStore) -> None:
    answers = service(store)
    contact = store.upsert_contact("491555000001@c.us")
    ticket = store.create_ticket(contact.id, "491555000001@c.us")
    answers.offer_candidate(ticket.id, "Where is it?", "It shipped yesterday.")
    saved = answers.save_latest_candidate(ticket.id)
    assert saved is not None and saved.enabled
    assert answers.save_latest_candidate(ticket.id) is None
    assert answers.search("shipped")
    with pytest.raises(ValueError):
        answers.offer_candidate(ticket.id, "", "answer")


def test_candidates_are_scoped_to_their_ticket(store: TicketStore) -> None:
    answers = service(store)
    first_contact = store.upsert_contact("491555000001@c.us")
    first = store.create_ticket(first_contact.id, "491555000001@c.us")
    second_contact = store.upsert_contact("491555000002@c.us")
    second = store.create_ticket(second_contact.id, "491555000002@c.us")
    answers.offer_candidate(first.id, "First?", "First answer.")
    answers.offer_candidate(second.id, "Second?", "Second answer.")

    saved = answers.save_latest_candidate(first.id)
    assert saved is not None and saved.question == "First?"
    assert answers.save_latest_candidate(first.id) is None
    assert answers.save_latest_candidate(second.id) is not None


def test_over_budget_answers_tell_the_model_to_search(store: TicketStore) -> None:
    answers = service(store)
    answers.add("Refund window", "Thirty days from delivery.")
    section = answers.prompt_section(token_limit=1)
    assert "search_knowledge" in section
    assert "too large" in section.lower()


def test_search_tool_caps_the_total_payload(store: TicketStore) -> None:
    answers = service(store)
    for index in range(5):
        answers.add(f"Refund question {index}", "x" * 7000)
    payload = json.loads(search_knowledge(answers).fn(query="refund"))
    total = sum(
        len(item["question"]) + len(item["answer"]) for item in payload["answers"]
    )
    assert payload["answers"]
    assert total <= 12_000
    assert "rendered" not in payload


def test_safety_line_is_present_without_answers() -> None:
    prompt = render_system_prompt("You are a personal assistant.")
    assert prompt.startswith("Never invent a price, deadline or policy.")
    assert prompt.endswith("You are a personal assistant.")


def test_search_tool_and_prompt_injection(store: TicketStore) -> None:
    answers = service(store)
    answers.add("Refund window", "Thirty days from delivery.")
    tool = search_knowledge(answers)
    payload = json.loads(tool.fn(query="refund"))
    assert payload["ok"] and payload["answers"][0]["question"] == "Refund window"

    prompt = render_system_prompt(
        "Instructions.", answers=answers.prompt_section(), knowledge_only=True
    )
    assert "Never invent a price, deadline or policy" in prompt
    assert "# Answers" in prompt
    assert "I don't know" in prompt

    empty = json.loads(search_knowledge(None).fn(query="refund"))
    assert empty["ok"] and empty["answers"] == []
