"""Teach: answer CRUD, on/off, undo, validation and CSRF."""

import re

from fastapi.testclient import TestClient
from tests.web.conftest import csrf_of


def add_answer(client: TestClient, question: str, answer: str) -> str:
    """Add one answer and return its rendered id."""
    token = csrf_of(client.get("/app/teach").text)
    response = client.post(
        "/app/teach/add",
        data={"question": question, "answer": answer, "csrf_token": token},
        follow_redirects=True,
    )
    assert response.status_code == 200
    match = re.search(r'data-answer-id="(\d+)"', response.text)
    assert match is not None
    return match.group(1)


def test_empty_page_is_purposeful(auth_client: TestClient) -> None:
    """An empty Teach page says what to do, without invented rows."""
    response = auth_client.get("/app/teach")
    assert response.status_code == 200
    assert "No answers yet" in response.text
    assert "Add answer" in response.text


def test_add_edit_toggle_delete_undo(auth_client: TestClient) -> None:
    """The whole operator answer lifecycle works through the forms."""
    answer_id = add_answer(
        auth_client, "When is the bridge club open?", "Mondays, 9 to 17."
    )
    page = auth_client.get("/app/teach")
    token = csrf_of(page.text)
    edited = auth_client.post(
        f"/app/teach/{answer_id}/edit",
        data={
            "question": "When is the bridge club open?",
            "answer": "Mondays and Wednesdays, 9 to 17.",
            "csrf_token": token,
        },
        follow_redirects=True,
    )
    assert "Mondays and Wednesdays" in edited.text
    toggled = auth_client.post(
        f"/app/teach/{answer_id}/toggle",
        data={"csrf_token": token},
        follow_redirects=True,
    )
    assert "Off" in toggled.text
    deleted = auth_client.post(
        f"/app/teach/{answer_id}/delete",
        data={"csrf_token": token},
        follow_redirects=True,
    )
    assert "When is the bridge club open?" not in deleted.text
    undone = auth_client.post(
        "/app/teach/undo", data={"csrf_token": token}, follow_redirects=True
    )
    assert "When is the bridge club open?" in undone.text


def test_validation_keeps_the_draft(auth_client: TestClient) -> None:
    """An empty question is refused without creating a row."""
    token = csrf_of(auth_client.get("/app/teach").text)
    response = auth_client.post(
        "/app/teach/add",
        data={"question": "", "answer": "orphan", "csrf_token": token},
    )
    assert response.status_code == 400
    assert "orphan" in response.text
    assert "data-answer-id" not in auth_client.get("/app/teach").text


def test_edit_link_opens_the_editor(auth_client: TestClient) -> None:
    """The edit link loads the saved text, not an empty form."""
    answer_id = add_answer(auth_client, "Question", "Answer")
    response = auth_client.get("/app/teach", params={"edit": answer_id})
    assert response.status_code == 200
    assert ">Answer</textarea>" in response.text
    assert "Edit answer" in response.text


def test_write_requires_csrf_and_same_origin(auth_client: TestClient) -> None:
    """Missing CSRF and cross-site Origin both refuse the write."""
    token = csrf_of(auth_client.get("/app/teach").text)
    missing = auth_client.post("/app/teach/add", data={"question": "Q", "answer": "A"})
    assert missing.status_code == 403
    cross = auth_client.post(
        "/app/teach/add",
        data={"question": "Q", "answer": "A", "csrf_token": token},
        headers={"Origin": "https://evil.invalid"},
    )
    assert cross.status_code == 403
