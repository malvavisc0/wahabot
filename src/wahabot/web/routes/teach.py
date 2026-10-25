"""Teach: add, edit, switch off, delete and undo answers."""

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response

from wahabot.core.answers.service import AnswerService
from wahabot.web.auth import csrf_matches, forbidden, origin_ok
from wahabot.web.deps import (
    answer_service_for,
    current_settings,
    error_page,
    page_auth,
    render,
)
from wahabot.web.services.teach import teach_view

router = APIRouter()


def teach_context(
    service: AnswerService,
    token: str,
    *,
    editor_mode: str = "add",
    answer_id: int | None = None,
    draft: dict[str, str] | None = None,
    error: str = "",
) -> dict[str, object]:
    """The shared template values; the editor is add or edit."""
    view = teach_view(service)
    action = f"/app/teach/{answer_id}/edit" if answer_id is not None else "/app/teach/add"
    return {
        "csrf_token": token,
        "answers": view.answers,
        "latest_deleted": view.latest_deleted,
        "editor_mode": editor_mode,
        "editor_action": action,
        "draft": draft or {"question": "", "answer": ""},
        "error": error,
    }


def service() -> AnswerService:
    """The Teach service for this request's settings."""
    return answer_service_for(current_settings())


@router.get("/app/teach")
def teach_page(request: Request, edit: str = "") -> Response:
    """The editor and the current answer list."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    answers = service()
    if edit:
        try:
            answer_id = int(edit)
        except ValueError:
            return RedirectResponse("/app/teach", status_code=303)
        found = answers.store.get(answer_id)
        if found is None:
            return RedirectResponse("/app/teach", status_code=303)
        return render(
            request,
            "teach.html",
            teach_context(
                answers,
                auth.csrf,
                editor_mode="edit",
                answer_id=answer_id,
                draft={"question": found.question, "answer": found.answer},
            ),
        )
    return render(request, "teach.html", teach_context(answers, auth.csrf))


@router.post("/app/teach/add")
async def teach_add(request: Request) -> Response:
    """Add one answer and return to the list."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    if not origin_ok(request, settings):
        return forbidden()
    form = await request.form()
    if not csrf_matches(request, settings, str(form.get("csrf_token", ""))):
        return forbidden()
    answers = service()
    draft = {
        "question": str(form.get("question", "")),
        "answer": str(form.get("answer", "")),
    }
    try:
        answers.add(draft["question"], draft["answer"])
    except ValueError as exc:
        return render(
            request,
            "teach.html",
            teach_context(answers, auth.csrf, draft=draft, error=str(exc)),
            status_code=400,
        )
    return RedirectResponse("/app/teach", status_code=303)


@router.post("/app/teach/{answer_id}/edit")
async def teach_edit(request: Request, answer_id: int) -> Response:
    """Edit one answer's question and answer."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    if not origin_ok(request, settings):
        return forbidden()
    form = await request.form()
    if not csrf_matches(request, settings, str(form.get("csrf_token", ""))):
        return forbidden()
    answers = service()
    draft = {
        "question": str(form.get("question", "")),
        "answer": str(form.get("answer", "")),
    }
    try:
        answers.edit(answer_id, question=draft["question"], answer=draft["answer"])
    except (KeyError, ValueError) as exc:
        return render(
            request,
            "teach.html",
            teach_context(
                answers,
                auth.csrf,
                editor_mode="edit",
                answer_id=answer_id,
                draft=draft,
                error=str(exc),
            ),
            status_code=400,
        )
    return RedirectResponse("/app/teach", status_code=303)


@router.post("/app/teach/{answer_id}/toggle")
async def teach_toggle(request: Request, answer_id: int) -> Response:
    """Switch one answer on or off."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    if not origin_ok(request, settings):
        return forbidden()
    form = await request.form()
    if not csrf_matches(request, settings, str(form.get("csrf_token", ""))):
        return forbidden()
    answers = service()
    found = answers.store.get(answer_id, include_deleted=True)
    if found is None:
        return error_page(request, "That answer was not found.", 404)
    if found.enabled:
        answers.disable(answer_id)
    else:
        answers.enable(answer_id)
    return RedirectResponse("/app/teach", status_code=303)


@router.post("/app/teach/{answer_id}/delete")
async def teach_delete(request: Request, answer_id: int) -> Response:
    """Delete one answer; the latest deletion is undoable."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    if not origin_ok(request, settings):
        return forbidden()
    form = await request.form()
    if not csrf_matches(request, settings, str(form.get("csrf_token", ""))):
        return forbidden()
    try:
        service().delete(answer_id)
    except KeyError:
        return error_page(request, "That answer was not found.", 404)
    return RedirectResponse("/app/teach", status_code=303)


@router.post("/app/teach/undo")
async def teach_undo(request: Request) -> Response:
    """Restore the latest deleted answer."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    if not origin_ok(request, settings):
        return forbidden()
    form = await request.form()
    if not csrf_matches(request, settings, str(form.get("csrf_token", ""))):
        return forbidden()
    service().undo_delete()
    return RedirectResponse("/app/teach", status_code=303)
