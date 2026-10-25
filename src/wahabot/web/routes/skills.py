"""Skills: list, one-URL draft, review, test gating and enable/disable."""

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response

from wahabot.core.skills.errors import SkillError
from wahabot.settings import Settings
from wahabot.web.auth import csrf_matches, forbidden, origin_ok
from wahabot.web.deps import (
    current_settings,
    error_page,
    page_auth,
    render,
    skill_runtime_for,
    ticket_store_for,
)
from wahabot.web.services.skills import (
    create_draft,
    delete_skill,
    row_view,
    save_skill,
    skill_page,
)

router = APIRouter()


def runtime(settings: Settings) -> Any:
    """The skill stores for one request."""
    return skill_runtime_for(settings)


def draft_overrides(form: dict[str, str]) -> dict[str, str]:
    """The submitted review fields, so a failed save keeps the input."""
    return {
        "base_url": form.get("base_url", "").strip(),
        "test_path": form.get("test_path", "").strip(),
        "customer_chats": form.get("customer_chats", "read"),
        "writes_need_operator": "on" if "writes_need_operator" in form else "",
        "instructions": form.get("instructions", ""),
        "skill_md": form.get("skill_md", ""),
    }


def detail_response(
    request: Request,
    name: str,
    token: str,
    *,
    feedback: str = "",
    feedback_class: str = "",
    form: dict[str, str] | None = None,
) -> Response:
    """Render the one-skill page with optional action feedback."""
    settings = current_settings()
    store, secrets, _drafts, _service = runtime(settings)
    try:
        page = skill_page(store, secrets, name, ticket_store_for(settings))
    except KeyError:
        return error_page(request, "That skill was not found.", 404)
    draft = {**page.draft, **draft_overrides(form)} if form else page.draft
    context: dict[str, Any] = {
        "csrf_token": token,
        "skill": page.skill,
        "stage": page.stage,
        "result": page.result,
        "key_set": page.key_set,
        "calls": page.calls,
        "draft": draft,
        "feedback": feedback,
        "feedback_class": feedback_class,
    }
    return render(request, "skill_detail.html", context)


def valid_write(request: Request, settings: Settings, form: Any) -> bool:
    """Origin and CSRF for one skill write."""
    if not origin_ok(request, settings):
        return False
    return csrf_matches(request, settings, str(form.get("csrf_token", "")))


@router.get("/app/skills")
def skills_list(request: Request) -> Response:
    """Configured skills, saved drafts and attention states."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    store, _secrets, _drafts, _service = runtime(settings)
    rows = [row_view(skill) for skill in store.list()]
    configured = [row for row in rows if row.lifecycle == "configured"]
    drafts = [row for row in rows if row.lifecycle != "configured"]
    attention = [row for row in configured if row.failure_count >= 3]
    return render(
        request,
        "skills_list.html",
        {
            "csrf_token": auth.csrf,
            "configured": configured,
            "drafts": drafts,
            "attention": attention,
        },
    )


@router.get("/app/skills/new")
def skill_new(request: Request) -> Response:
    """The one-URL Add skill form."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    return render(
        request,
        "skill_new.html",
        {"csrf_token": auth.csrf, "error": ""},
    )


@router.post("/app/skills/new")
async def skill_create(request: Request) -> Response:
    """Create the inactive draft and open Review."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    form = await request.form()
    if not valid_write(request, settings, form):
        return forbidden()
    store, _secrets, drafts, _service = runtime(settings)
    try:
        skill = create_draft(store, drafts, str(form.get("source_url", "")))
    except (ValueError, SkillError) as exc:
        return render(
            request,
            "skill_new.html",
            {"csrf_token": auth.csrf, "error": str(exc)},
            status_code=400,
        )
    return RedirectResponse(f"/app/skills/{skill.name}", status_code=303)


@router.get("/app/skills/{name}")
def skill_detail(request: Request, name: str) -> Response:
    """The review and test page for one skill."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    return detail_response(request, name, auth.csrf)


@router.post("/app/skills/{name}/save")
async def skill_save(request: Request, name: str) -> Response:
    """Save the review fields, folder and write-only key."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    form = await request.form()
    if not valid_write(request, settings, form):
        return forbidden()
    store, secrets, drafts, _service = runtime(settings)
    values = {key: str(value) for key, value in form.items()}
    try:
        save_skill(
            store,
            secrets,
            drafts,
            settings.skills_dir,
            name,
            values,
        )
    except (ValueError, SkillError) as exc:
        return detail_response(
            request,
            name,
            auth.csrf,
            feedback=f"Not saved: {exc}",
            feedback_class="error",
            form=values,
        )
    saved = "Saved draft." if values.get("action") == "save" else "Saved."
    return detail_response(
        request, name, auth.csrf, feedback=saved, feedback_class="saved"
    )


@router.post("/app/skills/{name}/test")
async def skill_test(request: Request, name: str) -> Response:
    """Run only the prepared safe check."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    form = await request.form()
    if not valid_write(request, settings, form):
        return forbidden()
    _store, _secrets, drafts, _service = runtime(settings)
    try:
        result = drafts.run_test(name)
    except SkillError as exc:
        return detail_response(
            request,
            name,
            auth.csrf,
            feedback=f"Test failed: {exc}",
            feedback_class="error",
        )
    message = "Test passed." if result.passed else f"Test failed: {result.reason}"
    return detail_response(
        request,
        name,
        auth.csrf,
        feedback=message,
        feedback_class="saved" if result.passed else "error",
    )


@router.post("/app/skills/{name}/enable")
async def skill_enable(request: Request, name: str) -> Response:
    """Enable only a current, passed test."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    form = await request.form()
    if not valid_write(request, settings, form):
        return forbidden()
    store, _secrets, drafts, _service = runtime(settings)
    if drafts.enable(name):
        return detail_response(
            request, name, auth.csrf, feedback="Enabled.", feedback_class="saved"
        )
    skill = store.get(name)
    reason = (skill.test_reason if skill is not None else "") or "the test is not current"
    return detail_response(
        request,
        name,
        auth.csrf,
        feedback=f"Not enabled: {reason}",
        feedback_class="error",
    )


@router.post("/app/skills/{name}/disable")
async def skill_disable(request: Request, name: str) -> Response:
    """Switch a skill off without deleting its configuration."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    form = await request.form()
    if not valid_write(request, settings, form):
        return forbidden()
    _store, _secrets, drafts, _service = runtime(settings)
    try:
        drafts.disable(name)
    except SkillError as exc:
        return detail_response(
            request, name, auth.csrf, feedback=str(exc), feedback_class="error"
        )
    return detail_response(
        request, name, auth.csrf, feedback="Paused.", feedback_class="saved"
    )


@router.post("/app/skills/{name}/delete")
async def skill_delete(request: Request, name: str) -> Response:
    """Delete one skill and its saved folder."""
    settings = current_settings()
    auth = page_auth(request, settings)
    if isinstance(auth, Response):
        return auth
    form = await request.form()
    if not valid_write(request, settings, form):
        return forbidden()
    store, _secrets, _drafts, _service = runtime(settings)
    if store.get(name) is None:
        return error_page(request, "That skill was not found.", 404)
    delete_skill(store, settings.skills_dir, name)
    return RedirectResponse("/app/skills", status_code=303)
