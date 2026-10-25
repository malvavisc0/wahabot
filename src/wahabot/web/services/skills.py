"""Skill view models and the Add-skill draft/edit workflow."""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from wahabot.core.skills.drafts import SkillDraftService
from wahabot.core.skills.models import SAFE_NAME, Skill
from wahabot.core.skills.secrets import SecretStore
from wahabot.core.skills.store import SkillStore
from wahabot.core.tickets.store import TicketStore
from wahabot.web.services import format_time

CAPABILITIES = {
    "none": "Operator requests only",
    "read": "Customer lookup (read only)",
    "read_write": "Look up and allow changes (approval in WhatsApp)",
}
PERMISSIONS = {
    True: "Changes need your approval in WhatsApp",
    False: "Read-only customer lookup",
}
TEST_LABELS = {
    "passed": "Passed",
    "failed": "Failed",
    "stale": "Stale",
    "untested": "Untested",
}


@dataclass(frozen=True, slots=True)
class SkillRow:
    """One row in the Skills list."""

    name: str
    host: str
    source_url: str
    capabilities: str
    permissions: str
    enabled: bool
    stage_label: str
    test_label: str
    test_state: str
    reason: str
    failure_count: int
    updated: str
    lifecycle: str


@dataclass(frozen=True, slots=True)
class SkillCallRow:
    """One recent skill call in the page."""

    method: str
    path: str
    outcome: str
    failed: bool
    started_by: str
    time: str
    duration_ms: int
    error: str
    ticket_number: int | None


@dataclass(frozen=True, slots=True)
class SkillPage:
    """The one-skill page's state."""

    skill: Skill
    stage: str
    result: dict[str, str]
    key_set: bool
    calls: list[SkillCallRow]
    draft: dict[str, str]


def host_of(urls: list[str]) -> str:
    """The configured service host, never a credential-bearing URL."""
    if not urls:
        return "No destination yet"
    return urlsplit(urls[0]).hostname or urls[0]


def slugify(value: str) -> str:
    """A safe skill name from a host or path segment."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return cleaned[:40].strip("-") or "skill"


def source_parts(source_url: str) -> tuple[str, str]:
    """Validate the one docs URL and return its origin and a name seed."""
    parsed = urlsplit(source_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("enter an http(s) documentation URL")
    if parsed.username or parsed.password:
        raise ValueError("the URL must not carry credentials")
    origin = f"{parsed.scheme}://{parsed.netloc}"
    seed = parsed.path.rstrip("/").rsplit("/", 1)[-1] or parsed.hostname
    return origin, seed


def unique_name(seed: str, store: SkillStore) -> str:
    """A safe name that does not collide with an existing skill."""
    base = slugify(seed)
    candidate = base
    suffix = 2
    while store.get(candidate) is not None:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate if SAFE_NAME.fullmatch(candidate) else "skill"


def create_draft(store: SkillStore, drafts: SkillDraftService, source_url: str) -> Skill:
    """Create the inactive draft and move it to Review."""
    origin, seed = source_parts(source_url)
    name = unique_name(seed, store)
    skill = drafts.create_draft(
        name,
        inputs={
            "source_url": source_url.strip(),
            "base_url": origin,
            "test_path": "/health",
        },
    )
    skill.source_url = source_url.strip()
    skill.draft_stage = "review"
    return store.upsert(skill)


def draft_values(skill: Skill) -> dict[str, str]:
    """The resumable inputs plus saved instructions."""
    inputs = skill.draft_inputs
    fallback = skill.base_urls[0] if skill.base_urls else ""
    base_url = str(inputs["base_url"]) if "base_url" in inputs else fallback
    return {
        "base_url": base_url,
        "test_path": str(inputs.get("test_path") or "/health"),
        "customer_chats": skill.customer_chats,
        "writes_need_operator": "on" if skill.writes_need_operator else "",
        "instructions": skill.instructions,
        "skill_md": skill.skill_md,
    }


@dataclass(frozen=True, slots=True)
class ReviewValues:
    """Validated review fields before they touch a skill row."""

    origin: str
    test_path: str
    customer_chats: str
    instructions: str
    skill_md: str
    api_key: str
    writes_need_operator: bool


def normalized_test_path(value: str) -> str:
    """A leading-slash safe-check path, defaulting to ``/health``."""
    path = value.strip() or "/health"
    return path if path.startswith("/") else f"/{path}"


def parse_review(form: dict[str, str]) -> ReviewValues:
    """Validate the review form without mutating anything."""
    base_url = form.get("base_url", "").strip()
    origin = source_parts(base_url)[0] if "://" in base_url else ""
    if not origin:
        raise ValueError("enter the service destination")
    test_path = normalized_test_path(form.get("test_path", ""))
    customer_chats = form.get("customer_chats", "read")
    if customer_chats not in CAPABILITIES:
        raise ValueError("choose what the skill may do")
    return ReviewValues(
        origin=origin,
        test_path=test_path,
        customer_chats=customer_chats,
        instructions=form.get("instructions", "").strip(),
        skill_md=form.get("skill_md", "").strip(),
        api_key=form.get("api_key", "").strip(),
        writes_need_operator=form.get("writes_need_operator", "") == "on",
    )


def apply_review(
    skill: Skill,
    values: ReviewValues,
    secrets: SecretStore,
    store: SkillStore,
) -> Skill:
    """Apply validated fields, including the write-only key reference."""
    secret_ref = skill.secret_ref or f"{skill.name.upper().replace('-', '_')}_KEY"
    if values.api_key:
        secrets.put(secret_ref, values.api_key)
    has_key = bool(values.api_key or store.secret_exists(secret_ref))
    auth_type = "bearer" if has_key else "none"
    skill.base_urls = [values.origin]
    skill.source_url = skill.source_url or values.origin
    skill.instructions = values.instructions
    skill.skill_md = values.skill_md or values.instructions
    skill.customer_chats = values.customer_chats
    skill.writes_need_operator = values.writes_need_operator
    skill.auth_type = auth_type
    skill.secret_ref = secret_ref if auth_type != "none" else ""
    skill.lifecycle = "configured"
    skill.draft_stage = "review"
    skill.draft_inputs.update(
        {
            "source_url": skill.source_url,
            "base_url": values.origin,
            "test_path": values.test_path,
        }
    )
    return skill


def save_draft_fields(skill: Skill, form: dict[str, str], store: SkillStore) -> Skill:
    """Persist draft edits without configuring, writing files or storing a key."""
    base_url = form.get("base_url", "").strip()
    origin = source_parts(base_url)[0] if base_url else ""
    customer_chats = form.get("customer_chats", skill.customer_chats)
    if customer_chats not in CAPABILITIES:
        raise ValueError("choose what the skill may do")
    skill.base_urls = [origin] if origin else skill.base_urls
    skill.source_url = skill.source_url or origin
    skill.instructions = form.get("instructions", skill.instructions).strip()
    skill.skill_md = form.get("skill_md", skill.skill_md).strip()
    skill.customer_chats = customer_chats
    skill.writes_need_operator = form.get("writes_need_operator", "") == "on"
    skill.draft_stage = "review"
    skill.draft_inputs.update(
        {
            "source_url": skill.source_url,
            "base_url": origin,
            "test_path": normalized_test_path(form.get("test_path", "")),
        }
    )
    return store.upsert(skill)


def save_skill(
    store: SkillStore,
    secrets: SecretStore,
    drafts: SkillDraftService,
    skills_dir: Path,
    name: str,
    form: dict[str, str],
) -> Skill:
    """Save the draft, or configure it and invalidate an old test."""
    skill = store.get(name)
    if skill is None:
        raise ValueError("unknown skill")
    if form.get("action") == "save":
        updated = save_draft_fields(skill, form, store)
        drafts.refresh_staleness(name)
        return store.get(name) or updated
    values = parse_review(form)
    updated = store.upsert(apply_review(skill, values, secrets, store))
    write_folder(skills_dir, updated, values.test_path)
    drafts.refresh_staleness(name)
    return store.get(name) or updated


def skill_document(skill: Skill, test_path: str) -> dict[str, Any]:
    """The validated ``skill.yaml`` for one saved draft."""
    document: dict[str, Any] = {
        "name": skill.name,
        "kind": "http",
        "base_url": skill.base_urls[0],
        "auth": (
            {"type": "bearer", "secret": skill.secret_ref}
            if skill.auth_type != "none"
            else {"type": "none"}
        ),
        "customer_chats": skill.customer_chats,
        "writes_need_operator": skill.writes_need_operator,
        "test": {"method": "GET", "path": test_path},
    }
    return document


def write_folder(skills_dir: Path, skill: Skill, test_path: str) -> None:
    """Write the saved skill folder that the safe test and runtime load."""
    folder = skills_dir / skill.name
    folder.mkdir(parents=True, exist_ok=True)
    body = skill.skill_md or skill.instructions or f"{skill.name} service"
    (folder / "SKILL.md").write_text(f"# {skill.name}\n\n{body}\n", encoding="utf-8")
    (folder / "skill.yaml").write_text(
        yaml.safe_dump(skill_document(skill, test_path), sort_keys=False),
        encoding="utf-8",
    )


def stage_for(skill: Skill) -> str:
    """The wizard stage shown as current."""
    if skill.test_state in {"passed", "failed", "stale"}:
        return "test"
    return "review" if skill.draft_stage == "review" else "connect"


def result_view(skill: Skill) -> dict[str, str]:
    """The stable test-result region's label, class and reason."""
    label = TEST_LABELS.get(skill.test_state, "Untested")
    classes = {
        "passed": "passed",
        "failed": "error",
        "stale": "unsaved",
        "untested": "muted",
    }
    reason = skill.test_reason or "No safe check has run for this configuration."
    return {
        "label": label,
        "css": classes.get(skill.test_state, "muted"),
        "tested_at": format_time(skill.last_tested_at),
        "reason": reason,
    }


def row_view(skill: Skill) -> SkillRow:
    """Map one stored skill to the list row."""
    return SkillRow(
        name=skill.name,
        host=host_of(skill.base_urls),
        source_url=skill.source_url,
        capabilities=CAPABILITIES.get(skill.customer_chats, skill.customer_chats),
        permissions=PERMISSIONS[skill.writes_need_operator],
        enabled=skill.enabled,
        stage_label=stage_for(skill).title(),
        test_label=TEST_LABELS.get(skill.test_state, "Untested"),
        test_state=skill.test_state,
        reason=skill.test_reason,
        failure_count=skill.failure_count,
        updated=format_time(skill.updated_at),
        lifecycle=skill.lifecycle,
    )


def call_rows(store: SkillStore, name: str, tickets: TicketStore) -> list[SkillCallRow]:
    """The recent calls for one skill page."""
    numbers = {ticket.id: ticket.number for ticket in tickets.all_tickets()}
    return [
        SkillCallRow(
            method=call.method,
            path=call.path,
            outcome=call.outcome,
            failed=call.outcome != "ok",
            started_by=call.started_by,
            time=format_time(call.created_at),
            duration_ms=call.duration_ms,
            error=call.error,
            ticket_number=numbers.get(call.ticket_id or 0),
        )
        for call in store.calls(name)
    ]


def skill_page(
    store: SkillStore,
    secrets: SecretStore,
    name: str,
    tickets: TicketStore,
) -> SkillPage:
    """Load the one-skill page state."""
    skill = store.get(name)
    if skill is None:
        raise KeyError(name)
    key_set = bool(skill.secret_ref and store.secret_exists(skill.secret_ref))
    return SkillPage(
        skill=skill,
        stage=stage_for(skill),
        result=result_view(skill),
        key_set=key_set,
        calls=call_rows(store, name, tickets),
        draft=draft_values(skill),
    )


def delete_skill(store: SkillStore, skills_dir: Path, name: str) -> None:
    """Remove one skill row and its saved folder."""
    store.delete(name)
    folder = skills_dir / name
    if folder.is_dir():
        shutil.rmtree(folder)
