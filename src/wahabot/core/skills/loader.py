"""Read, validate and import skill folders."""

import shutil
from pathlib import Path

import yaml
from pydantic import ValidationError

from wahabot.core.skills.errors import SkillError
from wahabot.core.skills.models import Skill, SkillConfig, skill_folder
from wahabot.core.skills.store import SkillStore

PREDEFINED_DIR = Path(__file__).resolve().parents[2] / "skills" / "predefined"


def load_skill(name: str, skills_dir: Path) -> SkillConfig:
    """Read and validate one installed skill folder."""
    folder = skill_folder(skills_dir, name)
    return read_config(folder)


def read_config(folder: Path) -> SkillConfig:
    """Parse ``SKILL.md`` and ``skill.yaml`` from one folder."""
    docs = folder / "SKILL.md"
    config = folder / "skill.yaml"
    if not docs.is_file() or not config.is_file():
        raise SkillError(f"skill folder {folder.name} needs SKILL.md and skill.yaml")
    try:
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SkillError(f"skill.yaml in {folder.name} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise SkillError(f"skill.yaml in {folder.name} must be a mapping")
    try:
        config = SkillConfig.model_validate({**raw, "skill_md": docs.read_text()})
    except ValidationError as exc:
        raise SkillError(f"skill {folder.name} is invalid: {exc}") from exc
    if (
        config.sql is not None
        and config.sql.engine == "sqlite"
        and not Path(config.sql.sqlite_path).is_absolute()
    ):
        config.sql.sqlite_path = str((folder / config.sql.sqlite_path).resolve())
    return config


def import_skill(source: Path, skills_dir: Path, store: SkillStore) -> Skill:
    """Validate one folder, copy it into place and write its configured row."""
    config = read_config(source)
    existing = store.get(config.name)
    if existing is not None and existing.enabled:
        raise SkillError(
            f"skill {config.name} is enabled; disable it before importing again"
        )
    _require_secrets(config, store)
    target = skills_dir / config.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)
    docs = (target / "SKILL.md").read_text(encoding="utf-8")
    return store.upsert(
        Skill(
            name=config.name,
            lifecycle="configured",
            enabled=False,
            base_urls=config.effective_base_urls,
            auth_type=config.auth.type,
            secret_ref=config.auth.secret,
            customer_chats=config.customer_chats,
            writes_need_operator=config.writes_need_operator,
            instructions=config.instructions,
            skill_md=docs,
            draft_stage="test",
            test_state="untested",
        )
    )


def _require_secrets(config: SkillConfig, store: SkillStore) -> None:
    missing = [name for name in config.secret_refs if not store.secret_exists(name)]
    if missing:
        raise SkillError(
            "skill references missing secret(s): " + ", ".join(sorted(missing))
        )
