"""The stable configuration fingerprint a safe test covers."""

import hashlib
import json
from pathlib import Path

from wahabot.core.skills.errors import SkillError
from wahabot.core.skills.loader import load_skill
from wahabot.core.skills.models import SkillConfig
from wahabot.core.skills.secrets import SecretStore
from wahabot.core.skills.store import SkillStore


def configuration_hash(
    store: SkillStore, secrets: SecretStore, skills_dir: Path, name: str
) -> str:
    """A hash of every live setting and secret a test result must cover."""
    skill = store.get(name)
    if skill is None:
        return ""
    config = config_or_none(name, skills_dir)
    secret_material = {
        ref: secrets.ciphertext(ref) for ref in (config.secret_refs if config else [])
    }
    material = {
        "skill": {
            "name": skill.name,
            "base_urls": skill.base_urls,
            "auth_type": skill.auth_type,
            "secret_ref": skill.secret_ref,
            "customer_chats": skill.customer_chats,
            "writes_need_operator": skill.writes_need_operator,
            "instructions": skill.instructions,
            "skill_md": skill.skill_md,
            "draft_inputs": skill.draft_inputs,
        },
        "config": config.model_dump() if config is not None else {},
        "secret": secret_material,
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, default=str).encode()
    ).hexdigest()


def configuration_changed(
    store: SkillStore, secrets: SecretStore, skills_dir: Path, name: str
) -> bool:
    """Whether an enabled skill's live configuration moved past its test."""
    skill = store.get(name)
    if skill is None or not skill.test_config_hash:
        return False
    return configuration_hash(store, secrets, skills_dir, name) != (
        skill.test_config_hash
    )


def config_or_none(name: str, skills_dir: Path) -> SkillConfig | None:
    """The live config, or None when the folder is missing or invalid."""
    try:
        return load_skill(name, skills_dir)
    except SkillError:
        return None
