"""Draft lifecycle: safe tests, configuration hashes and enable gating."""

import hashlib
import json
from pathlib import Path

from wahabot.core.skills.errors import SkillError
from wahabot.core.skills.guardrails import is_write_method
from wahabot.core.skills.loader import load_skill
from wahabot.core.skills.models import SAFE_NAME, Skill, SkillConfig, SkillTestResult
from wahabot.core.skills.secrets import SecretStore
from wahabot.core.skills.service import SkillService
from wahabot.core.skills.store import SkillStore


class SkillDraftService:
    """Manage the draft -> tested -> enabled path."""

    def __init__(
        self,
        store: SkillStore,
        secrets: SecretStore,
        skills_dir: Path,
        service: SkillService,
    ) -> None:
        self.store = store
        self.secrets = secrets
        self.skills_dir = skills_dir
        self.service = service

    def create_draft(
        self,
        name: str,
        *,
        inputs: dict[str, object] | None = None,
        instructions: str = "",
        skill_md: str = "",
    ) -> Skill:
        """Create a resumable draft that cannot be enabled yet."""
        if not SAFE_NAME.fullmatch(name):
            raise SkillError("skill name must match [a-z0-9-]+")
        if self.store.get(name) is not None:
            raise SkillError(f"skill {name} already exists")
        return self.store.upsert(
            Skill(
                name=name,
                lifecycle="draft",
                draft_stage="connect",
                draft_inputs=dict(inputs or {}),
                instructions=instructions,
                skill_md=skill_md,
                test_state="untested",
            )
        )

    def save_draft(
        self,
        name: str,
        *,
        inputs: dict[str, object] | None = None,
        stage: str | None = None,
        instructions: str | None = None,
        skill_md: str | None = None,
    ) -> Skill:
        """Save partial wizard inputs and a stage for Phase 5."""
        return self.store.save_draft(
            name,
            inputs=inputs,
            stage=stage,
            instructions=instructions,
            skill_md=skill_md,
        )

    def run_test(self, name: str) -> SkillTestResult:
        """Run only the prepared safe check from ``skill.yaml``."""
        try:
            config = load_skill(name, self.skills_dir)
        except SkillError as exc:
            return SkillTestResult(False, str(exc))
        if config.test is None or not config.test.configured:
            reason = (
                "No safe check is configured in skill.yaml; add "
                "test: {method: GET, path: /health} before enabling."
            )
            self.store.set_test(name, state="untested", reason=reason, config_hash="")
            return SkillTestResult(False, reason)
        method = config.test.method.strip().upper()
        if is_write_method(method):
            reason = (
                "Only a safe read check (GET/HEAD/OPTIONS) can be tested; "
                "a write test is never run automatically."
            )
            self.store.set_test(name, state="untested", reason=reason, config_hash="")
            return SkillTestResult(False, reason)
        try:
            result = self.service.test_probe(name, method, config.test.path)
        except SkillError as exc:
            reason = str(exc)
            self._record(name, False, reason)
            return SkillTestResult(False, reason)
        passed = result.status < 400
        reason = f"safe check returned HTTP {result.status}"
        self._record(name, passed, reason)
        return SkillTestResult(passed, reason)

    def refresh_staleness(self, name: str) -> None:
        """Mark a passed test stale when its covered configuration changed."""
        skill = self.store.get(name)
        if skill is None or skill.test_state != "passed":
            return
        current = self.configuration_hash(name)
        if current != skill.test_config_hash:
            self.store.set_test(
                name,
                state="stale",
                reason="the skill configuration changed after its last test",
                config_hash=skill.test_config_hash,
            )

    def enable(self, name: str) -> bool:
        """Enable only a configured, current, passing, tested skill."""
        self.refresh_staleness(name)
        skill = self.store.get(name)
        if skill is None:
            return False
        if skill.lifecycle != "configured":
            return False
        if skill.test_state != "passed" or not skill.test_config_hash:
            return False
        if self.configuration_hash(name) != skill.test_config_hash:
            self.store.set_test(
                name,
                state="stale",
                reason="the skill configuration changed after its last test",
                config_hash=skill.test_config_hash,
            )
            return False
        self.store.set_enabled(name, True)
        return True

    def disable(self, name: str) -> None:
        """Turn a skill off without deleting its configuration."""
        self.store.set_enabled(name, False)

    def configuration_hash(self, name: str) -> str:
        """A hash of everything a test result must cover."""
        skill = self.store.get(name)
        if skill is None:
            return ""
        config = _config_or_none(name, self.skills_dir)
        secret_material = "".join(
            self.secrets.ciphertext(ref) for ref in (config.secret_refs if config else [])
        )
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

    def _record(self, name: str, passed: bool, reason: str) -> None:
        state = "passed" if passed else "failed"
        config_hash = self.configuration_hash(name) if passed else ""
        self.store.set_test(name, state=state, reason=reason, config_hash=config_hash)


def _config_or_none(name: str, skills_dir: Path) -> SkillConfig | None:
    try:
        return load_skill(name, skills_dir)
    except SkillError:
        return None
