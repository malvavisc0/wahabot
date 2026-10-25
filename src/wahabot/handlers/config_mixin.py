import time

from wahabot.ai.context import (
    render_system_prompt,
)
from wahabot.ai.tools.runcontext import current_target
from wahabot.core.access import SessionConfig
from wahabot.core.schedule.hours import (
    effective_timezone,
    format_opening,
    opening_state,
)
from wahabot.core.teams.groups import group_participation, is_team_group
from wahabot.handlers.base import HandlerBase
from wahabot.status import state as status_state

#: The gate sees a bounded brief, never the pinned state block the agent gets.
GATE_BRIEF_CHARS = 600

#: What each participation mode asks the gate to judge.
_PARTICIPATION_BRIEF = {
    "never": "In groups, never reply.",
    "mentioned": "In groups, reply only when addressed or replied to.",
    "judicious": "In groups, judge each message for whether to reply.",
}


class ConfigMixin(HandlerBase):
    def render_prompt(self) -> str:
        """Re-render ``{{date}}``/``{{time}}`` and pick up config edits.

        The current config's prompt always wins over the startup
        snapshot, so prompt changes apply without a restart too. The
        bot's own ids (``{{own_jid}}``/``{{own_lid}}``/``{{own_identities}}``)
        come from ``status.state`` — captured at startup and re-captured
        on every session recovery — so identity in the prompt is
        knowledge, not inference; an unknown identity drops the
        prompt's identity lines instead of rendering a stale JID.
        """
        current = self.config_reloader.current_config()
        answers = (
            self.answer_service.prompt_section()
            if self.answer_service is not None
            else ""
        )
        skills = ""
        if self.skill_service is not None:
            try:
                skills = self.skill_service.prompt_section(current_target())
            except RuntimeError:
                skills = ""
        return render_system_prompt(
            current.system_prompt,
            current.timezone or self.settings.timezone,
            current.bot_name,
            current.goal,
            current.memory,
            own_jid=status_state.operator_jid,
            own_lid=status_state.operator_lid,
            operator_name=current.operator_name,
            answers=answers,
            knowledge_only=current.knowledge_only,
            skills=skills,
        )

    def participation_brief(self, current: SessionConfig, target: object) -> str:
        """The participation rule with the team-group override applied."""
        chat_id = str(getattr(target, "chat_id", "") or "")
        mode = (
            group_participation(current, chat_id)
            if chat_id
            else current.group_participation
        )
        return _PARTICIPATION_BRIEF.get(mode, "")

    def render_session_rules(self) -> str:
        """The session rules that gate behaviour, as one current line.

        Hot-reloaded like the prompt: the participation mode comes
        from the live config, so a config edit changes the next run's
        pinned state block without a restart. Support hours and reply
        windows are the v1.2.0 support product's settings; the engine
        states what it actually enforces today.
        """
        current = self.config_reloader.current_config()
        zone = effective_timezone(current.timezone, self.settings.timezone)
        try:
            target = current_target()
        except RuntimeError:
            target = None
        participation = self.participation_brief(current, target)
        parts = [f"Timezone: {zone}.", participation]
        if target is not None and is_team_group(current.team_groups, target.chat_id):
            parts.append(
                "Internal team group: read-only ticket facts, no ticket writes, "
                + "no customer messages, no customer-system writes, no TTS, "
                + "no greetings."
            )
        if current.voice_replies and self.settings.tts_url:
            parts.append(
                "Voice replies are enabled: when the customer sends a voice note, "
                + "deliver the answer as a voice note."
            )
        if current.hours:
            state = opening_state(current, int(time.time()), self.settings.timezone)
            if state.open:
                parts.append(
                    f"Open now until {format_opening(state.next_opening, zone)}."
                )
            else:
                label = format_opening(state.next_opening, zone)
                parts.append(
                    f"Closed now; next opening {label}. Non-urgent alerts are "
                    + "deferred; tell the customer when to expect a reply."
                )
        return " ".join(part for part in parts if part)

    def render_gate_brief(self) -> str:
        """The bounded session brief the triage gate sees, not the agent's.

        The gate judged generic ``why`` values for the same reason: it
        never knew the assistant's name, the operator, the goal, or the
        operator exception. This is the one-line fix — separate from
        ``render_session_rules``, which feeds the agent's pinned block.
        Hot-reloaded like the prompt.
        """
        current = self.config_reloader.current_config()
        try:
            target = current_target()
        except RuntimeError:
            target = None
        parts = [f"Assistant name: {current.bot_name or 'the assistant'}."]
        parts.append(
            f"Timezone: {effective_timezone(current.timezone, self.settings.timezone)}."
        )
        if current.operator_name:
            parts.append(f"Human operator: {current.operator_name}.")
        parts.append(self.participation_brief(current, target))
        exception = (
            "Messages for the operator's personal life, body, whereabouts, "
            + "plans, calendar, or tastes are not for the assistant."
        )
        parts.append(exception)
        if current.goal:
            parts.append(f"Goal: {current.goal}")
        return " ".join(part for part in parts if part)[:GATE_BRIEF_CHARS]
