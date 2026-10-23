from wahabot.ai.context import (
    render_system_prompt,
)
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
        return render_system_prompt(
            current.system_prompt,
            self.settings.timezone,
            current.bot_name,
            current.goal,
            current.memory,
            own_jid=status_state.operator_jid,
            own_lid=status_state.operator_lid,
            operator_name=current.operator_name,
        )

    def render_session_rules(self) -> str:
        """The session rules that gate behaviour, as one current line.

        Hot-reloaded like the prompt: the participation mode comes
        from the live config, so a config edit changes the next run's
        pinned state block without a restart. Support hours and reply
        windows are the v1.2.0 support product's settings; the engine
        states what it actually enforces today.
        """
        current = self.config_reloader.current_config()
        participation = _PARTICIPATION_BRIEF.get(current.group_participation, "")
        return " ".join(
            part
            for part in (f"Timezone: {self.settings.timezone}.", participation)
            if part
        )

    def render_gate_brief(self) -> str:
        """The bounded session brief the triage gate sees, not the agent's.

        The gate judged generic ``why`` values for the same reason: it
        never knew the assistant's name, the operator, the goal, or the
        operator exception. This is the one-line fix — separate from
        ``render_session_rules``, which feeds the agent's pinned block.
        Hot-reloaded like the prompt.
        """
        current = self.config_reloader.current_config()
        parts = [f"Assistant name: {current.bot_name or 'the assistant'}."]
        parts.append(f"Timezone: {self.settings.timezone}.")
        if current.operator_name:
            parts.append(f"Human operator: {current.operator_name}.")
        parts.append(_PARTICIPATION_BRIEF.get(current.group_participation, ""))
        exception = (
            "Messages for the operator's personal life, body, whereabouts, "
            + "plans, calendar, or tastes are not for the assistant."
        )
        parts.append(exception)
        if current.goal:
            parts.append(f"Goal: {current.goal}")
        return " ".join(part for part in parts if part)[:GATE_BRIEF_CHARS]
