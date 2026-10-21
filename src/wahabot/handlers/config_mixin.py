from wahabot.ai.context import (
    render_system_prompt,
)
from wahabot.handlers.base import HandlerBase
from wahabot.status import state as status_state


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
        participation = {
            "never": "In groups, never reply.",
            "mentioned": "In groups, reply only when addressed or replied to.",
            "judicious": "In groups, judge each message for whether to reply.",
        }.get(current.group_participation, "")
        return " ".join(
            part
            for part in (f"Timezone: {self.settings.timezone}.", participation)
            if part
        )
