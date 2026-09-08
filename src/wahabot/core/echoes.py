"""Self-chat echo tracking: the bot's own messages to itself.

The bot writes to its own "message yourself" chat from three senders —
self-chat command replies, ``escalate`` reports, and session up/down
notifications. WhatsApp echoes each one back as a fresh ``fromMe``
``message`` event, and a message in the self-chat that matches the
mention pattern is (by design) parsed as an operator command with
cross-chat reach. Without this module the bot could parse its own
output — or, far worse, prompt-injected content the model was tricked
into writing into an escalation report — as a trusted command.

Every sender to the self-chat marks the sent id here, and the message
entry point checks ``is_self_echo`` before classifying an event as a
command. Bounded like the seen-id cache; entries expire with the same
window WAHA redelivery uses.
"""

from wahabot.core.cache import TtlCache

#: How long a sent id stays marked — past this, a redelivery of the
#: echo is stale by the message-age guard anyway.
_ECHO_TTL_S = 300
_MAX_ECHOES = 1000

_echoes: TtlCache[str, bool] = TtlCache(_ECHO_TTL_S, _MAX_ECHOES)


def remember_self_echo(message_id: str) -> None:
    """Mark a message the bot sent to its own self-chat as run output."""
    if message_id:
        _echoes.put(message_id, True)


def is_self_echo(message_id: str) -> bool:
    """True when this message id is one the bot sent to itself."""
    return message_id in _echoes
