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

import time

_echoes: dict[str, float] = {}
#: How long a sent id stays marked — past this, a redelivery of the
#: echo is stale by the message-age guard anyway.
_ECHO_TTL_S = 300
_MAX_ECHOES = 1000


def _drop_expired(now: float) -> None:
    """Evict entries past the TTL (write-side sweep)."""
    for stale_id in [k for k, ts in _echoes.items() if now - ts > _ECHO_TTL_S]:
        del _echoes[stale_id]


def remember_self_echo(message_id: str) -> None:
    """Mark a message the bot sent to its own self-chat as run output."""
    if not message_id:
        return
    now = time.time()
    while len(_echoes) >= _MAX_ECHOES:
        # Insertion order is oldest first, same idiom as the seen cache.
        del _echoes[next(iter(_echoes))]
    _echoes[message_id] = now
    _drop_expired(now)


def is_self_echo(message_id: str) -> bool:
    """True when this message id is one the bot sent to itself.

    Expired entries are dropped on read: past the TTL a redelivery is
    stale by the message-age guard and can no longer re-enter the
    command path.
    """
    ts = _echoes.get(message_id)
    if ts is None:
        return False
    if time.time() - ts > _ECHO_TTL_S:
        del _echoes[message_id]
        return False
    return True
