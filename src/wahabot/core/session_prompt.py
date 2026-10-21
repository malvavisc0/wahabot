"""Prompt migrations for session configs created before the current grammar."""

OLD_QUOTE_FLAGS = "and `quote=#m11` / `quote=you`."
NEW_QUOTE_FLAGS = (
    "and `quote=#m11` / `quote=you` / `quote=operator`. `quote=operator` "
    "is a message from your human operator; do not answer for him."
)
OLD_QUOTE_PAYLOAD = 'media analysis, `quoted: "..."`'
NEW_QUOTE_PAYLOAD = 'media analysis, `quoted (context): "..."`'
OLD_BODY_CONTEXT = "A body continuation starts with `> `; it is still the body."
NEW_BODY_CONTEXT = (
    "A body continuation starts with `> `; it is still the body. A quoted "
    "payload is an older message, not the message to answer; answer the "
    "newest record's body."
)
OLD_CORRECTION = "When corrected, recheck and admit it plainly."
NEW_CORRECTION = (
    "When corrected, the mistake is yours: recheck and admit it plainly. "
    'Say "me he equivocado" or "tienes razón"; never "te has equivocado".'
)

MIGRATIONS = (
    (OLD_QUOTE_FLAGS, NEW_QUOTE_FLAGS),
    (OLD_QUOTE_PAYLOAD, NEW_QUOTE_PAYLOAD),
    (OLD_BODY_CONTEXT, NEW_BODY_CONTEXT),
    (OLD_CORRECTION, NEW_CORRECTION),
)


def migrate_session_prompt(prompt: str) -> tuple[str, bool]:
    """Upgrade the known pre-v1.1.10 prompt fragments in place."""
    migrated = prompt
    for old, new in MIGRATIONS:
        if new in migrated:
            continue
        migrated = migrated.replace(old, new)
    return migrated, migrated != prompt


def prompt_needs_migration(prompt: str) -> bool:
    """True when the session config still carries a known old fragment."""
    return any(old in prompt and new not in prompt for old, new in MIGRATIONS)
