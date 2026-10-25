"""Errors raised by the skills runtime."""


class SkillError(RuntimeError):
    """A skill operation was refused or failed with a clear reason."""


class SecretKeyError(SkillError):
    """The secret key is missing, invalid, or does not decrypt the stored value."""


class QueryRefused(SkillError):
    """A SQL query was refused before execution (shape, comments, mutation)."""
