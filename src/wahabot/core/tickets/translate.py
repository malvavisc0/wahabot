"""The LLM-backed translator shared by alerts and the operator reply loop."""

from collections.abc import Callable

_PROMPT = (
    "Translate the following text into {language}. Keep every number, date, "
    "amount and name exactly as written. Return only the translation.\n\n{text}"
)


class TicketTranslator:
    """A translator whose completer is attached after the agent is built.

    One channel per registration, so tools and handlers share the same
    lazily bound model. Without a completer the source text is returned
    unchanged — translation never blocks a ticket.
    """

    def __init__(self, completer: Callable[[str], str] | None = None) -> None:
        self.completer = completer

    def bind(self, completer: Callable[[str], str]) -> None:
        """Attach the model completion used for every translation."""
        self.completer = completer

    def __call__(self, text: str, language: str) -> str:
        """Translate one text, or return it unchanged when unbound."""
        if self.completer is None:
            return text
        return self.completer(_PROMPT.format(language=language, text=text))
