"""Cheap, deterministic language detection for customer messages.

The ticket layer only needs enough signal to pick the holding reply's
language and to fill the operator's ``Language:`` line. A stop-word
vote is deliberately used instead of a model call: it is free, cannot
fail, and the customer's own words are already in hand.
"""

import re
from types import MappingProxyType

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

#: Words that unambiguously identify one language in a short message.
_STRONG = MappingProxyType(
    {
        "hola": "es",
        "gracias": "es",
        "bonjour": "fr",
        "merci": "fr",
        "hallo": "de",
        "danke": "de",
        "obrigado": "pt",
        "obrigada": "pt",
        "olá": "pt",
        "ciao": "it",
        "grazie": "it",
    }
)

#: Function words for the languages the operator desk is used in.
_STOPWORDS = MappingProxyType(
    {
        "en": frozenset(
            {
                "the",
                "and",
                "is",
                "are",
                "my",
                "your",
                "i",
                "you",
                "we",
                "please",
                "hello",
                "thanks",
                "with",
                "for",
                "not",
                "have",
                "has",
                "can",
                "this",
                "that",
            }
        ),
        "es": frozenset(
            {
                "el",
                "la",
                "los",
                "las",
                "de",
                "del",
                "que",
                "y",
                "en",
                "un",
                "una",
                "por",
                "para",
                "con",
                "mi",
                "tu",
                "es",
                "está",
                "esta",
                "necesito",
                "ayuda",
                "pedido",
                "pero",
                "no",
                "sí",
                "si",
                "cuando",
                "dónde",
                "donde",
                "tengo",
                "quiero",
                "puede",
            }
        ),
        "fr": frozenset(
            {
                "le",
                "la",
                "les",
                "de",
                "des",
                "du",
                "et",
                "en",
                "un",
                "une",
                "pour",
                "avec",
                "mon",
                "ma",
                "mes",
                "est",
                "je",
                "j",
                "ai",
                "besoin",
                "aide",
                "commande",
                "mais",
                "pas",
                "oui",
                "quand",
                "où",
                "j'ai",
                "svp",
                "merci",
            }
        ),
        "de": frozenset(
            {
                "der",
                "die",
                "das",
                "den",
                "dem",
                "und",
                "ist",
                "sind",
                "ich",
                "du",
                "wir",
                "mein",
                "meine",
                "mit",
                "für",
                "nicht",
                "habe",
                "hat",
                "kann",
                "bitte",
                "hallo",
                "danke",
                "bestellung",
                "wann",
                "wo",
                "aber",
                "ja",
            }
        ),
        "pt": frozenset(
            {
                "o",
                "a",
                "os",
                "as",
                "de",
                "do",
                "da",
                "que",
                "e",
                "em",
                "um",
                "uma",
                "para",
                "com",
                "meu",
                "minha",
                "é",
                "está",
                "preciso",
                "ajuda",
                "pedido",
                "mas",
                "não",
                "sim",
                "quando",
                "onde",
                "tenho",
                "quero",
            }
        ),
        "it": frozenset(
            {
                "il",
                "lo",
                "la",
                "i",
                "gli",
                "le",
                "di",
                "del",
                "che",
                "e",
                "in",
                "un",
                "una",
                "per",
                "con",
                "mio",
                "mia",
                "è",
                "sono",
                "ho",
                "bisogno",
                "aiuto",
                "ordine",
                "ma",
                "non",
                "sì",
                "quando",
                "dove",
                "voglio",
            }
        ),
    }
)


def strong_language(words: list[str]) -> str:
    """The language of the first unambiguous marker, or ""."""
    for word in words:
        strong = _STRONG.get(word)
        if strong is not None:
            return strong
    return ""


def language_scores(words: list[str]) -> dict[str, int]:
    """Stop-word votes for every known language."""
    return {
        language: sum(1 for word in words if word in stops)
        for language, stops in _STOPWORDS.items()
    }


def best_language(scores: dict[str, int]) -> str:
    """The clear winner, or "" when the vote is weak or tied."""
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best, score = ranked[0]
    if score < 2:
        return ""
    if len(ranked) > 1 and ranked[1][1] == score:
        return ""
    return best


def detect_language(text: str) -> str:
    """The likely ISO-639-1 language of *text*, or "" when unsure."""
    words = [word for word in _WORD.findall(text.casefold()) if word]
    if not words:
        return ""
    return strong_language(words) or best_language(language_scores(words))


def operator_language_for(
    configured: str, detected: str = "", default: str = "en"
) -> str:
    """The operator-facing language: setting, then detection, then English."""
    return (configured or detected or default).strip() or default
