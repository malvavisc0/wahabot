"""Cached display translation for the ticket views."""

import hashlib
from collections.abc import Callable

from loguru import logger

from wahabot.core.tickets.store import TicketStore

Translator = Callable[[str, str], str]

TRANSLATION_LANGUAGE = "en"


def translate_cached(
    store: TicketStore,
    text: str,
    target: str,
    translate: Translator | None,
) -> str:
    """Translate once per source hash and language, caching the result.

    Without a configured translator the source is returned untouched and
    nothing is cached, so a later translator can still run.
    """
    if translate is None or not text.strip():
        return text
    source_hash = hashlib.sha256(text.encode()).hexdigest()
    cached = store.translation(source_hash, target)
    if cached is not None:
        return cached
    try:
        rendered = str(translate(text, target)).strip()
    except Exception as exc:
        logger.warning("Display translation failed: {error}", error=type(exc).__name__)
        return text
    if rendered:
        store.save_translation(source_hash, target, rendered)
    return rendered or text
