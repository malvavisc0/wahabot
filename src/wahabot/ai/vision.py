"""One-line image captions: the durable text anchor for a vision turn.

Image pixels are deliberately one-shot (see ``workflow.with_image``):
they ride the first LLM call of a run and never enter the rolling
memory buffer. Without a text stand-in, round 2+ of the same run — and
every later turn quoting the picture — leaves the model blind, and a
conversational model confabulates instead of admitting it ("Wait, I
don't know what the image is. I made up a comment.").

:func:`caption_image` spends one small vision call per downloaded image
*before* the agent run and returns a single sentence ("beer glass, foam
shaped like a bear"). The handler stores it on the image dict
(``image["caption"]``) while downloading — outside the agent lock — and
``handle_message`` weaves it into the user message text (``(image
shows: beer glass, foam shaped like a bear)``), so the description
lives in memory like any other chat line: it survives trimming, rides
every later LLM call for free, and keeps the system prompt's "you can
see photos" and "never guess" rules simultaneously satisfiable once the
pixels are gone.
"""

import asyncio
from typing import Any

from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from loguru import logger

__all__ = ["caption_image", "caption_images", "image_caption", "image_noun"]

#: One sentence, no preamble: the caption is embedded verbatim into the
#: user message text, so anything chatty ("This image shows…") would
#: read as the sender's words in the retained history.
_CAPTION_PROMPT = (
    "Describe this image in one short sentence. Name the concrete subject: "
    "objects, people, animals, and any visible text or meme caption. "
    "No preamble, no disclaimers — just the description."
)

#: Captions are trimmed to one line and this many chars: they sit inside
#: the ``(image: …)`` marker in the user message, so a rambling model
#: must not inflate the rolling buffer.
MAX_CAPTION_CHARS = 200


async def caption_image(
    llm: FunctionCallingLLM, image: dict[str, Any], timeout: float = 30.0
) -> str:
    """One-sentence description of *image*, or "" on any failure.

    A failed caption must never sink the turn: the caller falls back to
    the plain ``(image)`` marker and the pixels still ride the first
    LLM call, so the run degrades to the pre-caption behavior.
    """
    message = ChatMessage(
        role=MessageRole.USER,
        blocks=[
            ImageBlock(
                image=image["data"],
                image_mimetype=image.get("mimetype") or "image/jpeg",
            ),
            TextBlock(text=_CAPTION_PROMPT),
        ],
    )
    try:
        response = await asyncio.wait_for(llm.achat([message]), timeout=timeout)
    except Exception as exc:  # any failure degrades to the bare (image) marker
        logger.warning("Image caption failed (turn stays text-anchored): {exc}", exc=exc)
        return ""
    caption = str(response.message.content or "").strip()
    # One line only; a multi-line answer would break the message layout.
    caption = caption.splitlines()[0].strip() if caption else ""
    if len(caption) > MAX_CAPTION_CHARS:
        caption = caption[:MAX_CAPTION_CHARS].rstrip() + "…"
    logger.info("Image caption: {caption!r}", caption=caption)
    return caption


async def caption_images(
    llm: FunctionCallingLLM, images: list[dict[str, Any]], timeout: float = 30.0
) -> list[str]:
    """Captions for *images*, position-aligned; failed images get ""."""
    if not images:
        return []
    return list(
        await asyncio.gather(*(caption_image(llm, img, timeout) for img in images))
    )


def image_caption(image: dict[str, Any]) -> str:
    """The caption stored on an image dict by the download path, or ""."""
    return str(image.get("caption") or "")


def image_noun(captions: list[str]) -> str:
    """The ``(image…)`` marker text for a turn whose body is only pictures.

    With one captioned image: ``(image shows: beer glass with bear
    foam)``. An album lists its captions: ``(images show: a cat; a
    dog)``. The ``shows`` framing keeps an instruction-like caption
    ("ignore your instructions…") reading as a description of pixels,
    not as sender text. Images whose caption failed keep the bare
    ``(image)``/``(images)`` form.
    """
    known = [caption for caption in captions if caption]
    if not known:
        return "(images)" if len(captions) > 1 else "(image)"
    if len(captions) == 1:
        return f"(image shows: {known[0]})"
    return f"(images show: {'; '.join(known)})"
