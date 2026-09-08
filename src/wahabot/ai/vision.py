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
(``image["caption"]``) while downloading — before the chat's run lock — and
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

__all__ = [
    "MAX_CAPTION_CHARS",
    "caption_image",
    "caption_images",
    "clamp_caption_line",
    "image_caption",
    "image_noun",
]

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


def clamp_caption_line(text: str) -> str:
    """A caption answer reduced to one clamped line, "" when empty.

    Captions are embedded verbatim into the user message text, so
    anything chatty ("This image shows…") would read as the sender's
    words, a multi-line answer would break the message layout, and a
    rambling model must not inflate the rolling buffer. Shared by the
    image and video captioners.
    """
    caption = text.strip()
    caption = caption.splitlines()[0].strip() if caption else ""
    if len(caption) > MAX_CAPTION_CHARS:
        caption = caption[:MAX_CAPTION_CHARS].rstrip() + "…"
    return caption


async def caption_blocks(
    llm: FunctionCallingLLM,
    blocks: list[Any],
    timeout: float = 30.0,
    semaphore: asyncio.Semaphore | None = None,
    kind: str = "Image",
) -> str:
    """One-sentence caption of the message *blocks*, or "" on any failure.

    A failed caption must never sink the turn: the caller falls back to
    the plain marker and the pixels still ride the first LLM call, so
    the run degrades to the pre-caption behavior. *semaphore* (the
    workflow's LLM gate) bounds this call within the run's concurrency
    budget. Shared by the image and video captioners.
    """
    message = ChatMessage(role=MessageRole.USER, blocks=blocks)
    try:
        if semaphore is None:
            response = await asyncio.wait_for(llm.achat([message]), timeout=timeout)
        else:
            async with semaphore:
                response = await asyncio.wait_for(llm.achat([message]), timeout=timeout)
    except Exception as exc:  # any failure degrades to the bare marker
        logger.warning(
            "{kind} caption failed (turn stays text-anchored): {exc}",
            kind=kind,
            exc=exc,
        )
        return ""
    caption = clamp_caption_line(str(response.message.content or ""))
    logger.info("{kind} caption: {caption!r}", kind=kind, caption=caption)
    return caption


async def caption_image(
    llm: FunctionCallingLLM,
    image: dict[str, Any],
    timeout: float = 30.0,
    semaphore: asyncio.Semaphore | None = None,
) -> str:
    """One-sentence description of *image*, or "" on any failure."""
    return await caption_blocks(
        llm,
        [
            ImageBlock(
                image=image["data"],
                image_mimetype=image.get("mimetype") or "image/jpeg",
            ),
            TextBlock(text=_CAPTION_PROMPT),
        ],
        timeout,
        semaphore=semaphore,
        kind="Image",
    )


async def caption_images(
    llm: FunctionCallingLLM,
    images: list[dict[str, Any]],
    timeout: float = 30.0,
    semaphore: asyncio.Semaphore | None = None,
) -> list[str]:
    """Captions for *images*, position-aligned; failed images get "".

    *semaphore* (the workflow's LLM gate) is acquired per caption call:
    a burst of parallel chat runs must queue their vision calls like
    any other LLM traffic instead of fanning out past the cap.
    """
    if not images:
        return []
    return list(
        await asyncio.gather(
            *(caption_image(llm, img, timeout, semaphore=semaphore) for img in images)
        )
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
