"""Image generation and editing tools, registered only with an image API.

Both tools return a local file path plus metadata; the model sends the
file with ``send_media`` (kind=image, path=…) so the existing
one-send-per-run latch and delivery treatment apply unchanged.
"""

from llama_index.core.tools import BaseTool, FunctionTool

from wahabot.ai.tools.envelope import error, ok
from wahabot.ai.tools.schemas import EditImageSchema, ImageSchema
from wahabot.ai.tools.whatsapp import current_image_path
from wahabot.core import images
from wahabot.settings import Settings

__all__ = ["image_builder"]


def image_builder(settings: Settings) -> list[BaseTool]:
    """Build the text-to-image and image-edit tools over the image service."""

    def generate_image_fn(prompt: str, size: str = "", reason: str = "") -> str:
        """Render a full scene caption to a PNG and return its path and metadata."""
        render_size = (size or settings.image_size).strip()
        data = images.generate(settings, prompt.strip(), render_size)
        if data is None:
            return error("image generation failed; try a simpler caption")
        path = images.save(data)
        return ok(
            path=str(path),
            requested_size=render_size,
            **images.metadata(data),
            hint="send it with send_media kind=image path=<path>",
        )

    def edit_image_fn(
        prompt: str,
        source: str = "",
        strength: float | None = None,
        reason: str = "",
    ) -> str:
        """Repaint a source image to match a prompt; return path and metadata.

        The source is *source* (a local path or URL) or, when omitted,
        the image the sender just sent this turn.
        """
        chosen = source.strip() or current_image_path()
        if not chosen:
            return error("no source image: pass source=path/url or send an image")
        loaded = images.load_source(chosen)
        if isinstance(loaded, str):
            return error(loaded)
        name, data, mimetype = loaded
        result = images.edit(settings, prompt.strip(), name, data, mimetype, strength)
        if result is None:
            return error("image edit failed; try a simpler prompt")
        path = images.save(result)
        return ok(
            path=str(path),
            source=chosen,
            **images.metadata(result),
            hint="send it with send_media kind=image path=<path>",
        )

    return [
        FunctionTool.from_defaults(
            fn=generate_image_fn,
            fn_schema=ImageSchema,
            name="generate_image",
            description=(
                "Generate an image from a prompt and return the temp file's path "
                "plus format, pixel width/height and byte size; then send it with "
                "send_media kind=image path=…. The service does NOT expand "
                "prompts, so write a rich, full-scene caption yourself — subject "
                "and action, setting and background, lighting and mood, "
                "composition and camera/lens, style and colours; a short request "
                f"yields a weak image. Size is WxH (default {settings.image_size})."
            ),
        ),
        FunctionTool.from_defaults(
            fn=edit_image_fn,
            fn_schema=EditImageSchema,
            name="edit_image",
            description=(
                "Repaint an existing image to match a prompt and return the new "
                "file's path plus metadata; send it with send_media kind=image "
                "path=…. The service does NOT expand prompts: write a rich caption "
                "of the FINISHED scene — subject, setting, lighting, style — "
                "stating what changes and what stays the same; a short request "
                "yields a weak result. The source is a local path/URL, or the "
                "image the sender just sent when omitted; strength (0-1) sets how "
                "much is repainted."
            ),
        ),
    ]
