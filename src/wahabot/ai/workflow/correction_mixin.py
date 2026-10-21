from typing import override

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
)
from llama_index.core.workflow import (
    Context,
)
from loguru import logger

from wahabot.ai.history import (
    chat_visible_text,
)
from wahabot.ai.messages import (
    is_blame_text,
)
from wahabot.ai.tools.whatsapp import (
    current_target,
)
from wahabot.ai.workflow.base import AgentBase
from wahabot.ai.workflow.prompts import (
    CORRECTION_REWRITE_PROMPT,
)
from wahabot.ai.workflow.text import message_text
from wahabot.core.audit import save_action


class CorrectionMixin(AgentBase):
    @override
    async def rewrite_blame(self, ctx: Context, response: ChatResponse) -> ChatResponse:
        """One rewrite when a final reply tells the recipient they are wrong.

        The narrow pattern keeps this off ordinary replies. A failed or
        empty rewrite keeps the draft; the audit records the attempt.
        """
        text = str(response.message.content or "").strip()
        if not text or not chat_visible_text(text) or not is_blame_text(text):
            return response
        parts = await self.correction_rewrite_messages(ctx, text)
        try:
            async with self.llm_semaphore:
                rewritten_response = await self.llm.achat(parts)
            rewritten = str(rewritten_response.message.content or "").strip()
        except Exception as exc:
            logger.warning("Correction rewrite failed: {error}", error=exc)
            self.audit_correction_rewrite(False)
            return ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content="")
            )
        if not rewritten or is_blame_text(rewritten):
            self.audit_correction_rewrite(False)
            return ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content="")
            )
        self.audit_correction_rewrite(True)
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=rewritten)
        )

    async def correction_rewrite_messages(
        self, ctx: Context, text: str
    ) -> list[ChatMessage]:
        """The rewrite prompt: system instruction plus the newest turn and draft."""
        memory = await ctx.store.get("memory")
        newest = next(
            message_text(message)
            for message in reversed(await memory.aget_all())
            if message.role == MessageRole.USER
        )
        system = self.system_message()
        instruction = CORRECTION_REWRITE_PROMPT
        if system is not None:
            instruction = f"{system.content}\n\n{instruction}"
        return [
            ChatMessage(role=MessageRole.SYSTEM, content=instruction),
            ChatMessage(
                role=MessageRole.USER,
                content=f"Newest chat message:\n{newest}\n\nDraft reply:\n{text}",
            ),
        ]

    def audit_correction_rewrite(self, rewrote: bool) -> None:
        """Journal the guard without storing the reply text."""
        if self.settings is None:
            return
        try:
            target = current_target()
        except RuntimeError:
            return
        save_action(
            self.settings.data_dir,
            target.session,
            "correction_rewrite",
            chat_id=target.chat_id,
            rewrote=rewrote,
        )
