from llama_index.core.tools import BaseTool, FunctionTool

from wahabot.ai.tools.envelope import error, ok
from wahabot.ai.tools.runcontext import (
    current_target,
    delivered_to_self,
    delivery_message_error,
    delivery_target_error,
    fenced_chat,
    fenced_message_id,
    log_action_reason,
    send_failed_envelope,
    send_guard,
)
from wahabot.ai.tools.schemas import (
    ForwardMessageSchema,
)
from wahabot.core.waha import WahaClient


def forward_message(waha: WahaClient) -> BaseTool:
    """Build native forwarding from a source message id to a destination JID.

    Ordinary runs fence both destination and recognizable source ids;
    operator runs can cross chats. Success echoes the source message_id,
    not the new forwarded id. Rendering/attribution is not verified here.
    """

    def forward_message_fn(
        message_id: str, chat: str | None = None, reason: str = ""
    ) -> str:
        target = current_target()
        if refusal := send_guard(target):
            return refusal
        if not message_id.strip():
            return error("message_id is required")
        chat_id, fence_error = fenced_chat(chat, target)
        if fence_error:
            return error(fence_error)
        session = target.session
        if not session or not chat_id:
            return error("no active conversation context")
        delivery_error = delivery_target_error(chat_id, None, target)
        if delivery_error:
            return error(delivery_error)
        _, id_error = fenced_message_id(message_id, target)
        if id_error:
            return error(id_error)
        delivery_error = delivery_message_error(message_id, target)
        if delivery_error:
            return error(delivery_error)
        log_action_reason("forward_message", reason, chat=chat_id)
        try:
            sent_id = waha.forward_message(session, chat_id, message_id)
        except Exception as exc:
            return send_failed_envelope("forward_message", chat_id, exc)
        delivered_to_self(chat_id, sent_id)
        target.sent = chat_id
        target.sent_id = sent_id
        return ok(message_id=message_id, chat=chat_id)

    return FunctionTool.from_defaults(
        fn=forward_message_fn,
        fn_schema=ForwardMessageSchema,
        name="forward_message",
        description=(
            "Native WAHA forwarding without retyping: message_id is the "
            "existing source id, chat is the destination JID. Ordinary runs "
            "must keep both in the current chat; operator commands may cross "
            "chats. Success echoes the source id, not the new forwarded id. "
            "Media/rendering depends on WAHA; original-author attribution "
            "is not guaranteed. Shares the send allowance with text/media."
        ),
    )
