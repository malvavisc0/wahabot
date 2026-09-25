"""Message/prompt sizing helpers for the agent workflow.

``message_text`` and ``token_count`` feed the memory trim
(``trim_to_budget`` in ``wahabot.ai.history``): the budget is
denominated in real tokens, so the counter must share the tokenizer the
``ChatMemoryBuffer`` uses and fall back conservatively when it fails.
"""

from llama_index.core.base.llms.types import ChatMessage, ToolCallBlock


def message_text(msg: ChatMessage) -> str:
    """A message's full text: content plus tool-call names and arguments.

    Tool-call arguments ride on the message as blocks/kwargs, not in
    ``content`` — without them a long tool-call history looks nearly
    free and the true prompt size drifts far past the budget.
    """
    args = "".join(
        str(block.tool_kwargs) for block in msg.blocks if isinstance(block, ToolCallBlock)
    ) or str(msg.additional_kwargs.get("tool_calls", ""))
    return str(msg.content or "") + args


def token_count(msg: ChatMessage) -> int:
    """A message's honest token count via the tokenizer, chars/4 fallback.

    The budget (``memory_token_limit``) is denominated in real tokens,
    so the trim must count in the same currency: the llama-index default
    tokenizer (the same one ``ChatMemoryBuffer`` uses for its own
    eviction) puts one Spanish/English word at ~2-3 tokens, where the
    old 1-char≈1-token estimate overcounted ~4x and starved the model
    to a few visible turns per run. If the tokenizer raises for any
    reason, a chars/4 estimate keeps the trim conservative (still an
    overcount, never an undercount of true context).
    """
    text = message_text(msg)
    try:
        from llama_index.core.utils import get_tokenizer

        return len(get_tokenizer()(text))
    except Exception:
        return max(1, len(text) // 4)
