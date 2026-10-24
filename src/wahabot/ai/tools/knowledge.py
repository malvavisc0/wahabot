"""Read-only knowledge-base search tool."""

from llama_index.core.tools import BaseTool, FunctionTool

from wahabot.ai.tools.envelope import ok
from wahabot.ai.tools.runcontext import log_action_reason
from wahabot.ai.tools.schemas import SearchKnowledgeSchema
from wahabot.core.answers.retrieval import ANSWERS_SEARCH_LIMIT, render_search_results
from wahabot.core.answers.service import AnswerService


def search_knowledge(service: AnswerService | None) -> BaseTool:
    """Build the shared read-only answer search tool.

    Registered unconditionally: with no store it reports that there are
    no answers, so an answer added later needs no agent rebuild.
    """

    def search_knowledge_fn(query: str, reason: str = "") -> str:
        log_action_reason("search_knowledge", reason, query=query)
        if service is None:
            return ok(answers=[], message="No published answers are available.")
        hits = service.search(query, limit=ANSWERS_SEARCH_LIMIT)
        return ok(
            answers=[
                {
                    "id": hit.answer.id,
                    "question": hit.answer.question,
                    "answer": hit.answer.answer,
                }
                for hit in hits
            ],
            rendered=render_search_results(hits),
        )

    return FunctionTool.from_defaults(
        fn=search_knowledge_fn,
        fn_schema=SearchKnowledgeSchema,
        name="search_knowledge",
        description=(
            "Search the customer-maintained published answers when the whole "
            "answer set was not injected into the prompt. Read-only and shared "
            "across chats; reports no answers when the knowledge base is empty."
        ),
    )
