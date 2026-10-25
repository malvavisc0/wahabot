"""Tools that expose configured skills to the agent."""

from llama_index.core.tools import BaseTool, FunctionTool

from wahabot.ai.tools.envelope import error
from wahabot.ai.tools.runcontext import current_target, log_action_reason
from wahabot.ai.tools.schemas import HttpRequestSchema, ReadSkillSchema, SqlQuerySchema
from wahabot.core.skills.service import SkillService


def build_skill_tools(service: SkillService | None) -> list[BaseTool]:
    """Build the skill tools; they explain themselves when none are configured."""

    def http_request_fn(
        skill: str,
        method: str,
        path: str,
        query: dict[str, str] = {},  # noqa: B006
        body: dict[str, object] = {},  # noqa: B006
        reason: str = "",
    ) -> str:
        log_action_reason("http_request", reason, skill=skill, method=method)
        if service is None:
            return error("no skills are configured in this deployment")
        try:
            target = current_target()
        except RuntimeError:
            return error("http_request ran outside an agent run")
        return service.execute_http(
            skill,
            method,
            path,
            query or None,
            body or None,
            target,
        )

    def sql_query_fn(skill: str, query: str, reason: str = "") -> str:
        log_action_reason("sql_query", reason, skill=skill)
        if service is None:
            return error("no skills are configured in this deployment")
        try:
            target = current_target()
        except RuntimeError:
            return error("sql_query ran outside an agent run")
        return service.execute_sql(skill, query, target)

    def read_skill_fn(skill: str, reason: str = "") -> str:
        log_action_reason("read_skill", reason, skill=skill)
        if service is None:
            return error("no skills are configured in this deployment")
        try:
            target = current_target()
        except RuntimeError:
            return error("read_skill ran outside an agent run")
        return service.read_skill(skill, target)

    return [
        FunctionTool.from_defaults(
            fn=http_request_fn,
            fn_schema=HttpRequestSchema,
            name="http_request",
            description=(
                "Call one configured HTTP skill. The runtime locks the request to "
                "the skill's base URL, blocks private/metadata addresses, enforces "
                "the sender scope, logs the call, and stages customer/team-group "
                "writes for the operator's yes."
            ),
        ),
        FunctionTool.from_defaults(
            fn=sql_query_fn,
            fn_schema=SqlQuerySchema,
            name="sql_query",
            description=(
                "Run one read-only SQL query through a configured SQL skill. "
                "Operator runs only; SELECT or WITH...SELECT, one statement, "
                "bounded rows/columns/bytes and a statement timeout."
            ),
        ),
        FunctionTool.from_defaults(
            fn=read_skill_fn,
            fn_schema=ReadSkillSchema,
            name="read_skill",
            description=(
                "Read one active skill's SKILL.md when its docs were left out of "
                "the prompt budget. Read-only; it never runs the integration."
            ),
        ),
    ]
