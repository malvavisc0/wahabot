"""Self-chat skill confirmations and capability answers."""

import asyncio
from typing import override

from wahabot.core.models import WahaEvent
from wahabot.core.tickets.router import Route, RouteKind, is_affirmative
from wahabot.handlers.base import HandlerBase


class SkillMixin(HandlerBase):
    """Handle the self-chat routes owned by the skill runtime."""

    @override
    async def ticket_skill_route(self, event: WahaEvent, route: Route) -> bool:
        """Handle a staged-write yes/no or a capability question."""
        if route.kind == RouteKind.CAPABILITIES:
            summary = (
                self.skill_service.capability_summary()
                if self.skill_service is not None
                else "No skills are configured yet."
            )
            await self.ticket_send_operator(event, summary)
            return True
        if route.kind != RouteKind.SKILL_CONFIRM:
            return False
        service = self.skill_service
        request = route.skill_request
        if service is None:
            await self.ticket_send_operator(event, "No staged request is waiting.")
            return True
        if request is None:
            await self.ticket_send_operator(
                event,
                "More than one staged request is waiting; quote the one you mean.",
            )
            return True
        approve = is_affirmative(route.text.strip())
        text = await asyncio.to_thread(service.confirm, request.id, approve=approve)
        await self.ticket_send_operator(event, text)
        return True
