"""Ticket routing, pause handling and the operator reply loop."""

import asyncio
import time
from typing import override

from loguru import logger

from wahabot.ai.messages import bot_jids, jid_string
from wahabot.ai.tools.runcontext import RunTarget
from wahabot.commands import build_command_event
from wahabot.core.access import SessionConfig
from wahabot.core.audit import save_action
from wahabot.core.models import WahaEvent
from wahabot.core.notebook import AUTHOR_OPERATOR, KIND_FACT
from wahabot.core.runs import chat_lock
from wahabot.core.tickets.alerts import (
    alert_intent_key,
    operator_language,
    send_alert,
    translate_cached,
)
from wahabot.core.tickets.attachments import persist_event_attachments
from wahabot.core.tickets.identity import phone_alias
from wahabot.core.tickets.language import detect_language, operator_language_for
from wahabot.core.tickets.lifecycle import (
    GLOBAL_PAUSE_FOREVER,
    Inbound,
    apply_delivery,
    apply_split,
    close_ticket,
    closed_mid_run,
    holding_due,
    inbound_words,
    is_paused,
    is_thanks_only,
    queue_alert,
    record_inbound,
    reopen_ticket,
)
from wahabot.core.tickets.models import (
    Actor,
    DoneBy,
    EventKind,
    Ticket,
    TicketState,
    TicketUpdate,
)
from wahabot.core.tickets.replies import (
    Remember,
    ReplyOutcome,
    decision_fact,
    deliver_held,
    handle_operator_reply,
    localized_confirmation,
)
from wahabot.core.tickets.router import (
    Route,
    RouteKind,
    ask_which_ticket,
    is_self_chat_event,
    whats_open_text,
)
from wahabot.handlers.base import HandlerBase, run_delivered
from wahabot.handlers.memory import remember_customer_message, remember_sent_relay
from wahabot.handlers.replies import send_self_reply
from wahabot.status.state import state as status_state

_HOLDING_REPLY = "Thanks — your message is still with the team. Someone will reply here."
_HOLDING_COOLDOWN_S = 3600


class TicketMixin(HandlerBase):
    """Ticket lifecycle hooks shared by the single, burst and album paths."""

    @override
    async def ticket_gate_events(self, events: list[WahaEvent]) -> bool:
        """Record inbound events and handle paused, operator and thanks turns.

        Returns True when the turn needs no assistant run. The ticket is
        recorded for every customer 1:1 message, including paused ones.
        """
        if not events or not self.ticket_turn_is_customer(events[0]):
            return False
        inbound = await asyncio.to_thread(self.ticket_note_events, events)
        store = self.ticket_store
        if inbound is None or store is None:
            return False
        if await self.ticket_thanks_turn(events, inbound):
            return True
        if await self.ticket_deliver_held(inbound, events[0]):
            refreshed = store.ticket(inbound.ticket.id)
            if refreshed is not None:
                inbound.ticket = refreshed
        return await self.ticket_handled_without_run(events, inbound)

    async def ticket_handled_without_run(
        self, events: list[WahaEvent], inbound: Inbound
    ) -> bool:
        """Pause, handover and operator-solved turns skip the agent run."""
        if self.ticket_paused(events[0].session, inbound):
            await self.ticket_remember_and_settle(events, "operator_owned", "paused")
            return True
        if inbound.ticket.state == TicketState.NEEDS_YOU.value:
            await self.ticket_needs_you_turn(events, inbound)
            return True
        if inbound.operator_solved:
            await self.ticket_operator_reask(events, inbound)
            return True
        return False

    def ticket_turn_is_customer(self, event: WahaEvent) -> bool:
        """Whether the turn belongs to a 1:1 customer chat."""
        chat = str(event.payload.get("from", ""))
        return (
            bool(chat)
            and not chat.endswith("@g.us")
            and not event.payload.get("fromMe")
            and not is_self_chat_event(event)
        )

    def ticket_note_events(self, events: list[WahaEvent]) -> Inbound | None:
        """Create/reopen the ticket and store every event's attachments."""
        store = self.ticket_store
        if store is None:
            return None
        content, body = inbound_words(events)
        config = self.config_reloader.current_config()
        phone = phone_alias(store, content, self.waha.lid_phone)
        inbound = record_inbound(
            store,
            content,
            message_ids=[str(event.payload.get("id", "")) for event in events],
            close_after_days=config.close_after_days,
            body=body,
            phone=phone,
        )
        if not inbound.contact.language and body.strip():
            detected = detect_language(body)
            if detected:
                store.set_contact_language(inbound.contact.id, detected)
                inbound.contact.language = detected
        for event in events:
            persist_event_attachments(
                store, self.waha, inbound.ticket, event, self.settings
            )
        return inbound

    async def ticket_thanks_turn(self, events: list[WahaEvent], inbound: Inbound) -> bool:
        """Keep a done ticket done for a bare thanks and react instead."""
        if not inbound.reopened or not is_thanks_only(inbound.body):
            return False
        store = self.ticket_store
        if store is None:
            return False
        ticket = store.ticket(inbound.ticket.id)
        if ticket is None:
            return False
        store.update_ticket(
            ticket.id,
            TicketUpdate(
                state=TicketState.DONE.value,
                done_by=inbound.previous_done_by or DoneBy.ASSISTANT.value,
            ),
            actor=Actor.CUSTOMER.value,
            kind=EventKind.STATE.value,
            body="customer thanked; ticket stays done",
        )
        message_id = str(events[0].payload.get("id", ""))
        try:
            self.waha.send_reaction(events[0].session, ticket.chat_id, message_id, "👍")
        except Exception as exc:
            logger.warning("Thank-you reaction failed: {exc}", exc=exc)
        await self.ticket_remember_and_settle(events, "silent", "thanks")
        return True

    async def ticket_deliver_held(self, inbound: Inbound, event: WahaEvent) -> bool:
        """Send a held operator draft on the next customer message."""
        if self.ticket_store is None:
            return False
        outcome = await deliver_held(
            self.ticket_store,
            self.waha,
            inbound.ticket.chat_id,
            inbound.ticket,
            inbound.contact,
            remember=self.ticket_remember_relay(inbound.ticket),
            answer_service=self.answer_service,
        )
        if outcome is None or not outcome.sent:
            return False
        if outcome.save_ticket is not None:
            config = self.config_reloader.current_config()
            try:
                confirmation = await self.ticket_confirmation(outcome, config)
                await self.ticket_send_operator(event, confirmation)
            except Exception as exc:
                logger.warning("Held-draft confirmation failed: {exc}", exc=exc)
        return True

    async def ticket_needs_you_turn(
        self, events: list[WahaEvent], inbound: Inbound
    ) -> None:
        """Hold the customer with a short reply and nudge once per interval."""
        if self.ticket_holding_due(inbound.ticket.id):
            await self.ticket_send_holding(
                events,
                inbound.ticket.chat_id,
                inbound.contact.language,
                ticket_id=inbound.ticket.id,
            )
        if self.scheduler is not None:
            await asyncio.to_thread(self.scheduler.nudge, int(time.time()))
        await self.ticket_remember_and_settle(events, "operator_owned", "needs_you")

    def ticket_holding_due(self, ticket_id: int) -> bool:
        """Whether this needs-you ticket should send another holding reply."""
        store = self.ticket_store
        if store is None:
            return True
        return holding_due(store, ticket_id, int(time.time()), _HOLDING_COOLDOWN_S)

    async def ticket_operator_reask(
        self, events: list[WahaEvent], inbound: Inbound
    ) -> None:
        """Alert the operator again when a solved ticket reopens."""
        delivered = False
        try:
            delivered = await asyncio.to_thread(self.ticket_send_alert, inbound)
        except Exception as exc:
            logger.warning("Reopened ticket alert failed: {exc}", exc=exc)
        if not delivered:
            await self.ticket_send_holding(
                events,
                inbound.ticket.chat_id,
                inbound.contact.language,
                ticket_id=inbound.ticket.id,
            )
        await self.ticket_remember_and_settle(
            events, "operator_owned", "operator solved re-ask"
        )

    async def ticket_send_holding(
        self,
        events: list[WahaEvent],
        chat_id: str,
        language: str = "",
        *,
        ticket_id: int = 0,
    ) -> bool:
        """Send the holding reply in the customer's language, fail-soft.

        The text is fixed, so the translation is cached by source hash:
        one model call per language for the bot's lifetime.
        """
        text = _HOLDING_REPLY
        store = self.ticket_store
        if store is not None and language and language != "en":
            try:
                text = await asyncio.to_thread(
                    translate_cached,
                    store,
                    _HOLDING_REPLY,
                    language,
                    self.ticket_translator,
                )
            except Exception as exc:
                logger.warning("Holding translation failed: {exc}", exc=exc)
        try:
            sent_id = await asyncio.to_thread(
                self.waha.send_text, events[0].session, chat_id, text
            )
        except Exception as exc:
            logger.warning(
                "Holding reply failed for {chat}: {exc}", chat=chat_id, exc=exc
            )
            return False
        if sent_id and store is not None and ticket_id:
            store.add_event(
                ticket_id,
                Actor.ASSISTANT.value,
                EventKind.HOLDING.value,
                text,
                [sent_id],
            )
        return bool(sent_id)

    def ticket_send_alert(self, inbound: Inbound) -> bool:
        """Send the alert for a reopened operator-solved ticket."""
        store = self.ticket_store
        if store is None or not status_state.operator_jid:
            raise RuntimeError("no operator target")
        result = send_alert(
            store,
            self.waha,
            self.settings.session,
            status_state.operator_jid,
            inbound.ticket,
            inbound.contact,
            language=operator_language(
                self.config_reloader.current_config().operator_language
            ),
            translator=self.ticket_translator,
            alert_key=alert_intent_key(store, inbound.ticket, "reask"),
        )
        if result.pending:
            queue_alert(store, inbound.ticket)
            return False
        return result.delivered

    async def ticket_remember_and_settle(
        self, events: list[WahaEvent], outcome: str, reason: str
    ) -> None:
        """Save the customer messages to memory and settle their intake rows."""
        await self.ticket_remember_events(events)
        ids = [str(event.payload.get("id", "")) for event in events]
        self.intake.finish(
            events[0].session,
            str(events[0].payload.get("from", "")),
            ids,
            delivered=False,
            outcome=outcome,
            reason=reason,
        )

    async def ticket_remember_events(self, events: list[WahaEvent]) -> None:
        """Append non-empty customer bodies to the chat buffer."""
        for event in events:
            body = str(event.payload.get("body", "") or "").strip()
            if not body:
                continue
            async with chat_lock(event.session, str(event.payload.get("from", ""))):
                await remember_customer_message(event, self.agent, self.settings, body)

    def ticket_remember_relay(self, ticket: Ticket) -> Remember:
        """The memory callback for an operator relay to one ticket."""

        async def remember(text: str) -> None:
            await remember_sent_relay(
                self.settings.session, ticket.chat_id, self.agent, self.settings, text
            )

        return remember

    @override
    def ticket_active_id(self, chat_id: str) -> int:
        """The chat's active ticket id, or 0 when tickets are unavailable."""
        if self.ticket_store is None:
            return 0
        ticket = self.ticket_store.active_ticket(chat_id)
        return ticket.id if ticket is not None else 0

    @override
    def finalize_ticket(
        self, target: RunTarget | None, outcome: str, reply: str = ""
    ) -> None:
        """Apply staged ticket values after a confirmed assistant delivery."""
        if target is None or target.ticket_store is None or not target.ticket_id:
            return
        if not run_delivered(target, outcome):
            return
        store = target.ticket_store
        ticket = store.ticket(target.ticket_id)
        if ticket is None:
            return
        staged = target.ticket_update or TicketUpdate()
        if closed_mid_run(ticket, staged):
            logger.info(
                "Ticket {number} closed mid-run; discarding its staged update",
                number=ticket.number,
            )
            return
        staged = apply_split(store, ticket, target.ticket_split, staged)
        apply_delivery(store, ticket, staged, reply or target.sent_text)

    def ticket_paused(self, session: str, inbound: Inbound) -> bool:
        """Whether the customer chat or the whole session is paused."""
        store = self.ticket_store
        if store is None:
            return False
        return is_paused(store, session, inbound, int(time.time()))

    @override
    def pause_after_human_message(self, event: WahaEvent, config: SessionConfig) -> None:
        """Pause a customer chat after a human operator typed there."""
        if self.ticket_store is None or not event.payload.get("fromMe"):
            return
        if str(event.payload.get("source", "")) != "app":
            return
        chat = str(event.payload.get("from", ""))
        if not chat or chat.endswith("@g.us"):
            return
        if jid_string(chat) in bot_jids(event):
            return
        hours = config.pause_after_human_hours
        if hours <= 0:
            return
        until = int(time.time()) + hours * 3600
        contact = self.ticket_store.contact(chat)
        if contact is not None:
            self.ticket_store.set_contact_paused(contact.chat_id, until)
        self.ticket_store.set_chat_paused(chat, until)

    @override
    async def handle_ticket_route(
        self, event: WahaEvent, route: Route, config: SessionConfig
    ) -> bool:
        """Handle one parsed self-chat command; always answers the operator."""
        if self.ticket_store is None:
            return False
        if route.kind in (RouteKind.QUOTE_REPLY, RouteKind.TICKET_REPLY):
            await self.ticket_reply_route(event, route, config)
            return True
        if route.kind == RouteKind.WHATS_OPEN:
            await self.ticket_send_operator(event, whats_open_text(self.ticket_store))
            return True
        if route.kind == RouteKind.SAVE:
            await self.ticket_save_answer(event, route)
            return True
        if route.kind in (RouteKind.CLOSE, RouteKind.REOPEN):
            await self.ticket_state_route(event, route)
            return True
        if route.kind in (RouteKind.PAUSE, RouteKind.RESUME):
            await self.ticket_pause_route(event, route, config)
            return True
        if route.kind == RouteKind.ASK:
            text = route.text or ask_which_ticket(route.candidates)
            await self.ticket_send_operator(event, text)
            return True
        return False

    async def ticket_save_answer(self, event: WahaEvent, route: Route) -> None:
        """Save the newest answer candidate, or say there is none."""
        service = self.answer_service
        if service is None:
            await self.ticket_send_operator(event, "Teaching is not available.")
            return
        ticket_id = route.ticket.id if route.ticket is not None else None
        saved = service.save_latest_candidate(ticket_id)
        if saved is None:
            where = f" for ticket {route.ticket.number}" if route.ticket else ""
            await self.ticket_send_operator(event, f"There is nothing to save{where}.")
            return
        await self.ticket_send_operator(
            event, f"Saved answer {saved.id}: {saved.question}"
        )

    async def ticket_reply_route(
        self, event: WahaEvent, route: Route, config: SessionConfig
    ) -> None:
        """Run the classify/draft/verify/send loop for one operator reply."""
        store = self.ticket_store
        if store is None:
            return
        ticket = store.ticket(route.ticket.id) if route.ticket is not None else None
        if ticket is None:
            await self.ticket_send_operator(event, "No matching ticket found.")
            return
        if ticket.state == TicketState.CLOSED.value:
            await self.ticket_send_operator(
                event, f"Ticket {ticket.number} is closed; reopen it first."
            )
            return
        contact = store.contact_by_id(ticket.contact_id)
        if contact is None:
            await self.ticket_send_operator(event, "The ticket has no contact record.")
            return
        async with chat_lock(event.session, ticket.chat_id):
            outcome = await handle_operator_reply(
                store,
                self.waha,
                event.session,
                ticket,
                contact,
                route.text,
                complete=self.ticket_complete,
                reply_window_days=config.reply_window_days,
                remember=self.ticket_remember_relay(ticket),
                answer_service=self.answer_service,
            )
            if outcome.sent:
                await self.ticket_write_fact(event, ticket, route.text, outcome)
        text = await self.ticket_confirmation(outcome, config)
        await self.ticket_send_operator(event, text)

    def ticket_complete(self, prompt: str) -> str:
        """The synchronous model completion the reply loop uses."""
        return str(self.agent.llm.complete(prompt).text)

    async def ticket_write_fact(
        self,
        event: WahaEvent,
        ticket: Ticket,
        operator_text: str,
        outcome: ReplyOutcome,
    ) -> None:
        """Store the operator's decision as a chat-scoped typed fact."""
        if self.agent.notebook is None:
            return
        key = f"ticket-{ticket.number}-decision"
        text = decision_fact(ticket, operator_text, outcome.draft)
        await asyncio.to_thread(
            self.agent.notebook.write,
            event.session,
            ticket.chat_id,
            key,
            text,
            kind=KIND_FACT,
            author_kind=AUTHOR_OPERATOR,
        )

    async def ticket_confirmation(
        self, outcome: ReplyOutcome, config: SessionConfig
    ) -> str:
        """The operator-facing confirmation in the configured language."""
        language = operator_language_for(config.operator_language, outcome.detected)
        return await asyncio.to_thread(
            localized_confirmation, outcome, language, self.ticket_translator
        )

    async def ticket_state_route(self, event: WahaEvent, route: Route) -> None:
        """Close or reopen a numbered ticket."""
        if route.ticket is None:
            await self.ticket_send_operator(event, "No ticket with that number.")
            return
        store = self.ticket_store
        if store is None:
            return
        ok = True
        async with chat_lock(event.session, route.ticket.chat_id):
            try:
                if route.kind == RouteKind.CLOSE:
                    close_ticket(store, route.ticket)
                    message = f"Closed {route.ticket.number}."
                else:
                    reopen_ticket(store, route.ticket)
                    message = f"Reopened {route.ticket.number}."
            except ValueError as exc:
                ok = False
                message = str(exc)
        save_action(
            self.settings.data_dir,
            event.session,
            "ticket_command",
            ticket=route.ticket.number,
            command=route.kind.value,
            ok=ok,
        )
        await self.ticket_send_operator(event, message)

    async def ticket_pause_route(
        self, event: WahaEvent, route: Route, config: SessionConfig
    ) -> None:
        """Pause or resume one contact, or the whole session."""
        store = self.ticket_store
        if store is None:
            return
        if route.contact is not None:
            if route.kind == RouteKind.PAUSE:
                if config.pause_after_human_hours <= 0:
                    message = "Pause is disabled: set pause_after_human_hours above 0."
                else:
                    until = int(time.time()) + config.pause_after_human_hours * 3600
                    store.set_contact_paused(route.contact.chat_id, until)
                    message = f"Paused {route.contact.display}."
            else:
                store.set_contact_paused(route.contact.chat_id, None)
                for alias in (
                    route.contact.chat_id,
                    route.contact.phone,
                    route.contact.lid,
                ):
                    if alias:
                        store.set_chat_paused(alias, None)
                message = f"Resumed {route.contact.display}."
        else:
            until = GLOBAL_PAUSE_FOREVER if route.kind == RouteKind.PAUSE else None
            store.set_session_paused(event.session, until)
            message = (
                "Paused all customer chats." if until else "Resumed all customer chats."
            )
        save_action(
            self.settings.data_dir,
            event.session,
            "ticket_command",
            command=route.kind.value,
        )
        await self.ticket_send_operator(event, message)

    async def ticket_send_operator(self, event: WahaEvent, text: str) -> None:
        """Send a command answer back to the operator's self-chat."""
        command = WahaEvent.model_validate(build_command_event(event.session, text))
        command.payload["reply_chat_id"] = str(event.payload.get("from", ""))
        command.payload["reply_to"] = str(event.payload.get("id", ""))
        await send_self_reply(self.waha, command, text)
